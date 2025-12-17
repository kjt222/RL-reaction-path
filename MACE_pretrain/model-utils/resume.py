"""Resume training from an existing checkpoint without changing hyperparameters."""

from __future__ import annotations
import sitecustomize  # noqa: F401


import argparse
import logging
from pathlib import Path

import json
import sys
import torch
import torch.serialization
try:  # PyG>=2.3
    from torch_geometric.loader import DataLoader as PYGDataLoader
except ImportError:  # PyG<=2.2
    from torch_geometric.dataloader import DataLoader as PYGDataLoader
from torch_ema import ExponentialMovingAverage

from mace import tools
from mace.tools import torch_geometric as mace_tg

# Ensure project root (MACE_pretrain) is on sys.path to import optimizer/dataloader from root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import prepare_xyz_dataloaders, prepare_lmdb_dataloaders
from train_mace import train
from optimizer import build_optimizer, build_scheduler, load_optimizer_state, load_scheduler_state
from model_loader import (
    load_checkpoint_artifacts,
    build_model_with_json,
    save_checkpoint,
    save_best_model,
)

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume training from a checkpoint directory.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True, help="Directory containing checkpoint.pt and (optionally) best_model.pt/best.pt.")
    parser.add_argument("--output", type=Path, help="Directory to write resumed checkpoints (default: checkpoint_dir).")
    parser.add_argument("--model_json", type=Path, help="Path to model.json (default: checkpoint_dir/model.json).")
    parser.add_argument("--epochs", type=int, help="Total number of epochs to train to. Defaults to the value stored in the checkpoint config.")
    parser.add_argument("--progress", dest="progress", action="store_true", help="Force enable progress bar (overrides checkpoint config).")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Force disable progress bar (overrides checkpoint config).")
    parser.set_defaults(progress=None)
    return parser.parse_args()


def _build_lmdb_loaders_from_json(args, json_meta: dict, resume_indices=None):
    if "z_table" not in json_meta:
        raise ValueError("model.json 缺少 z_table，无法构建 dataloader。")
    z_table = tools.AtomicNumberTable([int(z) for z in json_meta["z_table"]])
    coverage = getattr(args, "elements", None)
    return prepare_lmdb_dataloaders(
        args,
        z_table=z_table,
        resume_indices=resume_indices,
        coverage_zs=coverage,
        seed=getattr(args, "seed", None),
    )


def main() -> None:
    args = parse_args()

    ckpt_dir = args.checkpoint_dir.expanduser().resolve()
    ckpt_path = ckpt_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint.pt: {ckpt_path}")

    model_json_path = args.model_json or (ckpt_dir / "model.json")
    if not model_json_path.exists():
        raise FileNotFoundError(f"未找到 model.json: {model_json_path}")

    ckpt_state_dict, ckpt_module, train_state, ckpt_raw = load_checkpoint_artifacts(ckpt_path)
    resume_config = train_state.get("config") or (ckpt_raw.get("raw", {}).get("config") if isinstance(ckpt_raw, dict) else None)
    if resume_config is None:
        raise ValueError("checkpoint 中缺少 config，无法恢复训练超参数。")

    config = dict(resume_config)
    if args.progress is not None:
        config["progress"] = args.progress

    # 构造与训练脚本一致的 Namespace
    cfg_ns = argparse.Namespace(**config)
    cfg_ns.output = args.output or ckpt_dir

    tools.set_seeds(config.get("seed", 42))

    resume_indices = train_state.get("lmdb_indices") or (ckpt_raw.get("lmdb_indices") if isinstance(ckpt_raw, dict) else None)
    if cfg_ns.data_format == "xyz":
        raise ValueError("resume 目前仅支持 LMDB 跳过统计量；XYZ 流程未实现无统计量加载。")
    elif cfg_ns.data_format == "lmdb":
        train_loader, valid_loader, train_indices, val_indices = _build_lmdb_loaders_from_json(
            cfg_ns, json.load(model_json_path.open("r", encoding="utf-8")), resume_indices=resume_indices
        )
        lmdb_indices = {"train": train_indices, "val": val_indices}
    else:
        raise ValueError(f"Unsupported data format: {cfg_ns.data_format}")

    # 权重与模型：仅使用指定 checkpoint 的内容（不自动读取同目录 best_model.pt）
    module_fallback = ckpt_module
    best_state_dict = (ckpt_raw.get("best_model_state_dict") if isinstance(ckpt_raw, dict) else None) or train_state.get("best_model_state_dict")
    best_epoch_saved = train_state.get("best_epoch", 0)

    model, _ = build_model_with_json(model_json_path, ckpt_path, ckpt_state_dict, module_fallback)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_optimizer(model, cfg_ns)
    if train_state.get("optimizer_state_dict") is not None:
        load_optimizer_state(optimizer, train_state.get("optimizer_state_dict"), LOGGER)

    scheduler_obj, scheduler_step = build_scheduler(optimizer, cfg_ns)
    if train_state.get("scheduler_state_dict") is not None:
        scheduler_state = train_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            load_scheduler_state(scheduler_obj, scheduler_state, LOGGER)

    ema = ExponentialMovingAverage(model.parameters(), decay=cfg_ns.ema_decay) if cfg_ns.ema else None
    if ema is not None:
        if train_state.get("ema_state_dict") is not None:
            ema.load_state_dict(train_state["ema_state_dict"])
            LOGGER.info("已加载 EMA 状态。")
        else:
            LOGGER.info("EMA 启用，但 checkpoint 未提供 EMA 状态。")
    else:
        LOGGER.info("EMA 关闭。")

    start_epoch = int(train_state.get("epoch", 0)) + 1
    best_val_loss = float(train_state.get("best_val_loss", float("inf")))
    if best_state_dict is None:
        best_state_dict = train_state.get("best_model_state_dict")

    target_epochs = args.epochs if args.epochs is not None else config.get("epochs", start_epoch)
    total_epochs = max(target_epochs, start_epoch)
    config["epochs"] = total_epochs

    best_state_dict, best_val_loss, last_ckpt = train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler_step,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=total_epochs,
        energy_weight=cfg_ns.energy_weight,
        force_weight=cfg_ns.force_weight,
        ema=ema,
        show_progress=cfg_ns.progress,
        early_stop_factor=cfg_ns.early_stop_factor,
        save_every=cfg_ns.save_every,
        save_dir=cfg_ns.output,
        config=config,
        lmdb_indices=lmdb_indices,
        start_epoch=start_epoch,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch_saved if best_epoch_saved is not None else start_epoch - 1,
    )

    if cfg_ns.output:
        cfg_ns.output.mkdir(parents=True, exist_ok=True)
        final_train_state = last_ckpt.get("train_state") or {}
        final_model_state = last_ckpt.get("model_state_dict") or {k: v.cpu() for k, v in model.state_dict().items()}
        # 统一使用 best_state_dict（train 返回的最佳权重，EMA 优先由 train 内部决定）
        final_best = last_ckpt.get("best_model_state_dict") or final_model_state
        final_train_state.setdefault("epoch", total_epochs)
        final_train_state.setdefault("config", config)
        final_train_state.setdefault("lmdb_indices", lmdb_indices)
        final_train_state["best_epoch"] = last_ckpt.get("best_epoch", best_epoch_saved)
        save_checkpoint(
            cfg_ns.output / "checkpoint.pt",
            model,
            final_train_state,
            model_state_dict=final_model_state,
            ema_state_dict=final_train_state.get("ema_state_dict"),
            best_state_dict=final_best,
        )
        # best_model.pt 只保存一份：使用最终 best_state_dict
        save_best_model(cfg_ns.output / "best_model.pt", model, final_best, model_state_dict=final_best)
        LOGGER.info(
            "Resume 完成，checkpoint 保存在 %s，best 模型保存在 %s，best_val_loss=%.6f",
            cfg_ns.output / "checkpoint.pt",
            cfg_ns.output / "best_model.pt",
            best_val_loss,
        )


if __name__ == "__main__":
    main()
