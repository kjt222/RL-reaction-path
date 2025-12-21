"""Resume training from an existing checkpoint file without changing hyperparameters."""

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
    canonical_json_text,
    hash_text,
    get_code_version,
    derive_run_name_from_checkpoint,
    resolve_input_json_path,
    resolve_output_json_paths,
    resolve_output_paths,
)

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume training from a checkpoint file.")
    parser.add_argument("--input_model", type=Path, required=True, help="Checkpoint path (.pt file).")
    parser.add_argument("--input_json", type=Path, help="model.json path (default: derived from input_model).")
    parser.add_argument("--output_checkpoint", type=Path, required=True, help="Checkpoint output path (.pt or directory).")
    parser.add_argument("--output_model", type=Path, required=True, help="Best model output path (.pt or directory).")
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

    model_path = args.input_model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {model_path}")
    ckpt_path = model_path
    code_version = get_code_version()

    output_checkpoint = args.output_checkpoint.expanduser()
    output_model = args.output_model.expanduser()
    resolved_ckpt, resolved_model = resolve_output_paths(
        output_checkpoint,
        output_model,
    )
    if resolved_ckpt is None or resolved_model is None:
        raise ValueError("必须提供 output_checkpoint 与 output_model。")

    model_json_path = resolve_input_json_path(model_path, args.input_json)
    output_json_paths = resolve_output_json_paths(output_checkpoint, output_model)

    ckpt_state_dict, ckpt_module, train_state, ckpt_raw = load_checkpoint_artifacts(ckpt_path)
    resume_config = train_state.get("config") or (ckpt_raw.get("raw", {}).get("config") if isinstance(ckpt_raw, dict) else None)
    if resume_config is None:
        raise ValueError("checkpoint 中缺少 config，无法恢复训练超参数。")

    config = dict(resume_config)
    if args.progress is not None:
        config["progress"] = args.progress

    # 构造与训练脚本一致的 Namespace
    cfg_ns = argparse.Namespace(**config)

    tools.set_seeds(config.get("seed", 42))

    resume_indices = train_state.get("lmdb_indices") or (ckpt_raw.get("lmdb_indices") if isinstance(ckpt_raw, dict) else None)
    with model_json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    for output_json in output_json_paths:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(json_meta, f, ensure_ascii=True, indent=2)
    model_json_text = canonical_json_text(json_meta)
    model_json_hash = hash_text(model_json_text)

    if cfg_ns.data_format == "xyz":
        raise ValueError("resume 目前仅支持 LMDB 跳过统计量；XYZ 流程未实现无统计量加载。")
    elif cfg_ns.data_format == "lmdb":
        train_loader, valid_loader, train_indices, val_indices = _build_lmdb_loaders_from_json(
            cfg_ns, json_meta, resume_indices=resume_indices
        )
        lmdb_indices = {"train": train_indices, "val": val_indices}
    else:
        raise ValueError(f"Unsupported data format: {cfg_ns.data_format}")

    # 权重与模型：仅使用指定 checkpoint 的内容（不自动读取同目录 bestmodel）
    module_fallback = ckpt_module
    best_state_dict = None
    best_epoch_saved = train_state.get("best_epoch", 0)

    model, _ = build_model_with_json(
        model_json_path,
        ckpt_path,
        ckpt_state_dict,
        module_fallback,
        checkpoint_obj=ckpt_raw if isinstance(ckpt_raw, dict) else None,
    )

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
    target_epochs = args.epochs if args.epochs is not None else config.get("epochs", start_epoch)
    total_epochs = max(target_epochs, start_epoch)
    config["epochs"] = total_epochs

    best_state_dict, best_val_loss, last_ckpt = train(
        model=model,
        optimizer=optimizer,
        scheduler_step=scheduler_step,
        scheduler_obj=scheduler_obj,
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
        output_checkpoint_path=resolved_ckpt,
        output_model_path=resolved_model,
        config=config,
        lmdb_indices=lmdb_indices,
        model_json_text=model_json_text,
        model_json_hash=model_json_hash,
        code_version=code_version,
        start_epoch=start_epoch,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch_saved if best_epoch_saved is not None else start_epoch - 1,
    )

    if resolved_ckpt is not None or resolved_model is not None:
        final_train_state = last_ckpt.get("train_state") or {}
        final_model_state = last_ckpt.get("model_state_dict") or {k: v.cpu() for k, v in model.state_dict().items()}
        # 统一使用 best_state_dict（train 返回的最佳权重，EMA 优先由 train 内部决定）
        final_best = best_state_dict or final_model_state
        final_train_state.setdefault("epoch", total_epochs)
        final_train_state.setdefault("config", config)
        final_train_state.setdefault("lmdb_indices", lmdb_indices)
        final_train_state["best_epoch"] = last_ckpt.get("best_epoch", best_epoch_saved)
        if resolved_ckpt is not None:
            ckpt_run_name = derive_run_name_from_checkpoint(resolved_ckpt)
            save_checkpoint(
                resolved_ckpt,
                model,
                final_train_state,
                model_state_dict=final_model_state,
                ema_state_dict=final_train_state.get("ema_state_dict"),
                model_json_text=model_json_text,
                model_json_hash=model_json_hash,
                code_version=code_version,
                run_name=ckpt_run_name,
            )
        # bestmodel 只保存一份：使用最终 best_state_dict
        if resolved_model is not None:
            model_run_name = derive_run_name_from_checkpoint(resolved_model)
            save_best_model(
                resolved_model,
                model,
                final_best,
                model_state_dict=final_best,
                model_json_text=model_json_text,
                model_json_hash=model_json_hash,
                code_version=code_version,
                run_name=model_run_name,
            )
        LOGGER.info(
            "Resume 完成，checkpoint 保存在 %s，best 模型保存在 %s，best_val_loss=%.6f",
            resolved_ckpt,
            resolved_model,
            best_val_loss,
        )


if __name__ == "__main__":
    main()
