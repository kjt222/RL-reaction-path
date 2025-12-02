"""Resume training from an existing checkpoint without changing hyperparameters."""

from __future__ import annotations
import sitecustomize  # noqa: F401


import argparse
import logging
from pathlib import Path

import copy
import json
import torch
import torch.serialization
try:  # PyG>=2.3
    from torch_geometric.loader import DataLoader as PYGDataLoader
except ImportError:  # PyG<=2.2
    from torch_geometric.dataloader import DataLoader as PYGDataLoader
from torch_ema import ExponentialMovingAverage

from mace import tools
from mace.tools import torch_geometric as mace_tg

from dataloader import prepare_xyz_dataloaders, prepare_lmdb_dataloaders
from read_model import validate_json_against_checkpoint
from models import build_model_from_json
from train_mace import train

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume training from a checkpoint directory.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing checkpoint.pt and (optionally) best_model.pt/best.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory to write resumed checkpoints (default: checkpoint_dir).",
    )
    parser.add_argument(
        "--model_json",
        type=Path,
        help="Path to model.json (default: checkpoint_dir/model.json).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total number of epochs to train to. Defaults to the value stored in the checkpoint config.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Force enable progress bar (overrides checkpoint config).",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Force disable progress bar (overrides checkpoint config).",
    )
    parser.set_defaults(progress=None)
    return parser.parse_args()


def _load_checkpoint_artifacts(path: Path) -> tuple[dict, torch.nn.Module | None, dict, dict]:
    """加载 checkpoint/best_model，返回 (state_dict, module, train_state, raw_dict)。"""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state_dict: dict | None = None
    module: torch.nn.Module | None = None
    train_state: dict = {}

    if isinstance(obj, dict):
        state_dict = obj.get("model_state_dict") or obj.get("state_dict")
        maybe_module = obj.get("model")
        if isinstance(maybe_module, torch.nn.Module):
            module = maybe_module
        train_state = obj.get("train_state") or {}
        if state_dict is None and isinstance(module, torch.nn.Module):
            state_dict = module.state_dict()
    elif isinstance(obj, torch.nn.Module):
        module = obj
        state_dict = obj.state_dict()
    else:
        raise ValueError(f"Unsupported checkpoint object type: {type(obj)} from {path}")

    if state_dict is None:
        raise ValueError(f"No state_dict found in checkpoint: {path}")
    return state_dict, module, train_state, obj if isinstance(obj, dict) else {}


def _build_model_with_json(
    json_path: Path,
    checkpoint_path: Path,
    state_dict: dict,
    module_fallback: torch.nn.Module | None,
) -> torch.nn.Module:
    """优先用 model.json + state_dict 构建，失败则回退到 checkpoint 内置 module。"""
    with json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    ok, diffs = validate_json_against_checkpoint(json_path, checkpoint_path)
    if not ok:
        raise ValueError(f"model.json 与 checkpoint 不一致: {diffs}")

    model: torch.nn.Module | None = None
    build_error: Exception | None = None
    try:
        model = build_model_from_json(json_meta)
        try:
            model.load_state_dict(state_dict, strict=True)
            LOGGER.info("严格按 model.json 加载权重成功。")
        except RuntimeError as e:
            LOGGER.warning("strict 加载失败，改为非严格加载：%s", e)
            model.load_state_dict(state_dict, strict=False)
    except Exception as e:  # pragma: no cover - best effort
        build_error = e
        model = None
        LOGGER.error("基于 model.json 构建/加载模型失败：%s", e)

    if model is None:
        if module_fallback is not None:
            LOGGER.warning("回退到 checkpoint 内的 nn.Module。")
            model = module_fallback
        else:
            raise ValueError(f"无法从 JSON 构建模型，且无可用回退模块：{build_error}") from build_error

    return model


def _build_lmdb_loaders_from_json(args, json_meta: dict, resume_indices=None):
    required = ("z_table", "cutoff", "e0_values", "avg_num_neighbors")
    missing = [k for k in required if k not in json_meta]
    if missing:
        raise ValueError(f"model.json 缺少字段: {missing}")
    return prepare_lmdb_dataloaders(
        args,
        resume_indices=resume_indices,
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

    ckpt_state_dict, ckpt_module, train_state, ckpt_raw = _load_checkpoint_artifacts(ckpt_path)
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
        (
            train_loader,
            valid_loader,
            z_table,
            avg_num_neighbors,
            e0_values,
            train_indices,
            val_indices,
        ) = _build_lmdb_loaders_from_json(cfg_ns, json.load(model_json_path.open("r", encoding="utf-8")), resume_indices=resume_indices)
        lmdb_indices = {"train": train_indices, "val": val_indices}
    else:
        raise ValueError(f"Unsupported data format: {cfg_ns.data_format}")

    # 权重与模型：优先使用 checkpoint 的最新状态
    module_fallback = ckpt_module
    best_model_path = None
    for candidate in ["best_model.pt", "best.pt"]:
        path = ckpt_dir / candidate
        if path.exists():
            best_model_path = path
            break
    best_state_dict = None
    if best_model_path is not None:
        try:
            best_state_dict, best_module, _, _ = _load_checkpoint_artifacts(best_model_path)
            module_fallback = best_module or module_fallback
        except Exception as e:
            LOGGER.warning("读取 best_model 失败，将忽略：%s", e)
    if best_state_dict is None:
        best_state_dict = (ckpt_raw.get("best_model_state_dict") if isinstance(ckpt_raw, dict) else None) or train_state.get("best_model_state_dict")

    model = _build_model_with_json(model_json_path, ckpt_path, ckpt_state_dict, module_fallback)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_ns.lr,
        weight_decay=cfg_ns.weight_decay,
    )
    if train_state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(train_state["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=50,
    )
    if train_state.get("scheduler_state_dict") is not None:
        scheduler_state = train_state["scheduler_state_dict"]
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

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
        scheduler=scheduler,
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
        best_epoch=start_epoch - 1,
    )

    if cfg_ns.output:
        cfg_ns.output.mkdir(parents=True, exist_ok=True)
        final_train_state = last_ckpt.get("train_state") or {}
        final_model_state = last_ckpt.get("model_state_dict") or {k: v.cpu() for k, v in model.state_dict().items()}
        final_train_state.setdefault("epoch", total_epochs)
        final_train_state.setdefault("config", config)
        final_train_state.setdefault("lmdb_indices", lmdb_indices)
        module_copy = copy.deepcopy(model).cpu()
        torch.save(
            {
                "model_state_dict": final_model_state,
                "train_state": final_train_state,
                "best_model_state_dict": best_state_dict,
                "model": module_copy,
            },
            cfg_ns.output / "checkpoint.pt",
        )
        best_model_copy = copy.deepcopy(model).cpu()
        best_model_copy.load_state_dict(best_state_dict if best_state_dict is not None else final_model_state, strict=False)
        torch.save(
            {
                "model_state_dict": best_state_dict if best_state_dict is not None else final_model_state,
                "model": best_model_copy,
            },
            cfg_ns.output / "best_model.pt",
        )
        LOGGER.info(
            "Resume 完成，checkpoint 保存在 %s，best 模型保存在 %s，best_val_loss=%.6f",
            cfg_ns.output / "checkpoint.pt",
            cfg_ns.output / "best_model.pt",
            best_val_loss,
        )


if __name__ == "__main__":
    main()
