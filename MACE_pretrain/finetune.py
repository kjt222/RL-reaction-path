"""Finetune a MACE model from a previously trained checkpoint/best model."""

from __future__ import annotations
import sitecustomize  # noqa: F401



import argparse
import contextlib
import copy
import json
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.serialization
try:  # PyG>=2.3
    from torch_geometric.loader import DataLoader as PYGDataLoader
except ImportError:  # PyG<=2.2
    from torch_geometric.dataloader import DataLoader as PYGDataLoader
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from mace import tools
from mace.tools import torch_geometric as mace_tg

from dataloader import prepare_xyz_dataloaders, prepare_lmdb_dataloaders
from read_model import validate_json_against_checkpoint
from models import build_model_from_json

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune a trained MACE model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="起始 checkpoint 路径（默认同目录寻找 model.json / best_model.pt）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="微调输出目录（默认：checkpoint_dir/finetune）",
    )
    parser.add_argument(
        "--model_json",
        type=Path,
        help="model.json 路径（默认使用 checkpoint_dir/model.json）",
    )
    parser.add_argument(
        "--use_best",
        dest="use_best",
        action="store_true",
        help="优先从 best_model.pt 或 best.pt 加载权重（默认开启）",
    )
    parser.add_argument(
        "--no-use_best",
        dest="use_best",
        action="store_false",
        help="仅使用 checkpoint.pt 内的 model_state_dict",
    )
    parser.add_argument(
        "--reuse_optimizer_state",
        action="store_true",
        help="加载旧优化器状态（默认不加载，重新创建并用新 lr）",
    )
    parser.add_argument(
        "--reuse_scheduler_state",
        action="store_true",
        help="加载旧调度器状态（默认不加载，重新创建）",
    )
    parser.add_argument(
        "--reuse_indices",
        action="store_true",
        help="复用 checkpoint 内保存的 lmdb_indices（仅 LMDB）",
    )
    parser.add_argument(
        "--data_format",
        choices=["xyz", "lmdb"],
        help="数据格式，不指定则使用 checkpoint 保存的 config",
    )
    parser.add_argument("--xyz_dir", type=Path, help="XYZ 数据路径")
    parser.add_argument("--lmdb_train", type=Path, help="训练 LMDB 目录")
    parser.add_argument("--lmdb_val", type=Path, help="验证 LMDB 目录")
    parser.add_argument("--lmdb_train_max_samples", type=int, default=None)
    parser.add_argument("--lmdb_val_max_samples", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--train_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（<=0 禁用），默认 1.0。",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cutoff", type=float, help="默认沿用 checkpoint 元数据")
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--force_weight", type=float, default=1000.0)
    parser.add_argument("--lr", type=float, default=1.0e-4, help="微调学习率")
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--num_interactions", type=int, help="默认沿用 checkpoint 元数据")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--ema", dest="ema", action="store_true", help="启用 EMA（默认）")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="禁用 EMA")
    parser.add_argument("--elements", type=int, nargs="+", help="可选元素列表（LMDB）")
    parser.add_argument("--progress", dest="progress", action="store_true", help="显示进度条（默认）")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="关闭进度条")
    parser.add_argument(
        "--plateau_patience",
        type=int,
        default=4,
        help="ReduceLROnPlateau 的 patience（默认 4）。",
    )
    parser.add_argument(
        "--early_stop_factor",
        type=int,
        default=5,
        help="early-stop 窗口 = 调度器 patience * early_stop_factor，设 0 关闭",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="每 N 个 epoch 保存 checkpoint，0 关闭周期保存",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(use_best=True, ema=True, progress=True)
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
) -> tuple[torch.nn.Module, dict]:
    """基于 model.json + state_dict 构建模型；JSON 不一致会报错，构建失败回退 checkpoint module。"""
    with json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    ok, diffs = validate_json_against_checkpoint(json_path, checkpoint_path)
    if not ok:
        non_e0 = [d for d in diffs if "e0_values" not in d]
        if non_e0:
            raise ValueError(f"model.json 与 checkpoint 不一致: {diffs}")
        LOGGER.warning("model.json 与 checkpoint 仅 E0 不一致，将继续（已记录差异）：%s", diffs)

    model: torch.nn.Module | None = None
    build_error: Exception | None = None
    try:
        model = build_model_from_json(json_meta)
        model.load_state_dict(state_dict, strict=True)
        LOGGER.info("严格按 model.json 加载权重成功。")
    except Exception as exc:
        build_error = exc
        LOGGER.error("严格按 model.json 构建模型失败：%s", exc)

    if model is None:
        if module_fallback is None:
            raise ValueError(
                "无法基于 model.json 构建模型，且 checkpoint 中无 nn.Module 回退。"
            ) from build_error
        if not hasattr(module_fallback, "avg_num_neighbors"):
            raise ValueError("回退模块缺少 avg_num_neighbors，无法用于训练。")
        LOGGER.warning("回退到 checkpoint 内的 nn.Module（假设包含统计量）。")
        model = module_fallback

    return model, json_meta




def compute_losses(
    outputs: dict,
    batch,
    energy_weight: float,
    force_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_energy = outputs["energy"].squeeze(-1)
    true_energy = batch.energy.squeeze(-1)
    energy_loss = F.mse_loss(pred_energy, true_energy)

    pred_forces = outputs["forces"]
    true_forces = batch.forces
    force_loss = F.mse_loss(pred_forces, true_forces)

    total_loss = energy_weight * energy_loss + force_weight * force_loss
    return total_loss, energy_loss, force_loss


def evaluate(
    model,
    loader,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_energy_rmse = 0.0
    total_force_rmse = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        # Force computation relies on autograd, so keep gradients enabled.
        outputs = model(batch.to_dict(), training=False, compute_force=True)
        loss, energy_loss, force_loss = compute_losses(
            outputs, batch, energy_weight, force_weight
        )

        batch_size = batch.energy.shape[0]
        total_loss += loss.item() * batch_size
        total_energy_rmse += torch.sqrt(energy_loss).item() * batch_size
        total_force_rmse += torch.sqrt(force_loss).item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / total_samples,
        total_energy_rmse / total_samples,
        total_force_rmse / total_samples,
    )


def train(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader,
    valid_loader,
    device: torch.device,
    epochs: int,
    energy_weight: float,
    force_weight: float,
    ema: ExponentialMovingAverage | None,
    show_progress: bool,
    early_stop_factor: int,
    save_every: int,
    save_dir: Path | None,
    config: dict | None,
    lmdb_indices: dict | None = None,
    clip_grad_norm: float | None = None,
    start_epoch: int = 1,
    best_state_dict: dict | None = None,
    best_val_loss: float = float("inf"),
    best_epoch: int = 0,
):
    last_state_dict: dict | None = None
    latest_train_state: dict | None = None
    last_epoch = start_epoch - 1

    scheduler_patience = getattr(scheduler, "patience", None)
    if scheduler_patience is not None and early_stop_factor > 0:
        early_stop_window = scheduler_patience * early_stop_factor
    else:
        early_stop_window = None

    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        iterator = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            if show_progress
            else train_loader
        )
        for batch in iterator:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.to_dict(), training=True, compute_force=True)
            loss, _, _ = compute_losses(outputs, batch, energy_weight, force_weight)
            loss.backward()
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update()

            total_train_loss += loss.item()
            total_batches += 1

        avg_train_loss = total_train_loss / max(total_batches, 1)

        context = ema.average_parameters() if ema is not None else contextlib.nullcontext()
        with context:
            with torch.enable_grad():
                val_loss, val_energy_rmse, val_force_rmse = evaluate(
                    model,
                    valid_loader,
                    device,
                    energy_weight,
                    force_weight,
                )

        scheduler.step(val_loss)
        LOGGER.info(
            "Epoch %4d | Train Loss %.6f | Val Loss %.6f | Val RMSE (E %.6f, F %.6f) | LR %.6e",
            epoch,
            avg_train_loss,
            val_loss,
            val_energy_rmse,
            val_force_rmse,
            optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with ema.average_parameters() if ema is not None else torch.no_grad():
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
        last_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        train_state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "lmdb_indices": lmdb_indices,
        }
        latest_train_state = train_state

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            if val_loss <= best_val_loss and best_state_dict is not None:
                best_path = save_dir / "best_model.pt"
                best_model_copy = copy.deepcopy(model).cpu()
                best_model_copy.load_state_dict(best_state_dict, strict=False)
                torch.save(
                    {
                        "model_state_dict": best_state_dict,
                        "model": best_model_copy,
                    },
                    best_path,
                )
                LOGGER.info("Saved new best model to %s", best_path)
            if save_every > 0 and epoch % save_every == 0:
                ckpt_path = save_dir / "checkpoint.pt"
                module_copy = copy.deepcopy(model).cpu()
                torch.save(
                    {
                        "model_state_dict": last_state_dict,
                        "train_state": train_state,
                        "best_model_state_dict": best_state_dict,
                        "model": module_copy,
                    },
                    ckpt_path,
                )
                LOGGER.info("Saved checkpoint at epoch %d to %s", epoch, ckpt_path)

        if early_stop_window is not None:
            epochs_since_best = epoch - best_epoch
            if epochs_since_best >= early_stop_window:
                LOGGER.info(
                    "Early stopping triggered after %d epochs without val improvement "
                    "(best epoch %d, window %d).",
                    epochs_since_best,
                    best_epoch,
                    early_stop_window,
                )
                break

    if best_state_dict is None and last_state_dict is not None:
        best_state_dict = last_state_dict
    if last_state_dict is None:
        last_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    if latest_train_state is None:
        latest_train_state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "epoch": last_epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "lmdb_indices": lmdb_indices,
        }

    return best_state_dict, best_val_loss, {
        "model_state_dict": last_state_dict,
        "best_model_state_dict": best_state_dict,
        "best_val_loss": best_val_loss,
        "train_state": latest_train_state,
    }


def _resolve_paths_from_config(args: argparse.Namespace, config: dict | None) -> None:
    if config is None:
        return
    if args.data_format is None and "data_format" in config:
        args.data_format = config["data_format"]
    if args.xyz_dir is None and config.get("xyz_dir"):
        args.xyz_dir = Path(config["xyz_dir"])
    if args.lmdb_train is None and config.get("lmdb_train"):
        args.lmdb_train = Path(config["lmdb_train"])
    if args.lmdb_val is None and config.get("lmdb_val"):
        args.lmdb_val = Path(config["lmdb_val"])
    if args.lmdb_train_max_samples is None and config.get("lmdb_train_max_samples") is not None:
        args.lmdb_train_max_samples = config["lmdb_train_max_samples"]
    if args.lmdb_val_max_samples is None and config.get("lmdb_val_max_samples") is not None:
        args.lmdb_val_max_samples = config["lmdb_val_max_samples"]
    if args.batch_size == parser_defaults()["batch_size"] and "batch_size" in config:
        args.batch_size = config["batch_size"]
    if args.num_workers == parser_defaults()["num_workers"] and "num_workers" in config:
        args.num_workers = config["num_workers"]


def parser_defaults() -> dict:
    return {
        "batch_size": 16,
        "num_workers": 0,
    }


def main() -> None:
    args = parse_args()
    tools.set_seeds(args.seed)

    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    base_dir = ckpt_path.parent
    if args.output is None:
        args.output = base_dir / "finetune"

    model_json_path = args.model_json or (base_dir / "model.json")
    if not model_json_path.exists():
        raise FileNotFoundError(f"未找到 model.json: {model_json_path}")

    ckpt_state_dict, ckpt_module, ckpt_train_state, ckpt_raw = _load_checkpoint_artifacts(ckpt_path)
    raw_config = ckpt_train_state.get("config") or (ckpt_raw.get("raw", {}).get("config") if isinstance(ckpt_raw, dict) else None)
    _resolve_paths_from_config(args, raw_config)

    # 如未显式提供，尽量用 model.json 填充关键超参
    with model_json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    if args.cutoff is None and "cutoff" in json_meta:
        args.cutoff = float(json_meta["cutoff"])
    if args.num_interactions is None and "num_interactions" in json_meta:
        args.num_interactions = int(json_meta["num_interactions"])
    model_zs = [int(z) for z in json_meta.get("z_table", [])]

    # 数据加载：完全信任 model.json，不再重新计算 E0/avg_num_neighbors
    resume_indices = ckpt_train_state.get("lmdb_indices") if args.reuse_indices else None
    lmdb_indices = None
    if args.data_format is None:
        raise ValueError("data_format 未指定，且 checkpoint config 中缺失。")

    if args.data_format == "xyz":
        raise ValueError("finetune 目前仅在 LMDB 流程下跳过统计计算；如需 XYZ 微调请先实现无统计量加载。")
    elif args.data_format == "lmdb":
        if args.lmdb_train is None or args.lmdb_val is None:
            raise ValueError("--lmdb_train/--lmdb_val 必须提供或在 checkpoint config 中存在")
        (
            train_loader,
            valid_loader,
            z_table,
            avg_num_neighbors,
            e0_values,
            train_indices,
            val_indices,
        ) = prepare_lmdb_dataloaders(
            args,
            resume_indices=resume_indices,
            z_table_override=model_zs if model_zs else None,
        )
        lmdb_indices = {"train": train_indices, "val": val_indices}
    else:
        raise ValueError(f"Unsupported data format: {args.data_format}")

    # 权重来源：仅使用指定的 checkpoint
    module_fallback = ckpt_module
    state_dict_for_load = ckpt_state_dict
    load_path = ckpt_path
    LOGGER.info("加载指定 checkpoint 中的模型权重: %s", ckpt_path)
    best_state_dict = (
        (ckpt_raw.get("best_model_state_dict") if isinstance(ckpt_raw, dict) else None)
        or ckpt_train_state.get("best_model_state_dict")
    )

    model, _ = _build_model_with_json(
        model_json_path,
        load_path,
        state_dict_for_load,
        module_fallback,
    )

    # 权重：优先 best_model.pt / best.pt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 AdamW 以改进权重衰减表现
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.reuse_optimizer_state and ckpt_train_state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt_train_state["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        LOGGER.info("已加载优化器状态，并将学习率覆盖为 %.3e", args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=args.plateau_patience,
    )
    if args.reuse_scheduler_state and ckpt_train_state.get("scheduler_state_dict") is not None:
        scheduler_state = ckpt_train_state["scheduler_state_dict"]
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            LOGGER.info("已加载调度器状态。")

    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay) if args.ema else None
    if ema is not None and args.reuse_optimizer_state:
        # 如果需要，可以在 future 扩展成单独开关；当前仅当旧状态存在时加载
        if ckpt_train_state.get("ema_state_dict") is not None:
            ema.load_state_dict(ckpt_train_state["ema_state_dict"])
            LOGGER.info("已加载 EMA 状态。")
    elif ema is not None:
        LOGGER.info("EMA 启用，但不加载旧状态。")
    else:
        LOGGER.info("EMA 关闭。")

    best_val_loss = float(ckpt_train_state.get("best_val_loss", float("inf")))

    best_state_dict, best_val_loss, last_ckpt = train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=args.epochs,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        ema=ema,
        show_progress=args.progress,
        early_stop_factor=args.early_stop_factor,
        save_every=args.save_every,
        save_dir=args.output,
        config=vars(args),
        lmdb_indices=lmdb_indices,
        clip_grad_norm=args.clip_grad_norm,
        start_epoch=1,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        best_epoch=0,
    )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        final_train_state = last_ckpt.get("train_state") or {}
        final_train_state.setdefault("config", vars(args))
        final_train_state.setdefault("lmdb_indices", lmdb_indices)
        final_train_state.setdefault("epoch", args.epochs)
        final_model_state = last_ckpt.get("model_state_dict") or {k: v.cpu() for k, v in model.state_dict().items()}

        module_copy = copy.deepcopy(model).cpu()
        torch.save(
            {
                "model_state_dict": final_model_state,
                "train_state": final_train_state,
                "best_model_state_dict": best_state_dict,
                "model": module_copy,
            },
            args.output / "checkpoint.pt",
        )
        best_model_copy = copy.deepcopy(model).cpu()
        best_model_copy.load_state_dict(best_state_dict if best_state_dict is not None else final_model_state, strict=False)
        torch.save(
            {
                "model_state_dict": best_state_dict if best_state_dict is not None else final_model_state,
                "model": best_model_copy,
            },
            args.output / "best_model.pt",
        )
        LOGGER.info(
            "Finetune 完成，checkpoint 保存在 %s，best 模型保存在 %s，best_val_loss=%.6f",
            args.output / "checkpoint.pt",
            args.output / "best_model.pt",
            best_val_loss,
        )


if __name__ == "__main__":
    main()
