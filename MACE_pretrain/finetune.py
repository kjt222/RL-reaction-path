"""Finetune a MACE model from a previously trained checkpoint/best model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from mace import tools

from dataloader import prepare_lmdb_dataloaders, prepare_xyz_dataloaders
from metadata import build_metadata, load_checkpoint, save_checkpoint
from models import instantiate_model

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
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="目录包含 checkpoint.pt 与 best_model.pt/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="微调输出目录（默认：checkpoint_dir/finetune）",
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
    parser.add_argument("--neighbor_sample_size", type=int, default=1024)
    parser.add_argument("--lmdb_e0_samples", type=int, default=2000)
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
        with torch.no_grad():
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
    metadata: dict,
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

        context = ema.average_parameters() if ema is not None else torch.no_grad()
        with context:
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
                torch.save(best_state_dict, best_path)
                LOGGER.info("Saved new best model to %s", best_path)
            if save_every > 0 and epoch % save_every == 0:
                ckpt_path = save_dir / "checkpoint.pt"
                save_checkpoint(
                    ckpt_path,
                    model_state_dict=last_state_dict,
                    metadata=metadata,
                    train_state=train_state,
                    best_model_state_dict=best_state_dict,
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

    if args.output is None:
        args.output = args.checkpoint_dir / "finetune"
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()

    ckpt_path = args.checkpoint_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint.pt: {ckpt_path}")

    bundle = load_checkpoint(ckpt_path, map_location="cpu")
    metadata = bundle.get("metadata") or {}
    if not metadata:
        raw = bundle.get("raw", {}) if isinstance(bundle, dict) else {}
        required = ("z_table", "avg_num_neighbors", "e0_values", "cutoff", "num_interactions")
        if all(k in raw for k in required):
            metadata = build_metadata(
                raw["z_table"],
                raw["avg_num_neighbors"],
                raw["e0_values"],
                raw["cutoff"],
                raw["num_interactions"],
                extra={"lmdb_indices": raw.get("lmdb_indices")},
            )
            LOGGER.warning("从旧版 checkpoint 字段恢复了元数据。")
    if not metadata:
        LOGGER.warning("checkpoint 缺少元数据，将根据当前数据重新估计统计量。")
        if args.cutoff is None:
            args.cutoff = 5.0
            LOGGER.warning("未提供 cutoff，使用默认 5.0 Å。")
        if args.num_interactions is None:
            args.num_interactions = 3
            LOGGER.warning("未提供 num_interactions，使用默认 3。")

    train_state = bundle.get("train_state") or {}
    raw_config = train_state.get("config") or bundle["raw"].get("config")
    _resolve_paths_from_config(args, raw_config)

    # 确保与元数据一致的关键超参
    args.cutoff = float(metadata.get("cutoff")) if metadata.get("cutoff") is not None else args.cutoff
    args.num_interactions = (
        int(metadata.get("num_interactions")) if metadata.get("num_interactions") is not None else args.num_interactions
    )

    # 数据加载
    resume_indices = train_state.get("lmdb_indices") if args.reuse_indices else None
    if args.data_format is None:
        raise ValueError("data_format 未指定，且 checkpoint config 中缺失。")

    if args.data_format == "xyz":
        if args.xyz_dir is None:
            raise ValueError("--xyz_dir 必须提供或在 checkpoint config 中存在")
        (
            train_loader,
            valid_loader,
            z_table,
            avg_num_neighbors,
            e0_values,
        ) = prepare_xyz_dataloaders(args)
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
        ) = prepare_lmdb_dataloaders(args, resume_indices=resume_indices)
        lmdb_indices = {"train": train_indices, "val": val_indices}
    else:
        raise ValueError(f"Unsupported data format: {args.data_format}")

    # 使用 checkpoint 元数据，避免与新数据统计不一致
    metadata = build_metadata(
        z_table=metadata.get("z_table", z_table),
        avg_num_neighbors=metadata.get("avg_num_neighbors", avg_num_neighbors),
        e0_values=metadata.get("e0_values", e0_values),
        cutoff=metadata.get("cutoff", args.cutoff),
        num_interactions=metadata.get("num_interactions", args.num_interactions),
        extra={"lmdb_indices": lmdb_indices} if args.data_format == "lmdb" else None,
    )

    model = instantiate_model(
        tools.AtomicNumberTable(metadata["z_table"]),
        float(metadata["avg_num_neighbors"]),
        float(metadata["cutoff"]),
        np.asarray(metadata["e0_values"], dtype=float),
        int(metadata["num_interactions"]),
    )

    # 权重：优先 best_model.pt / best.pt
    best_model_path = None
    for candidate in ["best_model.pt", "best.pt"]:
        path = args.checkpoint_dir / candidate
        if path.exists():
            best_model_path = path
            break

    if args.use_best and best_model_path is not None:
        LOGGER.info("加载最佳模型权重: %s", best_model_path)
        state_dict = torch.load(best_model_path, map_location="cpu")
    else:
        LOGGER.info("加载 checkpoint 中的模型权重: %s", ckpt_path)
        state_dict = bundle["model_state_dict"]

    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 AdamW 以改进权重衰减表现
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.reuse_optimizer_state and train_state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(train_state["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        LOGGER.info("已加载优化器状态，并将学习率覆盖为 %.3e", args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=args.plateau_patience,
    )
    if args.reuse_scheduler_state and train_state.get("scheduler_state_dict") is not None:
        scheduler_state = train_state["scheduler_state_dict"]
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            LOGGER.info("已加载调度器状态。")

    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay) if args.ema else None
    if ema is not None and args.reuse_optimizer_state:
        # 如果需要，可以在 future 扩展成单独开关；当前仅当旧状态存在时加载
        if train_state.get("ema_state_dict") is not None:
            ema.load_state_dict(train_state["ema_state_dict"])
            LOGGER.info("已加载 EMA 状态。")
    elif ema is not None:
        LOGGER.info("EMA 启用，但不加载旧状态。")
    else:
        LOGGER.info("EMA 关闭。")

    best_state_dict = None
    best_val_loss = float("inf")

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
        metadata=metadata,
        config=vars(args),
        lmdb_indices=metadata.get("lmdb_indices"),
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
        final_train_state.setdefault("lmdb_indices", metadata.get("lmdb_indices"))
        save_checkpoint(
            args.output / "checkpoint.pt",
            model_state_dict=last_ckpt.get("model_state_dict"),
            metadata=metadata,
            train_state=final_train_state,
            best_model_state_dict=best_state_dict,
        )
        if best_state_dict is not None:
            torch.save(best_state_dict, args.output / "best_model.pt")
        LOGGER.info(
            "Finetune 完成，checkpoint 保存在 %s，best 模型保存在 %s，best_val_loss=%.6f",
            args.output / "checkpoint.pt",
            args.output / "best_model.pt",
            best_val_loss,
        )


if __name__ == "__main__":
    main()
