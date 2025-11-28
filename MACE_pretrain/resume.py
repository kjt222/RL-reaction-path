"""Resume training from an existing checkpoint without changing hyperparameters."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.serialization
from torch_ema import ExponentialMovingAverage

from mace import tools

from dataloader import prepare_lmdb_dataloaders, prepare_xyz_dataloaders
from metadata import build_metadata, load_checkpoint, save_checkpoint
from models import instantiate_model
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


def main() -> None:
    args = parse_args()

    ckpt_dir = args.checkpoint_dir.expanduser().resolve()
    ckpt_path = ckpt_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint.pt: {ckpt_path}")

    bundle = load_checkpoint(ckpt_path, map_location="cpu")
    train_state = bundle.get("train_state") or {}
    resume_config = train_state.get("config") or bundle["raw"].get("config")
    if resume_config is None:
        raise ValueError("checkpoint 中缺少 config，无法恢复训练超参数。")

    config = dict(resume_config)
    if args.progress is not None:
        config["progress"] = args.progress

    # 构造与训练脚本一致的 Namespace
    cfg_ns = argparse.Namespace(**config)
    cfg_ns.output = args.output or ckpt_dir

    tools.set_seeds(config.get("seed", 42))

    resume_indices = train_state.get("lmdb_indices") or bundle["raw"].get("lmdb_indices")
    if cfg_ns.data_format == "xyz":
        (
            train_loader,
            valid_loader,
            z_table,
            avg_num_neighbors,
            e0_values,
        ) = prepare_xyz_dataloaders(cfg_ns)
        lmdb_indices = None
    elif cfg_ns.data_format == "lmdb":
        (
            train_loader,
            valid_loader,
            z_table,
            avg_num_neighbors,
            e0_values,
            train_indices,
            val_indices,
        ) = prepare_lmdb_dataloaders(cfg_ns, resume_indices=resume_indices)
        lmdb_indices = {"train": train_indices, "val": val_indices}
    else:
        raise ValueError(f"Unsupported data format: {cfg_ns.data_format}")

    resume_metadata = bundle.get("metadata") or {}
    metadata = build_metadata(
        z_table=resume_metadata.get("z_table", z_table),
        avg_num_neighbors=resume_metadata.get("avg_num_neighbors", avg_num_neighbors),
        e0_values=resume_metadata.get("e0_values", e0_values),
        cutoff=resume_metadata.get("cutoff", cfg_ns.cutoff),
        num_interactions=resume_metadata.get("num_interactions", cfg_ns.num_interactions),
        extra={"lmdb_indices": lmdb_indices} if cfg_ns.data_format == "lmdb" else None,
    )

    model = instantiate_model(
        tools.AtomicNumberTable(metadata["z_table"]),
        float(metadata["avg_num_neighbors"]),
        float(metadata["cutoff"]),
        np.asarray(metadata["e0_values"], dtype=float),
        int(metadata["num_interactions"]),
        architecture=metadata.get("architecture"),
    )
    model.load_state_dict(bundle["model_state_dict"])

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
    best_state_dict = (
        bundle.get("best_model_state_dict")
        or (train_state.get("best_model_state_dict") if train_state else None)
    )

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
        metadata=metadata,
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
        save_checkpoint(
            cfg_ns.output / "checkpoint.pt",
            model_state_dict=final_model_state,
            metadata=metadata,
            train_state=final_train_state,
            best_model_state_dict=best_state_dict,
        )
        if best_state_dict is not None:
            torch.save(best_state_dict, cfg_ns.output / "best_model.pt")
        LOGGER.info(
            "Resume 完成，checkpoint 保存在 %s，best 模型保存在 %s，best_val_loss=%.6f",
            cfg_ns.output / "checkpoint.pt",
            cfg_ns.output / "best_model.pt",
            best_val_loss,
        )


if __name__ == "__main__":
    main()
