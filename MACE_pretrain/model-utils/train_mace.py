"""Entry point for pretraining MACE models on multiple dataset formats."""

from __future__ import annotations
import sitecustomize  # noqa: F401

import argparse
import contextlib
import copy
import logging
import types
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.serialization
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

# Ensure project root (MACE_pretrain) is on sys.path to import optimizer/dataloader from root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mace import tools

from dataloader import prepare_lmdb_dataloaders, prepare_xyz_dataloaders
from models import build_model_from_json
from read_model import _export_model_json, _diff_json
from losses import compute_train_loss, init_metrics_state, accumulate_metrics, finalize_metrics
from model_loader import save_checkpoint, save_best_model
from optimizer import build_optimizer, build_scheduler

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MACE models")
    parser.add_argument("--data_format", choices=["xyz", "lmdb"], default="xyz", help="Input data format. Currently 'lmdb' is a placeholder.")
    parser.add_argument("--xyz_dir", type=Path, help="Path to a single .xyz file or directory containing .xyz files.")
    parser.add_argument("--lmdb_train", type=Path, help="Path to the training LMDB directory (future use).")
    parser.add_argument("--lmdb_val", type=Path, help="Path to the validation LMDB directory (future use).")
    parser.add_argument("--lmdb_train_max_samples", type=int, default=None, help="Optional limit on number of LMDB samples used for training (random subset).")
    parser.add_argument("--lmdb_val_max_samples", type=int, default=None, help="Optional limit on number of LMDB samples used for validation.")
    parser.add_argument("--output", type=Path, default=Path("mace_pretrain.pt"), help="Output directory for the trained model checkpoint.")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of configurations to reservoir-sample from xyz files.")
    parser.add_argument("--train_size", type=int, default=450, help="Number of configurations used for training (remainder used for validation).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling, shuffling and training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size for both training and validation loaders.")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Radial cutoff (Å) used to build neighborhoods.")
    parser.add_argument("--energy_weight", type=float, default=1.0, help="Weight applied to the energy MSE term.")
    parser.add_argument("--force_weight", type=float, default=1000.0, help="Weight applied to the force MSE term.")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Initial learning rate for Adam optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-6, help="Weight decay (L2 regularisation) for the optimizer.")
    parser.add_argument("--num_interactions", type=int, default=3, help="Number of message-passing interaction blocks in the MACE model.")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="Exponential moving average decay (ignored if EMA disabled).")
    parser.add_argument("--ema", dest="ema", action="store_true", help="Enable EMA smoothing of model weights (default).")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="Disable EMA smoothing of model weights.")
    parser.add_argument("--neighbor_sample_size", type=int, default=1024, help="Number of samples to estimate average neighbors for LMDB data.")
    parser.add_argument("--lmdb_e0_samples", type=int, default=2000, help="Number of LMDB entries to sample for E0 estimation and element detection.")
    parser.add_argument("--elements", type=int, nargs="+", help="Optional explicit list of atomic numbers for LMDB datasets.")
    parser.add_argument("--progress", dest="progress", action="store_true", help="Show tqdm progress bar during training (default).")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bar during training.")
    parser.add_argument("--early_stop_factor", type=int, default=5, help="Multiplier applied to the scheduler patience to determine the early-stopping window (set 0 to disable).")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs (0 to disable periodic saves).")
    parser.set_defaults(ema=True)
    parser.set_defaults(progress=True)
    return parser.parse_args()


def evaluate(
    model,
    loader,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
) -> dict[str, float]:
    """Return global per-atom / per-component metrics for validation."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics_state = init_metrics_state()

    for batch in loader:
        batch = batch.to(device)
        # Force computation relies on autograd, so keep gradients enabled.
        with torch.set_grad_enabled(True):
            outputs = model(batch.to_dict(), training=False, compute_force=True)
        # DEBUG: 打印哪些键为 None，避免 detach 崩溃
        print("\n[DEBUG] Checking outputs keys:")
        for key, value in outputs.items():
            if value is None:
                print(f"  -> ⚠️ 发现 None 值！Key: '{key}'")
            else:
                shape_info = list(value.shape) if hasattr(value, "shape") else "Scalar"
                print(f"  -> ✅ Key: '{key}', Shape: {shape_info}")
        # 评估阶段不需要反传，立刻切断图以节省显存；过滤掉 None 防止报错
        outputs = {k: v.detach() for k, v in outputs.items() if v is not None}
        loss = compute_train_loss(
            outputs,
            batch,
            types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight),
        )

        batch_size = batch.energy.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        accumulate_metrics(metrics_state, outputs, batch, cfg=types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight))

    if total_samples == 0:
        return {}

    finalized = finalize_metrics(metrics_state, energy_weight=energy_weight, force_weight=force_weight)
    finalized["loss"] = total_loss / total_samples
    return finalized


def train(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
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
    start_epoch: int = 1,
    best_state_dict: dict | None = None,
    best_val_loss: float = float("inf"),
    best_epoch: int = 0,
) -> Tuple[dict, float, dict]:
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
        train_metrics_state = init_metrics_state()
        train_loss_sum = 0.0
        train_sample_count = 0

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
            batch_size = batch.energy.shape[0]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.to_dict(), training=True, compute_force=True)
            loss = compute_train_loss(
                outputs,
                batch,
                types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight),
            )
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update()

            accumulate_metrics(train_metrics_state, outputs, batch, cfg=types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight))
            train_loss_sum += loss.item() * batch_size
            train_sample_count += batch_size

        finalized_train = finalize_metrics(train_metrics_state, energy_weight=energy_weight, force_weight=force_weight)
        avg_train_loss = train_loss_sum / train_sample_count if train_sample_count > 0 else 0.0

        context = ema.average_parameters() if ema is not None else contextlib.nullcontext()
        with context:
            with torch.enable_grad():
                val_metrics = evaluate(
                    model,
                    valid_loader,
                    device,
                    energy_weight,
                    force_weight,
                )

        val_loss = float(val_metrics.get("loss", 0.0))
        val_energy_rmse = float(val_metrics.get("energy_rmse", 0.0))
        val_force_rmse = float(val_metrics.get("force_rmse", 0.0))
        val_energy_mae = float(val_metrics.get("energy_mae", 0.0))
        val_force_mae = float(val_metrics.get("force_mae", 0.0))

        scheduler(val_loss)
        LOGGER.info(
            "Epoch %4d | Train Loss %.6f | Val Loss %.6f | Val RMSE (E/atom %.6f, F/comp %.6f) "
            "| Val MAE (E/atom %.6f, F/comp %.6f) | LR %.6e",
            epoch,
            avg_train_loss,
            val_loss,
            val_energy_rmse,
            val_force_rmse,
            val_energy_mae,
            val_force_mae,
            optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_snapshot = {k: v.cpu() for k, v in model.state_dict().items()}
            if ema is not None:
                with ema.average_parameters():
                    best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                best_state_dict = raw_snapshot
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
                save_best_model(best_path, model, best_state_dict, model_state_dict=best_state_dict)
                LOGGER.info("Saved new best model to %s", best_path)
            if save_every > 0 and save_dir is not None and epoch % save_every == 0:
                ckpt_path = save_dir / "checkpoint.pt"
                save_checkpoint(
                    ckpt_path,
                    model,
                    train_state,
                    model_state_dict=last_state_dict,
                    ema_state_dict=train_state.get("ema_state_dict"),
                    best_state_dict=best_state_dict,
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


def main() -> None:
    args = parse_args()
    tools.set_seeds(args.seed)

    import json
    import tempfile

    # 1) 强制依赖 model.json，缺统计量直接报错
    model_json = args.output / "model.json"
    if not model_json.exists():
        raise FileNotFoundError(f"model.json not found: {model_json}（train 需要完整架构与统计量）")
    with model_json.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    required_fields = [
        "z_table",
        "e0_values",
        "avg_num_neighbors",
        "cutoff",
        "num_interactions",
        "hidden_irreps",
        "num_radial_basis",
        "num_polynomial_cutoff",
    ]
    missing = [k for k in required_fields if k not in json_meta]
    if missing:
        raise ValueError(f"model.json 缺少必要字段: {missing}")

    # 用 JSON 构建模型
    model = build_model_from_json(json_meta)

    # 2) 用 read_model 的导出逻辑对比 JSON 与模型结构/统计量，严格一致（含 e0_values）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = Path(tmp.name)
    try:
        _export_model_json(model, tmp_path)
        with tmp_path.open("r", encoding="utf-8") as f:
            exported = json.load(f)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    # 若导出的 JSON 缺少显式字段（如 MLP_irreps/gate/max_ell 等），用原始 json_meta 补齐再比对，避免因模型未暴露属性导致的假冲突。
    for key in [
        "hidden_irreps",
        "MLP_irreps",
        "max_ell",
        "correlation",
        "gate",
        "radial_type",
        "num_radial_basis",
        "num_polynomial_cutoff",
        "interactions",
        "products",
        "readouts",
        "scale_shift",
    ]:
        if key not in exported and key in json_meta:
            exported[key] = json_meta[key]
    diffs = _diff_json(json_meta, exported)
    if diffs:
        raise ValueError(f"model.json 与按 JSON 构建的模型不一致: {diffs}")

    # 3) 从模型提取统计量传给 dataloader
    if hasattr(model, "atomic_numbers"):
        z_table = tools.AtomicNumberTable(sorted({int(z) for z in model.atomic_numbers.view(-1).tolist()}))
    else:
        z_table = tools.AtomicNumberTable([int(z) for z in json_meta["z_table"]])
    args.cutoff = float(json_meta["cutoff"])
    args.z_table = z_table

    if args.data_format == "xyz":
        if args.xyz_dir is None:
            raise ValueError("--xyz_dir must be set when data_format='xyz'")
        (
            train_loader,
            valid_loader,
        ) = prepare_xyz_dataloaders(args)
        train_indices = None
        val_indices = None
    elif args.data_format == "lmdb":
        train_loader, valid_loader, train_indices, val_indices = prepare_lmdb_dataloaders(
            args,
            z_table=z_table,
            resume_indices=None,
            coverage_zs=getattr(args, "elements", None) or z_table.zs,
            seed=args.seed,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported data format: {args.data_format}")

    lmdb_indices = {"train": train_indices, "val": val_indices}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_optimizer(model, args)
    _, scheduler_step = build_scheduler(optimizer, args)

    ema = (
        ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
        if args.ema
        else None
    )
    if ema is not None:
        LOGGER.info("EMA enabled with decay %.4f", args.ema_decay)
    else:
        LOGGER.info("EMA disabled.")

    start_epoch = 1
    best_val_loss = float("inf")
    best_state_dict = None

    best_state_dict, best_val_loss, last_ckpt = train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler_step,
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
        start_epoch=start_epoch,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        best_epoch=start_epoch - 1,
    )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        final_train_state = last_ckpt.get("train_state") or {}
        final_model_state = last_ckpt.get("model_state_dict") or {k: v.cpu() for k, v in model.state_dict().items()}
        final_best = last_ckpt.get("best_model_state_dict") or final_model_state
        final_train_state.setdefault("epoch", args.epochs)
        final_train_state.setdefault("config", vars(args))
        final_train_state.setdefault("lmdb_indices", lmdb_indices)
        ckpt_path = args.output / "checkpoint.pt"
        save_checkpoint(
            ckpt_path,
            model,
            final_train_state,
            model_state_dict=final_model_state,
            ema_state_dict=final_train_state.get("ema_state_dict"),
            best_state_dict=final_best,
        )
        save_best_model(args.output / "best_model.pt", model, final_best, model_state_dict=final_best)
        LOGGER.info(
            "Saved final checkpoint to %s and best model to %s (val_loss %.6f)",
            ckpt_path,
            args.output / "best_model.pt",
            best_val_loss,
        )


if __name__ == "__main__":
    main()
