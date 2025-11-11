"""Entry point for pretraining MACE models on multiple dataset formats."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.serialization
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from mace import tools

from dataloader import prepare_lmdb_dataloaders, prepare_xyz_dataloaders
from models import instantiate_model

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float64)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MACE models")
    parser.add_argument(
        "--data_format",
        choices=["xyz", "lmdb"],
        default="xyz",
        help="Input data format. Currently 'lmdb' is a placeholder.",
    )
    parser.add_argument(
        "--xyz_dir",
        type=Path,
        help="Path to a single .xyz file or directory containing .xyz files.",
    )
    parser.add_argument(
        "--lmdb_train",
        type=Path,
        help="Path to the training LMDB directory (future use).",
    )
    parser.add_argument(
        "--lmdb_val",
        type=Path,
        help="Path to the validation LMDB directory (future use).",
    )
    parser.add_argument(
        "--lmdb_train_max_samples",
        type=int,
        default=None,
        help="Optional limit on number of LMDB samples used for training (random subset).",
    )
    parser.add_argument(
        "--lmdb_val_max_samples",
        type=int,
        default=None,
        help="Optional limit on number of LMDB samples used for validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mace_pretrain.pt"),
        help="Output path for the trained model checkpoint.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Number of configurations to reservoir-sample from xyz files.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=450,
        help="Number of configurations used for training (remainder used for validation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling, shuffling and training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size for both training and validation loaders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Radial cutoff (Å) used to build neighborhoods.",
    )
    parser.add_argument(
        "--energy_weight",
        type=float,
        default=1.0,
        help="Weight applied to the energy MSE term.",
    )
    parser.add_argument(
        "--force_weight",
        type=float,
        default=1000.0,
        help="Weight applied to the force MSE term.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        help="Initial learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1.0e-6,
        help="Weight decay (L2 regularisation) for the optimizer.",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of message-passing interaction blocks in the MACE model.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="Exponential moving average decay (ignored if EMA disabled).",
    )
    parser.add_argument(
        "--ema",
        dest="ema",
        action="store_true",
        help="Enable EMA smoothing of model weights (default).",
    )
    parser.add_argument(
        "--no-ema",
        dest="ema",
        action="store_false",
        help="Disable EMA smoothing of model weights.",
    )
    parser.add_argument(
        "--neighbor_sample_size",
        type=int,
        default=1024,
        help="Number of samples to estimate average neighbors for LMDB data.",
    )
    parser.add_argument(
        "--lmdb_e0_samples",
        type=int,
        default=2000,
        help="Number of LMDB entries to sample for E0 estimation and element detection.",
    )
    parser.add_argument(
        "--elements",
        type=int,
        nargs="+",
        help="Optional explicit list of atomic numbers for LMDB datasets.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show tqdm progress bar during training (default).",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm progress bar during training.",
    )
    parser.set_defaults(ema=True)
    parser.set_defaults(progress=True)
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
) -> Tuple[dict, float]:
    best_val_loss = float("inf")
    best_state_dict: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        iterator = (
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            if show_progress
            else train_loader
        )
        for batch in iterator:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.to_dict(), training=True, compute_force=True)
            loss, _, _ = compute_losses(outputs, batch, energy_weight, force_weight)
            loss.backward()
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
            if ema is not None:
                with ema.average_parameters():
                    best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state_dict is None:
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_state_dict, best_val_loss


def main() -> None:
    args = parse_args()
    tools.set_seeds(args.seed)

    if args.data_format == "xyz":
        if args.xyz_dir is None:
            raise ValueError("--xyz_dir must be set when data_format='xyz'")
        (
            train_loader,
            valid_loader,
            z_table,
            avg_num_neighbors,
            e0_values,
        ) = prepare_xyz_dataloaders(args)
    elif args.data_format == "lmdb":
        (
            train_loader,
            valid_loader,
            z_table,
            avg_num_neighbors,
            e0_values,
        ) = prepare_lmdb_dataloaders(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported data format: {args.data_format}")

    model = instantiate_model(
        z_table,
        avg_num_neighbors,
        args.cutoff,
        e0_values,
        args.num_interactions,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=50,
    )

    ema = (
        ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
        if args.ema
        else None
    )
    if ema is not None:
        LOGGER.info("EMA enabled with decay %.4f", args.ema_decay)
    else:
        LOGGER.info("EMA disabled.")

    best_state_dict, best_val_loss = train(
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
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(
                f"Output checkpoint already exists: {args.output}. "
                "Please specify a new --output path to avoid overwriting."
            )
        checkpoint = {
            "model_state_dict": best_state_dict,
            "best_val_loss": best_val_loss,
            "avg_num_neighbors": float(avg_num_neighbors),
            "z_table": [int(z) for z in z_table.zs],
            "e0_values": e0_values.tolist(),
            "cutoff": args.cutoff,
            "num_interactions": args.num_interactions,
        }
        torch.save(checkpoint, args.output)
        LOGGER.info(
            "Saved best validation checkpoint to %s (val_loss %.6f)",
            args.output,
            best_val_loss,
        )


if __name__ == "__main__":
    main()
