"""Sample rMD17 XYZ data and train a MACE potential with a custom loop."""

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization
torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float64)

from ase import io as ase_io
from e3nn import o3
from torch_ema import ExponentialMovingAverage

from mace import data, modules, tools
from mace.tools import torch_geometric

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample rMD17 xyz data and train MACE."
    )
    parser.add_argument(
        "--xyz_dir",
        type=Path,
        required=True,
        help="Path to a single .xyz file or directory containing .xyz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mace_rmd17.pt"),
        help="Output path for the trained model checkpoint.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Number of configurations to reservoir-sample from all xyz files.",
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
    parser.set_defaults(ema=True)
    return parser.parse_args()


def ensure_xyz_files(path: Path) -> List[Path]:
    """Return list of xyz files given a file or directory path."""
    if path.is_file():
        if path.suffix.lower() != ".xyz":
            raise ValueError(f"Provided file is not an .xyz: {path}")
        logging.info("Using single xyz file: %s", path)
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"XYZ path does not exist: {path}")

    if not path.is_dir():
        raise ValueError(f"Path must be a .xyz file or directory: {path}")

    xyz_files = sorted(path.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {path}")
    logging.info("Found %d xyz files for sampling.", len(xyz_files))
    return xyz_files


def reservoir_sample_atoms(
    xyz_files: Sequence[Path],
    sample_size: int,
    seed: int,
) -> List["ase.Atoms"]:
    """Reservoir sample frames across multiple xyz files."""
    rng = np.random.default_rng(seed)
    reservoir: List["ase.Atoms"] = []
    total_seen = 0
    for xyz_path in xyz_files:
        logging.info("Reading frames from %s", xyz_path.name)
        for atoms in ase_io.iread(xyz_path, index=":"):
            energy = atoms.info.get("energy")
            calc_energy = None
            if atoms.calc is not None:
                calc_energy = atoms.calc.results.get("energy")
                if calc_energy is None:
                    try:
                        calc_energy = atoms.calc.get_potential_energy()
                    except Exception:  # pylint: disable=broad-except
                        calc_energy = None
            logging.debug(
                "Frame %d energy info=%s calc=%s",
                total_seen,
                energy,
                calc_energy,
            )
            if energy is None and calc_energy is not None:
                atoms.info["energy"] = float(calc_energy)
            total_seen += 1
            atoms_copy = atoms.copy()
            if len(reservoir) < sample_size:
                reservoir.append(atoms_copy)
            else:
                idx = rng.integers(0, total_seen)
                if idx < sample_size:
                    reservoir[idx] = atoms_copy
    if total_seen < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples but dataset only has {total_seen} frames."
        )
    rng.shuffle(reservoir)
    logging.info(
        "Reservoir sampling complete (sampled %d of %d frames).", sample_size, total_seen
    )
    return reservoir


def split_samples(
    atoms_list: Sequence["ase.Atoms"], train_size: int
) -> Tuple[List["ase.Atoms"], List["ase.Atoms"]]:
    if train_size >= len(atoms_list):
        raise ValueError(
            "train_size must be smaller than the total number of sampled frames."
        )
    train_atoms = list(atoms_list[:train_size])
    valid_atoms = list(atoms_list[train_size:])
    return train_atoms, valid_atoms


def build_key_specification() -> data.KeySpecification:
    key_spec = data.KeySpecification()
    key_spec.update(info_keys={"energy": "energy"})
    key_spec.update(arrays_keys={"forces": "force"})
    return key_spec


def atoms_to_configurations(
    atoms_list: Sequence["ase.Atoms"], key_spec: data.KeySpecification
) -> data.Configurations:
    prepared: List["ase.Atoms"] = []
    for atoms in atoms_list:
        energy = atoms.info.get("energy")
        if energy is None and atoms.calc is not None:
            try:
                energy = atoms.calc.results.get("energy")
                if energy is None:
                    energy = atoms.calc.get_potential_energy()
            except Exception:  # pylint: disable=broad-except
                energy = None
        atoms_copy = atoms.copy()
        if energy is not None:
            atoms_copy.info["energy"] = float(energy)
        prepared.append(atoms_copy)
    return data.config_from_atoms_list(prepared, key_specification=key_spec)


def gather_atomic_numbers(configs: Iterable[data.Configuration]) -> List[int]:
    unique_numbers = set()
    for cfg in configs:
        unique_numbers.update(cfg.atomic_numbers.tolist())
    return sorted(unique_numbers)


def configs_to_atomic_data(
    configs: Sequence[data.Configuration],
    z_table: tools.AtomicNumberTable,
    cutoff: float,
) -> List[data.AtomicData]:
    heads = ["Default"]
    return [
        data.AtomicData.from_config(cfg, z_table=z_table, cutoff=cutoff, heads=heads)
        for cfg in configs
    ]


class AtomicDataListDataset(torch.utils.data.Dataset):
    def __init__(self, atomic_data_list: Sequence[data.AtomicData]):
        self.atomic_data_list = list(atomic_data_list)

    def __len__(self) -> int:
        return len(self.atomic_data_list)

    def __getitem__(self, idx: int) -> data.AtomicData:
        return self.atomic_data_list[idx]


def instantiate_model(
    z_table: tools.AtomicNumberTable,
    avg_num_neighbors: float,
    cutoff: float,
    atomic_energies: np.ndarray,
    num_interactions: int,
) -> modules.MACE:
    model = modules.MACE(
        r_max=cutoff,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps("128x0e + 128x1o"),
        MLP_irreps=o3.Irreps("64x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=3,
        radial_type="bessel",
    )
    return model


def compute_losses(
    outputs: dict,
    batch: data.AtomicData,
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
    model: modules.MACE,
    loader: torch_geometric.dataloader.DataLoader,
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
        outputs = model(
            batch.to_dict(),
            training=False,
            compute_force=True,
        )
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
    model: modules.MACE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: torch_geometric.dataloader.DataLoader,
    valid_loader: torch_geometric.dataloader.DataLoader,
    device: torch.device,
    epochs: int,
    energy_weight: float,
    force_weight: float,
    ema: Optional[ExponentialMovingAverage],
) -> Tuple[dict, float]:
    best_val_loss = float("inf")
    best_state_dict: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch.to_dict(),
                training=True,
                compute_force=True,
            )
            loss, _, _ = compute_losses(outputs, batch, energy_weight, force_weight)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update()

            total_train_loss += loss.item()
            total_batches += 1

        avg_train_loss = total_train_loss / max(total_batches, 1)

        context = ema.average_parameters() if ema is not None else nullcontext()
        with context:
            val_loss, val_energy_rmse, val_force_rmse = evaluate(
                model,
                valid_loader,
                device,
                energy_weight,
                force_weight,
            )

        scheduler.step(val_loss)

        logging.info(
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
                    best_state_dict = deepcopy(model.state_dict())
            else:
                best_state_dict = deepcopy(model.state_dict())

    if best_state_dict is None:
        best_state_dict = deepcopy(model.state_dict())

    return best_state_dict, best_val_loss


def main() -> None:
    args = parse_args()
    tools.set_seeds(args.seed)

    xyz_files = ensure_xyz_files(args.xyz_dir)
    sampled_atoms = reservoir_sample_atoms(xyz_files, args.sample_size, args.seed)
    train_atoms, valid_atoms = split_samples(sampled_atoms, args.train_size)

    logging.info(
        "Training frames: %d | Validation frames: %d",
        len(train_atoms),
        len(valid_atoms),
    )

    key_spec = build_key_specification()
    train_configs = atoms_to_configurations(train_atoms, key_spec)
    valid_configs = atoms_to_configurations(valid_atoms, key_spec)

    if train_configs:
        logging.debug(
            "First training config properties: %s", train_configs[0].properties
        )

    all_numbers = gather_atomic_numbers(train_configs + valid_configs)
    z_table = tools.AtomicNumberTable(all_numbers)

    train_atomic_data = configs_to_atomic_data(train_configs, z_table, args.cutoff)
    valid_atomic_data = configs_to_atomic_data(valid_configs, z_table, args.cutoff)

    try:
        atomic_energies_dict = data.compute_average_E0s(train_configs, z_table)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(
            "Failed to compute per-element E0s (%s); falling back to global average.",
            exc,
        )
        atomic_energies_dict = {}

    e0_values = np.array(
        [atomic_energies_dict.get(z, np.nan) for z in z_table.zs],
        dtype=float,
    )
    if not np.all(np.isfinite(e0_values)):
        energies = [
            cfg.properties.get("energy")
            for cfg in train_configs
            if cfg.properties.get("energy") is not None
        ]
        if not energies:
            logging.error("No training configurations have energy labels; aborting.")
            raise ValueError(
                "Cannot compute fallback E0s: no configurations have energy labels."
            ) from None
        total_energy = float(np.sum(energies))
        total_atoms = sum(len(cfg.atomic_numbers) for cfg in train_configs)
        per_atom_shift = total_energy / total_atoms
        e0_values = np.full(len(z_table), per_atom_shift, dtype=float)
        logging.warning(
            "Using global per-atom energy shift %.6f for all elements.",
            per_atom_shift,
        )
    else:
        logging.info(
            "Fitted atomic E0s: %s",
            {int(z): float(atomic_energies_dict[z]) for z in z_table.zs},
        )

    train_dataset = AtomicDataListDataset(train_atomic_data)
    valid_dataset = AtomicDataListDataset(valid_atomic_data)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch_geometric.dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    stats_loader = torch_geometric.dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    avg_num_neighbors = modules.compute_avg_num_neighbors(stats_loader)
    logging.info("Estimated average number of neighbors: %.4f", avg_num_neighbors)

    logging.info("Using %d interaction layers.", args.num_interactions)
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

    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay) if args.ema else None
    if ema is not None:
        logging.info("EMA enabled with decay %.4f", args.ema_decay)
    else:
        logging.info("EMA disabled.")

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
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "best_val_loss": best_val_loss,
            },
            args.output,
        )
        logging.info(
            "Saved best validation checkpoint to %s (val_loss %.6f)",
            args.output,
            best_val_loss,
        )


if __name__ == "__main__":
    main()
