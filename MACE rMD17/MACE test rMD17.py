"""Evaluate a trained MACE model on sampled frames from XYZ files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.serialization

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float64)

from ase import io as ase_io
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MACE model on sampled XYZ frames."
    )
    parser.add_argument(
        "--xyz_path",
        type=Path,
        required=True,
        help="Path to a single .xyz file or directory containing .xyz files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of frames to reservoir-sample for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for evaluation loader.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Radial cutoff (Å) used to build neighborhoods (must match training).",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of message-passing layers (must match training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available).",
    )
    parser.add_argument(
        "--pred_output",
        type=Path,
        default=None,
        help="Optional output .xyz path to write predicted energies and forces.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch predictions (for debugging).",
    )
    return parser.parse_args()


def ensure_xyz_files(path: Path) -> List[Path]:
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
    rng = np.random.default_rng(seed)
    reservoir: List[Tuple[int, "ase.Atoms"]] = []
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
            if energy is None and calc_energy is not None:
                atoms.info["energy"] = float(calc_energy)

            total_seen += 1
            frame_index = total_seen - 1
            atoms_copy = atoms.copy()
            if len(reservoir) < sample_size:
                reservoir.append((frame_index, atoms_copy))
            else:
                idx = rng.integers(0, total_seen)
                if idx < sample_size:
                    reservoir[idx] = (frame_index, atoms_copy)
    if total_seen < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples but dataset only has {total_seen} frames."
        )
    reservoir.sort(key=lambda item: item[0])
    logging.info(
        "Reservoir sampling complete (sampled %d of %d frames).",
        sample_size,
        total_seen,
    )
    return [atoms for _, atoms in reservoir]


def split_samples(
    atoms_list: Sequence["ase.Atoms"], sample_size: int
) -> List["ase.Atoms"]:
    if sample_size > len(atoms_list):
        logging.warning(
            "Requested sample_size=%d but only %d frames available; using all frames.",
            sample_size,
            len(atoms_list),
        )
        sample_size = len(atoms_list)
    return list(atoms_list[:sample_size])


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
    zs = set()
    for cfg in configs:
        zs.update(cfg.atomic_numbers.tolist())
    return sorted(zs)


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


def write_predictions(
    atoms_list: Sequence["ase.Atoms"],
    energy_preds: np.ndarray,
    force_preds: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing predictions to %s", path)
    offset = 0
    with path.open("w", encoding="utf-8") as handle:
        for atoms, energy in zip(atoms_list, energy_preds):
            n_atoms = len(atoms)
            forces = force_preds[offset : offset + n_atoms]
            offset += n_atoms
            handle.write(f"{n_atoms}\n")
            comment = (
                "Properties=species:S:1:pos:R:3:pred_force:R:3 "
                f"pred_energy={energy:.16f}"
            )
            handle.write(comment + "\n")
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            for symbol, pos, force in zip(symbols, positions, forces):
                pos_str = " ".join(f"{value:.16f}" for value in pos)
                force_str = " ".join(f"{value:.16f}" for value in force)
                handle.write(f"{symbol} {pos_str} {force_str}\n")
    if offset != force_preds.shape[0]:
        logging.warning(
            "Predicted force array length (%d) differs from atoms written (%d).",
            force_preds.shape[0],
            offset,
        )


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
    return modules.MACE(
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


def compute_e0s(
    configs: Sequence[data.Configuration], z_table: tools.AtomicNumberTable
) -> np.ndarray:
    try:
        atomic_energies_dict = data.compute_average_E0s(configs, z_table)
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
            for cfg in configs
            if cfg.properties.get("energy") is not None
        ]
        if not energies:
            raise ValueError(
                "Cannot compute fallback E0s: no configurations have energy labels."
            )
        total_energy = float(np.sum(energies))
        total_atoms = sum(len(cfg.atomic_numbers) for cfg in configs)
        per_atom_shift = total_energy / total_atoms
        e0_values = np.full(len(z_table), per_atom_shift, dtype=float)
        logging.warning(
            "Using global per-atom energy shift %.6f for all elements.",
            per_atom_shift,
        )
    else:
        logging.info(
            "Fitted atomic E0s: %s",
            {int(z): float(e0_values[i]) for i, z in enumerate(z_table.zs)},
        )
    return e0_values


def evaluate_model(
    model: modules.MACE,
    loader: torch_geometric.dataloader.DataLoader,
    device: torch.device,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    energy_preds: List[np.ndarray] = []
    energy_targets: List[np.ndarray] = []
    force_preds: List[np.ndarray] = []
    force_targets: List[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        outputs = model(
            batch.to_dict(),
            training=False,
            compute_force=True,
        )

        pred_energy = outputs["energy"].squeeze(-1).detach().cpu().numpy()
        true_energy = batch.energy.squeeze(-1).detach().cpu().numpy()

        pred_forces = outputs["forces"].detach().cpu().numpy()
        true_forces = batch.forces.detach().cpu().numpy()

        energy_preds.append(pred_energy)
        energy_targets.append(true_energy)
        force_preds.append(pred_forces)
        force_targets.append(true_forces)

        if verbose:
            logging.info(
                "Batch %d energy prediction (first 5): %s",
                batch_idx,
                pred_energy[:5],
        )

    if not energy_preds:
        logging.warning("No batches evaluated; nothing to report.")
        empty = np.array([])
        return empty, empty, empty, empty

    energy_preds_arr = np.concatenate(energy_preds)
    energy_targets_arr = np.concatenate(energy_targets)
    force_preds_arr = np.concatenate(force_preds, axis=0)
    force_targets_arr = np.concatenate(force_targets, axis=0)

    energy_diff = energy_preds_arr - energy_targets_arr
    force_diff = force_preds_arr - force_targets_arr

    energy_mae = np.mean(np.abs(energy_diff))
    energy_rmse = np.sqrt(np.mean(np.square(energy_diff)))
    force_mae = np.mean(np.abs(force_diff))
    force_rmse = np.sqrt(np.mean(np.square(force_diff)))

    logging.info("Evaluated %d configurations.", energy_preds_arr.shape[0])
    logging.info(
        "Energy MAE: %.6f | Energy RMSE: %.6f | Force MAE: %.6f | Force RMSE: %.6f",
        energy_mae,
        energy_rmse,
        force_mae,
        force_rmse,
    )

    return energy_preds_arr, energy_targets_arr, force_preds_arr, force_targets_arr


def main() -> None:
    args = parse_args()
    tools.set_seeds(args.seed)

    xyz_files = ensure_xyz_files(args.xyz_path)
    sampled_atoms = reservoir_sample_atoms(xyz_files, args.sample_size, args.seed)
    sampled_atoms = split_samples(sampled_atoms, args.sample_size)

    key_spec = build_key_specification()
    configs = atoms_to_configurations(sampled_atoms, key_spec)

    if not configs:
        raise ValueError("No configurations obtained from the supplied xyz files.")

    z_table = tools.AtomicNumberTable(gather_atomic_numbers(configs))
    atomic_data_list = configs_to_atomic_data(configs, z_table, args.cutoff)

    dataset = AtomicDataListDataset(atomic_data_list)
    loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    e0_values = compute_e0s(configs, z_table)

    stats_loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    avg_num_neighbors = modules.compute_avg_num_neighbors(stats_loader)
    logging.info("Estimated average number of neighbors: %.4f", avg_num_neighbors)

    model = instantiate_model(
        z_table,
        avg_num_neighbors,
        args.cutoff,
        e0_values,
        args.num_interactions,
    )

    device = torch.device(args.device)
    model.to(device)

    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    logging.info("Loaded model weights from %s", args.model)

    energy_preds, _, force_preds, _ = evaluate_model(
        model,
        loader,
        device,
        args.verbose,
    )

    if energy_preds.size == 0:
        return

    if args.pred_output is not None:
        output_path = args.pred_output
    else:
        base_path = xyz_files[0] if xyz_files else args.xyz_path
        output_path = base_path.with_name(base_path.stem + "_pred.xyz")

    write_predictions(sampled_atoms, energy_preds, force_preds, output_path)


if __name__ == "__main__":
    main()
