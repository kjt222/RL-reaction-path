"""Utilities to build DataLoaders from Extended XYZ trajectories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from ase import io as ase_io
try:  # PyG>=2.3
    from torch_geometric.loader import DataLoader as PYGDataLoader
except ImportError:  # PyG<=2.2
    from torch_geometric.dataloader import DataLoader as PYGDataLoader

from mace import data, modules, tools
from mace.tools import torch_geometric

LOGGER = logging.getLogger(__name__)


class AtomicDataListDataset(torch.utils.data.Dataset):
    """Minimal Dataset wrapper around a list of AtomicData objects."""

    def __init__(self, atomic_data_list: Sequence[data.AtomicData]):
        self.atomic_data_list = list(atomic_data_list)

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return len(self.atomic_data_list)

    def __getitem__(self, idx: int) -> data.AtomicData:
        return self.atomic_data_list[idx]


def ensure_xyz_files(path: Path) -> List[Path]:
    """Return all XYZ files from the provided path."""

    if path.is_file():
        if path.suffix.lower() != ".xyz":
            raise ValueError(f"Provided file is not an .xyz: {path}")
        LOGGER.info("Using single xyz file: %s", path)
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"XYZ path does not exist: {path}")

    if not path.is_dir():
        raise ValueError(f"Path must be a .xyz file or directory: {path}")

    xyz_files = sorted(path.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {path}")
    LOGGER.info("Found %d xyz files for sampling.", len(xyz_files))
    return xyz_files


def reservoir_sample_atoms(
    xyz_files: Sequence[Path],
    sample_size: int,
    seed: int,
) -> List["ase.Atoms"]:
    """Reservoir-sample frames across multiple XYZ files."""

    rng = np.random.default_rng(seed)
    reservoir: List["ase.Atoms"] = []
    total_seen = 0
    for xyz_path in xyz_files:
        LOGGER.info("Reading frames from %s", xyz_path.name)
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
    LOGGER.info(
        "Reservoir sampling complete (sampled %d of %d frames).",
        sample_size,
        total_seen,
    )
    return reservoir


def split_samples(
    atoms_list: Sequence["ase.Atoms"],
    train_size: int,
) -> Tuple[List["ase.Atoms"], List["ase.Atoms"]]:
    """Split sampled frames into train/validation lists."""

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
    atoms_list: Sequence["ase.Atoms"],
    key_spec: data.KeySpecification,
) -> data.Configurations:
    """Convert ASE Atoms objects into MACE Configurations."""

    prepared: List["ase.Atoms"] = []
    for atoms in atoms_list:
        if "force" not in atoms.arrays:
            raise ValueError("XYZ frame missing required forces array 'force'.")
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


def compute_e0s(
    configs: Sequence[data.Configuration],
    z_table: tools.AtomicNumberTable,
) -> np.ndarray:
    """Compute fitted atomic reference energies with safe fallbacks."""

    try:
        atomic_energies_dict = data.compute_average_E0s(configs, z_table)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning(
            "Failed to compute per-element E0s (%s); falling back to global average.",
            exc,
        )
        atomic_energies_dict = {}

    e0_values = np.array(
        [atomic_energies_dict.get(z, np.nan) for z in z_table.zs],
        dtype=float,
    )
    if np.all(np.isfinite(e0_values)):
        LOGGER.info(
            "Fitted atomic E0s: %s",
            {int(z): float(atomic_energies_dict[z]) for z in z_table.zs},
        )
        return e0_values

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
    LOGGER.warning(
        "Using global per-atom energy shift %.6f for all elements.",
        per_atom_shift,
    )
    return np.full(len(z_table), per_atom_shift, dtype=float)


def build_dataloaders(
    train_atomic_data: Sequence[data.AtomicData],
    valid_atomic_data: Sequence[data.AtomicData],
    batch_size: int,
    seed: int,
    num_workers: int,
) -> Tuple[PYGDataLoader, PYGDataLoader]:
    """Create PyTorch Geometric DataLoaders for train/validation splits."""

    train_dataset = AtomicDataListDataset(train_atomic_data)
    valid_dataset = AtomicDataListDataset(valid_atomic_data)

    generator = torch.Generator().manual_seed(seed)
    train_loader = PYGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    valid_loader = PYGDataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, valid_loader


def prepare_xyz_dataloaders(args):
    """Build DataLoaders from Extended XYZ files, using provided z_table/cutoff."""

    xyz_dir = Path(args.xyz_dir)
    xyz_files = ensure_xyz_files(xyz_dir)
    sampled_atoms = reservoir_sample_atoms(xyz_files, args.sample_size, args.seed)
    train_atoms, valid_atoms = split_samples(sampled_atoms, args.train_size)

    LOGGER.info(
        "Training frames: %d | Validation frames: %d",
        len(train_atoms),
        len(valid_atoms),
    )

    key_spec = build_key_specification()
    train_configs = atoms_to_configurations(train_atoms, key_spec)
    valid_configs = atoms_to_configurations(valid_atoms, key_spec)

    if not train_configs or not valid_configs:
        raise ValueError("No configurations generated from xyz data.")

    all_numbers = gather_atomic_numbers(train_configs + valid_configs)
    # z_table 必须由外部提供（模型/metadata）；确保数据元素在映射内。
    if not hasattr(args, "z_table") or args.z_table is None:
        raise ValueError("prepare_xyz_dataloaders 需要外部提供 z_table (args.z_table)。")
    z_table = args.z_table
    missing = [z for z in all_numbers if z not in z_table.zs]
    if missing:
        raise ValueError(f"XYZ 数据包含模型 z_table 不支持的元素: {missing}")

    train_atomic_data = configs_to_atomic_data(train_configs, z_table, args.cutoff)
    valid_atomic_data = configs_to_atomic_data(valid_configs, z_table, args.cutoff)

    train_loader, valid_loader = build_dataloaders(
        train_atomic_data,
        valid_atomic_data,
        args.batch_size,
        args.seed,
        args.num_workers,
    )

    return train_loader, valid_loader
