"""Utilities to build DataLoaders backed by OC22-style LMDB shards."""

from __future__ import annotations

import bisect
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lmdb
import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data
from torch.utils.data import get_worker_info

from mace import data, modules, tools
from mace.tools import torch_geometric

from .xyz_loader import build_key_specification, compute_e0s

LOGGER = logging.getLogger(__name__)


def _list_lmdb_files(directory: Path) -> List[Path]:
    """Return sorted LMDB shard paths inside a directory."""

    if directory is None:
        raise ValueError("LMDB directory path is not provided.")
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"LMDB directory not found: {directory}")

    lmdb_files = sorted(directory.glob("data.*.lmdb"))
    if not lmdb_files:
        raise FileNotFoundError(f"No data.*.lmdb files found in {directory}")
    return lmdb_files


def _upgrade_pyg_data(pyg_data) -> Data:
    """Rebuild PyG Data from legacy storage without touching properties."""

    if isinstance(pyg_data, Data):
        raw_dict = object.__getattribute__(pyg_data, "__dict__")
        if "_store" in raw_dict:
            mapping = dict(raw_dict["_store"])
        else:
            mapping = dict(raw_dict)
    elif isinstance(pyg_data, dict):
        mapping = dict(pyg_data)
    else:
        raise TypeError(f"Unsupported LMDB entry type: {type(pyg_data)}")

    if "positions" in mapping and "pos" not in mapping:
        mapping["pos"] = mapping.pop("positions")
    if "pos" not in mapping:
        raise ValueError(
            "Legacy PyG data object missing 'pos'/'positions'. Keys: "
            f"{list(mapping.keys())}"
        )

    return Data.from_dict(mapping)


def _pyg_to_configuration(
    pyg_data,
    key_spec: data.KeySpecification,
) -> data.Configuration:
    """Convert a PyG Data object stored in the LMDB into a MACE Configuration."""

    pyg_data = _upgrade_pyg_data(pyg_data)

    numbers = pyg_data.atomic_numbers.detach().cpu().numpy()
    positions = pyg_data.pos.detach().cpu().numpy()

    cell = pyg_data.cell.detach().cpu().numpy()
    if cell.size == 9:
        cell = cell.reshape(3, 3)

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)

    if getattr(pyg_data, "y", None) is not None:
        energy_value = pyg_data.y
        if isinstance(energy_value, torch.Tensor):
            energy_value = energy_value.detach().cpu().item()
        atoms.info["energy"] = float(energy_value)

    forces_value = getattr(pyg_data, "force", None)
    if forces_value is None:
        raise ValueError("LMDB entry does not contain forces, cannot train.")
    if isinstance(forces_value, torch.Tensor):
        forces_value = forces_value.detach().cpu().numpy()
    else:
        forces_value = np.asarray(forces_value)
    atoms.arrays["force"] = forces_value

    configs = data.config_from_atoms_list([atoms], key_specification=key_spec)
    return configs[0]


def _sample_configs_and_elements(
    lmdb_files: Sequence[Path],
    key_spec: data.KeySpecification,
    sample_limit: int,
) -> Tuple[List[data.Configuration], List[int]]:
    """Sample a subset of configurations to estimate E0s and element coverage."""

    sampled_configs: List[data.Configuration] = []
    unique_numbers: set[int] = set()

    for lmdb_path in lmdb_files:
        env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=2048,
            subdir=False,
        )
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                pyg_data = pickle.loads(value)
                unique_numbers.update(
                    pyg_data.atomic_numbers.detach().cpu().numpy().tolist()
                )
                if len(sampled_configs) < sample_limit:
                    sampled_configs.append(_pyg_to_configuration(pyg_data, key_spec))
                if len(sampled_configs) >= sample_limit:
                    break
        env.close()
        if len(sampled_configs) >= sample_limit:
            break

    if not sampled_configs:
        raise ValueError("Failed to sample configurations from LMDB files.")

    return sampled_configs, sorted(unique_numbers)


class LmdbAtomicDataset(torch.utils.data.Dataset):
    """Dataset that streams AtomicData objects from LMDB shards."""

    def __init__(
        self,
        lmdb_files: Sequence[Path],
        z_table: tools.AtomicNumberTable,
        cutoff: float,
        key_spec: data.KeySpecification,
        max_samples: int | None = None,
    ) -> None:
        if not lmdb_files:
            raise ValueError("No LMDB files provided.")

        self.heads = ["Default"]
        self.z_table = z_table
        self.cutoff = cutoff
        self.key_spec = key_spec
        self.lmdb_files = list(lmdb_files)

        self.cumulative_sizes: List[int] = []
        total = 0
        self._shard_sizes: List[int] = []
        for lmdb_path in self.lmdb_files:
            entries = self._count_entries(lmdb_path)
            self._shard_sizes.append(entries)
            total += entries
            self.cumulative_sizes.append(total)

        if total == 0:
            raise ValueError("LMDB dataset appears empty.")

        self.total_samples = total
        self.max_samples = max_samples
        self._main_env_cache: Dict[int, lmdb.Environment] = {}
        self._worker_env_cache: Dict[int, Dict[int, lmdb.Environment]] = {}
        if max_samples is not None and max_samples < self.total_samples:
            self._indices = self._build_sample_indices(max_samples, total)
            self.total_samples = len(self._indices)
        else:
            self._indices = None

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return self.total_samples

    def __getitem__(self, idx: int) -> data.AtomicData:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for LMDB dataset.")
        if self._indices is not None:
            shard_idx, local_idx = self._indices[idx]
        else:
            shard_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            lower_bound = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
            local_idx = idx - lower_bound

        env = self._get_env(shard_idx)
        with env.begin(write=False) as txn:
            key = f"{local_idx}".encode("ascii")
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Entry {local_idx} missing in shard {shard_idx}.")
            pyg_data = pickle.loads(value)

        configuration = _pyg_to_configuration(pyg_data, self.key_spec)
        atomic_data = data.AtomicData.from_config(
            configuration,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=self.heads,
        )
        return atomic_data

    def _count_entries(self, lmdb_path: Path) -> int:
        env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            subdir=False,
            max_readers=1,
        )
        try:
            return env.stat().get("entries", 0)
        finally:
            env.close()

    def _get_env(self, shard_idx: int) -> lmdb.Environment:
        worker_info = get_worker_info()
        if worker_info is None:
            cache = self._main_env_cache
        else:
            cache = self._worker_env_cache.setdefault(worker_info.id, {})
        env = cache.get(shard_idx)
        if env is None:
            env = lmdb.open(
                str(self.lmdb_files[shard_idx]),
                readonly=True,
                lock=False,
                readahead=False,
                subdir=False,
                max_readers=1,
            )
            cache[shard_idx] = env
        return env

    def _close_cache(self, cache: Dict[int, lmdb.Environment]) -> None:
        for env in cache.values():
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass
        cache.clear()

    def __del__(self):  # pragma: no cover - GC hook
        self._close_cache(self._main_env_cache)
        for cache in self._worker_env_cache.values():
            self._close_cache(cache)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_main_env_cache"] = {}
        state["_worker_env_cache"] = {}
        return state

    def _build_sample_indices(self, max_samples: int, original_total: int) -> List[Tuple[int, int]]:
        max_samples = min(max_samples, original_total)
        rng = np.random.default_rng(seed=0)
        selected = rng.choice(original_total, size=max_samples, replace=False)
        selected.sort()

        indices: List[Tuple[int, int]] = []
        shard_idx = 0
        shard_start = 0
        shard_end = self.cumulative_sizes[0]
        for global_idx in selected:
            while global_idx >= shard_end:
                shard_idx += 1
                shard_start = shard_end
                shard_end = self.cumulative_sizes[shard_idx]
            local_idx = global_idx - shard_start
            indices.append((shard_idx, local_idx))
        LOGGER.info(
            "Randomly selected %d/%d LMDB samples for this split.",
            len(indices),
            original_total,
        )
        return indices


def prepare_lmdb_dataloaders(args):
    """Build DataLoaders and metadata from LMDB datasets."""

    if args.lmdb_train is None or args.lmdb_val is None:
        raise ValueError(
            "--lmdb_train and --lmdb_val must be provided for LMDB datasets."
        )

    train_files = _list_lmdb_files(args.lmdb_train)
    val_files = _list_lmdb_files(args.lmdb_val)

    key_spec = build_key_specification()

    sample_limit = max(1, args.lmdb_e0_samples)
    sampled_configs, detected_numbers = _sample_configs_and_elements(
        train_files, key_spec, sample_limit
    )

    if args.elements:
        element_list = sorted(set(args.elements))
    else:
        element_list = detected_numbers
        LOGGER.info(
            "Detected %d unique elements from samples: %s",
            len(element_list),
            element_list,
        )

    if not element_list:
        raise ValueError("Failed to determine element list for LMDB dataset.")

    z_table = tools.AtomicNumberTable(element_list)

    e0_values = compute_e0s(sampled_configs, z_table)

    train_dataset = LmdbAtomicDataset(
        train_files,
        z_table,
        args.cutoff,
        key_spec,
        max_samples=args.lmdb_train_max_samples,
    )
    valid_dataset = LmdbAtomicDataset(
        val_files,
        z_table,
        args.cutoff,
        key_spec,
        max_samples=args.lmdb_val_max_samples,
    )

    train_loader = torch_geometric.dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    neighbor_sample_size = min(len(train_dataset), args.neighbor_sample_size)
    stats_indices = list(range(neighbor_sample_size))
    stats_subset = torch.utils.data.Subset(train_dataset, stats_indices)
    stats_loader = torch_geometric.dataloader.DataLoader(
        stats_subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    avg_num_neighbors = modules.compute_avg_num_neighbors(stats_loader)
    LOGGER.info(
        "Estimated average number of neighbors from %d samples: %.4f",
        neighbor_sample_size,
        avg_num_neighbors,
    )

    return train_loader, valid_loader, z_table, avg_num_neighbors, e0_values
    def _build_sample_indices(self, max_samples: int) -> List[Tuple[int, int]]:
        rng = np.random.default_rng(seed=0)
        total = self.total_samples
        selected = rng.choice(total, size=max_samples, replace=False)
        selected.sort()
        indices: List[Tuple[int, int]] = []
        shard_idx = 0
        shard_start = 0
        for global_idx in selected:
            while shard_idx < len(self.cumulative_sizes) and global_idx >= self.cumulative_sizes[shard_idx]:
                shard_start = self.cumulative_sizes[shard_idx]
                shard_idx += 1
            local_idx = global_idx - shard_start
            indices.append((shard_idx, local_idx))
        return indices
