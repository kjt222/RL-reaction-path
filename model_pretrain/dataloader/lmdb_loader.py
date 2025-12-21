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

from .xyz_loader import build_key_specification

LOGGER = logging.getLogger(__name__)

OC22_ELEMENTS = [
    # Adsorbates
    1, 6, 7, 8,
    # Group 1 & 2
    3, 11, 19, 37, 55, 4, 12, 20, 38, 56,
    # Transition metals
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 44, 45, 46, 47, 48,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    # Lanthanides
    58, 71,
    # P-block metals/metalloids
    13, 14,
    31, 32, 34,
    49, 50, 51,
    81, 82, 83,
]


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

    pbc = getattr(pyg_data, "pbc", True)
    try:
        if hasattr(pbc, "cpu"):
            pbc = pbc.detach().cpu().numpy()
    except Exception:
        pass
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)

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


class LmdbAtomicDataset(torch.utils.data.Dataset):
    """Dataset that streams AtomicData objects from LMDB shards.

    z_table 决定元素索引映射；coverage_zs 仅用于采样时的元素覆盖校验，可小于 z_table。
    """

    def __init__(
        self,
        lmdb_files: Sequence[Path],
        z_table: tools.AtomicNumberTable,
        cutoff: float,
        key_spec: data.KeySpecification,
        max_samples: int | None = None,
        selected_indices: List[Tuple[int, int]] | None = None,
        coverage_zs: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        if not lmdb_files:
            raise ValueError("No LMDB files provided.")

        self.heads = ["Default"]
        self.z_table = z_table
        self.coverage_zs = sorted({int(z) for z in (coverage_zs or z_table.zs)})
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
        self._rng = np.random.default_rng(seed=seed)
        self._missing_warned = False
        if selected_indices is not None:
            self._indices = list(selected_indices)
            self.total_samples = len(self._indices)
            if max_samples is not None and max_samples > self.total_samples:
                LOGGER.warning(
                    "Requested max_samples=%d but only %d preselected indices provided; using provided indices.",
                    max_samples,
                    self.total_samples,
                )
        elif max_samples is not None and max_samples < self.total_samples:
            self._indices = self._build_sample_indices(max_samples, total)
            self.total_samples = len(self._indices)
        else:
            self._indices = None

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return self.total_samples

    def __getitem__(self, idx: int) -> data.AtomicData:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for LMDB dataset.")
        shard_idx, local_idx = self._resolve_index(idx)

        env = self._get_env(shard_idx)
        with env.begin(write=False) as txn:
            key = f"{local_idx}".encode("ascii")
            value = txn.get(key)
            if value is None:
                raise KeyError(f"LMDB entry {local_idx} missing in shard {shard_idx}; please rebuild indices or fix dataset.")
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

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        if self._indices is not None:
            return self._indices[idx]
        shard_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        lower_bound = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        local_idx = idx - lower_bound
        return shard_idx, local_idx

    def _entry_exists(self, shard_idx: int, local_idx: int) -> bool:
        env = self._get_env(shard_idx)
        with env.begin(write=False) as txn:
            key = f"{local_idx}".encode("ascii")
            return txn.get(key) is not None

    def _build_sample_indices(self, max_samples: int, original_total: int) -> List[Tuple[int, int]]:
        max_samples = min(max_samples, original_total)
        rng = self._rng

        # Pass 1: ensure coverage by sampling at least one entry per element.
        coverage_indices: List[int] = []
        coverage_by_z: dict[int, bool] = {int(z): False for z in self.coverage_zs}
        coverage_counts: dict[int, int] = {int(z): 0 for z in self.coverage_zs}
        shard_idx = 0
        shard_start = 0
        shard_end = self.cumulative_sizes[0]
        for global_idx in range(original_total):
            while global_idx >= shard_end:
                shard_idx += 1
                shard_start = shard_end
                shard_end = self.cumulative_sizes[shard_idx]
            local_idx = global_idx - shard_start
            if not self._entry_exists(shard_idx, local_idx):
                continue
            env = self._get_env(shard_idx)
            with env.begin(write=False) as txn:
                val = txn.get(f"{local_idx}".encode("ascii"))
                if val is None:
                    continue
                pyg_data = pickle.loads(val)
            numbers = set(int(z) for z in pyg_data.atomic_numbers.tolist())
            for z in numbers:
                if z in coverage_by_z and not coverage_by_z[z]:
                    coverage_by_z[z] = True
                    coverage_counts[z] += 1
                    coverage_indices.append(global_idx)
                    break
                elif z in coverage_counts:
                    coverage_counts[z] += 1
            if all(coverage_by_z.values()):
                break

        missing_coverage = [z for z, hit in coverage_by_z.items() if not hit]
        if missing_coverage:
            LOGGER.warning(
                "Could not find samples covering elements: %s. "
                "LMDB shards may be incomplete.",
                missing_coverage,
            )

        if len(coverage_indices) > max_samples:
            raise ValueError(
                f"Requested max_samples={max_samples} is smaller than number of unique elements "
                f"({len(coverage_indices)}); increase max_samples."
            )

        # Pass 2: fill the rest randomly (without replacement), excluding already chosen.
        remaining = max_samples - len(coverage_indices)
        pool = np.setdiff1d(np.arange(original_total), np.array(coverage_indices), assume_unique=True)
        if remaining > 0:
            extra = rng.choice(pool, size=remaining, replace=False)
            selected = np.concatenate([np.array(coverage_indices, dtype=int), extra])
        else:
            selected = np.array(coverage_indices, dtype=int)
        selected.sort()
        LOGGER.info(
            "Element coverage ensured with %d seed samples (pool %d total). Remaining %d randomly selected.",
            len(coverage_indices),
            original_total,
            remaining,
        )

        indices: List[Tuple[int, int]] = []
        shard_idx = 0
        shard_start = 0
        shard_end = self.cumulative_sizes[0]
        missing = 0
        for global_idx in selected:
            while global_idx >= shard_end:
                shard_idx += 1
                shard_start = shard_end
                shard_end = self.cumulative_sizes[shard_idx]
            local_idx = global_idx - shard_start
            if not self._entry_exists(shard_idx, local_idx):
                missing += 1
                continue
            indices.append((shard_idx, local_idx))
        LOGGER.info(
            "Randomly selected %d/%d LMDB samples for this split.",
            len(indices),
            original_total,
        )
        if missing:
            LOGGER.warning(
                "Skipped %d missing LMDB entries while building sampled indices; dataset size reduced to %d.",
                missing,
                len(indices),
            )
        return indices

    @property
    def selected_indices(self) -> List[Tuple[int, int]] | None:
        return self._indices


def prepare_lmdb_dataloaders(
    args,
    z_table: tools.AtomicNumberTable,
    resume_indices: Dict[str, List[Tuple[int, int]]] | None = None,
    coverage_zs: Sequence[int] | None = None,
    seed: int | None = None,
):
    """Build DataLoaders from LMDB shards using a fixed z_table.

    - z_table: 模型/JSON 提供的编码表（必需）。
    - coverage_zs: 覆盖校验元素列表（默认 OC22），缺失会警告但不中断。
    - resume_indices: 复用采样索引以保持子集一致。
    """

    if args.lmdb_train is None or args.lmdb_val is None:
        raise ValueError(
            "--lmdb_train and --lmdb_val must be provided for LMDB datasets."
        )

    train_files = _list_lmdb_files(args.lmdb_train)
    val_files = _list_lmdb_files(args.lmdb_val)

    key_spec = build_key_specification()

    coverage_elements = (
        sorted({int(z) for z in coverage_zs}) if coverage_zs is not None else OC22_ELEMENTS
    )
    LOGGER.info("Using coverage set (%d elems): %s", len(coverage_elements), coverage_elements)
    if not coverage_elements:
        raise ValueError("Failed to determine element list for LMDB dataset.")

    element_count = len(coverage_elements)
    if args.lmdb_train_max_samples is not None and args.lmdb_train_max_samples < element_count:
        raise ValueError(
            f"--lmdb_train_max_samples={args.lmdb_train_max_samples} is smaller than "
            f"the number of detected elements ({element_count}); increase it to cover all elements."
        )
    if args.lmdb_val_max_samples is not None and args.lmdb_val_max_samples < element_count:
        raise ValueError(
            f"--lmdb_val_max_samples={args.lmdb_val_max_samples} is smaller than "
            f"the number of detected elements ({element_count}); increase it to cover all elements."
        )

    mapping_elements = list(z_table.zs)
    missing_in_model = [z for z in coverage_elements if z not in mapping_elements]
    if missing_in_model:
        raise ValueError(f"Coverage elements not in mapping z_table: {missing_in_model}")

    # 如果传入 resume_indices，要求 train/val 均存在且非空，强制复用而不重采样
    train_sel = val_sel = None
    if resume_indices is not None:
        if "train" not in resume_indices or resume_indices.get("train") is None:
            raise ValueError("resume_indices provided but missing 'train' entries.")
        if "val" not in resume_indices or resume_indices.get("val") is None:
            raise ValueError("resume_indices provided but missing 'val' entries.")
        train_sel = resume_indices["train"]
        val_sel = resume_indices["val"]
        LOGGER.info("Reusing LMDB sampled indices: train=%d, val=%d", len(train_sel), len(val_sel))

    train_dataset = LmdbAtomicDataset(
        train_files,
        z_table,
        args.cutoff,
        key_spec,
        max_samples=args.lmdb_train_max_samples,
        selected_indices=train_sel,
        coverage_zs=coverage_elements,
        seed=seed if seed is not None else getattr(args, "seed", 0),
    )
    valid_dataset = LmdbAtomicDataset(
        val_files,
        z_table,
        args.cutoff,
        key_spec,
        max_samples=args.lmdb_val_max_samples,
        selected_indices=val_sel,
        coverage_zs=coverage_elements,
        seed=seed if seed is not None else getattr(args, "seed", 0),
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

    return (
        train_loader,
        valid_loader,
        train_dataset.selected_indices,
        valid_dataset.selected_indices,
    )
