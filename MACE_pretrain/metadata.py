"""Shared helpers for packing and loading model metadata and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from mace import tools

Metadata = Dict[str, Any]
TrainState = Dict[str, Any]


def _coerce_z_table(z_table) -> list[int]:
    if isinstance(z_table, tools.AtomicNumberTable):
        return [int(z) for z in z_table.zs]
    return [int(z) for z in z_table]


def build_metadata(
    z_table,
    avg_num_neighbors: float,
    e0_values,
    cutoff: float,
    num_interactions: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Metadata:
    metadata: Metadata = {
        "z_table": _coerce_z_table(z_table),
        "avg_num_neighbors": float(avg_num_neighbors),
        "e0_values": np.asarray(e0_values, dtype=float).tolist(),
        "cutoff": float(cutoff),
        "num_interactions": int(num_interactions),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _legacy_metadata(raw: Dict[str, Any]) -> Metadata:
    metadata: Metadata = {}
    if "metadata" in raw and raw["metadata"]:
        metadata.update(raw["metadata"])
    for key in ("z_table", "avg_num_neighbors", "e0_values", "cutoff", "num_interactions"):
        if key in raw and raw[key] is not None:
            metadata[key] = raw[key]
    if "z_table" in metadata:
        metadata["z_table"] = _coerce_z_table(metadata["z_table"])
    if "e0_values" in metadata:
        metadata["e0_values"] = np.asarray(metadata["e0_values"], dtype=float).tolist()
    return metadata


def _legacy_train_state(raw: Dict[str, Any]) -> Optional[TrainState]:
    keys = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "ema_state_dict",
        "epoch",
        "best_val_loss",
        "lmdb_indices",
        "config",
        "best_model_state_dict",
    }
    train_state = {k: raw[k] for k in keys if k in raw}
    return train_state or None


def load_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    raw = torch.load(checkpoint_path, map_location=map_location)
    model_state = raw.get("model_state_dict", raw)
    metadata = raw.get("metadata") or _legacy_metadata(raw)
    train_state = raw.get("train_state") or _legacy_train_state(raw)
    best_model_state = raw.get("best_model_state_dict")
    return {
        "model_state_dict": model_state,
        "best_model_state_dict": best_model_state,
        "metadata": metadata or {},
        "train_state": train_state,
        "raw": raw,
    }


def save_checkpoint(
    path: Path | str,
    model_state_dict: Dict[str, Any],
    metadata: Metadata,
    train_state: Optional[TrainState] = None,
    best_model_state_dict: Optional[Dict[str, Any]] = None,
) -> Path:
    checkpoint = {
        "model_state_dict": model_state_dict,
        "metadata": metadata,
    }
    # Mirror key metadata fields for backward compatibility with older loaders.
    for key in ("z_table", "avg_num_neighbors", "e0_values", "cutoff", "num_interactions"):
        if key in metadata:
            checkpoint[key] = metadata[key]
    if best_model_state_dict is not None:
        checkpoint["best_model_state_dict"] = best_model_state_dict
    if train_state:
        checkpoint["train_state"] = train_state
    torch.save(checkpoint, path)
    return Path(path)
