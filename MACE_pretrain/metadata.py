"""Shared helpers for packing and loading model metadata and checkpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from mace import tools

Metadata = Dict[str, Any]
TrainState = Dict[str, Any]

LOGGER = logging.getLogger(__name__)


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj

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


def _metadata_json_path(checkpoint_path: Path | str, json_path: Path | None = None) -> Path:
    if json_path is not None:
        return Path(json_path)
    cp = Path(checkpoint_path)
    base_dir = cp if cp.is_dir() else cp.parent
    return base_dir / "metadata.json"


def _load_metadata_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_metadata_json(json_path: Path, metadata: Metadata) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)
    LOGGER.info("Wrote metadata.json to %s", json_path)


def load_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
    json_path: Path | None = None,
    strict_json: bool = True,
    allow_json_to_metadata: bool = False,
    write_json_if_missing: bool = True,
    persist_metadata: bool = False,
) -> Dict[str, Any]:
    raw = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model_state = raw.get("model_state_dict", raw)
    metadata = raw.get("metadata") or _legacy_metadata(raw)
    train_state = raw.get("train_state") or _legacy_train_state(raw)
    best_model_state = raw.get("best_model_state_dict")

    json_file = _metadata_json_path(checkpoint_path, json_path)
    json_meta = None
    if json_file.exists():
        json_meta = _load_metadata_json(json_file)
        LOGGER.info("Loaded metadata.json from %s", json_file)

    if json_meta is not None:
        if allow_json_to_metadata:
            if metadata and _normalize(metadata) != _normalize(json_meta):
                LOGGER.warning("metadata differs from metadata.json; using JSON to override as requested.")
            metadata = json_meta
            if persist_metadata:
                try:
                    updated = dict(raw)
                    updated["metadata"] = metadata
                    torch.save(updated, checkpoint_path)
                    LOGGER.info("Persisted metadata into checkpoint: %s", checkpoint_path)
                except Exception:
                    LOGGER.warning("Failed to persist metadata into checkpoint %s", checkpoint_path)
        else:
            if metadata:
                if strict_json and _normalize(metadata) != _normalize(json_meta):
                    raise ValueError("metadata mismatch between checkpoint and JSON; please resolve before proceeding.")
            else:
                raise ValueError(
                    "metadata missing in checkpoint but metadata.json present; "
                    "reload with allow_json_to_metadata=True if you trust the JSON."
                )

    if metadata and write_json_if_missing and not json_file.exists():
        try:
            _write_metadata_json(json_file, metadata)
        except Exception:
            LOGGER.warning("Failed to write metadata.json to %s", json_file)

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
