"""Model-backed force_fn loader using manifest + weights."""

from __future__ import annotations

import json
import contextlib
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from adapters import equiformer_v2, gemnet_oc, mace
from core.contracts import CanonicalBatch, ModelBundle
from core.registry import get_adapter
from experiments.sampling.schema import Structure


def _read_manifest_backend(manifest_path: Path) -> str:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    backend = data.get("backend")
    if not backend:
        raise ValueError("Manifest missing backend")
    return str(backend)


def load_bundle(
    *,
    manifest_path: str | Path,
    weights_path: str | Path | None = None,
    device: str = "cuda",
    head: Optional[str] = None,
) -> Tuple[ModelBundle, object]:
    """Load a model bundle and adapter from manifest."""

    equiformer_v2.register()
    gemnet_oc.register()
    mace.register()

    manifest_path = Path(manifest_path).expanduser().resolve()
    backend = _read_manifest_backend(manifest_path)
    adapter_cls = get_adapter(backend)
    adapter = adapter_cls()
    bundle = adapter.load_from_manifest(str(manifest_path), device=device, weights_path=str(weights_path) if weights_path else None)
    extras = dict(bundle.extras or {})
    if head is not None:
        extras["head"] = head
    bundle.extras = extras
    return bundle, adapter


def _structure_to_batch(structure: Structure) -> CanonicalBatch:
    cbatch: CanonicalBatch = {
        "z": np.asarray(structure.numbers, dtype=np.int64),
        "pos": np.asarray(structure.positions, dtype=np.float32),
        "cell": None if structure.cell is None else np.asarray(structure.cell, dtype=np.float32),
        "pbc": None if structure.pbc is None else np.asarray(structure.pbc, dtype=np.int8),
        "tags": None if structure.tags is None else np.asarray(structure.tags, dtype=np.int16),
        "fixed": None if structure.fixed is None else np.asarray(structure.fixed, dtype=np.int8),
        "natoms": np.asarray([structure.natoms], dtype=np.int64),
    }
    return cbatch


def build_force_fn(
    *,
    manifest_path: str | Path,
    weights_path: str | Path | None = None,
    device: str = "cuda",
    head: Optional[str] = None,
    use_amp: bool = False,
) -> Callable[[Structure], Tuple[np.ndarray, np.ndarray]]:
    """Return a callable that maps Structure -> (energy, forces)."""

    bundle, adapter = load_bundle(
        manifest_path=manifest_path,
        weights_path=weights_path,
        device=device,
        head=head,
    )

    amp_enabled = bool(use_amp and str(device).startswith("cuda"))

    def _force_fn(structure: Structure) -> Tuple[float, np.ndarray]:
        cbatch = _structure_to_batch(structure)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16) if amp_enabled else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast:
            outputs = adapter.predict(bundle, cbatch)
        energy = outputs.get("energy")
        forces = outputs.get("forces")
        if energy is None or forces is None:
            raise ValueError("Model outputs missing energy or forces")
        if torch.is_tensor(energy):
            energy = energy.detach().cpu().numpy()
        if torch.is_tensor(forces):
            forces = forces.detach().cpu().numpy()
        energy_val = float(np.asarray(energy).reshape(-1)[0])
        return energy_val, np.asarray(forces)

    return _force_fn
