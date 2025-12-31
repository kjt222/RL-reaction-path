"""Shared contracts for adapters and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, TypedDict


class CanonicalBatch(TypedDict, total=False):
    z: Any
    pos: Any
    cell: Any
    pbc: Any
    energy: Any
    forces: Any
    tags: Any
    fixed: Any
    ptr: Any
    natoms: Any
    head: Any
    energy_only: Any


class ModelOutputs(TypedDict, total=False):
    energy: Any
    forces: Any
    node_embed: Any
    graph_embed: Any


class EmbeddingOutputs(TypedDict, total=False):
    node_embed: Any
    graph_embed: Any


class MetricOutputs(TypedDict, total=False):
    energy_rmse: float
    energy_mae: float
    energy_mae_cfg: float
    force_rmse: float
    force_mae: float
    loss: float


def require_energy_forces(batch: Mapping[str, Any]) -> None:
    if batch.get("energy") is None:
        raise ValueError("CanonicalBatch missing energy")
    if batch.get("forces") is None:
        raise ValueError("CanonicalBatch missing forces")


@dataclass
class ModelBundle:
    model: Any
    manifest: Mapping[str, Any]
    backend: str
    device: str
    extras: Optional[Mapping[str, Any]] = None
