"""Sampling data structures (experimental)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class Structure:
    """Lightweight structure container for sampling."""

    positions: np.ndarray  # (N, 3)
    numbers: np.ndarray  # (N,)
    tags: Optional[np.ndarray] = None
    fixed: Optional[np.ndarray] = None
    cell: Optional[np.ndarray] = None  # (3, 3)
    pbc: Optional[np.ndarray] = None  # (3,)
    info: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Structure":
        return Structure(
            positions=self.positions.copy(),
            numbers=self.numbers.copy(),
            tags=None if self.tags is None else self.tags.copy(),
            fixed=None if self.fixed is None else self.fixed.copy(),
            cell=None if self.cell is None else self.cell.copy(),
            pbc=None if self.pbc is None else self.pbc.copy(),
            info=dict(self.info),
        )

    @property
    def natoms(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class ActionOp:
    name: str
    params: Dict[str, Any]
    target_indices: Optional[np.ndarray] = None


@dataclass
class QuenchResult:
    structure: Structure
    converged: bool
    steps: int
    forces: Optional[np.ndarray] = None
    energy: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BasinResult:
    basin_id: str
    is_new: Optional[bool] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleRecord:
    structure_in: Structure
    structure_pre: Structure
    structure_min: Optional[Structure]
    action: ActionOp
    quench: Optional[QuenchResult]
    basin: Optional[BasinResult]
    valid: bool
    flags: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


def ensure_numpy(array: Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)
