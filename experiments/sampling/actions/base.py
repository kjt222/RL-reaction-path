"""Action base interfaces (experimental)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from experiments.sampling.schema import ActionOp, Structure


@dataclass
class ActionContext:
    structure: Structure
    selection_mask: Optional[np.ndarray] = None
    candidates: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class ActionBase:
    name: str = "base"

    def sample(self, ctx: ActionContext, rng: np.random.Generator) -> ActionOp:
        raise NotImplementedError

    def apply(self, structure: Structure, op: ActionOp) -> Structure:
        raise NotImplementedError

    @staticmethod
    def _indices_from_mask(structure: Structure, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return np.arange(structure.natoms)
        mask = np.asarray(mask).astype(bool)
        return np.where(mask)[0]
