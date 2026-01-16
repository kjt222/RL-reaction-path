"""Force/PES interface for sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from experiments.sampling.schema import Structure


@dataclass
class ForceFieldOutput:
    energy: float
    forces: np.ndarray  # (N, 3)


class ForceFieldBase(Protocol):
    name: str

    def compute(self, structure: Structure) -> ForceFieldOutput:
        """Return energy and forces for a structure."""
        raise NotImplementedError
