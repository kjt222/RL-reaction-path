"""Level-1 basin fingerprint (approximate)."""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np

from experiments.sampling.basin.base import BasinBase
from experiments.sampling.geometry import pairwise_distances
from experiments.sampling.schema import BasinResult, Structure


class FingerprintBasin(BasinBase):
    name = "fingerprint"

    def __init__(self, *, round_decimals: int = 3) -> None:
        self._round_decimals = round_decimals

    def identify(self, structure: Structure) -> BasinResult:
        distances = pairwise_distances(structure.positions)
        iu = np.triu_indices(distances.shape[0], k=1)
        dist_vals = np.round(distances[iu], self._round_decimals)
        numbers = np.sort(structure.numbers)
        payload = np.concatenate([numbers.astype(float), dist_vals])
        digest = hashlib.sha256(payload.tobytes()).hexdigest()[:16]
        return BasinResult(basin_id=f"fp:{digest}", is_new=None, info={"round": self._round_decimals})
