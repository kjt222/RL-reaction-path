"""Level-1 basin fingerprint using element-pair distance histograms."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from experiments.sampling.basin.base import BasinBase
from experiments.sampling.schema import BasinResult, Structure


@dataclass
class HistogramBasin(BasinBase):
    """Element-pair distance histogram basin ID (permutation-invariant)."""

    name = "histogram"

    bin_width: float = 0.1
    max_dist: float = 6.0
    def _pair_key(self, zi: int, zj: int) -> Tuple[int, int]:
        return (zi, zj) if zi <= zj else (zj, zi)

    def identify(self, structure: Structure) -> BasinResult:
        positions = np.asarray(structure.positions, dtype=float)
        numbers = np.asarray(structure.numbers, dtype=int)
        n = positions.shape[0]
        if n == 0:
            raise ValueError("Empty structure for basin histogram")

        bins = int(np.floor(self.max_dist / self.bin_width))
        hist: Dict[Tuple[int, int], np.ndarray] = {}

        for i in range(n):
            for j in range(i + 1, n):
                vec = positions[j] - positions[i]
                dist = float(np.linalg.norm(vec))
                if dist > self.max_dist:
                    continue
                b = int(dist / self.bin_width)
                key = self._pair_key(int(numbers[i]), int(numbers[j]))
                if key not in hist:
                    hist[key] = np.zeros(bins, dtype=int)
                hist[key][b] += 1

        payload_parts = [np.asarray([self.bin_width, self.max_dist], dtype=np.float32).tobytes()]
        for key in sorted(hist.keys()):
            payload_parts.append(np.asarray(key, dtype=np.int16).tobytes())
            payload_parts.append(hist[key].tobytes())
        payload = b"".join(payload_parts)
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return BasinResult(
            basin_id=f"hist:{digest}",
            is_new=None,
            info={"bin_width": self.bin_width, "max_dist": self.max_dist},
        )
