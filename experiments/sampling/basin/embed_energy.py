"""Level-2 basin check (embedding + energy)."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from experiments.sampling.basin.base import BasinBase
from experiments.sampling.schema import BasinResult, Structure


class EmbedEnergyBasin(BasinBase):
    name = "embed_energy"

    def __init__(self, *, energy_tol: float = 0.05, embed_round: int = 3) -> None:
        self._energy_tol = energy_tol
        self._embed_round = embed_round

    def identify(self, structure: Structure) -> BasinResult:
        if "energy" not in structure.info or "embedding" not in structure.info:
            raise ValueError("Structure.info must include 'energy' and 'embedding' for embed_energy basin")
        energy = float(structure.info["energy"])
        embedding = np.asarray(structure.info["embedding"], dtype=float)
        embed = np.round(embedding, self._embed_round)
        payload = np.concatenate([[energy], embed])
        digest = hashlib.sha256(payload.tobytes()).hexdigest()[:16]
        return BasinResult(
            basin_id=f"emb:{digest}",
            is_new=None,
            info={"energy_tol": self._energy_tol, "embed_round": self._embed_round},
        )
