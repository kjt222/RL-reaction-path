"""Push/pull pair action."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from experiments.sampling.actions.base import ActionBase, ActionContext
from experiments.sampling.schema import ActionOp, Structure


class PushPullAction(ActionBase):
    name = "push_pull"

    def sample(self, ctx: ActionContext, rng: np.random.Generator) -> ActionOp:
        cfg: Dict[str, float] = (ctx.config or {}).get("push_pull", {})
        min_delta = float(cfg.get("min_delta", 0.05))
        max_delta = float(cfg.get("max_delta", 0.25))

        candidates = None
        core_mask = None
        if ctx.candidates:
            candidates = ctx.candidates.get("pairs")
            core_mask = ctx.candidates.get("core_mask")
        if candidates is not None and core_mask is not None:
            candidates = self._filter_pairs(candidates, core_mask)
        if candidates is None:
            candidates = self._random_pairs(ctx.structure.natoms, rng, core_mask=core_mask)

        i, j = candidates[rng.integers(len(candidates))]
        sign = rng.choice([-1.0, 1.0])
        magnitude = rng.uniform(min_delta, max_delta)
        delta = sign * magnitude
        return ActionOp(name=self.name, params={"pair": (int(i), int(j)), "delta": delta})

    def apply(self, structure: Structure, op: ActionOp) -> Structure:
        new_struct = structure.copy()
        i, j = op.params["pair"]
        delta = float(op.params["delta"])
        vec = new_struct.positions[j] - new_struct.positions[i]
        norm = np.linalg.norm(vec)
        if norm == 0:
            return new_struct
        direction = vec / norm
        shift = 0.5 * delta * direction
        new_struct.positions[i] -= shift
        new_struct.positions[j] += shift
        return new_struct

    @staticmethod
    def _random_pairs(
        natoms: int,
        rng: np.random.Generator,
        *,
        core_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if natoms < 2:
            raise ValueError("Not enough atoms for push/pull pairs")
        if core_mask is not None:
            core_mask = np.asarray(core_mask, dtype=bool)
            if core_mask.all():
                raise ValueError("All atoms are core; no valid push/pull pairs")
        # sample a small pool of random pairs
        size = min(128, natoms * (natoms - 1) // 2)
        pairs = set()
        max_tries = size * 10
        tries = 0
        while len(pairs) < size and tries < max_tries:
            i = rng.integers(natoms)
            j = rng.integers(natoms)
            if i == j:
                tries += 1
                continue
            if core_mask is not None and core_mask[i] and core_mask[j]:
                tries += 1
                continue
            pair = (min(i, j), max(i, j))
            pairs.add(pair)
            tries += 1
        if not pairs:
            raise ValueError("No valid push/pull pairs after filtering")
        return np.array(list(pairs), dtype=int)

    @staticmethod
    def _filter_pairs(pairs: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
        core_mask = np.asarray(core_mask, dtype=bool)
        kept = []
        for i, j in pairs:
            if core_mask[int(i)] and core_mask[int(j)]:
                continue
            kept.append((int(i), int(j)))
        if not kept:
            raise ValueError("No valid push/pull pairs after core filtering")
        return np.asarray(kept, dtype=int)
