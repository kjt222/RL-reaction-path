"""Local jitter action."""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.sampling.actions.base import ActionBase, ActionContext
from experiments.sampling.schema import ActionOp, Structure


class JitterAction(ActionBase):
    name = "jitter"

    def sample(self, ctx: ActionContext, rng: np.random.Generator) -> ActionOp:
        cfg: Dict[str, float] = (ctx.config or {}).get("jitter", {})
        sigma = float(cfg.get("sigma", 0.03))
        indices = self._indices_from_mask(ctx.structure, ctx.selection_mask)
        return ActionOp(name=self.name, params={"sigma": sigma}, target_indices=indices)

    def apply(self, structure: Structure, op: ActionOp) -> Structure:
        new_struct = structure.copy()
        sigma = float(op.params["sigma"])
        indices = op.target_indices if op.target_indices is not None else np.arange(structure.natoms)
        noise = np.random.normal(scale=sigma, size=(indices.shape[0], 3))
        new_struct.positions[indices] += noise
        return new_struct
