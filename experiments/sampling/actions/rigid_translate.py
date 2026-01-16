"""Rigid translation action."""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.sampling.actions.base import ActionBase, ActionContext
from experiments.sampling.geometry import random_unit_vector
from experiments.sampling.schema import ActionOp, Structure


class RigidTranslateAction(ActionBase):
    name = "rigid_translate"

    def sample(self, ctx: ActionContext, rng: np.random.Generator) -> ActionOp:
        cfg: Dict[str, float] = (ctx.config or {}).get("rigid_translate", {})
        min_step = float(cfg.get("min_step", 0.05))
        max_step = float(cfg.get("max_step", 0.30))
        direction = random_unit_vector(rng)
        magnitude = rng.uniform(min_step, max_step)
        delta = direction * magnitude
        indices = self._indices_from_mask(ctx.structure, ctx.selection_mask)
        return ActionOp(name=self.name, params={"delta": delta}, target_indices=indices)

    def apply(self, structure: Structure, op: ActionOp) -> Structure:
        new_struct = structure.copy()
        indices = op.target_indices if op.target_indices is not None else np.arange(structure.natoms)
        delta = np.asarray(op.params["delta"], dtype=float)
        new_struct.positions[indices] += delta
        return new_struct
