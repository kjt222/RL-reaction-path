"""Rigid rotation action."""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.sampling.actions.base import ActionBase, ActionContext
from experiments.sampling.geometry import random_unit_vector, rotate_points
from experiments.sampling.schema import ActionOp, Structure


class RigidRotateAction(ActionBase):
    name = "rigid_rotate"

    def sample(self, ctx: ActionContext, rng: np.random.Generator) -> ActionOp:
        cfg: Dict[str, float] = (ctx.config or {}).get("rigid_rotate", {})
        min_deg = float(cfg.get("min_deg", 2.0))
        max_deg = float(cfg.get("max_deg", 15.0))
        axis = random_unit_vector(rng)
        angle_deg = rng.uniform(min_deg, max_deg)
        angle_rad = np.deg2rad(angle_deg)
        indices = self._indices_from_mask(ctx.structure, ctx.selection_mask)
        return ActionOp(
            name=self.name,
            params={"axis": axis, "angle_rad": angle_rad},
            target_indices=indices,
        )

    def apply(self, structure: Structure, op: ActionOp) -> Structure:
        new_struct = structure.copy()
        indices = op.target_indices if op.target_indices is not None else np.arange(structure.natoms)
        axis = np.asarray(op.params["axis"], dtype=float)
        angle_rad = float(op.params["angle_rad"])
        center = new_struct.positions[indices].mean(axis=0)
        new_struct.positions[indices] = rotate_points(new_struct.positions[indices], center, axis, angle_rad)
        return new_struct
