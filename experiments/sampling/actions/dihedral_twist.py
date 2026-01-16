"""Dihedral twist action."""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.sampling.actions.base import ActionBase, ActionContext
from experiments.sampling.actions.targets import candidate_dihedrals
from experiments.sampling.geometry import rotate_points
from experiments.sampling.schema import ActionOp, Structure


class DihedralTwistAction(ActionBase):
    name = "dihedral_twist"

    def sample(self, ctx: ActionContext, rng: np.random.Generator) -> ActionOp:
        cfg: Dict[str, float] = (ctx.config or {}).get("dihedral_twist", {})
        min_deg = float(cfg.get("min_deg", 5.0))
        max_deg = float(cfg.get("max_deg", 25.0))
        bond_factor = float(cfg.get("bond_factor", 1.25))
        bond_cap = cfg.get("bond_cap", None)
        bond_cap = float(bond_cap) if bond_cap is not None else None

        candidates = None
        core_mask = None
        if ctx.candidates:
            candidates = ctx.candidates.get("dihedrals")
            core_mask = ctx.candidates.get("core_mask")
        if candidates is None:
            candidates = candidate_dihedrals(ctx.structure, bond_factor=bond_factor, bond_cap=bond_cap)
        if candidates and core_mask is not None:
            candidates = self._filter_dihedrals(candidates, core_mask)
        if not candidates:
            raise ValueError("No dihedral candidates available")

        pick = candidates[rng.integers(len(candidates))]
        angle_deg = rng.uniform(min_deg, max_deg)
        angle_rad = np.deg2rad(angle_deg) * rng.choice([-1.0, 1.0])
        return ActionOp(
            name=self.name,
            params={
                "dihedral": pick["dihedral"],
                "rot_mask": pick["rot_mask"],
                "angle_rad": angle_rad,
            },
        )

    def apply(self, structure: Structure, op: ActionOp) -> Structure:
        new_struct = structure.copy()
        i, j, k, l = op.params["dihedral"]
        rot_mask = np.asarray(op.params["rot_mask"], dtype=int)
        angle_rad = float(op.params["angle_rad"])
        axis = new_struct.positions[k] - new_struct.positions[j]
        norm = np.linalg.norm(axis)
        if norm == 0 or rot_mask.size == 0:
            return new_struct
        axis = axis / norm
        center = new_struct.positions[j]
        new_struct.positions[rot_mask] = rotate_points(new_struct.positions[rot_mask], center, axis, angle_rad)
        return new_struct

    @staticmethod
    def _filter_dihedrals(candidates: list[dict], core_mask: np.ndarray) -> list[dict]:
        core_mask = np.asarray(core_mask, dtype=bool)
        kept: list[dict] = []
        for entry in candidates:
            i, j, k, l = entry["dihedral"]
            rot_mask = np.asarray(entry["rot_mask"], dtype=int)
            if core_mask[int(i)] or core_mask[int(j)] or core_mask[int(k)] or core_mask[int(l)]:
                continue
            if rot_mask.size and core_mask[rot_mask].any():
                continue
            kept.append(entry)
        return kept
