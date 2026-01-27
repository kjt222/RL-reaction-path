"""Short MD action using the current PES as a force provider."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.sampling.actions.base import ActionBase, ActionContext
from experiments.sampling.schema import ActionOp, Structure


class MDAction(ActionBase):
    name = "md"

    def __init__(self, calculator) -> None:
        self._calculator = calculator
        try:
            import ase  # noqa: F401

            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def sample(self, ctx: ActionContext, rng: np.random.Generator) -> ActionOp:
        if not self._available:
            raise RuntimeError("ASE is required for MDAction")
        cfg: Dict[str, Any] = {}
        if ctx.config and isinstance(ctx.config.get("md"), dict):
            cfg = dict(ctx.config.get("md") or {})

        min_temp = float(cfg.get("min_temp_K", 50.0))
        max_temp = float(cfg.get("max_temp_K", 400.0))
        if max_temp < min_temp:
            min_temp, max_temp = max_temp, min_temp
        temperature = float(rng.uniform(min_temp, max_temp))

        min_steps = int(cfg.get("min_steps", 5))
        max_steps = int(cfg.get("max_steps", 30))
        if max_steps < min_steps:
            min_steps, max_steps = max_steps, min_steps
        steps = int(rng.integers(min_steps, max_steps + 1))

        if "dt_fs" in cfg:
            dt_fs = float(cfg.get("dt_fs"))
        else:
            min_dt = float(cfg.get("min_dt_fs", 0.5))
            max_dt = float(cfg.get("max_dt_fs", min_dt))
            if max_dt < min_dt:
                min_dt, max_dt = max_dt, min_dt
            dt_fs = float(rng.uniform(min_dt, max_dt))

        friction = float(cfg.get("friction", 0.02))
        integrator = str(cfg.get("integrator", "langevin")).lower()

        params = {
            "temperature_K": temperature,
            "steps": steps,
            "dt_fs": dt_fs,
            "friction": friction,
            "integrator": integrator,
        }
        return ActionOp(name=self.name, params=params)

    def apply(self, structure: Structure, op: ActionOp) -> Structure:
        if not self._available:
            raise RuntimeError("ASE is required for MDAction")

        try:
            from ase import Atoms
            from ase.constraints import FixAtoms
            from ase.geometry import wrap_positions
            from ase.md.andersen import Andersen
            from ase.md.langevin import Langevin
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
            from ase import units
        except Exception as exc:  # pragma: no cover - defensive import guard
            raise RuntimeError(f"Failed to import ASE MD components: {exc}") from exc

        params: Dict[str, Any] = dict(op.params or {})
        temperature = float(params.get("temperature_K", 300.0))
        steps = int(params.get("steps", 10))
        dt_fs = float(params.get("dt_fs", 0.5))
        friction = float(params.get("friction", 0.02))
        integrator = str(params.get("integrator", "langevin")).lower()

        atoms = Atoms(
            numbers=np.asarray(structure.numbers, dtype=int),
            positions=np.asarray(structure.positions, dtype=float),
            cell=structure.cell,
            pbc=structure.pbc,
        )
        if structure.tags is not None:
            atoms.set_tags(np.asarray(structure.tags, dtype=int))
        if structure.fixed is not None:
            fixed_mask = np.asarray(structure.fixed, dtype=bool)
            if fixed_mask.any():
                atoms.set_constraint(FixAtoms(mask=fixed_mask))

        atoms.calc = self._calculator

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms)

        timestep = dt_fs * units.fs
        if integrator == "andersen":
            dyn = Andersen(atoms, timestep=timestep, temperature_K=temperature)
        else:
            dyn = Langevin(atoms, timestep=timestep, temperature_K=temperature, friction=friction)
            integrator = "langevin"

        dyn.run(steps)

        new_positions = atoms.get_positions()
        try:
            if structure.cell is not None and structure.pbc is not None:
                new_positions = wrap_positions(new_positions, cell=structure.cell, pbc=structure.pbc)
        except Exception:
            pass

        md_energy: Optional[float]
        md_force_max: Optional[float]
        try:
            md_energy = float(atoms.get_potential_energy())
        except Exception:
            md_energy = None
        try:
            forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
            md_force_max = float(np.linalg.norm(forces, axis=1).max()) if forces.size else None
        except Exception:
            md_force_max = None

        info = dict(structure.info)
        info.update(
            {
                "md_temperature_K": temperature,
                "md_steps": steps,
                "md_dt_fs": dt_fs,
                "md_friction": friction,
                "md_integrator": integrator,
            }
        )
        if md_energy is not None:
            info["md_energy"] = md_energy
        if md_force_max is not None:
            info["md_force_max"] = md_force_max

        return Structure(
            positions=new_positions,
            numbers=structure.numbers,
            tags=structure.tags,
            fixed=structure.fixed,
            cell=structure.cell,
            pbc=structure.pbc,
            info=info,
        )

