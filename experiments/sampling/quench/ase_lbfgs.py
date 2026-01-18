"""ASE L-BFGS quench."""

from __future__ import annotations

from typing import Any

from experiments.sampling.quench.base import QuenchBase
from experiments.sampling.schema import QuenchResult, Structure


class ASELBFGSQuench(QuenchBase):
    name = "ase_lbfgs"

    def __init__(self, calculator: Any, *, fmax: float = 0.1, steps: int = 100) -> None:
        self._calculator = calculator
        self._fmax = fmax
        self._steps = steps

    def run(self, structure: Structure, *, step_callback=None) -> QuenchResult:
        try:
            from ase import Atoms
            from ase.optimize import LBFGS
            from ase.constraints import FixAtoms
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("ASE is required for ASELBFGSQuench") from exc

        atoms = Atoms(
            numbers=structure.numbers.tolist(),
            positions=structure.positions,
            cell=structure.cell,
            pbc=structure.pbc,
        )
        if structure.tags is not None:
            atoms.set_tags(structure.tags.tolist())
        if structure.fixed is not None:
            mask = [bool(x) for x in structure.fixed.tolist()]
            if any(mask):
                atoms.set_constraint(FixAtoms(mask=mask))
        if self._calculator is None:
            raise ValueError("ASELBFGSQuench requires a calculator")
        atoms.set_calculator(self._calculator)

        opt = LBFGS(atoms, logfile=None)
        if step_callback is not None:
            def _on_step():
                step_struct = structure.copy()
                step_struct.positions = atoms.get_positions()
                calc = atoms.calc
                forces = None
                energy = None
                if calc is not None and getattr(calc, "results", None):
                    forces = calc.results.get("forces")
                    energy = calc.results.get("energy")
                if forces is None:
                    forces = atoms.get_forces()
                if energy is None:
                    energy = atoms.get_potential_energy()
                step_callback(step_struct, forces, float(energy), opt.nsteps)
            opt.attach(_on_step, interval=1)
        opt.run(fmax=self._fmax, steps=self._steps)

        new_struct = structure.copy()
        new_struct.positions = atoms.get_positions()
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()
        if forces.size:
            fmax_val = float((forces**2).sum(axis=1).max() ** 0.5)
        else:
            fmax_val = 0.0
        info = {"fmax": self._fmax, "steps": self._steps, "fmax_final": fmax_val}
        return QuenchResult(
            structure=new_struct,
            converged=fmax_val <= self._fmax,
            steps=opt.nsteps,
            forces=forces,
            energy=float(energy),
            info=info,
        )
