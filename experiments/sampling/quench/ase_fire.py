"""ASE FIRE quench."""

from __future__ import annotations

from typing import Any, Dict, Optional

from experiments.sampling.quench.base import QuenchBase
from experiments.sampling.schema import QuenchResult, Structure


class ASEFIREQuench(QuenchBase):
    name = "ase_fire"

    def __init__(self, calculator: Any, *, fmax: float = 0.1, steps: int = 100) -> None:
        self._calculator = calculator
        self._fmax = fmax
        self._steps = steps

    def run(self, structure: Structure) -> QuenchResult:
        try:
            from ase import Atoms
            from ase.optimize import FIRE
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("ASE is required for ASEFIREQuench") from exc

        atoms = Atoms(
            numbers=structure.numbers.tolist(),
            positions=structure.positions,
            cell=structure.cell,
            pbc=structure.pbc,
        )
        if self._calculator is None:
            raise ValueError("ASEFIREQuench requires a calculator")
        atoms.set_calculator(self._calculator)

        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=self._fmax, steps=self._steps)

        new_struct = structure.copy()
        new_struct.positions = atoms.get_positions()
        forces = atoms.get_forces()
        info = {"fmax": self._fmax, "steps": self._steps}
        return QuenchResult(
            structure=new_struct,
            converged=opt.converged(),
            steps=opt.nsteps,
            forces=forces,
            info=info,
        )
