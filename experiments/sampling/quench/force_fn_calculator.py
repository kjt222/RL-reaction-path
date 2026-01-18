"""ASE calculator that delegates energy/forces to a force_fn."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - ASE is optional
    from ase.calculators.calculator import Calculator, all_changes
except Exception:  # pragma: no cover - optional dependency
    Calculator = object  # type: ignore
    all_changes = None  # type: ignore

from experiments.sampling.schema import Structure


ForceFn = Callable[[Structure], object]


class ForceFnCalculator(Calculator):
    """ASE calculator wrapper for a force_fn returning (energy, forces)."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        force_fn: ForceFn,
        *,
        fixed: Optional[np.ndarray] = None,
        tags: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self._force_fn = force_fn
        self._fixed = None if fixed is None else np.asarray(fixed, dtype=int)
        self._tags = None if tags is None else np.asarray(tags, dtype=int)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:  # type: ignore[override]
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("ForceFnCalculator requires ASE Atoms")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = atoms.get_cell().array if atoms.get_cell() is not None else None
        pbc = atoms.get_pbc()

        if self._fixed is not None and self._fixed.shape[0] != numbers.shape[0]:
            raise ValueError("fixed mask length does not match number of atoms")
        if self._tags is not None and self._tags.shape[0] != numbers.shape[0]:
            raise ValueError("tags length does not match number of atoms")

        structure = Structure(
            positions=positions,
            numbers=numbers,
            cell=None if cell is None else np.asarray(cell, dtype=float),
            pbc=None if pbc is None else np.asarray(pbc, dtype=bool),
            tags=self._tags,
            fixed=self._fixed,
        )
        result = self._force_fn(structure)
        energy, forces = _split_force_output(result)
        if forces is None:
            raise ValueError("force_fn did not return forces")
        if energy is None:
            energy = float("nan")
        self.results["energy"] = float(energy)
        self.results["forces"] = np.asarray(forces, dtype=float)


def _split_force_output(result: object) -> Tuple[Optional[float], Optional[np.ndarray]]:
    if result is None:
        return None, None
    if isinstance(result, dict):
        return _as_scalar(result.get("energy")), _as_forces(result.get("forces"))
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return _as_scalar(result[0]), _as_forces(result[1])
    return None, _as_forces(result)


def _as_scalar(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _as_forces(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr
