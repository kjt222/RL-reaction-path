"""Structure validity checks (optional)."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


def validate_min_distance(positions: np.ndarray, min_dist: float) -> Tuple[bool, Dict[str, float]]:
    n = positions.shape[0]
    if n < 2:
        return True, {"min_dist": min_dist}
    min_seen = None
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(positions[j] - positions[i]))
            if min_seen is None or dist < min_seen:
                min_seen = dist
            if dist < min_dist:
                return False, {"min_dist": min_dist, "min_seen": dist}
    return True, {"min_dist": min_dist, "min_seen": min_seen or float("inf")}


def _covalent_radii() -> np.ndarray:
    try:
        from ase.data import covalent_radii
    except Exception:
        return np.ones(119, dtype=float)
    return np.asarray(covalent_radii, dtype=float)


def validate_min_distance_structure(
    positions: np.ndarray,
    numbers: np.ndarray,
    *,
    min_factor: float = 0.7,
    hard_min: float = 0.7,
) -> Tuple[bool, Dict[str, float]]:
    n = positions.shape[0]
    if n < 2:
        return True, {"min_factor": min_factor, "hard_min": hard_min}
    radii = _covalent_radii()
    min_seen = None
    for i in range(n):
        ri = radii[int(numbers[i])] if int(numbers[i]) < radii.size else 1.0
        for j in range(i + 1, n):
            rj = radii[int(numbers[j])] if int(numbers[j]) < radii.size else 1.0
            threshold = max(hard_min, min_factor * (ri + rj))
            dist = float(np.linalg.norm(positions[j] - positions[i]))
            if min_seen is None or dist < min_seen:
                min_seen = dist
            if dist < threshold:
                return False, {"min_factor": min_factor, "hard_min": hard_min, "min_seen": dist}
    return True, {"min_factor": min_factor, "hard_min": hard_min, "min_seen": min_seen or float("inf")}

def build_min_distance_validator(
    *,
    min_factor: float = 0.7,
    hard_min: float = 0.7,
) -> Callable[["Structure"], Tuple[bool, Dict[str, float]]]:
    def _validator(structure):
        return validate_min_distance_structure(structure.positions, structure.numbers, min_factor=min_factor, hard_min=hard_min)

    return _validator


def build_fixed_mask_validator(
    ref_positions: np.ndarray,
    fixed_mask: np.ndarray,
    *,
    tol: float = 1e-3,
) -> Callable[["Structure"], Tuple[bool, Dict[str, float]]]:
    fixed_mask = np.asarray(fixed_mask).astype(bool)
    ref_positions = np.asarray(ref_positions)

    def _validator(structure):
        if fixed_mask.size == 0:
            return True, {"fixed_tol": tol}
        delta = np.linalg.norm(structure.positions[fixed_mask] - ref_positions[fixed_mask], axis=1)
        max_move = float(delta.max()) if delta.size else 0.0
        return max_move <= tol, {"fixed_tol": tol, "fixed_max_move": max_move}

    return _validator


def build_bond_stretch_validator(
    ref_positions: np.ndarray,
    bond_pairs: Iterable[tuple[int, int]],
    *,
    max_factor: float = 1.5,
    max_abs: float | None = None,
) -> Callable[["Structure"], Tuple[bool, Dict[str, float]]]:
    ref_positions = np.asarray(ref_positions)
    ref_pairs = list(bond_pairs)
    if not ref_pairs:
        def _noop(_structure):
            return True, {"bond_max_factor": max_factor}
        return _noop

    ref_dists = []
    for i, j in ref_pairs:
        ref_dists.append(float(np.linalg.norm(ref_positions[j] - ref_positions[i])))

    def _validator(structure):
        max_seen = 0.0
        for (i, j), ref_d in zip(ref_pairs, ref_dists):
            dist = float(np.linalg.norm(structure.positions[j] - structure.positions[i]))
            max_allowed = max_factor * ref_d
            if max_abs is not None:
                max_allowed = min(max_allowed, max_abs)
            if dist > max_allowed:
                return False, {"bond_max_factor": max_factor, "bond_max_abs": max_abs or 0.0, "bond_seen": dist}
            max_seen = max(max_seen, dist)
        return True, {"bond_max_factor": max_factor, "bond_max_abs": max_abs or 0.0, "bond_max_seen": max_seen}

    return _validator
