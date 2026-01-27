"""Reusable structure validators."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple, Optional, Mapping, Any

import numpy as np


ValidatorFn = Callable[["Structure"], tuple[bool, Dict[str, float]]]


def min_dist(positions: np.ndarray, min_dist: float) -> Tuple[bool, Dict[str, float]]:
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


def min_dist_struct(
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


def make_min_dist(
    *,
    min_factor: float = 0.7,
    hard_min: float = 0.7,
) -> ValidatorFn:
    def _validator(structure):
        return min_dist_struct(structure.positions, structure.numbers, min_factor=min_factor, hard_min=hard_min)

    return _validator


def make_fixed(
    ref_positions: np.ndarray,
    fixed_mask: np.ndarray,
    *,
    tol: float = 1e-3,
) -> ValidatorFn:
    fixed_mask = np.asarray(fixed_mask).astype(bool)
    ref_positions = np.asarray(ref_positions)

    def _validator(structure):
        if fixed_mask.size == 0:
            return True, {"fixed_tol": tol}
        delta = np.linalg.norm(structure.positions[fixed_mask] - ref_positions[fixed_mask], axis=1)
        max_move = float(delta.max()) if delta.size else 0.0
        return max_move <= tol, {"fixed_tol": tol, "fixed_max_move": max_move}

    return _validator


def make_bond(
    ref_positions: np.ndarray,
    bond_pairs: Iterable[tuple[int, int]],
    *,
    max_factor: float = 1.5,
    max_abs: float | None = None,
) -> ValidatorFn:
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


def _action_magnitude(action: Optional[str], params: Mapping[str, Any]) -> Optional[float]:
    if not action:
        return None
    try:
        if action == "rigid_translate":
            delta = params.get("delta")
            if isinstance(delta, (list, tuple)):
                return float(sum(float(x) ** 2 for x in delta) ** 0.5)
        if action == "rigid_rotate":
            angle = params.get("angle_rad")
            if angle is not None:
                return abs(float(angle)) * 180.0 / 3.141592653589793
        if action == "dihedral_twist":
            angle = params.get("angle_rad")
            if angle is not None:
                return abs(float(angle)) * 180.0 / 3.141592653589793
        if action == "push_pull":
            delta = params.get("delta")
            if delta is not None:
                return abs(float(delta))
        if action == "jitter":
            sigma = params.get("sigma")
            if sigma is not None:
                return abs(float(sigma))
        if action == "md":
            temp = params.get("temperature_K")
            if temp is not None:
                return abs(float(temp))
    except Exception:
        return None
    return None


class QualityGate:
    """Score-and-reject gate using action/force diagnostics."""

    requires_force = True

    def __init__(
        self,
        *,
        force_source: str = "force_pre",
        max_force: Optional[float] = None,
        min_force: Optional[float] = None,
        score_threshold: float = 1.0,
        action_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._force_source = force_source
        self._max_force = max_force
        self._min_force = min_force
        self._score_threshold = float(score_threshold)
        self._action_ranges = action_ranges or {}

    def __call__(self, structure):
        info: Dict[str, float] = {}
        action_name = structure.info.get("action_name")
        action_params = structure.info.get("action_params", {})
        action_mag = structure.info.get("action_magnitude")
        if action_mag is None and isinstance(action_params, dict):
            action_mag = _action_magnitude(str(action_name) if action_name else None, action_params)
        if action_mag is not None:
            info["quality_action_mag"] = float(action_mag)

        force_stats = structure.info.get(self._force_source)
        force_max = None
        if isinstance(force_stats, dict):
            try:
                force_max = float(force_stats.get("max"))
            except Exception:
                force_max = None
        if force_max is not None:
            info["quality_force_max"] = float(force_max)

        components = []
        reasons = []

        if self._max_force is not None or self._min_force is not None:
            if force_max is None:
                reasons.append("missing_force")
                components.append(0.0)
            else:
                ok = True
                if self._max_force is not None and force_max > self._max_force:
                    ok = False
                    reasons.append("force_max")
                if self._min_force is not None and force_max < self._min_force:
                    ok = False
                    reasons.append("force_min")
                components.append(1.0 if ok else 0.0)
            if self._max_force is not None:
                info["quality_force_max_thr"] = float(self._max_force)
            if self._min_force is not None:
                info["quality_force_min_thr"] = float(self._min_force)

        if self._action_ranges:
            if action_mag is None:
                reasons.append("missing_action_mag")
                components.append(0.0)
            else:
                cfg = self._action_ranges.get(str(action_name), self._action_ranges.get("default", {}))
                min_mag = cfg.get("min")
                max_mag = cfg.get("max")
                ok = True
                if min_mag is not None and action_mag < float(min_mag):
                    ok = False
                    reasons.append("action_min")
                if max_mag is not None and action_mag > float(max_mag):
                    ok = False
                    reasons.append("action_max")
                components.append(1.0 if ok else 0.0)
                if min_mag is not None:
                    info["quality_action_min_thr"] = float(min_mag)
                if max_mag is not None:
                    info["quality_action_max_thr"] = float(max_mag)

        if components:
            score = float(sum(components)) / float(len(components))
        else:
            score = 1.0
        info["quality_score"] = score
        info["quality_threshold"] = self._score_threshold
        ok = score >= self._score_threshold
        info["quality_pass"] = 1.0 if ok else 0.0
        if not ok:
            info["quality_reject"] = 1.0
            if reasons:
                info["quality_reason"] = 1.0  # marker for action_quality; detail kept in logs
        return ok, info


def make_quality_gate(
    *,
    force_source: str = "force_pre",
    max_force: Optional[float] = None,
    min_force: Optional[float] = None,
    score_threshold: float = 1.0,
    action_ranges: Optional[Dict[str, Dict[str, float]]] = None,
) -> ValidatorFn:
    return QualityGate(
        force_source=force_source,
        max_force=max_force,
        min_force=min_force,
        score_threshold=score_threshold,
        action_ranges=action_ranges,
    )
