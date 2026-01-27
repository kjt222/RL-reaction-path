"""Action plugins for small post-action position noise."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from experiments.sampling.schema import Structure


def _movable_mask(structure: Structure, selection_mask: Optional[np.ndarray]) -> np.ndarray:
    if selection_mask is not None:
        mask = np.asarray(selection_mask, dtype=bool)
        if mask.shape[0] == structure.positions.shape[0]:
            return mask
    if structure.fixed is not None:
        fixed = np.asarray(structure.fixed, dtype=bool)
        return ~fixed
    return np.ones(structure.positions.shape[0], dtype=bool)


def make_position_noise(
    *,
    sigma: float,
    clip: Optional[float] = None,
    movable_only: bool = True,
):
    """Create a post-action noise plugin.

    The noise is applied after an action is taken but before validation/quench.
    """

    sigma = float(sigma)
    clip_val = float(clip) if clip is not None else None

    def _plugin(
        structure: Structure,
        _op,
        rng: np.random.Generator,
        selection_mask: Optional[np.ndarray],
    ) -> Tuple[Structure, Dict[str, object]]:
        if sigma <= 0.0:
            return structure, {"noise_sigma": 0.0, "noise_applied": 0.0}

        mask = _movable_mask(structure, selection_mask) if movable_only else np.ones(structure.positions.shape[0], dtype=bool)
        if not np.any(mask):
            return structure, {"noise_sigma": sigma, "noise_applied": 0.0, "noise_count": 0.0}

        noise = rng.normal(loc=0.0, scale=sigma, size=structure.positions.shape)
        if clip_val is not None:
            noise = np.clip(noise, -clip_val, clip_val)
        noise[~mask] = 0.0

        new_positions = structure.positions + noise

        # Wrap under PBC when possible to avoid runaway coordinates.
        try:
            from ase.geometry import wrap_positions

            if structure.cell is not None and structure.pbc is not None:
                new_positions = wrap_positions(new_positions, cell=structure.cell, pbc=structure.pbc)
        except Exception:
            pass

        moved = noise[mask]
        rms = float(np.sqrt(np.mean(np.sum(moved * moved, axis=1)))) if moved.size else 0.0
        max_mag = float(np.sqrt(np.max(np.sum(moved * moved, axis=1)))) if moved.size else 0.0

        info = dict(structure.info)
        info.update(
            {
                "noise_sigma": sigma,
                "noise_clip": clip_val,
                "noise_rms": rms,
                "noise_max": max_mag,
            }
        )

        new_struct = Structure(
            positions=new_positions,
            numbers=structure.numbers,
            tags=structure.tags,
            fixed=structure.fixed,
            cell=structure.cell,
            pbc=structure.pbc,
            info=info,
        )

        flags: Dict[str, object] = {
            "noise_sigma": sigma,
            "noise_applied": 1.0,
            "noise_count": float(np.count_nonzero(mask)),
            "noise_rms": rms,
            "noise_max": max_mag,
        }
        if clip_val is not None:
            flags["noise_clip"] = clip_val
        return new_struct, flags

    return _plugin
