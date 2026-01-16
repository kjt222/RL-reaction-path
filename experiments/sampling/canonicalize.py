"""Canonicalize structures into a slab-fixed reference frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from experiments.sampling.schema import Structure


def _normalize(vec: np.ndarray, *, eps: float = 1e-12) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def _cell_axes(cell: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if cell is None:
        return None
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        return None
    a = _normalize(cell[0])
    b = _normalize(cell[1])
    if a is None or b is None:
        return None
    e3 = _normalize(np.cross(a, b))
    if e3 is None:
        return None
    e2 = np.cross(e3, a)
    e2 = _normalize(e2)
    if e2 is None:
        return None
    return a, e2, e3


def _pca_axes(positions: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if positions.shape[0] < 3:
        return None
    centered = positions - positions.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    if vals.size < 3:
        return None
    order = np.argsort(vals)
    # smallest variance as normal (e3), largest as in-plane (e1)
    e3 = _normalize(vecs[:, order[0]])
    e1 = _normalize(vecs[:, order[-1]])
    if e3 is None or e1 is None:
        return None
    e2 = _normalize(np.cross(e3, e1))
    if e2 is None:
        return None
    return e1, e2, e3


def _choose_axes(positions: np.ndarray, cell: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    axes_cell = _cell_axes(cell) if cell is not None else None
    axes_pca = _pca_axes(positions)
    if axes_cell is not None and axes_pca is not None:
        if abs(float(np.dot(axes_cell[2], axes_pca[2]))) < 0.7:
            return axes_pca
        return axes_cell
    return axes_cell or axes_pca


def _estimate_layer_spacing(z: np.ndarray) -> Optional[float]:
    z_sorted = np.sort(z)
    dz = np.diff(z_sorted)
    dz = dz[dz > 1e-6]
    if dz.size == 0:
        return None
    med = float(np.median(dz))
    small = dz[dz <= 1.5 * med]
    if small.size == 0:
        return med
    return float(np.median(small))


def _anchor_indices(structure: Structure, e3: np.ndarray) -> np.ndarray:
    if structure.fixed is not None:
        fixed = np.asarray(structure.fixed, dtype=bool)
        idx = np.where(fixed)[0]
        if idx.size > 0:
            return idx
    positions = np.asarray(structure.positions, dtype=float)
    z = positions @ e3
    z_min = float(np.min(z))
    d_layer = _estimate_layer_spacing(z)
    thickness = 2.5 * d_layer if d_layer is not None else 3.0
    return np.where(z <= z_min + thickness)[0]


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # Solve P -> Q rotation
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


@dataclass
class SlabCanonicalizer:
    """Canonicalize a structure to a slab-fixed reference frame."""

    reference_anchor: Optional[np.ndarray] = None

    def __call__(self, structure: Structure) -> Structure:
        positions = np.asarray(structure.positions, dtype=float)
        axes = _choose_axes(positions, structure.cell)
        if axes is None:
            return structure
        e1, e2, e3 = axes
        anchor_idx = _anchor_indices(structure, e3)
        if anchor_idx.size == 0:
            return structure
        anchor = positions[anchor_idx]
        center = anchor.mean(axis=0, keepdims=True)
        centered = positions - center
        if self.reference_anchor is not None:
            ref = np.asarray(self.reference_anchor, dtype=float)
            if ref.shape == anchor.shape:
                R = _kabsch(anchor - center, ref - ref.mean(axis=0, keepdims=True))
                rotated = centered @ R
            else:
                rotated = centered @ np.stack([e1, e2, e3], axis=1)
        else:
            rotated = centered @ np.stack([e1, e2, e3], axis=1)
        new_struct = structure.copy()
        new_struct.positions = rotated
        return new_struct
