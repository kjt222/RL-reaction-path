"""Slab-aware selection masks for sampling actions."""

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
    e2 = _normalize(np.cross(e3, a))
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
    # smallest variance as normal, largest as in-plane axis
    e3 = _normalize(vecs[:, order[0]])
    e1 = _normalize(vecs[:, order[-1]])
    if e3 is None or e1 is None:
        return None
    e2 = _normalize(np.cross(e3, e1))
    if e2 is None:
        return None
    return e1, e2, e3


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


def _choose_normal(positions: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    normal = None
    axes_cell = _cell_axes(cell) if cell is not None else None
    axes_pca = _pca_axes(positions)
    if axes_cell is not None:
        normal = axes_cell[2]
        if axes_pca is not None:
            pca_normal = axes_pca[2]
            if abs(float(np.dot(normal, pca_normal))) < 0.7:
                normal = pca_normal
    elif axes_pca is not None:
        normal = axes_pca[2]
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    return normal


@dataclass
class SlabMasks:
    core_mask: np.ndarray
    movable_mask: np.ndarray
    normal: np.ndarray
    layer_spacing: Optional[float]


def infer_slab_masks(
    structure: Structure,
    *,
    core_layers: float = 2.5,
) -> SlabMasks:
    """Infer slab core mask using fixed atoms or geometric layers."""

    positions = np.asarray(structure.positions, dtype=float)
    normal = _choose_normal(positions, structure.cell)
    if structure.fixed is not None:
        fixed = np.asarray(structure.fixed, dtype=bool)
        if fixed.any():
            movable = ~fixed
            return SlabMasks(core_mask=fixed, movable_mask=movable, normal=normal, layer_spacing=None)

    z = positions @ normal
    z_min = float(np.min(z))
    d_layer = _estimate_layer_spacing(z)
    thickness = core_layers * d_layer if d_layer is not None else 3.0
    core_mask = z <= (z_min + thickness)
    movable_mask = ~core_mask
    return SlabMasks(core_mask=core_mask, movable_mask=movable_mask, normal=normal, layer_spacing=d_layer)


def build_action_inputs(structure: Structure, *, core_layers: float = 2.5) -> Tuple[np.ndarray, dict]:
    """Helper to build selection_mask and candidates for sampling actions."""
    masks = infer_slab_masks(structure, core_layers=core_layers)
    candidates = {"core_mask": masks.core_mask, "slab_normal": masks.normal, "layer_spacing": masks.layer_spacing}
    return masks.movable_mask, candidates
