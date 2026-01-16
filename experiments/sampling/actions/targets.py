"""Target generation helpers for actions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from experiments.sampling.geometry import split_by_bond
from experiments.sampling.schema import Structure


def candidate_pairs(structure: Structure, min_dist: float, max_dist: float) -> np.ndarray:
    positions = structure.positions
    n = positions.shape[0]
    if n < 2:
        return np.zeros((1, 2), dtype=int)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[j] - positions[i])
            if min_dist <= dist <= max_dist:
                pairs.append((i, j))
    if not pairs:
        return np.zeros((1, 2), dtype=int)
    return np.array(pairs, dtype=int)


def _covalent_radii() -> np.ndarray:
    try:
        from ase.data import covalent_radii
    except Exception:
        return np.ones(119, dtype=float)
    return np.asarray(covalent_radii, dtype=float)


def _bonded_pairs(structure: Structure, bond_factor: float, bond_cap: float | None) -> list[tuple[int, int]]:
    positions = structure.positions
    numbers = structure.numbers
    n = positions.shape[0]
    radii = _covalent_radii()
    pairs = []
    for i in range(n):
        ri = radii[int(numbers[i])] if int(numbers[i]) < radii.size else 1.0
        for j in range(i + 1, n):
            rj = radii[int(numbers[j])] if int(numbers[j]) < radii.size else 1.0
            thresh = bond_factor * (ri + rj)
            if bond_cap is not None:
                thresh = min(thresh, bond_cap)
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist <= thresh:
                pairs.append((i, j))
    return pairs


def _build_adjacency(natoms: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(natoms)]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def candidate_dihedrals(
    structure: Structure,
    *,
    bond_factor: float = 1.25,
    bond_cap: float | None = None,
) -> List[dict]:
    positions = structure.positions
    n = positions.shape[0]
    if n < 4:
        return []
    pairs = _bonded_pairs(structure, bond_factor=bond_factor, bond_cap=bond_cap)
    adjacency = _build_adjacency(n, pairs)
    dihedrals = []
    seen = set()
    for j in range(n):
        for k in adjacency[j]:
            if j >= k:
                continue
            neighbors_j = [i for i in adjacency[j] if i != k]
            neighbors_k = [l for l in adjacency[k] if l != j]
            if not neighbors_j or not neighbors_k:
                continue
            for i in neighbors_j:
                for l in neighbors_k:
                    key = (i, j, k, l)
                    if key in seen:
                        continue
                    seen.add(key)
                    comp_j, comp_k = split_by_bond(adjacency, j, k)
                    rot_mask = comp_k if l in comp_k else comp_j
                    rot_mask = set(rot_mask)
                    # keep j,k anchored
                    rot_mask.discard(j)
                    rot_mask.discard(k)
                    dihedrals.append({
                        "dihedral": (i, j, k, l),
                        "rot_mask": np.array(sorted(rot_mask), dtype=int),
                    })
    return dihedrals
