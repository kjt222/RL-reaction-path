"""Geometry helpers for sampling actions."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    return vec / norm


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rotate_points(points: np.ndarray, center: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    rot = rotation_matrix(axis, angle_rad)
    shifted = points - center
    return shifted @ rot.T + center


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_adjacency(positions: np.ndarray, cutoff: float) -> list[list[int]]:
    n = positions.shape[0]
    dists = pairwise_distances(positions)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= cutoff:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def split_by_bond(adjacency: list[list[int]], i: int, j: int) -> Tuple[set[int], set[int]]:
    """Split graph into two components by removing bond i-j."""
    n = len(adjacency)
    visited = [False] * n

    def bfs(start: int, blocked: tuple[int, int]) -> set[int]:
        queue = [start]
        comp = set()
        visited[start] = True
        while queue:
            node = queue.pop()
            comp.add(node)
            for nb in adjacency[node]:
                if (node, nb) == blocked or (nb, node) == blocked:
                    continue
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return comp

    comp_i = bfs(i, (i, j))
    comp_j = bfs(j, (i, j))
    return comp_i, comp_j
