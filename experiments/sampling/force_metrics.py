"""Force statistics helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def force_stats(forces: np.ndarray, *, topk: Iterable[int] = (3, 5)) -> Dict[str, object]:
    """Compute scalar force statistics.

    Returns keys: max, mean, topk_mean, n.
    """
    if forces is None:
        return {}
    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError("forces must be shaped (N, 3)")
    if forces.size == 0:
        return {"max": 0.0, "mean": 0.0, "topk_mean": {}, "n": 0}
    norms = np.linalg.norm(forces, axis=1)
    stats: Dict[str, object] = {
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
        "topk_mean": {},
        "n": int(norms.shape[0]),
    }
    topk_mean: Dict[str, float] = {}
    for k in topk:
        k = int(k)
        if k <= 0:
            continue
        k_eff = min(k, norms.shape[0])
        if k_eff == 0:
            topk_mean[str(k)] = 0.0
            continue
        idx = np.argpartition(norms, -k_eff)[-k_eff:]
        topk_mean[str(k)] = float(np.mean(norms[idx]))
    stats["topk_mean"] = topk_mean
    return stats

