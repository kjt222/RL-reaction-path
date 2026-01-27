"""Action quality diagnostics from sampling logs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _force_metrics(step: Dict[str, object], source: str) -> Optional[Dict[str, object]]:
    metrics = step.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(source)
    if isinstance(value, dict):
        return value
    return None


def _force_max(step: Dict[str, object], source: str) -> Optional[float]:
    metrics = _force_metrics(step, source)
    if metrics is None:
        return None
    value = metrics.get("max")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _force_topk(step: Dict[str, object], source: str, k: int) -> Optional[float]:
    metrics = _force_metrics(step, source)
    if metrics is None:
        return None
    topk = metrics.get("topk_mean")
    if not isinstance(topk, dict):
        return None
    value = topk.get(str(int(k)))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_invalid_reason(flags: Dict[str, object]) -> str:
    if not flags:
        return "unknown"
    if "quality_reject" in flags or "quality_score" in flags:
        return "quality_gate"
    if "min_seen" in flags or "min_factor" in flags:
        return "min_dist"
    if "fixed_max_move" in flags:
        return "fixed"
    if "bond_max_seen" in flags or "bond_seen" in flags:
        return "bond"
    return "unknown"


def _action_magnitude(action: str, params: Dict[str, object]) -> Optional[float]:
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


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    n = len(vals)
    def _pct(p: float) -> float:
        idx = int(round((n - 1) * p))
        return vals[idx]
    return {
        "count": n,
        "min": vals[0],
        "p50": _pct(0.5),
        "p90": _pct(0.9),
        "max": vals[-1],
        "mean": sum(vals) / n,
    }


def summarize(
    steps: Iterable[Dict[str, object]],
    *,
    max_threshold: float = 0.7,
    topk_threshold: float = 0.35,
    topk: int = 5,
    source: str = "force_pre",
) -> Dict[str, object]:
    total = 0
    valid = 0
    quench_converged = 0
    quench_total = 0
    new_basin = 0
    trigger_max = 0
    trigger_topk = 0
    quality_reject = 0
    action_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    action_mags: Dict[str, List[float]] = defaultdict(list)
    force_max_vals: List[float] = []
    force_topk_vals: List[float] = []

    for step in steps:
        total += 1
        if step.get("valid") is True:
            valid += 1
        elif step.get("valid") is False:
            flags = step.get("flags", {})
            if isinstance(flags, dict):
                if "quality_reject" in flags:
                    quality_reject += 1
                invalid_reasons[_infer_invalid_reason(flags)] += 1
        action = step.get("action")
        if isinstance(action, str):
            action_counts[action] += 1
            params = step.get("action_params")
            if isinstance(params, dict):
                mag = _action_magnitude(action, params)
                if mag is not None:
                    action_mags[action].append(mag)

        if "quench_converged" in step:
            quench_total += 1
            if step.get("quench_converged") is True:
                quench_converged += 1

        if step.get("basin_is_new") is True:
            new_basin += 1

        max_val = _force_max(step, source)
        if max_val is not None and max_val > max_threshold:
            trigger_max += 1
        if max_val is not None:
            force_max_vals.append(max_val)

        topk_val = _force_topk(step, source, topk)
        if topk_val is not None and topk_val > topk_threshold:
            trigger_topk += 1
        if topk_val is not None:
            force_topk_vals.append(topk_val)

    def _rate(n: int, d: int) -> float:
        return float(n) / float(d) if d else 0.0

    return {
        "total_steps": total,
        "valid_rate": _rate(valid, total),
        "quench_converged_rate": _rate(quench_converged, quench_total),
        "new_basin_rate": _rate(new_basin, total),
        "trigger_rate_max": _rate(trigger_max, total),
        "trigger_rate_topk": _rate(trigger_topk, total),
        "quality_reject_rate": _rate(quality_reject, total),
        "action_counts": dict(action_counts),
        "invalid_reason_counts": dict(invalid_reasons),
        "action_magnitude_stats": {k: _summary_stats(v) for k, v in action_mags.items()},
        "force_max_stats": _summary_stats(force_max_vals),
        "force_topk_stats": _summary_stats(force_topk_vals),
        "params": {
            "source": source,
            "max_threshold": max_threshold,
            "topk_threshold": topk_threshold,
            "topk": topk,
        },
    }


def summarize_run_dir(
    run_dir: Path,
    *,
    max_threshold: float = 0.7,
    topk_threshold: float = 0.35,
    topk: int = 5,
    source: str = "force_pre",
) -> Dict[str, object]:
    steps_path = run_dir / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"steps.jsonl not found under {run_dir}")
    steps = list(_iter_jsonl(steps_path))
    return summarize(
        steps,
        max_threshold=max_threshold,
        topk_threshold=topk_threshold,
        topk=topk,
        source=source,
    )
