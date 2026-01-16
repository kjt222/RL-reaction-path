"""Finetune AL selector (max/mean force)."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from experiments.sampling.schema import SampleRecord


TriggerFn = Callable[[SampleRecord], Tuple[bool, Dict[str, object]]]


def _get_force_metrics(record: SampleRecord, *, source: str) -> Dict[str, object] | None:
    metrics = record.metrics.get(source)
    if isinstance(metrics, dict):
        return metrics
    return None


def build_max_force_trigger(*, threshold: float, source: str = "force_pre") -> TriggerFn:
    def _trigger(record: SampleRecord) -> Tuple[bool, Dict[str, object]]:
        metrics = _get_force_metrics(record, source=source)
        if metrics is None or "max" not in metrics:
            return False, {"reason": "missing_force_metrics", "source": source}
        value = float(metrics["max"])
        if value > threshold:
            return True, {"reason": "max_F", "value": value, "threshold": threshold, "source": source}
        return False, {"reason": "below_threshold", "value": value, "threshold": threshold, "source": source}

    return _trigger


def build_topk_force_trigger(
    *,
    threshold: float,
    k: int = 3,
    source: str = "force_pre",
) -> TriggerFn:
    key = str(int(k))

    def _trigger(record: SampleRecord) -> Tuple[bool, Dict[str, object]]:
        metrics = _get_force_metrics(record, source=source)
        if metrics is None or "topk_mean" not in metrics:
            return False, {"reason": "missing_force_metrics", "source": source}
        topk = metrics.get("topk_mean", {})
        if not isinstance(topk, dict) or key not in topk:
            return False, {"reason": "missing_topk", "k": k, "source": source}
        value = float(topk[key])
        if value > threshold:
            return True, {"reason": "topk_mean", "k": k, "value": value, "threshold": threshold, "source": source}
        return False, {"reason": "below_threshold", "k": k, "value": value, "threshold": threshold, "source": source}

    return _trigger


def build_any_trigger(triggers: List[TriggerFn]) -> TriggerFn:
    def _trigger(record: SampleRecord) -> Tuple[bool, Dict[str, object]]:
        metas: List[Dict[str, object]] = []
        for idx, trig in enumerate(triggers):
            ok, meta = trig(record)
            metas.append(meta)
            if ok:
                return True, {"reason": "any", "selected": idx, "checks": metas}
        return False, {"reason": "any", "selected": None, "checks": metas}

    return _trigger


def build_default_trigger(
    *,
    max_threshold: float = 0.7,
    topk_threshold: float = 0.35,
    k: int = 5,
    source: str = "force_pre",
) -> TriggerFn:
    trig_max = build_max_force_trigger(threshold=max_threshold, source=source)
    trig_topk = build_topk_force_trigger(threshold=topk_threshold, k=k, source=source)
    return build_any_trigger([trig_max, trig_topk])
