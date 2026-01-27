"""Recorder hooks for sampling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np

from experiments.sampling.graph.registry import BasinRegistry
from experiments.sampling.schema import SampleRecord, Structure
from experiments.sampling.structure_store import StructureStore


class RecorderBase(Protocol):
    def on_sample(self, record: SampleRecord) -> None:
        raise NotImplementedError

    def on_quench_step(self, record: SampleRecord) -> None:
        raise NotImplementedError


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _load_last_queue_idx(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    last_idx = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "queue_idx" in payload:
                try:
                    value = int(payload["queue_idx"])
                except (TypeError, ValueError):
                    continue
                last_idx = max(last_idx, value)
    return last_idx


def structure_to_dict(structure: Structure, *, round_decimals: Optional[int] = None) -> Dict[str, Any]:
    pos = structure.positions
    if round_decimals is not None:
        pos = np.round(pos, round_decimals)
    payload = {
        "numbers": structure.numbers,
        "positions": pos,
        "tags": structure.tags,
        "fixed": structure.fixed,
        "cell": structure.cell,
        "pbc": structure.pbc,
        "info": structure.info,
    }
    return {k: _to_jsonable(v) for k, v in payload.items() if v is not None}


class JSONLWriter:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")


class NoOpRecorder:
    def on_sample(self, record: SampleRecord) -> None:
        return None


class StepTraceRecorder:
    """Record every sampling step (for distribution/debug)."""

    def __init__(
        self,
        path: str | Path,
        *,
        include_structures: bool = False,
        round_decimals: Optional[int] = 4,
        structure_store: Optional[StructureStore] = None,
    ) -> None:
        self._writer = JSONLWriter(path)
        self._step = 0
        self._include_structures = include_structures
        self._round = round_decimals
        self._store = structure_store

    def on_sample(self, record: SampleRecord) -> None:
        self._step += 1
        payload: Dict[str, Any] = {
            "step_idx": self._step,
            "action": record.action.name,
            "action_params": record.action.params,
            "valid": record.valid,
            "flags": record.flags,
            "metrics": record.metrics,
        }
        if record.quench is not None:
            payload.update(
                {
                    "quench_converged": record.quench.converged,
                    "quench_steps": record.quench.steps,
                }
            )
        if record.basin is not None:
            payload["basin_id"] = record.basin.basin_id
            payload["basin_is_new"] = record.basin.is_new
        if self._include_structures:
            if self._store is not None:
                ref_pre = self._store.put(record.structure_pre, kind="pre")
                payload["structure_ref_pre"] = ref_pre.to_dict()
                if record.structure_min is not None:
                    ref_min = self._store.put(record.structure_min, kind="min")
                    payload["structure_ref_min"] = ref_min.to_dict()
            else:
                payload["structure_pre"] = structure_to_dict(record.structure_pre, round_decimals=self._round)
                if record.structure_min is not None:
                    payload["structure_min"] = structure_to_dict(record.structure_min, round_decimals=self._round)
        self._writer.write(payload)


class BasinRegistryRecorder:
    """Record only new basin entries with representative x_min."""

    def __init__(
        self,
        path: str | Path,
        *,
        registry: Optional[BasinRegistry] = None,
        round_decimals: Optional[int] = 4,
        structure_store: Optional[StructureStore] = None,
    ) -> None:
        self._writer = JSONLWriter(path)
        self._registry = registry or BasinRegistry()
        self._round = round_decimals
        self._store = structure_store

    def on_sample(self, record: SampleRecord) -> None:
        if record.basin is None or record.structure_min is None:
            return None
        basin = self._registry.register(record)
        if not basin.is_new:
            return None
        payload = {
            "basin_id": basin.basin_id,
            "first_seen_step": None,
        }
        if self._store is not None:
            ref_min = self._store.put(record.structure_min, kind="min")
            payload["structure_ref_min"] = ref_min.to_dict()
        else:
            payload["structure_min"] = structure_to_dict(record.structure_min, round_decimals=self._round)
        if record.metrics:
            payload["metrics"] = record.metrics
            if "energy_min" in record.metrics:
                payload["energy_min"] = record.metrics["energy_min"]
        if record.structure_min.info and "embedding" in record.structure_min.info:
            payload["embedding"] = record.structure_min.info["embedding"]
        self._writer.write(payload)


TriggerFn = Callable[[SampleRecord], Tuple[bool, Dict[str, Any]]]


class ALCandidateRecorder:
    """Record AL candidate structures when trigger conditions are met."""

    def __init__(
        self,
        path: str | Path,
        trigger_fn: TriggerFn,
        *,
        stages: Optional[Tuple[str, ...]] = None,
        round_decimals: Optional[int] = 4,
        structure_store: Optional[StructureStore] = None,
    ) -> None:
        self._writer = JSONLWriter(path)
        self._trigger = trigger_fn
        self._stages = set(stages) if stages else {"sample", "quench_step"}
        self._round = round_decimals
        self._store = structure_store
        self._queue_idx = _load_last_queue_idx(path)

    def on_sample(self, record: SampleRecord) -> None:
        if "sample" not in self._stages:
            return None
        ok, meta = self._trigger(record)
        if not ok:
            return None
        self._queue_idx += 1
        stage = record.flags.get("stage") if record.flags else None
        quench_step = record.flags.get("quench_step") if record.flags else None
        payload = {
            "queue_idx": self._queue_idx,
            "trigger": meta,
            "action": record.action.name,
            "action_params": record.action.params,
            "basin_id": record.basin.basin_id if record.basin else None,
        }
        if stage is not None:
            payload["stage"] = stage
        if quench_step is not None:
            payload["quench_step"] = quench_step
        if self._store is not None:
            ref_pre = self._store.put(record.structure_pre, kind="pre")
            payload["structure_ref_pre"] = ref_pre.to_dict()
        else:
            payload["structure_pre"] = structure_to_dict(record.structure_pre, round_decimals=self._round)
        self._writer.write(payload)

    def on_quench_step(self, record: SampleRecord) -> None:
        if "quench_step" not in self._stages:
            return None
        ok, meta = self._trigger(record)
        if not ok:
            return None
        self._queue_idx += 1
        stage = record.flags.get("stage") if record.flags else None
        quench_step = record.flags.get("quench_step") if record.flags else None
        payload = {
            "queue_idx": self._queue_idx,
            "trigger": meta,
            "action": record.action.name,
            "action_params": record.action.params,
            "basin_id": record.basin.basin_id if record.basin else None,
        }
        if stage is not None:
            payload["stage"] = stage
        if quench_step is not None:
            payload["quench_step"] = quench_step
        if self._store is not None:
            ref_pre = self._store.put(record.structure_pre, kind="pre")
            payload["structure_ref_pre"] = ref_pre.to_dict()
        else:
            payload["structure_pre"] = structure_to_dict(record.structure_pre, round_decimals=self._round)
        self._writer.write(payload)


class VizRecorder:
    """Record every frame (action/quench/min) for visualization."""

    def __init__(
        self,
        path: str | Path,
        *,
        structure_store: StructureStore,
        trigger_fn: Optional[TriggerFn] = None,
    ) -> None:
        self._writer = JSONLWriter(path)
        self._store = structure_store
        self._trigger = trigger_fn
        self._frame_idx = 0

    def _next_idx(self) -> int:
        self._frame_idx += 1
        return self._frame_idx

    def _trigger_meta(self, record: SampleRecord) -> tuple[bool, Optional[Dict[str, Any]]]:
        if self._trigger is None:
            return False, None
        ok, meta = self._trigger(record)
        if not ok:
            return False, None
        return True, meta

    def _write_frame(
        self,
        *,
        record: SampleRecord,
        structure: Structure,
        stage: str,
        quench_step: Optional[int] = None,
        basin_id: Optional[str] = None,
        basin_is_new: Optional[bool] = None,
    ) -> None:
        ref = self._store.put(structure, kind=stage)
        triggered, meta = self._trigger_meta(record)
        payload: Dict[str, Any] = {
            "frame_idx": self._next_idx(),
            "stage": stage,
            "action": record.action.name,
            "action_params": record.action.params,
            "metrics": record.metrics,
            "basin_id": basin_id,
            "basin_is_new": basin_is_new,
            "structure_ref": ref.to_dict(),
            "triggered": triggered,
        }
        if quench_step is not None:
            payload["quench_step"] = int(quench_step)
        if triggered and meta is not None:
            payload["trigger"] = meta
        self._writer.write(payload)

    def on_sample(self, record: SampleRecord) -> None:
        self._write_frame(record=record, structure=record.structure_pre, stage="action")
        if record.structure_min is not None:
            basin_id = record.basin.basin_id if record.basin else None
            basin_is_new = record.basin.is_new if record.basin else None
            self._write_frame(
                record=record,
                structure=record.structure_min,
                stage="min",
                basin_id=basin_id,
                basin_is_new=basin_is_new,
            )

    def on_quench_step(self, record: SampleRecord) -> None:
        quench_step = record.flags.get("quench_step") if record.flags else None
        self._write_frame(
            record=record,
            structure=record.structure_pre,
            stage="quench_step",
            quench_step=quench_step,
        )
