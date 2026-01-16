"""Action -> quench -> basin sampling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from experiments.sampling.actions.base import ActionBase, ActionContext
from experiments.sampling.force_metrics import force_stats
from experiments.sampling.recorders import RecorderBase
from experiments.sampling.schema import BasinResult, QuenchResult, SampleRecord, Structure


ValidatorFn = Callable[[Structure], tuple[bool, Dict[str, float]]]
ForceFn = Callable[[Structure], np.ndarray]


@dataclass
class PipelineConfig:
    max_attempts: int = 5


class SamplingPipeline:
    def __init__(
        self,
        actions: List[ActionBase],
        *,
        quench: Optional[Callable[[Structure], QuenchResult]] = None,
        force_fn: Optional[ForceFn] = None,
        basin: Optional[Callable[[Structure], BasinResult]] = None,
        validators: Optional[List[ValidatorFn]] = None,
        recorders: Optional[List[RecorderBase]] = None,
        rng: Optional[np.random.Generator] = None,
        config: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        if not actions:
            raise ValueError("SamplingPipeline requires at least one action")
        self._actions = actions
        self._quench = quench
        self._force_fn = force_fn
        self._basin = basin
        self._validators = validators or []
        self._recorders = recorders or []
        self._rng = rng or np.random.default_rng()
        self._config = config or {}
        self._cfg = PipelineConfig()

    def _emit(self, record: SampleRecord) -> None:
        for recorder in self._recorders:
            recorder.on_sample(record)

    def run_one(
        self,
        structure: Structure,
        *,
        selection_mask: Optional[np.ndarray] = None,
        candidates: Optional[Dict[str, object]] = None,
    ) -> SampleRecord:
        record = self._run_one(structure, selection_mask=selection_mask, candidates=candidates)
        self._emit(record)
        return record

    def _run_one(
        self,
        structure: Structure,
        *,
        selection_mask: Optional[np.ndarray] = None,
        candidates: Optional[Dict[str, object]] = None,
    ) -> SampleRecord:
        rng = self._rng
        action = None
        op = None
        errors: Dict[str, str] = {}
        for _ in range(self._cfg.max_attempts):
            action = rng.choice(self._actions)
            try:
                ctx = ActionContext(structure=structure, selection_mask=selection_mask, candidates=candidates, config=self._config)
                op = action.sample(ctx, rng)
                break
            except Exception as exc:
                errors[getattr(action, "name", "unknown")] = str(exc)
                action = None
                op = None
        if action is None or op is None:
            raise RuntimeError(f"Failed to sample action after {self._cfg.max_attempts} attempts: {errors}")

        structure_pre = action.apply(structure, op)

        valid = True
        flags: Dict[str, object] = {}
        for validator in self._validators:
            ok, info = validator(structure_pre)
            flags.update(info)
            if not ok:
                valid = False
                break

        quench_result = None
        basin_result = None
        structure_min = None
        metrics: Dict[str, object] = {}

        if valid:
            if self._force_fn is not None:
                try:
                    forces_pre = self._force_fn(structure_pre)
                    metrics["force_pre"] = force_stats(forces_pre, topk=(3, 5))
                except Exception as exc:
                    flags["force_pre_error"] = str(exc)
            if self._quench is not None:
                quench_result = self._quench(structure_pre)
                structure_min = quench_result.structure
                if quench_result.forces is not None:
                    metrics["force_min"] = force_stats(quench_result.forces, topk=(3, 5))
            else:
                structure_min = structure_pre
            if "force_min" not in metrics and self._force_fn is not None and structure_min is not None:
                try:
                    forces_min = self._force_fn(structure_min)
                    metrics["force_min"] = force_stats(forces_min, topk=(3, 5))
                except Exception as exc:
                    flags["force_min_error"] = str(exc)
            if self._basin is not None and structure_min is not None:
                basin_result = self._basin(structure_min)

        return SampleRecord(
            structure_in=structure,
            structure_pre=structure_pre,
            structure_min=structure_min,
            action=op,
            quench=quench_result,
            basin=basin_result,
            valid=valid,
            flags=flags,
            metrics=metrics,
        )
