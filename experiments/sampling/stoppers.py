"""Sampling stop conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from experiments.sampling.schema import SampleRecord


@dataclass
class SamplingState:
    step_idx: int = 0
    basins: int = 0


class StopperBase(Protocol):
    def on_sample(self, record: SampleRecord, state: SamplingState) -> Optional[str]:
        raise NotImplementedError


class MaxStepsStopper:
    def __init__(self, max_steps: int) -> None:
        self._max_steps = int(max_steps)

    def on_sample(self, record: SampleRecord, state: SamplingState) -> Optional[str]:
        if state.step_idx >= self._max_steps:
            return "max_steps"
        return None


class TargetBasinsStopper:
    def __init__(self, target_basins: int) -> None:
        self._target = int(target_basins)

    def on_sample(self, record: SampleRecord, state: SamplingState) -> Optional[str]:
        if state.basins >= self._target:
            return "target_basins"
        return None
