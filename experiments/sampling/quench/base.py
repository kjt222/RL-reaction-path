"""Quench base interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from experiments.sampling.schema import QuenchResult, Structure


QuenchStepCallback = Callable[[Structure, Optional[object], Optional[float], int], None]


class QuenchBase:
    name: str = "base"

    def run(self, structure: Structure, *, step_callback: Optional[QuenchStepCallback] = None) -> QuenchResult:
        raise NotImplementedError


class NoOpQuench(QuenchBase):
    name = "noop"

    def run(self, structure: Structure, *, step_callback: Optional[QuenchStepCallback] = None) -> QuenchResult:
        if step_callback is not None:
            step_callback(structure.copy(), None, None, 0)
        return QuenchResult(structure=structure.copy(), converged=True, steps=0, info={})
