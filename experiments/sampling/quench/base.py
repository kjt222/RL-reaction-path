"""Quench base interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from experiments.sampling.schema import QuenchResult, Structure


class QuenchBase:
    name: str = "base"

    def run(self, structure: Structure) -> QuenchResult:
        raise NotImplementedError


class NoOpQuench(QuenchBase):
    name = "noop"

    def run(self, structure: Structure) -> QuenchResult:
        return QuenchResult(structure=structure.copy(), converged=True, steps=0, info={})
