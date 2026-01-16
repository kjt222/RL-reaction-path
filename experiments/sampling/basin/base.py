"""Basin identification interfaces."""

from __future__ import annotations

from experiments.sampling.schema import BasinResult, Structure


class BasinBase:
    name: str = "base"

    def identify(self, structure: Structure) -> BasinResult:
        raise NotImplementedError
