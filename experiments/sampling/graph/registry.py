"""Global basin registry (experimental)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from experiments.sampling.schema import BasinResult, SampleRecord


@dataclass
class BasinRegistry:
    """In-memory basin registry; can be extended to build a basin graph."""

    basins: Dict[str, dict] = field(default_factory=dict)

    def has(self, basin_id: str) -> bool:
        return basin_id in self.basins

    def register(self, record: SampleRecord) -> BasinResult:
        basin = record.basin
        if basin is None:
            raise ValueError("SampleRecord has no basin result")
        if basin.basin_id not in self.basins:
            self.basins[basin.basin_id] = {"count": 0}
            basin.is_new = True
        else:
            basin.is_new = False
        self.basins[basin.basin_id]["count"] += 1
        return basin
