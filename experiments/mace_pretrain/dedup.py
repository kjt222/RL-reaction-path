"""DFT queue near-duplicate filtering (coarse bucket + RMSD)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from experiments.sampling.schema import SampleRecord, Structure


def _default_select_indices(structure: Structure) -> np.ndarray:
    if structure.fixed is not None:
        fixed = np.asarray(structure.fixed, dtype=bool)
        movable = np.where(~fixed)[0]
        if movable.size > 0:
            return movable
    return np.arange(len(structure.numbers))


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


@dataclass
class DedupDecision:
    accepted: bool
    bucket: str
    rmsd: Optional[float]
    reason: str


class DFTQueueDeduper:
    """Near-duplicate filter using coarse bucketing + RMSD.

    Notes:
      - Assumes structures share a common frame (e.g., fixed slab), no alignment.
      - Uses only selected indices (movable by default).
    """

    def __init__(
        self,
        *,
        round_decimals: int = 2,
        rmsd_threshold: float = 0.08,
        select_indices: Optional[Callable[[Structure], np.ndarray]] = None,
    ) -> None:
        self._round = round_decimals
        self._rmsd = rmsd_threshold
        self._select = select_indices or _default_select_indices
        self._buckets: Dict[str, List[np.ndarray]] = {}

    @property
    def round_decimals(self) -> int:
        return self._round

    @property
    def rmsd_threshold(self) -> float:
        return self._rmsd

    def _bucket_key(self, structure: Structure, indices: np.ndarray) -> str:
        numbers = np.asarray(structure.numbers, dtype=np.int64)[indices]
        positions = np.asarray(structure.positions, dtype=np.float32)[indices]
        positions_q = np.round(positions, self._round)
        cell = np.asarray(structure.cell, dtype=np.float32) if structure.cell is not None else np.zeros((0,))
        cell_q = np.round(cell, self._round)
        pbc = np.asarray(structure.pbc, dtype=np.int8) if structure.pbc is not None else np.zeros((0,))
        payload = b"".join([numbers.tobytes(), positions_q.tobytes(), cell_q.tobytes(), pbc.tobytes()])
        return hashlib.sha1(payload).hexdigest()

    def _iter_bucket(self, bucket: str) -> Iterable[np.ndarray]:
        return self._buckets.get(bucket, [])

    def decide(self, structure: Structure) -> DedupDecision:
        indices = self._select(structure)
        positions = np.asarray(structure.positions, dtype=np.float32)[indices]
        bucket = self._bucket_key(structure, indices)
        best_rmsd: Optional[float] = None
        for ref in self._iter_bucket(bucket):
            value = _rmsd(positions, ref)
            best_rmsd = value if best_rmsd is None else min(best_rmsd, value)
            if value <= self._rmsd:
                return DedupDecision(
                    accepted=False,
                    bucket=bucket,
                    rmsd=value,
                    reason="duplicate",
                )
        return DedupDecision(
            accepted=True,
            bucket=bucket,
            rmsd=best_rmsd,
            reason="new",
        )

    def add(self, structure: Structure) -> DedupDecision:
        decision = self.decide(structure)
        if decision.accepted:
            indices = self._select(structure)
            positions = np.asarray(structure.positions, dtype=np.float32)[indices]
            self._buckets.setdefault(decision.bucket, []).append(positions)
        return decision

    def size(self) -> int:
        return sum(len(items) for items in self._buckets.values())


class GlobalRMSDDeduper:
    """Global RMSD deduper (no bucketing)."""

    def __init__(
        self,
        *,
        rmsd_threshold: float = 0.18,
        select_indices: Optional[Callable[[Structure], np.ndarray]] = None,
    ) -> None:
        self._rmsd = rmsd_threshold
        self._select = select_indices or _default_select_indices
        self._accepted: List[np.ndarray] = []

    @property
    def rmsd_threshold(self) -> float:
        return self._rmsd

    def decide(self, structure: Structure) -> DedupDecision:
        indices = self._select(structure)
        positions = np.asarray(structure.positions, dtype=np.float32)[indices]
        best_rmsd: Optional[float] = None
        for ref in self._accepted:
            value = _rmsd(positions, ref)
            best_rmsd = value if best_rmsd is None else min(best_rmsd, value)
            if value <= self._rmsd:
                return DedupDecision(
                    accepted=False,
                    bucket="global",
                    rmsd=value,
                    reason="duplicate",
                )
        return DedupDecision(
            accepted=True,
            bucket="global",
            rmsd=best_rmsd,
            reason="new",
        )

    def add(self, structure: Structure) -> DedupDecision:
        decision = self.decide(structure)
        if decision.accepted:
            indices = self._select(structure)
            positions = np.asarray(structure.positions, dtype=np.float32)[indices]
            self._accepted.append(positions)
        return decision

    def size(self) -> int:
        return len(self._accepted)


TriggerFn = Callable[[SampleRecord], Tuple[bool, Dict[str, object]]]


def wrap_trigger_with_dedup(trigger_fn: TriggerFn, deduper: DFTQueueDeduper) -> TriggerFn:
    def _wrapped(record: SampleRecord) -> Tuple[bool, Dict[str, object]]:
        ok, meta = trigger_fn(record)
        if not ok:
            return False, meta
        decision = deduper.add(record.structure_pre)
        meta = dict(meta) if meta is not None else {}
        meta["dedup"] = {
            "accepted": decision.accepted,
            "bucket": decision.bucket,
            "rmsd": decision.rmsd,
            "reason": decision.reason,
        }
        return (decision.accepted, meta)

    return _wrapped
