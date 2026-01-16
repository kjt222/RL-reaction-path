"""DFT outbox processing: canonicalize + dedup + submit/skip logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from experiments.mace_pretrain.dedup import DFTQueueDeduper
from experiments.sampling.canonicalize import SlabCanonicalizer
import numpy as np

from experiments.sampling.schema import Structure
from experiments.sampling.structure_store import StructureStore


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return []
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


def _write_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _last_queue_idx(*paths: Path) -> int:
    last_idx = -1
    for path in paths:
        for payload in _iter_jsonl(path):
            if "queue_idx" in payload:
                try:
                    value = int(payload["queue_idx"])
                except (TypeError, ValueError):
                    continue
                last_idx = max(last_idx, value)
    return last_idx


def _structure_from_ref(queue_dir: Path, ref: Dict[str, object]) -> Structure:
    rel = ref.get("path")
    if not isinstance(rel, str):
        raise ValueError("structure_ref missing 'path'")
    return StructureStore.load_npz(queue_dir / rel)


def _structure_from_inline(payload: Dict[str, object]) -> Structure:
    numbers = np.asarray(payload["numbers"], dtype=np.int64)
    positions = np.asarray(payload["positions"], dtype=np.float32)
    return Structure(
        numbers=numbers,
        positions=positions,
        tags=np.asarray(payload["tags"], dtype=np.int16) if "tags" in payload else None,
        fixed=np.asarray(payload["fixed"], dtype=np.int8) if "fixed" in payload else None,
        cell=np.asarray(payload["cell"], dtype=np.float32) if "cell" in payload else None,
        pbc=np.asarray(payload["pbc"], dtype=np.int8) if "pbc" in payload else None,
        info=payload.get("info") if isinstance(payload.get("info"), dict) else {},
    )


def process_dft_queue(
    *,
    queue_path: str | Path,
    submit_path: str | Path | None = None,
    skip_path: str | Path | None = None,
    canonicalize: bool = True,
    deduper: Optional[DFTQueueDeduper] = None,
) -> Tuple[int, int]:
    """Process DFT queue and emit submit/skip logs.

    Returns:
        (submitted_count, skipped_count)
    """

    queue_path = Path(queue_path)
    queue_dir = queue_path.parent
    submit_path = Path(submit_path) if submit_path else queue_dir / "dft_submit.jsonl"
    skip_path = Path(skip_path) if skip_path else queue_dir / "dft_skip.jsonl"

    last_idx = _last_queue_idx(submit_path, skip_path)
    deduper = deduper or DFTQueueDeduper()
    canonicalizer = SlabCanonicalizer() if canonicalize else None

    submitted = 0
    skipped = 0

    for payload in _iter_jsonl(queue_path):
        queue_idx = payload.get("queue_idx")
        if queue_idx is None:
            continue
        try:
            queue_idx = int(queue_idx)
        except (TypeError, ValueError):
            continue
        if queue_idx <= last_idx:
            continue

        if "structure_ref_pre" in payload and isinstance(payload["structure_ref_pre"], dict):
            structure = _structure_from_ref(queue_dir, payload["structure_ref_pre"])
        elif "structure_pre" in payload and isinstance(payload["structure_pre"], dict):
            structure = _structure_from_inline(payload["structure_pre"])
        else:
            skip_payload = dict(payload)
            skip_payload["skip_reason"] = "missing_structure"
            _write_jsonl(skip_path, skip_payload)
            skipped += 1
            continue

        structure_canon = canonicalizer(structure) if canonicalizer is not None else structure
        decision = deduper.add(structure_canon)

        out_payload = dict(payload)
        out_payload["dedup"] = {
            "accepted": decision.accepted,
            "bucket": decision.bucket,
            "rmsd": decision.rmsd,
            "reason": decision.reason,
        }
        if decision.accepted:
            _write_jsonl(submit_path, out_payload)
            submitted += 1
        else:
            _write_jsonl(skip_path, out_payload)
            skipped += 1

    return submitted, skipped
