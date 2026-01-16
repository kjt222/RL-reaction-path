"""Structure store for deduplicated NPZ persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from experiments.sampling.schema import Structure


@dataclass(frozen=True)
class StructureRef:
    id: str
    path: str
    n_atoms: int
    round_decimals: int
    kind: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "path": self.path,
            "n_atoms": self.n_atoms,
            "round_decimals": self.round_decimals,
        }
        if self.kind is not None:
            payload["kind"] = self.kind
        return payload


def _as_array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if value is None:
        return np.array([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


class StructureStore:
    """Store structures as NPZ files with quantized hashing for dedup."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        round_decimals: int = 4,
        index_path: str | Path | None = None,
    ) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._round = round_decimals
        self._index = Path(index_path) if index_path else self._root / "index.jsonl"
        self._known: set[str] = set()
        self._load_index()

    @property
    def root_dir(self) -> Path:
        return self._root

    @property
    def round_decimals(self) -> int:
        return self._round

    def _load_index(self) -> None:
        if not self._index.exists():
            return
        with self._index.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and "id" in payload:
                    self._known.add(str(payload["id"]))

    def _quantize(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        return np.round(arr.astype(np.float32), self._round)

    def _hash_structure(self, structure: Structure) -> str:
        numbers = _as_array(structure.numbers, dtype=np.int64)
        positions = self._quantize(_as_array(structure.positions, dtype=np.float32))
        tags = _as_array(structure.tags, dtype=np.int16)
        fixed = _as_array(structure.fixed, dtype=np.int8)
        cell = self._quantize(_as_array(structure.cell, dtype=np.float32))
        pbc = _as_array(structure.pbc, dtype=np.int8)
        chunks = [
            numbers.tobytes(),
            positions.tobytes(),
            tags.tobytes(),
            fixed.tobytes(),
            cell.tobytes(),
            pbc.tobytes(),
        ]
        return hashlib.sha1(b"".join(chunks)).hexdigest()

    def _structure_path(self, structure_id: str) -> Path:
        return self._root / f"{structure_id}.npz"

    def _relative_path(self, structure_id: str) -> str:
        return f"{self._root.name}/{structure_id}.npz"

    def _write_index(self, ref: StructureRef) -> None:
        with self._index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ref.to_dict(), ensure_ascii=False) + "\n")

    def put(self, structure: Structure, *, kind: Optional[str] = None) -> StructureRef:
        structure_id = self._hash_structure(structure)
        path = self._structure_path(structure_id)
        ref = StructureRef(
            id=structure_id,
            path=self._relative_path(structure_id),
            n_atoms=len(structure.numbers),
            round_decimals=self._round,
            kind=kind,
        )
        if structure_id not in self._known:
            if not path.exists():
                payload = {
                    "numbers": np.asarray(structure.numbers, dtype=np.int64),
                    "positions": np.asarray(structure.positions, dtype=np.float32),
                }
                if structure.tags is not None:
                    payload["tags"] = np.asarray(structure.tags, dtype=np.int16)
                if structure.fixed is not None:
                    payload["fixed"] = np.asarray(structure.fixed, dtype=np.int8)
                if structure.cell is not None:
                    payload["cell"] = np.asarray(structure.cell, dtype=np.float32)
                if structure.pbc is not None:
                    payload["pbc"] = np.asarray(structure.pbc, dtype=np.int8)
                if structure.info is not None:
                    info_json = json.dumps(structure.info, ensure_ascii=False, default=str)
                    payload["info_json"] = np.asarray(info_json)
                np.savez_compressed(path, **payload)
            self._write_index(ref)
            self._known.add(structure_id)
        return ref

    @staticmethod
    def load_npz(path: str | Path) -> Structure:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            numbers = np.asarray(data["numbers"], dtype=np.int64)
            positions = np.asarray(data["positions"], dtype=np.float32)
            tags = np.asarray(data["tags"], dtype=np.int16) if "tags" in data else None
            fixed = np.asarray(data["fixed"], dtype=np.int8) if "fixed" in data else None
            cell = np.asarray(data["cell"], dtype=np.float32) if "cell" in data else None
            pbc = np.asarray(data["pbc"], dtype=np.int8) if "pbc" in data else None
            info = None
            if "info_json" in data:
                raw = data["info_json"].item()
                try:
                    info = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    info = None
        return Structure(
            numbers=numbers,
            positions=positions,
            tags=tags,
            fixed=fixed,
            cell=cell,
            pbc=pbc,
            info=info,
        )
