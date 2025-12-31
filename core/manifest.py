"""Manifest schema for portable model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class ArtifactRef:
    path: str
    format: str
    sha256: Optional[str] = None
    dtype: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "format": self.format,
        }
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            path=str(data["path"]),
            format=str(data["format"]),
            sha256=data.get("sha256"),
            dtype=data.get("dtype"),
        )


@dataclass
class Manifest:
    schema_version: str
    backend: str
    backend_version: str
    source: ArtifactRef
    weights: ArtifactRef
    rebuildable: bool
    io: Mapping[str, Any]
    embedding: Mapping[str, Any]
    config: Optional[Mapping[str, Any]] = None
    head: Optional[str] = None
    normalizer: Optional[Mapping[str, Any]] = None
    backend_state: Optional[Mapping[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "source": self.source.to_dict(),
            "weights": self.weights.to_dict(),
            "rebuildable": self.rebuildable,
            "io": dict(self.io),
            "embedding": dict(self.embedding),
        }
        if self.config is not None:
            payload["config"] = self.config
        if self.head is not None:
            payload["head"] = self.head
        if self.normalizer is not None:
            payload["normalizer"] = self.normalizer
        if self.backend_state is not None:
            payload["backend_state"] = self.backend_state
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Manifest":
        return cls(
            schema_version=str(data["schema_version"]),
            backend=str(data["backend"]),
            backend_version=str(data["backend_version"]),
            source=ArtifactRef.from_dict(data["source"]),
            weights=ArtifactRef.from_dict(data["weights"]),
            rebuildable=bool(data["rebuildable"]),
            io=data.get("io", {}),
            embedding=data.get("embedding", {}),
            config=data.get("config"),
            head=data.get("head"),
            normalizer=data.get("normalizer"),
            backend_state=data.get("backend_state"),
            notes=data.get("notes"),
        )
