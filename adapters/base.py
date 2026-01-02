"""Base adapter interface for core training/eval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from core.contracts import CanonicalBatch, ModelOutputs


class AdapterBase(ABC):
    backend_name: str = ""

    def __init__(self, train_cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.train_cfg = dict(train_cfg or {})

    @abstractmethod
    def build_model(self, cfg: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def model_spec(self, cfg: Mapping[str, Any], model: Any | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def select_head(self, cfg: Mapping[str, Any], model: Any) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def make_backend_batch(self, cbatch: CanonicalBatch, device: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, model: Any, backend_batch: Any) -> ModelOutputs:
        raise NotImplementedError

    @abstractmethod
    def loss(self, outputs: ModelOutputs, cbatch: CanonicalBatch) -> tuple[Any, dict[str, float]]:
        raise NotImplementedError

    def head_parameters(self, _model: Any):
        raise NotImplementedError("head_parameters is not implemented for this adapter")

    def native_train(self, _spec: Any) -> Any:
        raise NotImplementedError("native_train is not implemented for this adapter")

    def export_artifacts(self, _native_run_dir: Any, _run_dir: Any) -> None:
        raise NotImplementedError("export_artifacts is not implemented for this adapter")
