"""Dispatch tasks to backend-specific runners or core trainer."""

from __future__ import annotations

from pathlib import Path

from adapters.base import AdapterBase
from core.registry import get_adapter
from core.runner.layout import standard_artifact_paths, standard_checkpoint_paths
from core.runner.spec import BackendRunResult, CommonTaskSpec
from core.train.trainer import run_task as run_core_train


def dispatch(spec: CommonTaskSpec, export_artifacts: bool = True) -> BackendRunResult:
    adapter_cls = get_adapter(spec.backend)
    adapter = adapter_cls(train_cfg=spec.train)

    force_core = bool(spec.train.get("use_core") or spec.train.get("force_core") or spec.model_manifest)
    if not force_core and adapter.__class__.native_train is not AdapterBase.native_train:
        native_run_dir = adapter.native_train(spec)
        if export_artifacts and adapter.__class__.export_artifacts is not AdapterBase.export_artifacts:
            adapter.export_artifacts(native_run_dir, spec.run_dir)
        artifacts = standard_artifact_paths(spec.run_dir)
        checkpoints = standard_checkpoint_paths(spec.run_dir)
        best_model_path = (
            Path(artifacts["best_model"]) if export_artifacts and artifacts.get("best_model") else None
        )
        checkpoint_path = (
            Path(checkpoints["checkpoint"]) if export_artifacts and checkpoints.get("checkpoint") else None
        )
        return BackendRunResult(run_dir=spec.run_dir, best_model_path=best_model_path, checkpoint_path=checkpoint_path)

    return run_core_train(spec, adapter, export_artifacts=export_artifacts)
