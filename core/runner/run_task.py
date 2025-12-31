"""High-level backend task runner."""

from __future__ import annotations

from pathlib import Path

from core.ckpt.export import export_bundle
from core.registry import get_adapter
from core.runner.dispatch import dispatch
from core.runner.layout import standard_artifact_paths
from core.runner.spec import BackendRunResult, CommonTaskSpec


def run_backend_task(spec: CommonTaskSpec, export_artifacts: bool = True) -> BackendRunResult:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    if spec.mode == "export":
        if spec.model_in is None:
            raise ValueError("export requires model_in")
        adapter_cls = get_adapter(spec.backend)
        adapter = adapter_cls()
        bundle = adapter.load(spec.model_in, device=spec.device)
        paths = standard_artifact_paths(spec.run_dir)
        export_bundle(
            bundle,
            output_dir=Path(paths["best_model"]).parent,
            weights_name=Path(paths["best_model"]).name,
            manifest_name=Path(paths["manifest"]).name,
        )
        return BackendRunResult(run_dir=spec.run_dir, best_model_path=Path(spec.model_in))

    return dispatch(spec, export_artifacts=export_artifacts)
