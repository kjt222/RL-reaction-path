"""Common task spec for backend runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _coerce_str_list(name: str, value: Any | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [str(item) for item in value]


@dataclass
class CommonTaskSpec:
    backend: str
    mode: str
    run_dir: Path
    model_in: Optional[str] = None
    model_manifest: Optional[str] = None
    model_weights: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    backend_args: list[str] = field(default_factory=list)
    backend_python: Optional[str] = None
    backend_env: dict[str, str] = field(default_factory=dict)
    device: str = "cuda"
    config_dir: Optional[Path] = None

    @classmethod
    def from_run_dict(cls, run: Mapping[str, Any], config_dir: Path) -> "CommonTaskSpec":
        backend = str(run.get("backend") or "")
        if not backend:
            raise ValueError("backend is required")
        mode = str(run.get("task") or run.get("mode") or "train").lower()
        run_dir_raw = run.get("run_dir")
        if not run_dir_raw:
            raise ValueError("run_dir is required for backend tasks")
        run_dir = Path(_resolve_path(config_dir, str(run_dir_raw)) or str(run_dir_raw))

        model_in = run.get("model_in") or None
        model_manifest = None
        model_weights = None
        model_spec = run.get("model")
        if isinstance(model_spec, dict):
            model_manifest = model_spec.get("manifest")
            model_weights = model_spec.get("weights") or model_spec.get("path") or model_spec.get("source")
        if isinstance(model_in, dict):
            model_manifest = model_manifest or model_in.get("manifest")
            model_weights = model_weights or model_in.get("weights") or model_in.get("path") or model_in.get("source")
            model_in = model_in.get("path") or model_in.get("source") or model_in.get("weights")
        model_in = _resolve_path(config_dir, str(model_in)) if model_in else None
        model_manifest = _resolve_path(config_dir, str(model_manifest)) if model_manifest else None
        model_weights = _resolve_path(config_dir, str(model_weights)) if model_weights else None

        data = dict(run.get("data") or {})
        for key in ("train", "val", "test", "lmdb_train", "lmdb_val", "lmdb_path", "xyz_dir"):
            if key in data and data[key] is not None:
                data[key] = _resolve_path(config_dir, str(data[key]))
        if "indices_path" in data and data["indices_path"] is not None:
            data["indices_path"] = _resolve_path(config_dir, str(data["indices_path"]))
        for indices_key in ("train_indices", "val_indices", "test_indices"):
            if indices_key in data and isinstance(data[indices_key], dict):
                indices_spec = dict(data[indices_key])
                if "indices_path" in indices_spec and indices_spec["indices_path"] is not None:
                    indices_spec["indices_path"] = _resolve_path(config_dir, str(indices_spec["indices_path"]))
                data[indices_key] = indices_spec

        train = dict(run.get("train") or {})

        backend_args = _coerce_str_list("backend_args", run.get("backend_args") or run.get("backend_extra"))
        backend_python = run.get("backend_python") or run.get("python")
        backend_env = run.get("backend_env") or {}
        if not isinstance(backend_env, dict):
            raise ValueError("backend_env must be a mapping")

        device = str(run.get("device", "cuda"))

        return cls(
            backend=backend,
            mode=mode,
            run_dir=run_dir,
            model_in=model_in,
            model_manifest=model_manifest,
            model_weights=model_weights,
            data=data,
            train=train,
            backend_args=backend_args,
            backend_python=str(backend_python) if backend_python else None,
            backend_env={str(k): str(v) for k, v in backend_env.items()},
            device=device,
            config_dir=config_dir,
        )


@dataclass
class BackendRunResult:
    run_dir: Path
    checkpoint_path: Optional[Path] = None
    best_model_path: Optional[Path] = None
