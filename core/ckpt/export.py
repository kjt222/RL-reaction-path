"""Export standardized artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from core.contracts import ModelBundle
from core.manifest import ArtifactRef, Manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(
    bundle: ModelBundle,
    output_dir: Path,
    weights_name: str = "best_model.pt",
    manifest_name: str = "manifest.json",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / weights_name
    manifest_path = output_dir / manifest_name

    state_dict = {k: v.detach().cpu() for k, v in bundle.model.state_dict().items()}
    torch.save(state_dict, weights_path)

    manifest = dict(bundle.manifest)
    weights_meta = dict(manifest.get("weights", {}))
    weights_meta["path"] = str(weights_path)
    weights_meta["format"] = "torch"
    weights_meta["sha256"] = _sha256_file(weights_path)
    manifest["weights"] = weights_meta

    extras = bundle.extras or {}
    normalizers = extras.get("normalizers")
    scale_dict = extras.get("scale_dict")
    aux_files = extras.get("aux_files")
    if normalizers or scale_dict or aux_files:
        state_payload: dict[str, Any] = {}
        if normalizers:
            state_payload["normalizers"] = {k: v.state_dict() for k, v in normalizers.items()}
        if scale_dict:
            state_payload["scale_dict"] = scale_dict
        if aux_files:
            state_payload["aux_files"] = aux_files
        backend_state_path = output_dir / "backend_state.pt"
        torch.save(state_payload, backend_state_path)
        manifest["backend_state"] = {
            "path": str(backend_state_path),
            "format": "torch",
            "sha256": _sha256_file(backend_state_path),
        }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    return weights_path, manifest_path


def export_standard_artifacts(
    adapter: Any,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    output_dir: Path,
    weights_name: str = "best_model.pt",
    manifest_name: str = "manifest.json",
    normalizer: dict[str, Any] | None = None,
    head: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / weights_name
    manifest_path = output_dir / manifest_name

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, weights_path)
    weights_sha = _sha256_file(weights_path)

    backend_state: dict[str, Any] | None = None
    if adapter is not None:
        state_payload: dict[str, Any] = {}
        normalizers = getattr(adapter, "_normalizers", None)
        if normalizers:
            try:
                state_payload["normalizers"] = {k: v.state_dict() for k, v in normalizers.items()}
            except Exception:
                state_payload["normalizers"] = None
        scale_dict = getattr(adapter, "_scale_dict", None)
        if scale_dict:
            state_payload["scale_dict"] = scale_dict
        aux_files = getattr(adapter, "_aux_files", None)
        if aux_files:
            state_payload["aux_files"] = aux_files
        if state_payload:
            backend_state_path = output_dir / "backend_state.pt"
            torch.save(state_payload, backend_state_path)
            backend_state = {
                "path": str(backend_state_path),
                "format": "torch",
                "sha256": _sha256_file(backend_state_path),
            }

    spec = adapter.model_spec(cfg, model=model) if adapter is not None else {}
    backend = getattr(adapter, "backend_name", "unknown")
    backend_version = spec.get("backend_version", "unknown")
    config = spec.get("config") or cfg
    io = spec.get("io", {"energy_unit": "eV", "force_unit": "eV/Angstrom", "energy_is_total": True})
    embedding = spec.get("embedding", {"node_embed_layer": None, "graph_pool": "mean"})
    manifest_head = spec.get("head") or head
    manifest_normalizer = spec.get("normalizer") or normalizer

    source = ArtifactRef(path=str(weights_path), format="torch", sha256=weights_sha)
    weights = ArtifactRef(path=str(weights_path), format="torch", sha256=weights_sha)
    manifest = Manifest(
        schema_version="v1",
        backend=str(backend),
        backend_version=str(backend_version),
        source=source,
        weights=weights,
        rebuildable=bool(config),
        io=io,
        embedding=embedding,
        config=config,
        head=manifest_head,
        normalizer=manifest_normalizer,
        backend_state=backend_state,
    )

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, ensure_ascii=True)

    return weights_path, manifest_path
