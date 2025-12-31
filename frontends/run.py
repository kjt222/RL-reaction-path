"""Unified evaluation/inference runner for model adapters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from adapters import equiformer_v2, gemnet_oc, mace
from core.ckpt.export import export_bundle
from core.contracts import ModelBundle
from core.eval import run_lmdb_task
from core.registry import get_adapter
from core.runner.run_task import run_backend_task
from core.runner.spec import CommonTaskSpec
from core.transforms import build_transform


equiformer_v2.register()
gemnet_oc.register()
mace.register()


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("run.yaml must define a mapping")
    return data


def _load_runs(config_path: Path) -> list[dict[str, Any]]:
    config = _load_yaml(config_path)
    runs = config.get("runs")
    if runs is None:
        return [config]
    if not isinstance(runs, list):
        raise ValueError("runs must be a list")
    return runs


def _coerce_str_list(name: str, value: Any | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [str(item) for item in value]


def _run_backend_task(run: dict[str, Any], config_dir: Path) -> None:
    command = run.get("backend_command")
    script = run.get("backend_script")
    backend_args = _coerce_str_list("backend_args", run.get("backend_args"))

    if command is not None and script is not None:
        raise ValueError("Provide only one of backend_command or backend_script")
    if command is None and script is None:
        raise ValueError("backend_command or backend_script is required for task=backend")

    if command is not None:
        cmd = _coerce_str_list("backend_command", command)
        if cmd and cmd[0].endswith(".py"):
            cmd[0] = _resolve_path(config_dir, cmd[0]) or cmd[0]
        if backend_args:
            cmd.extend(backend_args)
    else:
        script_path = _resolve_path(config_dir, str(script))
        if script_path is None:
            raise ValueError("backend_script is empty")
        python = run.get("backend_python") or sys.executable
        cmd = [str(python), script_path, *backend_args]

    cwd = run.get("backend_cwd")
    if cwd:
        cwd = _resolve_path(config_dir, str(cwd))
    else:
        cwd = str(config_dir)

    env = os.environ.copy()
    extra_env = run.get("backend_env") or {}
    if not isinstance(extra_env, dict):
        raise ValueError("backend_env must be a mapping")
    env.update({str(k): str(v) for k, v in extra_env.items()})

    print(f"Running backend command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def _load_bundle(
    adapter,
    backend: str,
    model_spec: dict[str, Any],
    config_dir: Path,
    device: str,
) -> ModelBundle:
    manifest_path = _resolve_path(config_dir, model_spec.get("manifest"))
    weights_path = _resolve_path(config_dir, model_spec.get("weights"))
    source_path = _resolve_path(config_dir, model_spec.get("source") or model_spec.get("path"))

    if manifest_path:
        load_from_manifest = getattr(adapter, "load_from_manifest", None)
        if load_from_manifest is None:
            raise ValueError(f"Backend {backend} does not support manifest loading")
        bundle = load_from_manifest(manifest_path, device=device, weights_path=weights_path)
    elif source_path:
        bundle = adapter.load(source_path, device=device)
    else:
        raise ValueError("model.source or model.manifest is required")

    return bundle


def _print_metrics(metrics: dict[str, float]) -> None:
    for key in (
        "energy_rmse",
        "energy_mae",
        "energy_mae_cfg",
        "force_rmse",
        "force_mae",
    ):
        if key in metrics:
            print(f"{key}: {metrics[key]:.6e}")


def run_one(run: dict[str, Any], config_dir: Path) -> None:
    name = run.get("name") or run.get("id") or "run"
    task = str(run.get("task", "evaluate")).lower()
    if task == "backend":
        _run_backend_task(run, config_dir)
        return
    if task in {"train", "finetune", "resume", "export"}:
        spec = CommonTaskSpec.from_run_dict(run, config_dir)
        export_spec = run.get("export", {}) or {}
        export_artifacts = bool(export_spec.get("standard_artifacts", True))
        run_backend_task(spec, export_artifacts=export_artifacts)
        return
    if task not in {"evaluate", "infer"}:
        raise ValueError(f"Unsupported task: {task}")

    backend = run.get("backend")
    if not backend:
        raise ValueError("backend is required")

    adapter_cls = get_adapter(backend)
    adapter = adapter_cls()

    device = str(run.get("device", "cuda"))
    model_spec = run.get("model", {}) or {}
    data_spec = dict(run.get("data", {}) or {})
    eval_spec = run.get("eval", {}) or {}
    output_spec = run.get("output", {}) or {}

    energy_only = bool(eval_spec.get("energy_only", False) or run.get("energy_only", False))
    use_amp = bool(eval_spec.get("amp", False) or run.get("amp", False))
    if task == "evaluate" and energy_only:
        raise ValueError("evaluate requires forces; use task=infer for energy-only runs")
    head = model_spec.get("head") or run.get("head")

    bundle = _load_bundle(adapter, backend, model_spec, config_dir, device)
    extras = dict(bundle.extras or {})
    if head is not None:
        extras["head"] = head
    if energy_only:
        extras["energy_only"] = True
    bundle.extras = extras

    export_spec = model_spec.get("export", {}) or {}
    export_dir = _resolve_path(config_dir, export_spec.get("dir"))
    if export_dir:
        weights_name = export_spec.get("weights_name") or "model.pt"
        manifest_name = export_spec.get("manifest_name") or "manifest.json"
        export_bundle(bundle, Path(export_dir), weights_name, manifest_name)

    if data_spec.get("indices_path") is not None:
        data_spec["indices_path"] = _resolve_path(config_dir, str(data_spec["indices_path"]))

    lmdb_path = _resolve_path(config_dir, data_spec.get("lmdb") or data_spec.get("path") or data_spec.get("src"))
    if not lmdb_path:
        raise ValueError("data.lmdb is required")


    normalizer_cfg = (bundle.manifest or {}).get("normalizer") if isinstance(bundle.manifest, dict) else None
    transform = build_transform(
        {"transform": normalizer_cfg} if normalizer_cfg else {},
        manifest=bundle.manifest,
        extras=bundle.extras,
    )
    metrics, predictions, total_configs, total_atoms = run_lmdb_task(
        adapter=adapter,
        model=bundle.model,
        lmdb_path=lmdb_path,
        data_spec=data_spec,
        device=device,
        task=task,
        transform=transform,
        head=head,
        energy_only=energy_only,
        use_amp=use_amp,
    )

    output_dir = _resolve_path(config_dir, output_spec.get("dir"))
    if task == "evaluate":
        print(f"Run: {name} backend={backend} samples={total_configs} atoms={total_atoms}")
        _print_metrics(metrics)

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = out_dir / (output_spec.get("metrics_json") or "metrics.json")
            with metrics_path.open("w", encoding="utf-8") as handle:
                payload = {
                    "name": name,
                    "backend": backend,
                    "samples": total_configs,
                    "atoms": total_atoms,
                    **metrics,
                }
                json.dump(payload, handle, indent=2, ensure_ascii=True)

            metrics_csv = out_dir / (output_spec.get("metrics_csv") or "metrics.csv")
            _write_csv(metrics_csv, [payload])
    else:
        print(f"Run: {name} backend={backend} samples={total_configs} atoms={total_atoms}")
        print(f"Predictions: {len(predictions)}")
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_csv = out_dir / (output_spec.get("predictions_csv") or "predictions.csv")
            _write_csv(pred_csv, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified adapter runner")
    parser.add_argument("--config", required=True, help="Path to run.yaml")
    parser.add_argument("--run", action="append", help="Run name to execute (repeatable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    runs = _load_runs(config_path)
    selected = set(args.run or [])
    for run in runs:
        name = run.get("name") or run.get("id") or "run"
        if selected and name not in selected:
            continue
        run_one(run, config_path.parent)


if __name__ == "__main__":
    main()
