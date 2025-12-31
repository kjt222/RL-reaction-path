"""Backend runner for FairChem-based tasks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.runner.spec import BackendRunResult, CommonTaskSpec


def _resolve_path(base_dir: Path | None, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_root = run_dir / "checkpoints"
    if not checkpoints_root.exists():
        return None
    candidates = [p for p in checkpoints_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    best = latest / "best_checkpoint.pt"
    if best.exists():
        return best
    fallback = latest / "checkpoint.pt"
    if fallback.exists():
        return fallback
    return None


def _run_command(cmd: list[str], spec: CommonTaskSpec) -> None:
    python = spec.backend_python or sys.executable
    full_cmd = [python, *cmd]
    env = os.environ.copy()
    env.update(spec.backend_env)
    subprocess.run(full_cmd, check=True, env=env, cwd=str(spec.config_dir or Path.cwd()))


def run_task(spec: CommonTaskSpec) -> BackendRunResult:
    config_yml = spec.train.get("config_yml") or spec.train.get("config") or spec.data.get("config_yml")
    if not config_yml:
        raise ValueError("fairchem backends require train.config_yml")
    config_yml = _resolve_path(spec.config_dir, str(config_yml))

    mode = spec.mode
    if mode in {"train", "finetune", "resume"}:
        mode = "train"
    elif mode == "evaluate":
        mode = "validate"

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "backends" / "equiformer_v2" / "fairchem" / "main.py"

    cmd = [
        str(script),
        "--mode",
        mode,
        "--config-yml",
        str(config_yml),
        "--run-dir",
        str(spec.run_dir),
    ]

    if spec.model_in:
        cmd.extend(["--checkpoint", spec.model_in])

    identifier = spec.train.get("identifier")
    if identifier:
        cmd.extend(["--identifier", str(identifier)])
    seed = spec.train.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if spec.backend_args:
        cmd.extend(spec.backend_args)

    _run_command(cmd, spec)

    best_path = _find_latest_checkpoint(spec.run_dir)
    return BackendRunResult(run_dir=spec.run_dir, best_model_path=best_path, checkpoint_path=best_path)
