"""Run directory layout helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_dir(run_dir: Path) -> Path:
    return ensure_dir(run_dir / "artifacts")


def checkpoints_dir(run_dir: Path) -> Path:
    return ensure_dir(run_dir / "checkpoints")


def logs_dir(run_dir: Path) -> Path:
    return ensure_dir(run_dir / "logs")


def standard_artifact_paths(run_dir: Path) -> dict[str, Path]:
    artifacts = artifacts_dir(run_dir)
    return {
        "best_model": artifacts / "best_model.pt",
        "manifest": artifacts / "manifest.json",
    }


def standard_checkpoint_paths(run_dir: Path) -> dict[str, Path]:
    checkpoints = checkpoints_dir(run_dir)
    return {
        "checkpoint": checkpoints / "checkpoint.pt",
        "best_model": checkpoints / "best_model.pt",
    }
