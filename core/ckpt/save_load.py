"""Checkpoint save/load helpers for core trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def _extract_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "state_dict" in obj:
            return obj["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    raise ValueError("Unsupported checkpoint format; expected state_dict or checkpoint dict.")


def load_weights(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    obj = torch.load(Path(path), map_location=map_location, weights_only=False)
    return _extract_state_dict(obj)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler_state: Mapping[str, Any] | None,
    scaler_state: Mapping[str, Any] | None,
    epoch: int,
    best_metric: float | None,
    best_epoch: int | None,
    config: Mapping[str, Any] | None,
    normalizer: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": dict(scheduler_state) if scheduler_state is not None else None,
        "scaler_state_dict": dict(scaler_state) if scaler_state is not None else None,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "config": dict(config or {}),
        "normalizer": dict(normalizer or {}),
    }
    if extra:
        payload["extra"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    obj = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint must be a dict")
    return obj


def save_best_model(path: str | Path, model: torch.nn.Module) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, path)
    return path
