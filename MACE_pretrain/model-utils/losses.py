"""Shared loss and metric helpers with per-atom energy/error accounting."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def _get_cfg_attr(cfg: Any, name: str, default: float) -> float:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return float(cfg.get(name, default))
    return float(getattr(cfg, name, default))


def _atom_counts(batch: Any) -> torch.Tensor:
    """Return per-config atom counts from a batched AtomicData/Batch."""
    if hasattr(batch, "ptr"):
        ptr = torch.as_tensor(batch.ptr)
        counts = ptr[1:] - ptr[:-1]
    elif hasattr(batch, "natoms"):
        counts = torch.as_tensor(batch.natoms)
    elif hasattr(batch, "n_atoms"):
        counts = torch.as_tensor(batch.n_atoms)
    else:
        raise ValueError("Batch missing atom counts (ptr/natoms/n_atoms).")
    if counts.dim() == 0:
        counts = counts.unsqueeze(0)
    return counts


def compute_train_loss(outputs: Mapping[str, torch.Tensor], batch: Any, cfg: Any) -> torch.Tensor:
    """Return weighted loss for backprop; energy is per-atom MSE, force per-component MSE."""
    energy_weight = _get_cfg_attr(cfg, "energy_weight", 1.0)
    force_weight = _get_cfg_attr(cfg, "force_weight", 1.0)

    energy_pred = outputs["energy"].view(-1)
    energy_true = batch.energy.view(-1)
    counts = _atom_counts(batch).to(energy_pred)
    if energy_pred.shape[0] != counts.shape[0]:
        raise ValueError(f"Energy batch size {energy_pred.shape[0]} != atom counts {counts.shape[0]}")

    # Per-atom energy error: delta_E / num_atoms, averaged over atoms.
    energy_error_per_atom = (energy_pred - energy_true) / counts
    energy_mse = (energy_error_per_atom.pow(2) * counts).sum() / counts.sum()

    force_pred = outputs["forces"]
    force_true = batch.forces
    force_mse = torch.mean((force_pred - force_true) ** 2)

    return energy_weight * energy_mse + force_weight * force_mse


def init_metrics_state() -> dict[str, float]:
    return {
        "energy_sse": 0.0,
        "energy_mae": 0.0,
        "energy_count": 0.0,
        "force_sse": 0.0,
        "force_mae": 0.0,
        "force_count": 0.0,
    }


def accumulate_metrics(state: dict[str, float], outputs: Mapping[str, torch.Tensor], batch: Any, cfg: Any | None = None) -> None:
    """Accumulate SSE/MAE counts for global RMSE/MAE; energy is per-atom, force per-component."""
    counts = _atom_counts(batch).to(outputs["energy"])
    energy_pred = outputs["energy"].view(-1)
    energy_true = batch.energy.view(-1)
    energy_error_per_atom = (energy_pred - energy_true) / counts
    energy_sse_inc = float((energy_error_per_atom.pow(2) * counts).sum().item())
    energy_mae_inc = float((energy_error_per_atom.abs() * counts).sum().item())
    energy_count_inc = float(counts.sum().item())
    state["energy_sse"] += energy_sse_inc
    state["energy_mae"] += energy_mae_inc
    state["energy_count"] += energy_count_inc

    force_diff = outputs["forces"] - batch.forces
    force_sse_inc = float(force_diff.pow(2).sum().item())
    force_mae_inc = float(force_diff.abs().sum().item())
    force_count_inc = float(force_diff.numel())
    state["force_sse"] += force_sse_inc
    state["force_mae"] += force_mae_inc
    state["force_count"] += force_count_inc


def finalize_metrics(state: dict[str, float], energy_weight: float = 1.0, force_weight: float = 1.0) -> dict[str, float]:
    """Compute global RMSE/MAE; loss 需在训练/验证循环中按 batch_size 累计计算."""
    metrics: dict[str, float] = {}
    energy_mse = None
    force_mse = None
    if state["energy_count"] > 0:
        energy_mse = state["energy_sse"] / state["energy_count"]
        metrics["energy_rmse"] = energy_mse**0.5
        metrics["energy_mae"] = state["energy_mae"] / state["energy_count"]
    if state["force_count"] > 0:
        force_mse = state["force_sse"] / state["force_count"]
        metrics["force_rmse"] = force_mse**0.5
        metrics["force_mae"] = state["force_mae"] / state["force_count"]
    return metrics


__all__ = [
    "compute_train_loss",
    "init_metrics_state",
    "accumulate_metrics",
    "finalize_metrics",
]
