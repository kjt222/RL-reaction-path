"""Metric interface for unified evaluation across backends."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .contracts import CanonicalBatch, MetricOutputs, ModelOutputs


METRIC_KEYS = (
    "energy_rmse",
    "energy_mae",
    "energy_mae_cfg",
    "force_rmse",
    "force_mae",
    "loss",
)


def compute_metrics(_outputs: ModelOutputs, _batch: CanonicalBatch) -> MetricOutputs:
    state = init_metrics_state()
    accumulate_metrics(state, _outputs, _batch)
    return finalize_metrics(state)


def compute_loss(_outputs: ModelOutputs, _batch: CanonicalBatch) -> float:
    raise NotImplementedError("Loss is backend-owned; frontend provides interface only")


def _atom_counts(batch: Mapping[str, Any]) -> torch.Tensor:
    if "ptr" in batch and batch["ptr"] is not None:
        ptr = torch.as_tensor(batch["ptr"])
        counts = ptr[1:] - ptr[:-1]
    elif "natoms" in batch and batch["natoms"] is not None:
        counts = torch.as_tensor(batch["natoms"])
    else:
        raise ValueError("Batch missing atom counts (ptr/natoms).")
    if counts.dim() == 0:
        counts = counts.unsqueeze(0)
    return counts


def init_metrics_state() -> dict[str, float]:
    return {
        "energy_sse": 0.0,
        "energy_mae": 0.0,
        "energy_count": 0.0,
        "energy_abs_sum_cfg": 0.0,
        "energy_cfg_count": 0.0,
        "force_sse": 0.0,
        "force_mae": 0.0,
        "force_count": 0.0,
    }


def accumulate_metrics(state: dict[str, float], outputs: ModelOutputs, batch: CanonicalBatch) -> None:
    if "energy" in outputs and "energy" in batch and outputs["energy"] is not None and batch["energy"] is not None:
        counts = _atom_counts(batch).to(outputs["energy"])
        energy_pred = torch.as_tensor(outputs["energy"]).view(-1)
        energy_true = torch.as_tensor(batch["energy"]).view(-1)
        if energy_pred.shape[0] != counts.shape[0]:
            raise ValueError(f"Energy batch size {energy_pred.shape[0]} != atom counts {counts.shape[0]}")

        energy_error_per_atom = (energy_pred - energy_true) / counts
        state["energy_sse"] += float((energy_error_per_atom.pow(2) * counts).sum().item())
        state["energy_mae"] += float((energy_error_per_atom.abs() * counts).sum().item())
        state["energy_count"] += float(counts.sum().item())

        state["energy_abs_sum_cfg"] += float((energy_pred - energy_true).abs().sum().item())
        state["energy_cfg_count"] += float(energy_pred.shape[0])

    if "forces" in outputs and "forces" in batch and outputs["forces"] is not None and batch["forces"] is not None:
        force_pred = torch.as_tensor(outputs["forces"])
        force_true = torch.as_tensor(batch["forces"])
        force_diff = force_pred - force_true
        state["force_sse"] += float(force_diff.pow(2).sum().item())
        state["force_mae"] += float(force_diff.abs().sum().item())
        state["force_count"] += float(force_diff.numel())


def finalize_metrics(state: Mapping[str, float]) -> MetricOutputs:
    metrics: MetricOutputs = {}
    if state.get("energy_count", 0.0) > 0:
        energy_mse = state["energy_sse"] / state["energy_count"]
        metrics["energy_rmse"] = energy_mse**0.5
        metrics["energy_mae"] = state["energy_mae"] / state["energy_count"]
    if state.get("energy_cfg_count", 0.0) > 0:
        metrics["energy_mae_cfg"] = state["energy_abs_sum_cfg"] / state["energy_cfg_count"]
    if state.get("force_count", 0.0) > 0:
        force_mse = state["force_sse"] / state["force_count"]
        metrics["force_rmse"] = force_mse**0.5
        metrics["force_mae"] = state["force_mae"] / state["force_count"]
    return metrics
