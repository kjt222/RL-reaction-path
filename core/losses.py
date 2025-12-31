"""Shared loss helpers for adapter implementations."""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import torch

from core.batch.batch_utils import build_ptr
from core.contracts import CanonicalBatch, ModelOutputs


def _as_loss_name(value: object | None, default: str) -> str:
    if value is None:
        return default
    name = str(value).strip().lower()
    return name or default


def _as_float(value: object | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return total / count.clamp_min(1)


def _energy_loss(
    energy_pred: torch.Tensor,
    energy_true: torch.Tensor,
    counts: torch.Tensor,
    *,
    loss_name: str,
    per_atom: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_name = _as_loss_name(loss_name, "mse")
    if energy_pred.shape[0] != counts.shape[0]:
        raise ValueError(f"Energy batch size {energy_pred.shape[0]} != atom counts {counts.shape[0]}")
    if per_atom:
        error = (energy_pred - energy_true) / counts
        if loss_name == "mae":
            loss_sum = (error.abs() * counts).sum()
        elif loss_name == "mse":
            loss_sum = (error.pow(2) * counts).sum()
        else:
            raise ValueError(f"Unsupported energy loss: {loss_name}")
        loss_count = counts.sum()
    else:
        error = energy_pred - energy_true
        if loss_name == "mae":
            loss_sum = error.abs().sum()
        elif loss_name == "mse":
            loss_sum = error.pow(2).sum()
        else:
            raise ValueError(f"Unsupported energy loss: {loss_name}")
        loss_count = torch.as_tensor(energy_pred.numel(), device=energy_pred.device, dtype=energy_pred.dtype)

    loss_mean = _safe_mean(loss_sum, loss_count)
    return loss_sum, loss_count, loss_mean


def _force_loss(
    force_pred: torch.Tensor,
    force_true: torch.Tensor,
    counts: torch.Tensor,
    *,
    loss_name: str,
    train_on_free_atoms: bool,
    fixed: torch.Tensor | None,
    tag_specific_weights: Iterable[float] | None,
    tags: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_name = _as_loss_name(loss_name, "mse")
    diff = force_pred - force_true
    mask = None
    if train_on_free_atoms and fixed is not None:
        mask = fixed == 0
        diff = diff[mask]
        if tags is not None:
            tags = tags[mask]

    if loss_name == "l2mae":
        dists = torch.norm(diff, p=2, dim=-1)
        if tag_specific_weights:
            if tags is None:
                raise ValueError("tag_specific_weights requires batch tags")
            weights = torch.zeros_like(dists)
            weights[tags == 0] = float(tag_specific_weights[0])
            weights[tags == 1] = float(tag_specific_weights[1])
            weights[tags == 2] = float(tag_specific_weights[2])
            loss_sum = (dists * weights).sum()
        else:
            loss_sum = dists.sum()
        loss_count = torch.as_tensor(dists.numel(), device=dists.device, dtype=dists.dtype)
    elif loss_name == "atomwisel2":
        dists = torch.norm(diff, p=2, dim=-1)
        counts_long = counts.to(torch.long)
        natoms = torch.repeat_interleave(counts_long, counts_long)
        if mask is not None:
            natoms = natoms[mask]
        natoms = natoms.to(dists.device, dtype=dists.dtype)
        loss_sum = (natoms * dists).sum()
        loss_count = torch.as_tensor(counts_long.numel(), device=dists.device, dtype=dists.dtype)
    elif loss_name == "mae":
        loss_sum = diff.abs().sum()
        loss_count = torch.as_tensor(diff.numel(), device=diff.device, dtype=diff.dtype)
    elif loss_name == "mse":
        loss_sum = diff.pow(2).sum()
        loss_count = torch.as_tensor(diff.numel(), device=diff.device, dtype=diff.dtype)
    else:
        raise ValueError(f"Unsupported force loss: {loss_name}")

    loss_mean = _safe_mean(loss_sum, loss_count)
    return loss_sum, loss_count, loss_mean


def compute_energy_force_loss(
    outputs: ModelOutputs,
    cbatch: CanonicalBatch,
    *,
    energy_weight: float,
    force_weight: float,
    energy_loss: str,
    force_loss: str,
    energy_loss_mode: str,
    train_on_free_atoms: bool = False,
    tag_specific_weights: Iterable[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    energy_weight = _as_float(energy_weight, 1.0)
    force_weight = _as_float(force_weight, 1.0)
    energy_loss = _as_loss_name(energy_loss, "mse")
    force_loss = _as_loss_name(force_loss, "mse")

    ptr = build_ptr(cbatch)
    counts = (ptr[1:] - ptr[:-1]).to(torch.long)
    if counts.dim() == 0:
        counts = counts.unsqueeze(0)

    energy_sum = torch.tensor(0.0, device=counts.device)
    energy_count = torch.tensor(0.0, device=counts.device)
    energy_mean = torch.tensor(0.0, device=counts.device)
    if outputs.get("energy") is not None and cbatch.get("energy") is not None:
        energy_pred = torch.as_tensor(outputs["energy"], device=counts.device).view(-1)
        energy_true = torch.as_tensor(cbatch["energy"], device=counts.device).view(-1)
        per_atom = energy_loss_mode == "per_atom"
        energy_sum, energy_count, energy_mean = _energy_loss(
            energy_pred,
            energy_true,
            counts.to(energy_pred),
            loss_name=energy_loss,
            per_atom=per_atom,
        )

    force_sum = torch.tensor(0.0, device=counts.device)
    force_count = torch.tensor(0.0, device=counts.device)
    force_mean = torch.tensor(0.0, device=counts.device)
    if outputs.get("forces") is not None and cbatch.get("forces") is not None and force_weight > 0:
        force_pred = torch.as_tensor(outputs["forces"], device=counts.device)
        force_true = torch.as_tensor(cbatch["forces"], device=counts.device)
        fixed = cbatch.get("fixed")
        tags = cbatch.get("tags")
        fixed_t = torch.as_tensor(fixed, device=counts.device) if fixed is not None else None
        tags_t = torch.as_tensor(tags, device=counts.device) if tags is not None else None
        force_sum, force_count, force_mean = _force_loss(
            force_pred,
            force_true,
            counts,
            loss_name=force_loss,
            train_on_free_atoms=train_on_free_atoms,
            fixed=fixed_t,
            tag_specific_weights=tag_specific_weights,
            tags=tags_t,
        )

    total_loss = energy_weight * energy_mean + force_weight * force_mean
    logs = {
        "energy_loss": float(energy_mean.item()),
        "force_loss": float(force_mean.item()),
        "energy_loss_sum": float(energy_sum.item()),
        "energy_count": float(energy_count.item()),
        "force_loss_sum": float(force_sum.item()),
        "force_count": float(force_count.item()),
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "loss": float(total_loss.item()),
    }
    return total_loss, logs


__all__ = ["compute_energy_force_loss"]
