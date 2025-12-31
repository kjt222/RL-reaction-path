"""Batch helpers shared across adapters."""

from __future__ import annotations

from typing import Mapping, Tuple

import torch


def _as_tensor(value, device=None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value if device is None else value.to(device)
    return torch.as_tensor(value, device=device)


def build_ptr(batch: Mapping[str, object]) -> torch.Tensor:
    if "ptr" in batch and batch["ptr"] is not None:
        ptr = _as_tensor(batch["ptr"])
    elif "natoms" in batch and batch["natoms"] is not None:
        natoms = _as_tensor(batch["natoms"])
        if natoms.dim() == 0:
            natoms = natoms.unsqueeze(0)
        ptr = torch.zeros(natoms.numel() + 1, dtype=natoms.dtype, device=natoms.device)
        ptr[1:] = torch.cumsum(natoms, dim=0)
    else:
        z = _as_tensor(batch["z"])
        ptr = torch.tensor([0, int(z.shape[0])], device=z.device)
    return ptr.to(torch.long)


def batch_index_from_ptr(ptr: torch.Tensor, device: torch.device | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = ptr.device
    counts = (ptr[1:] - ptr[:-1]).to(torch.long)
    if counts.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=device), counts
    batch_index = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device, dtype=torch.long),
        counts,
    )
    return batch_index, counts


def mean_pool(node_embed: torch.Tensor, batch_index: torch.Tensor, num_graphs: int | None = None) -> torch.Tensor:
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    pooled = torch.zeros((num_graphs, node_embed.shape[-1]), device=node_embed.device, dtype=node_embed.dtype)
    if batch_index.numel():
        pooled.index_add_(0, batch_index, node_embed)
        counts = torch.bincount(batch_index, minlength=num_graphs).clamp_min(1).to(node_embed.dtype)
        pooled = pooled / counts.unsqueeze(-1)
    return pooled
