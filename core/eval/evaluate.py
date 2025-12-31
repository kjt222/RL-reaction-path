"""Evaluation/inference helpers for core pipeline."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from core.metrics import accumulate_metrics, finalize_metrics, init_metrics_state
from core.transforms import TargetTransform


def _as_tensor(value, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if value is None:
            moved[key] = None
        elif isinstance(value, (int, float, bool, str)):
            moved[key] = value
        else:
            moved[key] = _as_tensor(value, device)
    return moved


def evaluate(
    adapter: Any,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    task: str = "evaluate",
    transform: TargetTransform | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metrics_state = init_metrics_state()
    predictions: list[dict[str, Any]] = []
    transform_obj = transform or TargetTransform()

    for batch_indices, cbatch_raw in enumerate(loader):
        cbatch = transform_obj.apply_batch(cbatch_raw)
        cbatch = _move_batch_to_device(cbatch, device)
        backend_batch = adapter.make_backend_batch(cbatch, device)
        needs_grad = cbatch.get("forces") is not None and task == "evaluate"
        with torch.set_grad_enabled(needs_grad):
            outputs = adapter.forward(model, backend_batch)

        if task == "evaluate":
            outputs_detached = {k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()}
            outputs_physical = transform_obj.inverse_outputs(outputs_detached, cbatch_raw)
            accumulate_metrics(metrics_state, outputs_physical, cbatch_raw)
        else:
            outputs_physical = transform_obj.inverse_outputs(outputs, cbatch_raw)
            energies = torch.as_tensor(outputs_physical.get("energy", []), device="cpu").view(-1)
            ptr = torch.as_tensor(cbatch_raw.get("ptr"))
            counts = ptr[1:] - ptr[:-1]
            for i in range(int(counts.numel())):
                energy = float(energies[i].item()) if i < energies.numel() else float("nan")
                natoms = int(counts[i].item())
                per_atom = energy / natoms if natoms else float("nan")
                predictions.append(
                    {
                        "index": int(batch_indices * counts.numel() + i),
                        "energy": energy,
                        "energy_per_atom": per_atom,
                        "natoms": natoms,
                    }
                )

    metrics = finalize_metrics(metrics_state) if task == "evaluate" else {}
    return metrics, predictions
