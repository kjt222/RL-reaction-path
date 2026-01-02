"""LMDB evaluation/inference runner with optional AMP."""

from __future__ import annotations

import contextlib
from typing import Any, Mapping

import torch

from core.data.indices import build_indices
from core.data.io_lmdb import oc22_data_to_canonical
from core.data.lmdb_reader import LmdbReader, iter_batches
from core.metrics import accumulate_metrics, finalize_metrics, init_metrics_state
from core.transforms import TargetTransform


def _get_counts_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
    counts = ptr[1:] - ptr[:-1]
    if counts.dim() == 0:
        counts = counts.unsqueeze(0)
    return counts


def run_lmdb_task(
    adapter: Any,
    model: torch.nn.Module,
    lmdb_path: str,
    data_spec: Mapping[str, Any],
    device: str | torch.device,
    *,
    task: str = "evaluate",
    transform: TargetTransform | None = None,
    head: str | None = None,
    energy_only: bool = False,
    use_amp: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]], int, int]:
    if task not in {"evaluate", "infer"}:
        raise ValueError(f"Unsupported task: {task}")

    device_obj = torch.device(device)
    transform_obj = transform or TargetTransform()

    metrics_state = init_metrics_state()
    predictions: list[dict[str, Any]] = []
    total_configs = 0
    total_atoms = 0

    require_energy = task == "evaluate"
    require_forces = task == "evaluate" and not energy_only
    needs_grad = require_forces and task == "evaluate"

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device_obj.type == "cuda"
        else contextlib.nullcontext()
    )

    with LmdbReader(lmdb_path) as reader:
        indices = build_indices(len(reader), data_spec)
        for batch_indices, samples in iter_batches(reader, indices, batch_size=int(data_spec.get("batch_size", 1))):
            batch_raw = oc22_data_to_canonical(
                samples,
                device=device_obj,
                require_energy=require_energy,
                require_forces=require_forces,
            )
            if head is not None:
                batch_raw["head"] = head
            if energy_only:
                batch_raw["energy_only"] = True

            batch = transform_obj.apply_batch(batch_raw)
            backend_batch = adapter.make_backend_batch(batch, device_obj)
            with torch.set_grad_enabled(needs_grad):
                with amp_ctx:
                    outputs = adapter.forward(model, backend_batch)
            outputs = transform_obj.inverse_outputs(outputs, batch_raw)

            ptr = torch.as_tensor(batch_raw.get("ptr"))
            counts = _get_counts_from_ptr(ptr)
            total_configs += int(counts.numel())
            total_atoms += int(counts.sum().item())

            if task == "evaluate":
                accumulate_metrics(metrics_state, outputs, batch_raw)
            else:
                energies = torch.as_tensor(outputs.get("energy", []), device="cpu").view(-1)
                counts_cpu = counts.to("cpu")
                for i, idx in enumerate(batch_indices):
                    energy = float(energies[i].item()) if i < energies.numel() else float("nan")
                    natoms = int(counts_cpu[i].item()) if i < counts_cpu.numel() else 0
                    per_atom = energy / natoms if natoms else float("nan")
                    predictions.append(
                        {
                            "index": int(idx),
                            "energy": energy,
                            "energy_per_atom": per_atom,
                            "natoms": natoms,
                        }
                    )

    metrics = finalize_metrics(metrics_state) if task == "evaluate" else {}
    return metrics, predictions, total_configs, total_atoms
