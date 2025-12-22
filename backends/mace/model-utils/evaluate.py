"""Evaluate a trained model using stored metadata (no recomputation)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stub NVML / cuequivariance to avoid NVML/double-free issues on WSL/CPU runs.
# ---------------------------------------------------------------------------
import sys
from types import SimpleNamespace
import os

os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
os.environ.setdefault("CUEQUIVARIANCE_DISABLE_NVML", "1")

class _NVMLStub:
    class _FakeMem:
        total = 24 * 1024**3
        free = 20 * 1024**3
        used = total - free
    def nvmlInit(self): return None
    def nvmlShutdown(self): return None
    def nvmlDeviceGetCount(self): return 0
    def nvmlDeviceGetHandleByIndex(self, idx): return idx
    def nvmlDeviceGetCudaComputeCapability(self, handle): return (0, 0)
    def nvmlDeviceGetPowerManagementLimit(self, handle): return 0
    def nvmlDeviceGetName(self, handle): return b"Mock GPU"
    def nvmlDeviceGetMemoryInfo(self, handle): return self._FakeMem()
    def __getattr__(self, name): return lambda *a, **k: 0

sys.modules.setdefault("pynvml", _NVMLStub())
_STUB = SimpleNamespace()
for _name in [
    "cuequivariance_torch",
    "cuequivariance_ops_torch",
    "cuequivariance_ops",
    "cuequivariance_ops.triton",
    "cuequivariance_ops.triton.cache_manager",
    "cuequivariance_ops.triton.tuning_decorator",
    "cuequivariance_ops.triton.autotune_aot",
]:
    sys.modules.setdefault(_name, _STUB)
# ---------------------------------------------------------------------------

import argparse
import logging
from pathlib import Path
from typing import List, Sequence, Tuple
import lmdb
import sys

import numpy as np
import torch
import torch.serialization
from ase import io as ase_io
from mace import tools
from mace.tools import torch_geometric

# Ensure project root (backends/mace) is on sys.path to import dataloader/losses from root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader.lmdb_loader import LmdbAtomicDataset, _list_lmdb_files
from dataloader.xyz_loader import (
    AtomicDataListDataset,
    atoms_to_configurations,
    build_key_specification,
    configs_to_atomic_data,
    ensure_xyz_files,
    gather_atomic_numbers,
    reservoir_sample_atoms,
)
from losses import compute_train_loss, init_metrics_state, accumulate_metrics, finalize_metrics
from model_loader import load_for_eval

LOGGER = logging.getLogger(__name__)

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float32)
# 强制配置日志，防止被其他模块覆盖
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO, force=True)
# 确保未被全局 disable
logging.disable(logging.NOTSET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models using stored metadata (no recomputation).")
    parser.add_argument("--input_model", type=Path, required=True, help="Path to .pt model")
    parser.add_argument("--input_json", type=Path, help="Path to model.json (optional)")
    parser.add_argument("--data_format", choices=["xyz", "lmdb"], required=True)
    parser.add_argument("--xyz_dir", type=Path, help="Directory or file with XYZ data")
    parser.add_argument("--lmdb_path", type=Path, help="Directory containing LMDB shards")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of XYZ frames to sample (<=0 means use all frames)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lmdb_val_max_samples", type=int, default=None, help="Optional limit on number of LMDB validation samples (evaluation only).")
    parser.add_argument("--elements", type=int, nargs="+", help="Optional explicit list of atomic numbers for LMDB datasets")
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--force_weight", type=float, default=1000.0)
    return parser.parse_args()


def _random_indices(total_per_shard: List[int], max_samples: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(total_per_shard)
    total = cumulative[-1]
    count = min(max_samples, total)
    selected = rng.choice(total, size=count, replace=False)
    selected.sort()
    indices: List[Tuple[int, int]] = []
    shard_idx = 0
    shard_start = 0
    shard_end = cumulative[0]
    for gidx in selected:
        while gidx >= shard_end:
            shard_idx += 1
            shard_start = shard_end
            shard_end = cumulative[shard_idx]
        indices.append((shard_idx, int(gidx - shard_start)))
    return indices


def build_xyz_eval_loader(args, elements_override: Sequence[int] | None = None, cutoff: float | None = None):
    xyz_files = ensure_xyz_files(args.xyz_dir)
    if args.sample_size and args.sample_size > 0:
        sampled_atoms = reservoir_sample_atoms(xyz_files, args.sample_size, args.seed)
    else:
        sampled_atoms = []
        for xyz_path in xyz_files:
            LOGGER.info("Reading frames from %s", xyz_path.name)
            for atoms in ase_io.iread(xyz_path, index=":"):
                sampled_atoms.append(atoms.copy())
    if not sampled_atoms:
        raise ValueError("No atoms extracted from XYZ files for evaluation.")

    key_spec = build_key_specification()
    configs = atoms_to_configurations(sampled_atoms, key_spec)
    all_numbers = gather_atomic_numbers(configs)

    if elements_override is not None:
        provided = sorted({int(z) for z in elements_override})
        missing = set(all_numbers) - set(provided)
        if missing:
            raise ValueError(
                f"Dataset contains elements {sorted(missing)} not present in model z_table {provided}."
            )
        z_elements = provided
    else:
        z_elements = all_numbers

    z_table = tools.AtomicNumberTable(z_elements)
    atomic_data = configs_to_atomic_data(configs, z_table, cutoff)
    dataset = AtomicDataListDataset(atomic_data)
    loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return loader


def build_lmdb_eval_loader(args, elements_override: Sequence[int] | None = None, cutoff: float | None = None):
    if args.lmdb_path is None:
        raise ValueError("--lmdb_path must be provided for LMDB evaluation")
    lmdb_files = _list_lmdb_files(args.lmdb_path)
    key_spec = build_key_specification()

    element_override = elements_override or args.elements
    if element_override:
        element_list = sorted({int(z) for z in element_override})
    else:
        raise ValueError("elements must be provided for LMDB evaluation when using stored metadata.")

    z_table = tools.AtomicNumberTable(element_list)

    # Count entries per shard and random sample to avoid full coverage scan
    total_per_shard = []
    for lmdb_path in lmdb_files:
        env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            subdir=False,
            max_readers=1,
        )
        try:
            total_per_shard.append(env.stat().get("entries", 0))
        finally:
            env.close()
    if not total_per_shard or sum(total_per_shard) == 0:
        raise ValueError("LMDB dataset appears empty.")
    if args.lmdb_val_max_samples is None:
        selected_indices = None
    else:
        selected_indices = _random_indices(total_per_shard, args.lmdb_val_max_samples, args.seed)

    dataset = LmdbAtomicDataset(
        lmdb_files,
        z_table,
        cutoff,
        key_spec,
        max_samples=None,
        selected_indices=selected_indices,
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    return loader


def evaluate_model(
    model,
    loader,
    device,
    energy_weight,
    force_weight,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics_state = init_metrics_state()

    energy_preds: List[torch.Tensor] = []
    energy_targets: List[torch.Tensor] = []
    try:
        per_sample_enabled = len(loader.dataset) <= 10  # type: ignore[attr-defined]
    except Exception:
        per_sample_enabled = False
    energy_errors: List[float] = []
    force_rmse_samples: List[float] = []

    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(True):
            outputs = model(batch.to_dict(), training=False, compute_force=True)
            # 评估不需要反传，立刻切断图节省显存，并过滤 None
            outputs = {k: v.detach() for k, v in outputs.items() if v is not None}
            loss = compute_train_loss(
                outputs,
                batch,
                SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight),
            )
            pred_energy = outputs["energy"].view(-1)
            true_energy = batch.energy.view(-1)
            pred_forces = outputs["forces"]
            true_forces = batch.forces

        batch_size = batch.energy.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        energy_preds.append(pred_energy.detach().cpu().view(-1))
        energy_targets.append(true_energy.detach().cpu().view(-1))
        accumulate_metrics(metrics_state, outputs, batch, cfg=SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight))

        if per_sample_enabled:
            counts = None
            if hasattr(batch, "ptr"):
                ptr = batch.ptr.detach().cpu()
                counts = (ptr[1:] - ptr[:-1]).view(-1)
            elif hasattr(batch, "natoms"):
                counts = torch.as_tensor(batch.natoms).detach().cpu().view(-1)
            elif hasattr(batch, "n_atoms"):
                counts = torch.as_tensor(batch.n_atoms).detach().cpu().view(-1)
            if counts is None:
                raise ValueError("Batch missing atom counts (ptr/natoms/n_atoms) for per-sample metrics.")
            per_atom_energy_error = (pred_energy.detach().cpu().view(-1) - true_energy.detach().cpu().view(-1)) / counts
            energy_errors.extend(per_atom_energy_error.tolist())
            atom_batch = getattr(batch, "batch", None)
            if atom_batch is None:
                LOGGER.warning("Batch is missing 'batch' indices; disabling per-sample force RMSE calculation.")
                per_sample_enabled = False
                energy_errors = []
                force_rmse_samples = []
                continue
            per_atom_mse = torch.mean((pred_forces - true_forces) ** 2, dim=1)
            force_sum = torch.zeros(batch_size, device=per_atom_mse.device, dtype=per_atom_mse.dtype)
            force_count = torch.zeros(batch_size, device=per_atom_mse.device, dtype=per_atom_mse.dtype)
            force_sum.scatter_add_(0, atom_batch, per_atom_mse)
            force_count.scatter_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=per_atom_mse.dtype))
            batch_force_rmse = torch.sqrt(force_sum / force_count.clamp_min(1))
            force_rmse_samples.extend(batch_force_rmse.detach().cpu().tolist())

    if total_samples == 0:
        raise ValueError("Evaluation loader produced zero samples.")

    energy_preds_tensor = torch.cat(energy_preds)
    energy_targets_tensor = torch.cat(energy_targets)

    residual = energy_preds_tensor - energy_targets_tensor
    sse = torch.sum(residual ** 2)
    sst = torch.sum((energy_targets_tensor - torch.mean(energy_targets_tensor)) ** 2)
    r2_score = 1 - sse / sst if sst > 0 else torch.tensor(0.0)

    finalized = finalize_metrics(metrics_state, energy_weight=energy_weight, force_weight=force_weight)
    metrics = {
        "loss": float(total_loss / total_samples),
        "energy_rmse": float(finalized.get("energy_rmse", 0.0)),
        "force_rmse": float(finalized.get("force_rmse", 0.0)),
        "energy_mae": float(finalized.get("energy_mae", 0.0)),
        "force_mae": float(finalized.get("force_mae", 0.0)),
        "energy_mae_cfg": float(finalized.get("energy_mae_cfg", 0.0)),
        "r2": r2_score.item(),
    }
    per_sample = None
    if per_sample_enabled and total_samples <= 10:
        # 逐样本能量误差与力 RMSE（按 batch.batch 聚合）
        per_sample = {
            "energy_error": energy_errors[:total_samples],
            "force_rmse": force_rmse_samples[:total_samples],
        }
    return metrics, per_sample


def main() -> None:
    print(">>> evaluate.py starting", flush=True)
    args = parse_args()
    torch.manual_seed(args.seed)

    model_path = args.input_model.expanduser().resolve()
    model_json = args.input_json.expanduser().resolve() if args.input_json is not None else None
    model, _ = load_for_eval(model_path, input_json=model_json)

    # 提取 z_table 与 cutoff 供 dataloader 使用（仅信任模型，不再信任 metadata）
    if hasattr(model, "atomic_numbers"):
        z_table = tools.AtomicNumberTable(sorted({int(z) for z in model.atomic_numbers.view(-1).tolist()}))
    else:
        raise ValueError("无法从模型中获取 z_table。")

    if hasattr(model, "r_max"):
        cutoff = float(torch.as_tensor(model.r_max).item())
    else:
        cutoff = None

    if args.data_format == "xyz":
        loader = build_xyz_eval_loader(args, elements_override=z_table.zs, cutoff=cutoff)
    else:
        loader = build_lmdb_eval_loader(args, elements_override=z_table.zs, cutoff=cutoff)

    device = torch.device(args.device)
    model.to(device)

    metrics, per_sample = evaluate_model(
        model,
        loader,
        device,
        args.energy_weight,
        args.force_weight,
    )

    LOGGER.info(
        "Evaluation | Loss %.6f | E/atom RMSE %.6f MAE %.6f | |E| cfg %.6f | F/comp RMSE %.6f MAE %.6f | R^2 %.6f",
        metrics["loss"],
        metrics["energy_rmse"],
        metrics["energy_mae"],
        metrics["energy_mae_cfg"],
        metrics["force_rmse"],
        metrics["force_mae"],
        metrics["r2"],
    )
    if per_sample is not None:
        for idx, (de, frmse) in enumerate(zip(per_sample["energy_error"], per_sample["force_rmse"])):
            LOGGER.info("Sample %d | dE %.6f | F RMSE %.6f", idx, de, frmse)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO, force=True)
    main()
