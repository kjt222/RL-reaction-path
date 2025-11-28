"""Evaluate a trained model using stored metadata (no recomputation)."""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
from typing import List, Sequence, Tuple
import lmdb

import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization
from ase import io as ase_io
from mace import tools
from mace.tools import torch_geometric

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
from metadata import load_checkpoint
from models import instantiate_model

LOGGER = logging.getLogger(__name__)

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models using stored metadata (no recomputation).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_format", choices=["xyz", "lmdb"], required=True)
    parser.add_argument("--xyz_dir", type=Path, help="Directory or file with XYZ data")
    parser.add_argument("--lmdb_path", type=Path, help="Directory containing LMDB shards")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of XYZ frames to sample (<=0 means use all frames)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lmdb_val_max_samples",
        type=int,
        default=None,
        help="Optional limit on number of LMDB validation samples (evaluation only).",
    )
    parser.add_argument("--elements", type=int, nargs="+", help="Optional explicit list of atomic numbers for LMDB datasets")
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--force_weight", type=float, default=1000.0)
    return parser.parse_args()


def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def compute_losses(outputs, batch, energy_weight, force_weight):
    pred_energy = outputs["energy"].view(-1)
    true_energy = batch.energy.view(-1)
    energy_loss = F.mse_loss(pred_energy, true_energy)

    pred_forces = outputs["forces"]
    true_forces = batch.forces
    force_loss = F.mse_loss(pred_forces, true_forces)

    total_loss = energy_weight * energy_loss + force_weight * force_loss
    return total_loss, energy_loss, force_loss, pred_energy, true_energy


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
                f"Dataset contains elements {sorted(missing)} not present in checkpoint z_table {provided}."
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
    total_energy_rmse = 0.0
    total_force_rmse = 0.0
    total_samples = 0

    energy_preds: List[torch.Tensor] = []
    energy_targets: List[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        with torch.enable_grad():
            outputs = model(batch.to_dict(), training=False, compute_force=True)
            loss, energy_loss, force_loss, pred_energy, true_energy = compute_losses(
                outputs, batch, energy_weight, force_weight
            )

        batch_size = batch.energy.shape[0]
        total_loss += loss.item() * batch_size
        total_energy_rmse += torch.sqrt(energy_loss).item() * batch_size
        total_force_rmse += torch.sqrt(force_loss).item() * batch_size
        total_samples += batch_size

        energy_preds.append(pred_energy.detach().cpu().view(-1))
        energy_targets.append(true_energy.detach().cpu().view(-1))

    if total_samples == 0:
        raise ValueError("Evaluation loader produced zero samples.")

    energy_preds_tensor = torch.cat(energy_preds)
    energy_targets_tensor = torch.cat(energy_targets)

    residual = energy_preds_tensor - energy_targets_tensor
    sse = torch.sum(residual ** 2)
    sst = torch.sum((energy_targets_tensor - torch.mean(energy_targets_tensor)) ** 2)
    r2_score = 1 - sse / sst if sst > 0 else torch.tensor(0.0)

    metrics = {
        "loss": total_loss / total_samples,
        "energy_rmse": total_energy_rmse / total_samples,
        "force_rmse": total_force_rmse / total_samples,
        "r2": r2_score.item(),
    }
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    base_dir = checkpoint_path.parent
    json_path = base_dir / "metadata.json"

    # If the provided checkpoint is best_model.pt, prefer metadata from sibling checkpoint.pt
    meta_source = checkpoint_path
    meta_bundle = None
    ckpt_with_meta = base_dir / "checkpoint.pt"
    if checkpoint_path.name != "checkpoint.pt" and ckpt_with_meta.exists():
        meta_source = ckpt_with_meta
    try:
        meta_bundle = load_checkpoint(
            meta_source,
            map_location="cpu",
            allow_json_to_metadata=False,
            strict_json=False,
            ignore_json=True,
            require_metadata=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint metadata from {meta_source}: {e}")

    metadata = (meta_bundle.get("metadata") if meta_bundle else {}) or {}

    # 如果存在 metadata.json，要求与 checkpoint 内的 metadata 一致
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            json_meta = json.load(f)
        LOGGER.info("Loaded metadata.json from %s", json_path)
        if _normalize(metadata) != _normalize(json_meta):
            raise ValueError("Checkpoint metadata and metadata.json do not match; aborting evaluation.")
        metadata = json_meta

    # Load state_dict from the user-specified checkpoint (best_model or checkpoint)
    state_dict = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        allow_json_to_metadata=False,
        strict_json=False,
        ignore_json=True,
    ).get("model_state_dict", {})

    required_keys = ("z_table", "avg_num_neighbors", "e0_values", "cutoff", "num_interactions", "architecture")
    missing = [k for k in required_keys if metadata.get(k) is None]
    if missing:
        raise ValueError(f"Checkpoint/metadata.json missing required fields: {missing}")

    z_table = tools.AtomicNumberTable(sorted({int(z) for z in metadata["z_table"]}))
    avg_num_neighbors = float(metadata["avg_num_neighbors"])
    e0_values = np.asarray(metadata["e0_values"], dtype=float)
    cutoff = float(metadata["cutoff"])
    num_interactions = int(metadata["num_interactions"])
    architecture = metadata["architecture"]

    if args.data_format == "xyz":
        loader = build_xyz_eval_loader(args, elements_override=z_table.zs, cutoff=cutoff)
    else:
        loader = build_lmdb_eval_loader(args, elements_override=z_table.zs, cutoff=cutoff)

    model = instantiate_model(
        z_table,
        avg_num_neighbors,
        cutoff,
        e0_values,
        num_interactions,
        architecture=architecture,
    )

    device = torch.device(args.device)
    model.to(device)
    # 移除会覆盖元数据的缓冲区（如旧的 E0/avg_num_neighbors/atomic_numbers）
    strip_keys = [
        "atomic_energies_fn.atomic_energies",
        "atomic_numbers",
    ]
    strip_keys += [k for k in state_dict.keys() if k.endswith("avg_num_neighbors")]
    for k in strip_keys:
        state_dict.pop(k, None)
    model.load_state_dict(state_dict, strict=False)
    # 再次写入元数据中的 E0 / avg_num_neighbors，确保不被旧权重覆盖
    if hasattr(model, "atomic_energies_fn") and hasattr(model.atomic_energies_fn, "atomic_energies"):
        ae_dev = model.atomic_energies_fn.atomic_energies.device
        ae = torch.as_tensor(e0_values, dtype=model.atomic_energies_fn.atomic_energies.dtype, device=ae_dev)
        model.atomic_energies_fn.atomic_energies.data.copy_(ae)
    if hasattr(model, "avg_num_neighbors"):
        model.avg_num_neighbors = torch.as_tensor(
            avg_num_neighbors, dtype=torch.float64, device=model.parameters().__next__().device
        )
    LOGGER.info("Loaded checkpoint from %s", args.checkpoint)

    metrics = evaluate_model(
        model,
        loader,
        device,
        args.energy_weight,
        args.force_weight,
    )

    LOGGER.info(
        "Evaluation | Loss %.6f | Energy RMSE %.6f | Force RMSE %.6f | R^2 %.6f",
        metrics["loss"],
        metrics["energy_rmse"],
        metrics["force_rmse"],
        metrics["r2"],
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
    main()
