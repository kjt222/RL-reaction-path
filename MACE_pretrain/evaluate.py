"""Evaluate a trained MACE model on XYZ or LMDB datasets."""

from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization
import torch.utils.data
from ase import io as ase_io

from mace import modules, tools
from mace.tools import torch_geometric
from metadata import load_checkpoint

from dataloader.xyz_loader import (
    AtomicDataListDataset,
    atoms_to_configurations,
    build_key_specification,
    compute_e0s,
    configs_to_atomic_data,
    ensure_xyz_files,
    gather_atomic_numbers,
    reservoir_sample_atoms,
)
from dataloader.lmdb_loader import LmdbAtomicDataset, _list_lmdb_files, _sample_configs_and_elements
from models import instantiate_model

LOGGER = logging.getLogger(__name__)

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MACE models")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_format", choices=["xyz", "lmdb"], required=True)
    parser.add_argument("--xyz_dir", type=Path, help="Directory or file with XYZ data")
    parser.add_argument("--lmdb_path", type=Path, help="Directory containing LMDB shards")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of XYZ frames to sample (<=0 means use all frames)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num_interactions", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lmdb_e0_samples", type=int, default=2000)
    parser.add_argument("--neighbor_sample_size", type=int, default=1024)
    parser.add_argument("--elements", type=int, nargs="+", help="Optional explicit list of atomic numbers for LMDB datasets")
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--force_weight", type=float, default=1000.0)
    return parser.parse_args()


def compute_losses(outputs, batch, energy_weight, force_weight):
    pred_energy = outputs["energy"].squeeze(-1)
    true_energy = batch.energy.squeeze(-1)
    energy_loss = F.mse_loss(pred_energy, true_energy)

    pred_forces = outputs["forces"]
    true_forces = batch.forces
    force_loss = F.mse_loss(pred_forces, true_forces)

    total_loss = energy_weight * energy_loss + force_weight * force_loss
    return total_loss, energy_loss, force_loss, pred_energy, true_energy


def build_xyz_eval_loader(args, elements_override: Sequence[int] | None = None):
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
    atomic_data = configs_to_atomic_data(configs, z_table, args.cutoff)
    dataset = AtomicDataListDataset(atomic_data)
    loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    stats_loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    avg_num_neighbors = modules.compute_avg_num_neighbors(stats_loader)
    e0_values = compute_e0s(configs, z_table)
    return loader, z_table, avg_num_neighbors, e0_values


def build_lmdb_eval_loader(args, elements_override: Sequence[int] | None = None):
    if args.lmdb_path is None:
        raise ValueError("--lmdb_path must be provided for LMDB evaluation")
    lmdb_files = _list_lmdb_files(args.lmdb_path)
    key_spec = build_key_specification()
    sample_limit = max(1, args.lmdb_e0_samples)
    sampled_configs, detected_numbers = _sample_configs_and_elements(
        lmdb_files, key_spec, sample_limit
    )

    element_override = elements_override or args.elements
    if element_override:
        element_list = sorted({int(z) for z in element_override})
    else:
        element_list = detected_numbers
    if not element_list:
        raise ValueError("Could not determine element list for LMDB evaluation")

    z_table = tools.AtomicNumberTable(element_list)
    e0_values = compute_e0s(sampled_configs, z_table)

    dataset = LmdbAtomicDataset(lmdb_files, z_table, args.cutoff, key_spec)
    loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    neighbor_sample_size = min(len(dataset), args.neighbor_sample_size)
    stats_subset = torch.utils.data.Subset(dataset, list(range(neighbor_sample_size)))
    stats_loader = torch_geometric.dataloader.DataLoader(
        stats_subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    avg_num_neighbors = modules.compute_avg_num_neighbors(stats_loader)
    return loader, z_table, avg_num_neighbors, e0_values


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

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.to_dict(), training=False, compute_force=True)
            loss, energy_loss, force_loss, pred_energy, true_energy = compute_losses(
                outputs, batch, energy_weight, force_weight
            )

            batch_size = batch.energy.shape[0]
            total_loss += loss.item() * batch_size
            total_energy_rmse += torch.sqrt(energy_loss).item() * batch_size
            total_force_rmse += torch.sqrt(force_loss).item() * batch_size
            total_samples += batch_size

            energy_preds.append(pred_energy.detach().cpu())
            energy_targets.append(true_energy.detach().cpu())

    if total_samples == 0:
        raise ValueError("Evaluation loader produced zero samples.")

    energy_preds_tensor = torch.cat(energy_preds)
    energy_targets_tensor = torch.cat(energy_targets)

    residual = energy_preds_tensor - energy_targets_tensor
    sse = torch.sum(residual ** 2)
    tgt_mean = torch.mean(energy_targets_tensor)
    sst = torch.sum((energy_targets_tensor - tgt_mean) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else torch.tensor(float("nan"))

    metrics = {
        "loss": total_loss / total_samples,
        "energy_rmse": total_energy_rmse / total_samples,
        "force_rmse": total_force_rmse / total_samples,
        "r2": r2.item(),
    }
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    bundle = load_checkpoint(args.checkpoint, map_location="cpu")
    state_dict = bundle.get("model_state_dict", bundle)
    metadata = bundle.get("metadata") or {}
    ckpt_avg_neighbors = metadata.get("avg_num_neighbors")
    ckpt_z_table = metadata.get("z_table")
    ckpt_e0_values = metadata.get("e0_values")
    ckpt_cutoff = metadata.get("cutoff")
    ckpt_num_interactions = metadata.get("num_interactions")

    if ckpt_z_table is not None:
        elements_override = sorted({int(z) for z in ckpt_z_table})
        LOGGER.info(
            "Using element list from checkpoint (%d elements).",
            len(elements_override),
        )
    else:
        elements_override = None
        LOGGER.warning(
            "Checkpoint %s does not contain z_table; falling back to dataset elements."
            " 请确认评估数据的元素集合与训练时一致。",
            args.checkpoint,
        )

    if args.data_format == "xyz":
        if args.xyz_dir is None:
            raise ValueError("--xyz_dir must be provided for XYZ evaluation")
        loader, data_z_table, data_avg_neighbors, data_e0_values = build_xyz_eval_loader(
            args,
            elements_override=elements_override,
        )
    else:
        loader, data_z_table, data_avg_neighbors, data_e0_values = build_lmdb_eval_loader(
            args,
            elements_override=elements_override,
        )

    if ckpt_z_table is not None:
        z_table = tools.AtomicNumberTable(elements_override)
    else:
        z_table = data_z_table

    if ckpt_avg_neighbors is not None:
        avg_num_neighbors = float(ckpt_avg_neighbors)
    else:
        avg_num_neighbors = float(data_avg_neighbors)
        LOGGER.warning(
            "Checkpoint缺少 avg_num_neighbors，改用当前评估数据统计值 %.4f，结果可能与训练阶段不一致。",
            avg_num_neighbors,
        )

    if ckpt_e0_values is not None:
        e0_values = np.array(ckpt_e0_values, dtype=float)
    else:
        e0_values = data_e0_values
        LOGGER.warning(
            "Checkpoint缺少 e0_values，已根据评估数据重新拟合，会与训练设置略有差异。"
        )

    if ckpt_cutoff is not None:
        cutoff = ckpt_cutoff
    else:
        cutoff = args.cutoff
        LOGGER.warning(
            "Checkpoint缺少 cutoff，使用命令行提供的 %.2f", cutoff
        )

    if ckpt_num_interactions is not None:
        num_interactions = ckpt_num_interactions
    else:
        num_interactions = args.num_interactions
        LOGGER.warning(
            "Checkpoint缺少 num_interactions，使用命令行提供的 %d", num_interactions
        )

    model = instantiate_model(
        z_table,
        avg_num_neighbors,
        cutoff,
        e0_values,
        num_interactions,
    )

    device = torch.device(args.device)
    model.to(device)

    model.load_state_dict(state_dict)
    LOGGER.info("Loaded checkpoint from %s", args.checkpoint)

    # 评估力时需要保留计算图，保持 eval 模式但启用梯度
    context = contextlib.nullcontext()
    with context:
        with torch.enable_grad():
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
    main()
