"""Metadata helpers: override E0s and recompute E0s from a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from core.data.lmdb_reader import LmdbReader


def override_e0_from_json(
    state_dict: dict,
    json_meta: Mapping,
    module: "torch.nn.Module | None" = None,
    *,
    checkpoint_obj: dict | None = None,
    save_path: Path | None = None,
) -> dict:
    """
    Override atomic energies in a state_dict using e0_values from model.json，
    默认会同步到 nn.Module（若提供）并可选直接写回 .pt。

    - Requires json_meta 包含 e0_values。
    - 长度不匹配会抛出 ValueError。
    - 返回新的 state_dict 拷贝，不修改传入的 state_dict；如提供 checkpoint_obj 和 save_path，会把更新后的权重写入文件。
    """
    import torch

    if "e0_values" not in json_meta:
        raise ValueError("model.json 缺少 e0_values，无法覆盖 E0。")
    e0_values = json_meta["e0_values"]
    new_sd = dict(state_dict)
    applied = False
    for key in ("atomic_energies_fn.atomic_energies", "atomic_energies"):
        if key in new_sd:
            old = torch.as_tensor(new_sd[key])
            e0_tensor = torch.as_tensor(e0_values, dtype=old.dtype)
            if e0_tensor.numel() != old.numel():
                raise ValueError(
                    f"e0_values 长度与 state_dict[{key}] 不一致：json={e0_tensor.numel()} state={old.numel()}"
                )
            new_sd[key] = e0_tensor
            applied = True
    if not applied:
        raise ValueError("state_dict 中未找到原子能键，无法覆盖 E0。")

    if module is not None:
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module when provided.")
        # 允许非严格加载，避免无关键导致失败；这里只关心 E0 覆盖。
        module.load_state_dict(new_sd, strict=False)

    if checkpoint_obj is not None:
        if not isinstance(checkpoint_obj, dict):
            raise TypeError("checkpoint_obj must be a dict when provided.")
        checkpoint_obj = dict(checkpoint_obj)
        checkpoint_obj["model_state_dict"] = new_sd
        # 兼容可能存在的 state_dict 键
        if "state_dict" in checkpoint_obj and not isinstance(checkpoint_obj["state_dict"], torch.nn.Module):
            checkpoint_obj["state_dict"] = new_sd
        if module is not None:
            checkpoint_obj["model"] = module
        if save_path is not None:
            save_path = Path(save_path).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_obj, save_path)
    elif save_path is not None:
        # 没有 checkpoint_obj，则保存 state_dict 或 module
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if module is not None:
            torch.save(module, save_path)
        else:
            torch.save(new_sd, save_path)

    return new_sd


def _solve_e0(counts: list[np.ndarray], energies: list[float], n: int) -> np.ndarray:
    """最小二乘解 E0，带轻微 ridge。"""
    AtA = np.zeros((n, n), dtype=np.float64)
    Atb = np.zeros(n, dtype=np.float64)
    for c, y in zip(counts, energies):
        AtA += np.outer(c, c)
        Atb += c * y
    ridge = 1e-8
    AtA += ridge * np.eye(n, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(AtA, Atb, rcond=None)
    return sol


def _get(sample: object, name: str, default=None):
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _extract_atomic_numbers(sample: object) -> np.ndarray | None:
    z = _get(sample, "atomic_numbers", None)
    if z is None:
        z = _get(sample, "z", None)
    if z is None:
        return None
    return np.asarray(z, dtype=int)


def _extract_energy(sample: object) -> float | None:
    y = _get(sample, "y", None)
    if y is None:
        y = _get(sample, "y_relaxed", None)
    if y is None:
        y = _get(sample, "energy", None)
    if y is None:
        return None
    arr = np.asarray(y).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def compute_e0s_from_lmdb(
    lmdb_dir: Path,
    z_table: list[int],
    max_samples: int = 500_000,
    log_every: int = 0,
) -> tuple[list[float], list[bool]]:
    lmdb_dir = lmdb_dir.expanduser().resolve()
    n = len(z_table)

    max_z = int(max(z_table))
    idx_lookup = np.full(max_z + 1, -1, dtype=np.int64)
    for i, z in enumerate(z_table):
        zi = int(z)
        if zi <= max_z:
            idx_lookup[zi] = i

    AtA = np.zeros((n, n), dtype=np.float64)
    Atb = np.zeros(n, dtype=np.float64)
    coverage = np.zeros(n, dtype=np.int64)

    seen = 0
    with LmdbReader(lmdb_dir) as reader:
        total = len(reader)
        limit = min(total, int(max_samples))
        for idx in range(limit):
            if seen >= max_samples:
                break
            obj = reader[idx]
            zs = _extract_atomic_numbers(obj)
            if zs is None:
                continue
            y_val = _extract_energy(obj)
            if y_val is None:
                continue
            zs = np.asarray(zs, dtype=np.int64).reshape(-1)
            if zs.size == 0:
                continue
            if zs.max() > max_z:
                zs = zs[zs <= max_z]
            if zs.size == 0:
                continue
            idxs = idx_lookup[zs]
            valid = idxs >= 0
            if not np.any(valid):
                continue
            idxs = idxs[valid]
            counts = np.bincount(idxs, minlength=n).astype(np.float64, copy=False)
            if counts.sum() == 0:
                continue
            AtA += np.outer(counts, counts)
            Atb += counts * np.float64(y_val)
            coverage += counts > 0
            seen += 1
            if log_every and seen % int(log_every) == 0:
                print(f"E0 fit progress: {seen}/{limit}", flush=True)

    if seen == 0:
        raise RuntimeError("未从 LMDB 采到任何样本，无法重算 E0。")

    ridge = 1e-8
    AtA += ridge * np.eye(n, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(AtA, Atb, rcond=None)
    e0_values = [float(sol[i]) for i in range(n)]
    covered = [bool(x) for x in coverage.tolist()]
    return e0_values, covered


def recompute_e0s_from_lmdb(
    lmdb_dir: Path,
    model_json: Path,
    output_json: Path | None = None,
    max_samples: int = 500_000,
) -> Path:
    """
    用 LMDB 数据重算 E0：
    - 至少尝试覆盖 z_table 中的所有元素；未覆盖的元素保留旧值。
    - 采样上限 max_samples。
    """
    lmdb_dir = lmdb_dir.expanduser().resolve()
    model_json = model_json.expanduser().resolve()
    if output_json is None:
        output_json = model_json

    with model_json.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    z_table = meta["z_table"]
    n = len(z_table)
    e0_values, covered = compute_e0s_from_lmdb(lmdb_dir, z_table, max_samples)
    new_e0: list[float] = []
    old_e0 = meta.get("e0_values", [0.0] * n)
    for i in range(n):
        if covered[i]:
            new_e0.append(float(e0_values[i]))
        else:
            new_e0.append(float(old_e0[i]))

    meta["e0_values"] = new_e0
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    return output_json


def main():
    parser = argparse.ArgumentParser(description="Recompute E0s from LMDB and write to model.json")
    parser.add_argument("--lmdb_dir", type=Path, required=True, help="LMDB 目录")
    parser.add_argument("--model_json", type=Path, required=True, help="输入 model.json（提供 z_table 等）")
    parser.add_argument("--output_json", type=Path, help="输出路径（默认覆盖原文件）")
    parser.add_argument("--max_samples", type=int, default=500_000, help="采样上限")
    args = parser.parse_args()
    out = recompute_e0s_from_lmdb(args.lmdb_dir, args.model_json, args.output_json, args.max_samples)
    print(f"Wrote updated E0 to {out}")


if __name__ == "__main__":
    main()
