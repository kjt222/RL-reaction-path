"""Metadata helpers: override E0s and recompute E0s from a dataset."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping

import lmdb
import numpy as np
import torch

try:
    # 优先使用项目内的升级函数，兼容旧版 PyG 存储
    from dataloader.lmdb_loader import _upgrade_pyg_data, _list_lmdb_files  # type: ignore
except Exception:  # pragma: no cover
    _upgrade_pyg_data = None
    _list_lmdb_files = None


def override_e0_from_json(state_dict: dict, json_meta: Mapping) -> dict:
    """
    Override atomic energies in a state_dict using e0_values from model.json.
    - Requires json_meta 包含 e0_values。
    - 长度不匹配会抛出 ValueError。
    - 返回新的 state_dict 拷贝，不修改原对象。
    """
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
    if _upgrade_pyg_data is None or _list_lmdb_files is None:
        raise ImportError("dataloader.lmdb_loader 未可用，无法重算 E0。")

    lmdb_dir = lmdb_dir.expanduser().resolve()
    model_json = model_json.expanduser().resolve()
    if output_json is None:
        output_json = model_json

    with model_json.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    z_table = meta["z_table"]
    idx_map = {int(z): i for i, z in enumerate(z_table)}
    n = len(z_table)

    counts_list: list[np.ndarray] = []
    energies: list[float] = []
    coverage = np.zeros(n, dtype=np.int64)

    lmdb_files = _list_lmdb_files(lmdb_dir)
    seen = 0
    for shard in lmdb_files:
        if seen >= max_samples:
            break
        env = lmdb.open(
            str(shard),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=4,
            subdir=False,
        )
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, val in cursor:
                if seen >= max_samples:
                    break
                obj = pickle.loads(val)
                if _upgrade_pyg_data is not None:
                    try:
                        obj = _upgrade_pyg_data(obj)
                    except TypeError:
                        # 非 PyG/意外条目，跳过
                        continue
                zs = np.asarray(obj.atomic_numbers.cpu().numpy(), dtype=int)
                y = getattr(obj, "y", None)
                if y is None:
                    continue
                y_val = float(np.asarray(y).reshape(-1)[0])
                counts = np.zeros(n, dtype=np.float64)
                for z in zs:
                    idx = idx_map.get(int(z))
                    if idx is not None:
                        counts[idx] += 1.0
                if counts.sum() == 0:
                    continue
                counts_list.append(counts)
                energies.append(y_val)
                coverage += counts > 0
                seen += 1
        env.close()

    if not counts_list:
        raise RuntimeError("未从 LMDB 采到任何样本，无法重算 E0。")

    sol = _solve_e0(counts_list, energies, n)
    new_e0 = []
    old_e0 = meta.get("e0_values", [0.0] * n)
    for i in range(n):
        if coverage[i] > 0:
            new_e0.append(float(sol[i]))
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
