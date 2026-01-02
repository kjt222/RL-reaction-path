"""Prepare a new OC22 head for a multi-head MACE model.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .metadata import compute_e0s_from_lmdb


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _load_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_float_list(values: list[Any]) -> list[float]:
    return [float(v) for v in values]


def _update_scale_shift(meta: dict[str, Any], base_idx: int) -> None:
    scale_shift = meta.get("scale_shift")
    if not isinstance(scale_shift, dict):
        return

    scale = scale_shift.get("scale")
    shift = scale_shift.get("shift")

    if isinstance(scale, (list, tuple)):
        scale_list = _as_list(scale)
        base_scale = scale_list[base_idx] if base_idx < len(scale_list) else scale_list[-1]
        scale_list.append(float(base_scale))
        scale_shift["scale"] = scale_list
    elif scale is not None:
        scale_shift["scale"] = [float(scale), float(scale)]

    if isinstance(shift, (list, tuple)):
        shift_list = _as_list(shift)
        base_shift = shift_list[base_idx] if base_idx < len(shift_list) else shift_list[-1]
        shift_list.append(float(base_shift))
        scale_shift["shift"] = shift_list
    elif shift is not None:
        scale_shift["shift"] = [float(shift), float(shift)]


def prepare_oc22_head(
    base_model_json: Path,
    lmdb_dir: Path,
    output_json: Path,
    head_name: str = "oc22",
    fallback_head: str = "omat_pbe",
    max_samples: int = 500_000,
    log_every: int = 0,
) -> Path:
    meta = _load_meta(base_model_json)
    heads = list(meta.get("heads") or [])
    if head_name in heads:
        raise ValueError(f"Head {head_name} already present in model.json")
    if not heads:
        heads = [fallback_head]

    e0_values = meta.get("e0_values")
    if not isinstance(e0_values, list) or not e0_values:
        raise ValueError("model.json missing e0_values")

    multi_head = isinstance(e0_values[0], (list, tuple))
    if multi_head:
        if fallback_head not in heads:
            raise ValueError(f"fallback_head {fallback_head} not found in heads: {heads}")
        base_idx = heads.index(fallback_head)
        base_e0 = _ensure_float_list(e0_values[base_idx])
    else:
        base_idx = 0
        base_e0 = _ensure_float_list(e0_values)

    z_table = meta.get("z_table")
    if not isinstance(z_table, list) or not z_table:
        raise ValueError("model.json missing z_table")
    if len(base_e0) != len(z_table):
        raise ValueError("e0_values length does not match z_table length")

    e0_fit, covered = compute_e0s_from_lmdb(
        lmdb_dir,
        z_table,
        max_samples=max_samples,
        log_every=log_every,
    )
    oc22_e0 = [float(e0_fit[i]) if covered[i] else float(base_e0[i]) for i in range(len(z_table))]

    if multi_head:
        new_e0_values = [list(_ensure_float_list(v)) for v in e0_values]
        new_e0_values.append(oc22_e0)
    else:
        new_e0_values = [base_e0, oc22_e0]

    meta["heads"] = heads + [head_name]
    meta["e0_values"] = new_e0_values
    _update_scale_shift(meta, base_idx)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True, indent=2)
    return output_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OC22 head for multi-head MACE model.json")
    parser.add_argument("--base_model_json", type=Path, required=True, help="Base model.json with existing heads")
    parser.add_argument("--lmdb_dir", type=Path, required=True, help="OC22 LMDB dir (train shard dir)")
    parser.add_argument("--output_json", type=Path, required=True, help="Output model.json path")
    parser.add_argument("--head_name", type=str, default="oc22", help="New head name")
    parser.add_argument("--fallback_head", type=str, default="omat_pbe", help="Head to copy E0/scale for missing elems")
    parser.add_argument("--max_samples", type=int, default=500_000, help="Max LMDB samples to fit E0s")
    parser.add_argument("--log_every", type=int, default=0, help="Print progress every N samples")
    args = parser.parse_args()

    out = prepare_oc22_head(
        base_model_json=args.base_model_json,
        lmdb_dir=args.lmdb_dir,
        output_json=args.output_json,
        head_name=args.head_name,
        fallback_head=args.fallback_head,
        max_samples=args.max_samples,
        log_every=args.log_every,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
