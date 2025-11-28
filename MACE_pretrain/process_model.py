"""Wrap a downloaded MACE model file with metadata into our checkpoint format."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.serialization
import shutil

from metadata import save_checkpoint
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _load_e0s(model_obj: Any, mapping: Mapping[str, Any], override_path: Path | None) -> list[float]:
    if override_path is not None:
        text = override_path.read_text().strip()
        try:
            data = json.loads(text)
            e0_values = [float(x) for x in data]
        except json.JSONDecodeError:
            # Fallback: whitespace/comma-separated numbers
            parts = text.replace(",", " ").split()
            e0_values = [float(x) for x in parts]
        LOGGER.info("Loaded %d E0 values from %s", len(e0_values), override_path)
        return e0_values

    if hasattr(model_obj, "atomic_energies_fn") and hasattr(model_obj.atomic_energies_fn, "atomic_energies"):
        e0 = model_obj.atomic_energies_fn.atomic_energies
        if isinstance(e0, torch.Tensor):
            e0 = e0.detach().cpu().numpy()
        e0 = np.asarray(e0, dtype=float).tolist()
        LOGGER.info("Extracted E0 values from model (len=%d).", len(e0))
        return e0

    if "atomic_energies_fn.atomic_energies" in mapping:
        e0 = mapping["atomic_energies_fn.atomic_energies"]
        if isinstance(e0, torch.Tensor):
            e0 = e0.detach().cpu().numpy()
        e0_values = np.asarray(e0, dtype=float).tolist()
        LOGGER.info("Extracted E0 values from state_dict (len=%d).", len(e0_values))
        return e0_values

    raise ValueError("E0 values not found. Provide --e0_file with a list of numbers.")


def _coerce_atomic_numbers(model_obj: Any, mapping: Mapping[str, Any], z_max: int) -> list[int]:
    if hasattr(model_obj, "atomic_numbers"):
        try:
            z_list = [int(z) for z in model_obj.atomic_numbers.tolist()]
            LOGGER.info("Using atomic_numbers from model (len=%d).", len(z_list))
            return z_list
        except Exception:
            pass
    if "atomic_numbers" in mapping:
        vals = mapping["atomic_numbers"]
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        z_list = [int(z) for z in np.asarray(vals).tolist()]
        LOGGER.info("Using atomic_numbers from mapping (len=%d).", len(z_list))
        return z_list
    LOGGER.info("atomic_numbers not stored; defaulting to 1..%d", z_max)
    return list(range(1, z_max + 1))


def _coerce_state_dict(obj: Any) -> tuple[Mapping[str, Any], Any]:
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict(), obj
    if isinstance(obj, Mapping):
        if "model_state_dict" in obj:
            return obj["model_state_dict"], obj
        return obj, obj
    raise TypeError(f"Unsupported object type: {type(obj)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap a downloaded MACE .pt/.model with metadata as checkpoint.pt")
    parser.add_argument("--input", type=Path, required=True, help="Path to downloaded .pt/.model file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write checkpoint.pt/best_model.pt")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Model cutoff/r_max (default 6.0)")
    parser.add_argument("--num_interactions", type=int, default=2, help="Number of interaction blocks (default 2)")
    parser.add_argument("--avg_num_neighbors", type=float, default=15.0, help="Average neighbors to store in metadata")
    parser.add_argument("--z_max", type=int, default=89, help="If atomic_numbers missing, use 1..z_max")
    parser.add_argument("--e0_file", type=Path, help="Optional path to JSON or txt with E0 list to override model values")
    parser.add_argument("--num_radial_basis", type=int, default=10, help="Radial basis count (stored as extra metadata)")
    parser.add_argument("--num_cutoff_basis", type=int, default=5, help="Cutoff polynomial basis (extra metadata)")
    parser.add_argument("--hidden_irreps", type=str, default="128x0e + 128x1o", help="Hidden irreps string (extra metadata)")
    parser.add_argument("--max_ell", type=int, default=3, help="Max ell (extra metadata)")
    parser.add_argument("--correlation", type=int, default=3, help="Correlation/order (extra metadata)")
    parser.add_argument("--mlp_irreps", type=str, default="16x0e", help="MLP irreps (extra metadata)")
    parser.add_argument("--metadata_json", type=Path, help="Optional metadata.json to read fields from (defaults to input dir/metadata.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.input.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Input model not found: {model_path}")

    # Load required metadata.json
    json_path = args.metadata_json or (model_path.parent / "metadata.json")
    if not json_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    LOGGER.info("Loaded metadata.json from %s", json_path)

    required = ("z_table", "avg_num_neighbors", "e0_values", "cutoff", "num_interactions")
    missing = [k for k in required if k not in json_meta]
    if missing:
        raise ValueError(f"metadata.json missing required fields: {missing}")

    LOGGER.info("Loading %s", model_path)
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict, _ = _coerce_state_dict(obj)

    # Build/attach architecture
    arch_keys = [
        "architecture",
        "model_type",
        "hidden_irreps",
        "MLP_irreps",
        "correlation",
        "max_ell",
        "num_radial_basis",
        "num_bessel",
        "num_polynomial_cutoff",
        "num_cutoff_basis",
        "radial_type",
        "gate",
        "num_channels",
        "max_L",
        "scaling",
        "atomic_inter_scale",
        "atomic_inter_shift",
    ]
    architecture = json_meta.get("architecture")
    if architecture is None:
        architecture = {k: json_meta[k] for k in arch_keys if k in json_meta and k != "architecture"}
    if not architecture or "model_type" not in architecture:
        raise ValueError("metadata.json must include architecture or sufficient architecture fields (e.g., model_type, hidden_irreps, etc.).")

    # Use metadata from JSON with normalized types
    metadata = dict(json_meta)
    metadata["architecture"] = architecture
    metadata["z_table"] = [int(z) for z in metadata["z_table"]]
    metadata["e0_values"] = [float(x) for x in metadata["e0_values"]]
    metadata["avg_num_neighbors"] = float(metadata["avg_num_neighbors"])
    metadata["cutoff"] = float(metadata["cutoff"])
    metadata["num_interactions"] = int(metadata["num_interactions"])

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata.json to output_dir (normalized with architecture attached)
    target_meta = out_dir / "metadata.json"
    with target_meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)
    LOGGER.info("Wrote metadata.json to %s", target_meta)

    ckpt_path = out_dir / "checkpoint.pt"
    save_checkpoint(
        ckpt_path,
        model_state_dict=state_dict,
        metadata=metadata,
        train_state=None,
        best_model_state_dict=state_dict,
    )
    LOGGER.info("Saved checkpoint with metadata to %s", ckpt_path)
    best_path = out_dir / "best_model.pt"
    torch.save(state_dict, best_path)
    LOGGER.info("Also wrote best_model.pt to %s", best_path)


if __name__ == "__main__":
    main()
