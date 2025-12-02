"""Post-process a downloaded MACE model: inject E0s from model.json into the state_dict and save clean weights (no metadata)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.serialization
torch.set_default_dtype(torch.float32)

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _load_e0s_from_json(json_meta: Mapping[str, Any], override_path: Path | None) -> list[float]:
    """优先从 --e0_file 读取，否则从 model.json 中读取 e0_values。"""
    if override_path is not None:
        text = override_path.read_text().strip()
        try:
            data = json.loads(text)
            e0_values = [float(x) for x in data]
        except json.JSONDecodeError:
            parts = text.replace(",", " ").split()
            e0_values = [float(x) for x in parts]
        LOGGER.info("Loaded %d E0 values from %s", len(e0_values), override_path)
        return e0_values

    if "e0_values" not in json_meta:
        raise ValueError("model.json 缺少 e0_values，无法写入 E0。")
    e0 = json_meta["e0_values"]
    e0 = np.asarray(e0, dtype=float).tolist()
    LOGGER.info("Loaded %d E0 values from model.json.", len(e0))
    return e0


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
    parser = argparse.ArgumentParser(description="Inject E0s from model.json into a MACE .pt/.model state_dict (no metadata).")
    parser.add_argument("--input", type=Path, required=True, help="Path to downloaded .pt/.model file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write checkpoint.pt/best_model.pt")
    parser.add_argument("--model_json", type=Path, help="Path to model.json (default: alongside input)")
    parser.add_argument("--e0_file", type=Path, help="Optional path to JSON/txt with E0 list to override model.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.input.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Input model not found: {model_path}")

    # Load model.json（唯一的配置来源）
    json_path = args.model_json or (model_path.parent / "model.json")
    if not json_path.exists():
        raise FileNotFoundError(f"model.json not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    LOGGER.info("Loaded model.json from %s", json_path)

    LOGGER.info("Loading %s", model_path)
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict, _ = _coerce_state_dict(obj)

    # 用 model.json/e0_file 中的 E0 覆盖 state_dict
    e0_values = _load_e0s_from_json(json_meta, args.e0_file)
    key_e0 = "atomic_energies_fn.atomic_energies"
    if key_e0 in state_dict:
        state_dict[key_e0] = torch.tensor(e0_values, dtype=state_dict[key_e0].dtype)
        LOGGER.info("Overrode %s in state_dict using E0 from JSON/e0_file.", key_e0)
    else:
        LOGGER.warning("%s not found in state_dict; skipped E0 override.", key_e0)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model_state_dict": state_dict}, ckpt_path)
    LOGGER.info("Saved checkpoint (state_dict only) to %s", ckpt_path)
    best_path = out_dir / "best_model.pt"
    torch.save(state_dict, best_path)
    LOGGER.info("Also wrote best_model.pt to %s", best_path)


if __name__ == "__main__":
    main()
