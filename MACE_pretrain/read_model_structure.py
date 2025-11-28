"""Inspect a MACE checkpoint/.pt file and print structural metadata (no writes)."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Mapping

import torch
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _to_list(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() <= 32:
            return x.tolist()
        return f"tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    if isinstance(x, (list, tuple)):
        return list(x)
    return x


def _log_fields(container: Mapping[str, Any], name: str, keys: list[str]) -> None:
    LOGGER.info("--- %s ---", name)
    for k in keys:
        if k in container:
            LOGGER.info("%-24s %s", k + ":", _to_list(container[k]))


def _inspect_module(model: nn.Module) -> None:
    LOGGER.info("Detected nn.Module / TorchScript object: %s", type(model).__name__)
    attrs = [
        "r_max",
        "cutoff",
        "num_interactions",
        "avg_num_neighbors",
        "max_ell",
        "num_channels",
        "num_radial_basis",
        "num_bessel_basis",
        "num_polynomial_cutoff_basis",
        "correlation",
        "interaction_cls",
    ]
    LOGGER.info("--- Module attributes ---")
    for a in attrs:
        if hasattr(model, a):
            LOGGER.info("%-24s %s", a + ":", getattr(model, a))
    if hasattr(model, "atomic_numbers"):
        z = getattr(model, "atomic_numbers")
        try:
            z_list = z.tolist()
        except Exception:
            z_list = z
        LOGGER.info("atomic_numbers (len=%d): %s", len(z_list), z_list)
    if hasattr(model, "atomic_energies_fn") and hasattr(model.atomic_energies_fn, "atomic_energies"):
        e0s = model.atomic_energies_fn.atomic_energies
        LOGGER.info("atomic_energies_fn.atomic_energies: %s", _to_list(e0s))
    # Dump kwargs/config if present
    for name in ("model_kwargs", "kwargs", "config"):
        if hasattr(model, name):
            val = getattr(model, name)
            if isinstance(val, Mapping):
                LOGGER.info("--- %s ---", name)
                for k, v in val.items():
                    LOGGER.info("%-24s %s", k + ":", _to_list(v))
    # Fallback: dump visible __dict__ keys for manual inspection
    visible = {k: v for k, v in vars(model).items() if not k.startswith("_")}
    if visible:
        LOGGER.info("--- __dict__ (visible keys) ---")
        for k, v in visible.items():
            LOGGER.info("%-24s %s", k + ":", _to_list(v))
    # Inspect interactions.* attributes if present
    if hasattr(model, "interactions"):
        try:
            inter = getattr(model, "interactions")
            if isinstance(inter, (list, tuple)):
                inter = list(inter)
            if isinstance(inter, nn.ModuleList):
                inter = list(inter)
            for idx, blk in enumerate(inter):
                if blk is None:
                    continue
                attrs = {}
                for name in ("avg_num_neighbors", "r_max", "num_bessel", "num_polynomial_cutoff", "max_ell"):
                    if hasattr(blk, name):
                        attrs[name] = getattr(blk, name)
                if attrs:
                    LOGGER.info("interactions.%d attributes: %s", idx, {k: _to_list(v) for k, v in attrs.items()})
        except Exception:
            pass


def _inspect_dict(obj: Mapping[str, Any]) -> None:
    LOGGER.info("Detected mapping/dict with keys: %s", list(obj.keys()))

    # Common top-level fields
    common_keys = [
        "r_max",
        "cutoff",
        "num_interactions",
        "avg_num_neighbors",
        "z_table",
        "atomic_numbers",
        "e0s",
        "hidden_irreps",
        "correlation",
        "interaction_cls",
        "num_channels",
    ]
    _log_fields(obj, "Top-level fields", common_keys)

    # model_kwargs often contains structural hyperparams
    mk = obj.get("model_kwargs") or obj.get("config") or {}
    if isinstance(mk, Mapping):
        _log_fields(
            mk,
            "model_kwargs/config",
            [
                "r_max",
                "cutoff",
                "num_interactions",
                "avg_num_neighbors",
                "hidden_irreps",
                "max_ell",
                "num_channels",
                "num_radial_basis",
                "num_bessel_basis",
                "num_polynomial_cutoff_basis",
                "correlation",
                "interaction_cls",
                "atomic_numbers",
                "z_table",
                "e0s",
            ],
        )

    # state_dict shapes can hint at architecture depth
    sd = obj.get("state_dict")
    if sd is None and any(k.startswith("interactions.") for k in obj):
        sd = obj
    if isinstance(sd, Mapping):
        interaction_layers = {
            int(k.split(".")[1])
            for k in sd.keys()
            if k.startswith("interactions.") and k.split(".")[1].isdigit()
        }
        if interaction_layers:
            LOGGER.info("Detected interaction layers: %s (count=%d)", sorted(interaction_layers), len(interaction_layers))
        sample_keys = [k for k in sd.keys() if "interactions" in k][:5]
        if sample_keys:
            LOGGER.info("Sample state_dict keys: %s", sample_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a MACE .pt/.model file and print structural metadata.")
    parser.add_argument("path", type=str, help="Path to the .pt/.model checkpoint")
    args = parser.parse_args()

    LOGGER.info("Loading %s", args.path)
    obj = torch.load(args.path, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        _inspect_module(obj)
    elif isinstance(obj, Mapping):
        _inspect_dict(obj)
    else:
        LOGGER.info("Unknown object type: %s", type(obj))
        LOGGER.info("Object repr: %s", repr(obj)[:500])


if __name__ == "__main__":
    main()
