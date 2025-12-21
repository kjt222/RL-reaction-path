"""Model builder registry and metadata helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

import torch

ModelBuilder = Callable[[Mapping[str, Any]], torch.nn.Module]
MODEL_BUILDERS: Dict[str, ModelBuilder] = {}


def register_model(model_type: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(fn: ModelBuilder) -> ModelBuilder:
        MODEL_BUILDERS[model_type] = fn
        return fn

    return decorator


def attach_model_metadata(model: torch.nn.Module, meta: Mapping[str, Any]) -> None:
    """Attach key architecture/statistics metadata to a model instance."""
    if model is None:
        return

    def _set_attr(name: str, value: Any) -> None:
        try:
            setattr(model, name, value)
        except Exception:
            pass

    def _register_buffer(name: str, value: Any, dtype: torch.dtype) -> None:
        if value is None:
            return
        try:
            tensor = torch.tensor(value, dtype=dtype)
            existing = dict(model.named_buffers())
            if name in existing:
                try:
                    getattr(model, name).data = tensor
                    return
                except Exception:
                    pass
            model.register_buffer(name, tensor)
        except Exception:
            _set_attr(name, value)

    for key in ["max_ell", "correlation", "num_interactions", "num_radial_basis", "num_polynomial_cutoff"]:
        if key in meta and meta[key] is not None:
            _register_buffer(f"{key}_meta", int(meta[key]), dtype=torch.int64)

    if "cutoff" in meta and meta["cutoff"] is not None:
        _register_buffer("cutoff_meta", float(meta["cutoff"]), dtype=torch.float32)
    if "avg_num_neighbors" in meta and meta["avg_num_neighbors"] is not None:
        _register_buffer("avg_num_neighbors_meta", float(meta["avg_num_neighbors"]), dtype=torch.float32)

    for key in ["hidden_irreps", "MLP_irreps", "radial_type", "gate"]:
        if key in meta and meta[key] is not None:
            _set_attr(f"{key}_str", str(meta[key]))

    try:
        _set_attr("arch_meta", dict(meta))
    except Exception:
        pass


def build_model_from_json(meta: Mapping[str, Any]) -> torch.nn.Module:
    """Dispatch to the registered builder by model_type."""
    model_type = meta.get("model_type")
    if not model_type:
        raise ValueError("model.json missing 'model_type'")
    builder = MODEL_BUILDERS.get(model_type)
    if builder is None:
        raise ValueError(f"Unsupported model_type '{model_type}'. Known: {list(MODEL_BUILDERS)}")
    return builder(meta)


# Ensure built-in builders are registered when importing registry directly.
from . import model_mace as _mace  # noqa: F401
from . import model_equiformerv2 as _equiformer  # noqa: F401

__all__ = [
    "ModelBuilder",
    "MODEL_BUILDERS",
    "register_model",
    "attach_model_metadata",
    "build_model_from_json",
]
