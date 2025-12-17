"""Model factory with registry-based builders.

- default_architecture(): minimal ScaleShiftMACE defaults for训练便捷。
- build_model_from_json(meta): 根据 model_type 分发到注册表构建模型。
- instantiate_model: 复用旧的统一超参构建（暂用于 ScaleShiftMACE）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence
import logging

import numpy as np
import torch
from e3nn import o3
from mace import modules, tools

ModelBuilder = Callable[[Mapping[str, Any]], torch.nn.Module]
MODEL_BUILDERS: Dict[str, ModelBuilder] = {}


def register_model(model_type: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(fn: ModelBuilder) -> ModelBuilder:
        MODEL_BUILDERS[model_type] = fn
        return fn

    return decorator


def default_architecture() -> dict:
    """Return a minimal default architecture for ScaleShiftMACE."""
    return {
        "model_type": "ScaleShiftMACE",
        # 注意：builder 现已取消默认值，这个函数仅用于占位或测试。
        "hidden_irreps": None,
        "MLP_irreps": None,
        "correlation": None,
        "max_ell": None,
        "num_radial_basis": None,
        "num_polynomial_cutoff": None,
        "radial_type": None,
        "gate": None,
        "scaling": None,
    }


def instantiate_model(
    z_table: tools.AtomicNumberTable,
    avg_num_neighbors: float,
    cutoff: float,
    atomic_energies: np.ndarray,
    num_interactions: int,
    architecture: dict | None = None,
):
    """
    Instantiate a model strictly from provided architecture metadata (统一超参版).
    architecture 必须至少包含：
      model_type, hidden_irreps, MLP_irreps, correlation, max_ell,
      num_radial_basis/num_bessel, num_polynomial_cutoff/num_cutoff_basis,
      radial_type, gate
    """
    if architecture is None:
        raise ValueError("architecture must be provided via metadata/json; no defaults are assumed.")

    model_type = architecture.get("model_type")
    if model_type is None:
        raise ValueError("architecture missing 'model_type'; e.g., 'MACE' or 'ScaleShiftMACE'.")

    gate = architecture.get("gate")
    if gate is None:
        raise ValueError("architecture missing 'gate'")
    if isinstance(gate, str):
        gate_lower = gate.lower()
        if gate_lower == "silu":
            gate_fn = torch.nn.functional.silu
        elif gate_lower == "relu":
            gate_fn = torch.nn.functional.relu
        else:
            raise ValueError(f"Unsupported gate string: {gate}")
    else:
        gate_fn = gate

    num_bessel = architecture.get("num_bessel", architecture.get("num_radial_basis"))
    num_poly = architecture.get("num_polynomial_cutoff", architecture.get("num_cutoff_basis"))
    if num_bessel is None or num_poly is None:
        raise ValueError("architecture must specify num_bessel/num_radial_basis and num_polynomial_cutoff/num_cutoff_basis.")

    common_kwargs = dict(
        r_max=cutoff,
        num_bessel=int(num_bessel),
        num_polynomial_cutoff=int(num_poly),
        max_ell=int(architecture["max_ell"]),
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(architecture["hidden_irreps"]),
        MLP_irreps=o3.Irreps(architecture["MLP_irreps"]),
        gate=gate_fn,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=int(architecture["correlation"]),
        radial_type=architecture.get("radial_type") or "bessel",
    )

    if model_type == "ScaleShiftMACE":
        from mace.modules.models import ScaleShiftMACE

        return ScaleShiftMACE(
            atomic_inter_scale=float(architecture.get("atomic_inter_scale", 1.0)),
            atomic_inter_shift=float(architecture.get("atomic_inter_shift", 0.0)),
            **common_kwargs,
        )
    elif model_type == "MACE":
        return modules.MACE(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model_type in architecture: {model_type}")


@register_model("ScaleShiftMACE")
def build_scale_shift_mace(meta: Mapping[str, Any]) -> torch.nn.Module:
    """构建 ScaleShiftMACE，要求 JSON 完整且不做任何补齐/写回。"""
    required = [
        "hidden_irreps",
        "MLP_irreps",
        "max_ell",
        "correlation",
        "num_radial_basis",
        "num_polynomial_cutoff",
        "avg_num_neighbors",
        "z_table",
        "e0_values",
        "cutoff",
        "num_interactions",
        "gate",
        "radial_type",
    ]
    missing = [k for k in required if k not in meta or meta[k] is None]
    if missing:
        raise ValueError(f"model.json missing required fields: {missing}")

    z_table = tools.AtomicNumberTable(sorted({int(z) for z in meta["z_table"]}))
    cutoff = float(meta["cutoff"])
    e0_values = np.asarray(meta["e0_values"], dtype=float)
    num_interactions = int(meta["num_interactions"])

    arch = dict(meta)
    if "scale_shift" in meta and isinstance(meta["scale_shift"], Mapping):
        arch["atomic_inter_scale"] = float(meta["scale_shift"].get("scale", 1.0))
        arch["atomic_inter_shift"] = float(meta["scale_shift"].get("shift", 0.0))

    model = instantiate_model(
        z_table,
        float(meta["avg_num_neighbors"]),
        cutoff,
        e0_values,
        num_interactions,
        architecture=arch,
    )
    attach_model_metadata(model, meta)
    return model


@register_model("MACE")
def build_mace(meta: Mapping[str, Any]) -> torch.nn.Module:
    """构建标准 MACE（无 scale/shift），要求 JSON 完整且不做补齐/写回。"""
    required = [
        "hidden_irreps",
        "MLP_irreps",
        "max_ell",
        "correlation",
        "num_radial_basis",
        "num_polynomial_cutoff",
        "avg_num_neighbors",
        "z_table",
        "e0_values",
        "cutoff",
        "num_interactions",
        "gate",
        "radial_type",
    ]
    missing = [k for k in required if k not in meta or meta[k] is None]
    if missing:
        raise ValueError(f"model.json missing required fields: {missing}")

    z_table = tools.AtomicNumberTable(sorted({int(z) for z in meta["z_table"]}))
    cutoff = float(meta["cutoff"])
    e0_values = np.asarray(meta["e0_values"], dtype=float)
    num_interactions = int(meta["num_interactions"])

    arch = dict(meta)
    model = instantiate_model(
        z_table,
        float(meta["avg_num_neighbors"]),
        cutoff,
        e0_values,
        num_interactions,
        architecture=arch,
    )
    attach_model_metadata(model, meta)
    return model


def attach_model_metadata(model: torch.nn.Module, meta: Mapping[str, Any]) -> None:
    """Attach关键架构/统计超参到模型，便于严格导出/对比。"""
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
    """Generic entry: dispatch to registered builder by model_type."""
    model_type = meta.get("model_type")
    if not model_type:
        raise ValueError("model.json missing 'model_type'")
    builder = MODEL_BUILDERS.get(model_type)
    if builder is None:
        raise ValueError(f"Unsupported model_type '{model_type}'. Known: {list(MODEL_BUILDERS)}")
    return builder(meta)


__all__ = [
    "instantiate_model",
    "default_architecture",
    "build_model_from_json",
    "attach_model_metadata",
    "MODEL_BUILDERS",
    "register_model",
]
