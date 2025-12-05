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
    """
    构建 ScaleShiftMACE，优先使用 model.json 中的显式字段；若缺失则尝试从
    interactions/products 字段中反推（适配从 read_model.py 导出的详细 JSON）。
    """
    log = logging.getLogger(__name__)

    def _first(seq: Sequence[Any], key: str) -> Any | None:
        if not seq:
            return None
        v = seq[0].get(key)
        return v

    interactions = meta.get("interactions") or []
    products = meta.get("products") or []

    # 必填字段（不再提供默认值）
    hidden_irreps = meta.get("hidden_irreps") or _first(interactions, "hidden_irreps")
    mlp_irreps = meta.get("MLP_irreps") or _first(meta.get("readouts") or [], "hidden_irreps")
    max_ell = meta.get("max_ell")
    correlation = meta.get("correlation")
    num_radial_basis = meta.get("num_radial_basis")
    num_polynomial_cutoff = meta.get("num_polynomial_cutoff")
    avg_num_neighbors = meta.get("avg_num_neighbors")
    scale_shift = meta.get("scale_shift") or {}
    gate = meta.get("gate")
    radial_type = meta.get("radial_type")

    required_top = [
        ("hidden_irreps", hidden_irreps),
        ("MLP_irreps", mlp_irreps),
        ("max_ell", max_ell),
        ("correlation", correlation),
        ("num_radial_basis", num_radial_basis),
        ("num_polynomial_cutoff", num_polynomial_cutoff),
        ("avg_num_neighbors", avg_num_neighbors),
        ("z_table", meta.get("z_table")),
        ("e0_values", meta.get("e0_values")),
        ("cutoff", meta.get("cutoff")),
        ("num_interactions", meta.get("num_interactions")),
        ("gate", gate),
        ("radial_type", radial_type),
        ("scale_shift.scale", scale_shift.get("scale") if isinstance(scale_shift, Mapping) else None),
        ("scale_shift.shift", scale_shift.get("shift") if isinstance(scale_shift, Mapping) else None),
    ]
    missing = [k for k, v in required_top if v is None]
    if missing:
        raise ValueError(f"model.json missing required fields: {missing}")

    # 写回标准架构字段，供 instantiate_model 使用
    arch = dict(meta)
    arch["hidden_irreps"] = hidden_irreps
    arch["MLP_irreps"] = mlp_irreps
    arch["max_ell"] = max_ell
    arch["correlation"] = correlation
    arch["num_radial_basis"] = num_radial_basis
    arch["num_polynomial_cutoff"] = num_polynomial_cutoff
    arch["gate"] = gate
    arch["radial_type"] = radial_type
    if "scale_shift" in meta:
        arch["atomic_inter_scale"] = float(scale_shift.get("scale", 1.0))
        arch["atomic_inter_shift"] = float(scale_shift.get("shift", 0.0))

    z_table = tools.AtomicNumberTable(sorted({int(z) for z in meta["z_table"]}))
    cutoff = float(meta["cutoff"])
    e0_values = np.asarray(meta["e0_values"], dtype=float)
    num_interactions = int(meta["num_interactions"])
    return instantiate_model(
        z_table,
        float(avg_num_neighbors),
        cutoff,
        e0_values,
        num_interactions,
        architecture=arch,
    )


@register_model("MACE")
def build_mace(meta: Mapping[str, Any]) -> torch.nn.Module:
    """
    构建标准 MACE（无 scale/shift）。要求字段与 ScaleShiftMACE 类似，但不需要 scale_shift。
    """
    log = logging.getLogger(__name__)

    def _first(seq: Sequence[Any], key: str) -> Any | None:
        if not seq:
            return None
        return seq[0].get(key)

    interactions = meta.get("interactions") or []
    hidden_irreps = meta.get("hidden_irreps") or _first(interactions, "hidden_irreps")
    mlp_irreps = meta.get("MLP_irreps") or _first(meta.get("readouts") or [], "hidden_irreps")
    max_ell = meta.get("max_ell")
    correlation = meta.get("correlation")
    num_radial_basis = meta.get("num_radial_basis")
    num_polynomial_cutoff = meta.get("num_polynomial_cutoff")
    avg_num_neighbors = meta.get("avg_num_neighbors")
    gate = meta.get("gate")
    radial_type = meta.get("radial_type")

    required_top = [
        ("hidden_irreps", hidden_irreps),
        ("MLP_irreps", mlp_irreps),
        ("max_ell", max_ell),
        ("correlation", correlation),
        ("num_radial_basis", num_radial_basis),
        ("num_polynomial_cutoff", num_polynomial_cutoff),
        ("avg_num_neighbors", avg_num_neighbors),
        ("z_table", meta.get("z_table")),
        ("e0_values", meta.get("e0_values")),
        ("cutoff", meta.get("cutoff")),
        ("num_interactions", meta.get("num_interactions")),
        ("gate", gate),
        ("radial_type", radial_type),
    ]
    missing = [k for k, v in required_top if v is None]
    if missing:
        raise ValueError(f"model.json missing required fields: {missing}")

    arch = dict(meta)
    arch["hidden_irreps"] = hidden_irreps
    arch["MLP_irreps"] = mlp_irreps
    arch["max_ell"] = max_ell
    arch["correlation"] = correlation
    arch["num_radial_basis"] = num_radial_basis
    arch["num_polynomial_cutoff"] = num_polynomial_cutoff
    arch["gate"] = gate
    arch["radial_type"] = radial_type

    z_table = tools.AtomicNumberTable(sorted({int(z) for z in meta["z_table"]}))
    cutoff = float(meta["cutoff"])
    e0_values = np.asarray(meta["e0_values"], dtype=float)
    num_interactions = int(meta["num_interactions"])

    model = instantiate_model(
        z_table,
        float(avg_num_neighbors),
        cutoff,
        e0_values,
        num_interactions,
        architecture=arch,
    )
    return model


def build_model_from_json(meta: Mapping[str, Any]) -> torch.nn.Module:
    """Generic entry: dispatch to registered builder by model_type."""
    model_type = meta.get("model_type")
    if not model_type:
        raise ValueError("model.json missing 'model_type'")
    builder = MODEL_BUILDERS.get(model_type)
    if builder is None:
        raise ValueError(f"Unsupported model_type '{model_type}'. Known: {list(MODEL_BUILDERS)}")
    model = builder(meta)

    # 补充元数据到模型上，便于导出/对比（训练构建的模型缺少这些属性时会导致 JSON 校验失败）
    # 顶层字段
    for key in ["hidden_irreps", "max_ell", "correlation", "avg_num_neighbors"]:
        if key in meta:
            setattr(model, key, meta[key])

    # interactions/readouts 的 irreps 信息如果缺失，按 JSON 写回
    try:
        interactions = meta.get("interactions") or []
        if hasattr(model, "interactions"):
            for blk, info in zip(model.interactions, interactions):
                for attr in ["node_feats_irreps", "edge_attrs_irreps", "edge_feats_irreps", "target_irreps", "hidden_irreps"]:
                    if isinstance(info, Mapping) and attr in info:
                        setattr(blk, attr, info[attr])
                if isinstance(info, Mapping) and "avg_num_neighbors" in info:
                    setattr(blk, "avg_num_neighbors", info["avg_num_neighbors"])
    except Exception:
        pass

    try:
        readouts = meta.get("readouts") or []
        if hasattr(model, "readouts"):
            for rd, info in zip(model.readouts, readouts):
                for attr in ["irreps_in", "irreps_out", "hidden_irreps"]:
                    if isinstance(info, Mapping) and attr in info:
                        setattr(rd, attr, info[attr])
    except Exception:
        pass

    return model


__all__ = [
    "instantiate_model",
    "default_architecture",
    "build_model_from_json",
    "MODEL_BUILDERS",
    "register_model",
]
