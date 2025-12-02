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
        "hidden_irreps": "128x0e + 128x1o",
        "MLP_irreps": "16x0e",
        "correlation": 3,
        "max_ell": 3,
        "num_radial_basis": 10,
        "num_polynomial_cutoff": 5,
        "radial_type": "bessel",
        "gate": "silu",
        "scaling": "rms_forces_scaling",
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

    gate = architecture.get("gate", "silu")
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
        radial_type=architecture.get("radial_type", "bessel"),
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

    hidden_irreps = meta.get("hidden_irreps") or _first(interactions, "hidden_irreps")
    mlp_irreps = meta.get("MLP_irreps") or _first(meta.get("readouts") or [], "hidden_irreps")
    if hidden_irreps is None:
        raise ValueError("model.json missing 'hidden_irreps'; please提供或使用 read_model.py 导出详细 JSON")
    if mlp_irreps is None:
        mlp_irreps = "16x0e"  # 合理默认

    # max_ell: 优先显式字段，否则从 edge_attrs_irreps 推断
    max_ell = meta.get("max_ell")
    if max_ell is None:
        edge_irreps = _first(interactions, "edge_attrs_irreps")
        if edge_irreps:
            max_ell = max(ir.l for ir in o3.Irreps(edge_irreps))
            log.warning("max_ell 缺失，已从 edge_attrs_irreps 推断为 %s", max_ell)
    if max_ell is None:
        max_ell = 3

    correlation = meta.get("correlation")
    if correlation is None:
        correlation = 3
        log.warning("correlation 缺失，已使用默认值 3")

    num_radial_basis = meta.get("num_radial_basis")
    num_polynomial_cutoff = meta.get("num_polynomial_cutoff")
    if num_radial_basis is None or num_polynomial_cutoff is None:
        raise ValueError("model.json missing 'num_radial_basis' or 'num_polynomial_cutoff'")

    avg_num_neighbors = meta.get("avg_num_neighbors")
    if avg_num_neighbors is None:
        avg_num_neighbors = _first(interactions, "avg_num_neighbors")
        if avg_num_neighbors is not None:
            log.warning("avg_num_neighbors 缺失，已从 interactions 推断为 %.6f", avg_num_neighbors)
    if avg_num_neighbors is None:
        raise ValueError("model.json missing 'avg_num_neighbors'")

    required_top = ["z_table", "e0_values", "cutoff", "num_interactions"]
    missing = [k for k in required_top if meta.get(k) is None]
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
    "MODEL_BUILDERS",
    "register_model",
]
