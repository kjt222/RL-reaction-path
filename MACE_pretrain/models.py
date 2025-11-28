"""Model factory helpers for MACE pretraining."""

from __future__ import annotations

import numpy as np
import torch
from e3nn import o3
from mace import modules, tools


def instantiate_model(
    z_table: tools.AtomicNumberTable,
    avg_num_neighbors: float,
    cutoff: float,
    atomic_energies: np.ndarray,
    num_interactions: int,
    architecture: dict | None = None,
):
    """
    Instantiate a model strictly from provided architecture metadata.

    No default architecture is assumed; architecture must include at least:
    - model_type: e.g., "MACE" or "ScaleShiftMACE"
    - hidden_irreps, MLP_irreps, correlation, max_ell
    - num_radial_basis/num_bessel, num_polynomial_cutoff/num_cutoff_basis
    - radial_type, gate
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
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
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
