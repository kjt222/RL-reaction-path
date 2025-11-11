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
) -> modules.MACE:
    """Instantiate a MACE model with the standard training hyperparameters."""

    model = modules.MACE(
        r_max=cutoff,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps("128x0e + 128x1o"),
        MLP_irreps=o3.Irreps("64x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=3,
        radial_type="bessel",
    )
    return model
