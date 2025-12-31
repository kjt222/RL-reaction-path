"""EquiformerV2 adapter wrapper."""

from __future__ import annotations

from adapters.fairchem import FairchemAdapterBase
from core.registry import register_adapter


class EquiformerV2Adapter(FairchemAdapterBase):
    backend_name = "equiformer_v2"
    default_node_embed_layer = "norm_output"


def register() -> None:
    register_adapter(EquiformerV2Adapter.backend_name, EquiformerV2Adapter)
