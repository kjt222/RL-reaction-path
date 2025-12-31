"""GemNet-OC adapter wrapper (FairChem backend)."""

from __future__ import annotations

from adapters.fairchem import FairchemAdapterBase
from core.registry import register_adapter


class GemNetOCAdapter(FairchemAdapterBase):
    backend_name = "gemnet_oc"
    default_node_embed_layer = "out_mlp_E"


def register() -> None:
    register_adapter(GemNetOCAdapter.backend_name, GemNetOCAdapter)
