"""EquiformerV2 builder stub (to be implemented)."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .registry import register_model


@register_model("EquiformerV2")
def build_equiformer_v2(_meta: Mapping[str, Any]) -> torch.nn.Module:
    raise NotImplementedError("EquiformerV2 backend is not implemented in this repo yet.")


__all__ = ["build_equiformer_v2"]
