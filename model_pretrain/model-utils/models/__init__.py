"""Model registry package."""

from __future__ import annotations

from .registry import (
    ModelBuilder,
    MODEL_BUILDERS,
    register_model,
    attach_model_metadata,
    build_model_from_json,
)
from .mace import instantiate_model, default_architecture
from . import equiformer  # noqa: F401

__all__ = [
    "ModelBuilder",
    "MODEL_BUILDERS",
    "register_model",
    "attach_model_metadata",
    "build_model_from_json",
    "instantiate_model",
    "default_architecture",
]
