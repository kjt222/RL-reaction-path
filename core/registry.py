"""Adapter registry."""

from __future__ import annotations

from typing import Dict, Type

_ADAPTERS: Dict[str, type] = {}


def register_adapter(name: str, adapter_cls: type) -> None:
    if not name:
        raise ValueError("Adapter name must be non-empty")
    if name in _ADAPTERS:
        raise ValueError(f"Adapter already registered: {name}")
    _ADAPTERS[name] = adapter_cls


def get_adapter(name: str) -> type:
    if name not in _ADAPTERS:
        raise KeyError(f"Unknown adapter: {name}")
    return _ADAPTERS[name]


def list_adapters() -> Dict[str, type]:
    return dict(_ADAPTERS)
