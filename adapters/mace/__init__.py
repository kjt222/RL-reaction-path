"""MACE adapter package."""

try:
    from .adapter import MaceAdapter, register
except Exception:
    MaceAdapter = None
    register = None

__all__ = ["MaceAdapter", "register"]
