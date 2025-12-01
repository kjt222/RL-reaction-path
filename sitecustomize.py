"""
Global stub to disable NVML/cuequivariance imports in environments that don't support them (e.g., WSL CPU runs).
This file is auto-imported when its directory is on PYTHONPATH.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# Relax unsafe load env and disable NVML probing
os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
os.environ.setdefault("CUEQUIVARIANCE_DISABLE_NVML", "1")


class _NVMLStub:
    class _FakeMemoryInfo:
        def __init__(self):
            self.total = 24 * 1024**3  # pretend 24GB
            self.free = 20 * 1024**3
            self.used = self.total - self.free

    def nvmlInit(self):
        return None

    def nvmlShutdown(self):
        return None

    def nvmlDeviceGetCount(self):
        return 0

    def nvmlDeviceGetHandleByIndex(self, idx):
        return idx

    def nvmlDeviceGetCudaComputeCapability(self, handle):
        return (0, 0)

    def nvmlDeviceGetPowerManagementLimit(self, handle):
        return 0

    def nvmlDeviceGetName(self, handle):
        return b"Mock GPU"

    def nvmlDeviceGetMemoryInfo(self, handle):
        return self._FakeMemoryInfo()

    def __getattr__(self, name):  # Fallback for any other NVML calls
        return lambda *args, **kwargs: 0


sys.modules.setdefault("pynvml", _NVMLStub())

# Stub cuequivariance-related modules to avoid loading native extensions when unavailable
_STUB = SimpleNamespace()
for _name in [
    "cuequivariance_torch",
    "cuequivariance_ops_torch",
    "cuequivariance_ops",
    "cuequivariance_ops.triton",
    "cuequivariance_ops.triton.cache_manager",
    "cuequivariance_ops.triton.tuning_decorator",
    "cuequivariance_ops.triton.autotune_aot",
]:
    sys.modules.setdefault(_name, _STUB)

