"""
Stub pynvml so cuequivariance imports don't hit unsupported NVML calls (e.g., on WSL).
Python auto-imports this module when its directory is on PYTHONPATH.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace


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


if "pynvml" not in sys.modules:
    sys.modules["pynvml"] = _NVMLStub()
