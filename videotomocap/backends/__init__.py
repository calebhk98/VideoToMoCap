"""HMR (human mesh recovery) backends.

Each backend wraps one upstream research tool behind a common interface so the
orchestration layer never has to know which one is in use.  Pick with
``PipelineConfig.backend``.

Availability of the underlying code was verified 2026-07 (see README):
    gvhmr -> https://github.com/zju3dv/GVHMR   (SIGGRAPH Asia 2024)  [default]
    wham  -> https://github.com/yohanshin/WHAM  (CVPR 2024)
    tram  -> https://github.com/yufu-wang/tram  (ECCV 2024)
    noop  -> synthetic backend for tests / dry runs (no GPU, no weights)
"""

from __future__ import annotations

from .base import HMRBackend, BackendError
from .gvhmr import GVHMRBackend
from .wham import WHAMBackend
from .tram import TRAMBackend
from .noop import NoopBackend

_REGISTRY = {
    "gvhmr": GVHMRBackend,
    "wham": WHAMBackend,
    "tram": TRAMBackend,
    "noop": NoopBackend,
}


def get_backend(cfg) -> HMRBackend:
    name = cfg.backend.lower()
    if name not in _REGISTRY:
        raise BackendError(f"Unknown backend {cfg.backend!r}. Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[name](cfg)


__all__ = [
    "HMRBackend",
    "BackendError",
    "GVHMRBackend",
    "WHAMBackend",
    "TRAMBackend",
    "NoopBackend",
    "get_backend",
]
