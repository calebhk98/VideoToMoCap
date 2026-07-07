"""HMR (human mesh recovery) backends.

Each backend wraps one upstream research tool behind a common interface so the
orchestration layer never has to know which one is in use.  Pick with
``PipelineConfig.backend`` -- swapping is one config line.

Availability of the underlying code was verified 2026-07 (see README):

Body-only (fast bulk; hands zero-padded):
    gvhmr -> https://github.com/zju3dv/GVHMR      (SIGGRAPH Asia 2024)  [default]
    wham  -> https://github.com/yohanshin/WHAM     (CVPR 2024)
    tram  -> https://github.com/yufu-wang/tram     (ECCV 2024)

Whole-body SMPL-X (recovers articulated hands):
    smplestx   -> https://github.com/SMPLCap/SMPLest-X            (TPAMI 2025)
    whac       -> https://github.com/SMPLCap/WHAC                 (ECCV 2024, moving cam)
    osx        -> https://github.com/IDEA-Research/OSX            (CVPR 2023, MIT)
    hand4whole -> https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE (CVPR 2026, MIT)
    multihmr   -> https://github.com/naver/multi-hmr              (ECCV 2024)

Combined:
    fusion -> body backend + hand net (WiLoR/HaMeR) grafted into SMPL-X hands

Testing:
    noop   -> synthetic backend (no GPU, no weights)
"""

from __future__ import annotations

from .base import HMRBackend, BackendError
from .gvhmr import GVHMRBackend
from .wham import WHAMBackend
from .tram import TRAMBackend
from .noop import NoopBackend
from .smplx_frames import (
    SMPLestXBackend,
    WHACBackend,
    OSXBackend,
    Hand4WholePlusBackend,
    MultiHMRBackend,
)
from .fusion import FusionBackend

_REGISTRY = {
    # body-only
    "gvhmr": GVHMRBackend,
    "wham": WHAMBackend,
    "tram": TRAMBackend,
    # whole-body SMPL-X (with hands)
    "smplestx": SMPLestXBackend,
    "smplerx": SMPLestXBackend,   # alias
    "whac": WHACBackend,
    "osx": OSXBackend,
    "hand4whole": Hand4WholePlusBackend,
    "multihmr": MultiHMRBackend,
    # combined
    "fusion": FusionBackend,
    # testing
    "noop": NoopBackend,
}


def get_backend(cfg) -> HMRBackend:
    name = cfg.backend.lower()
    if name not in _REGISTRY:
        raise BackendError(f"Unknown backend {cfg.backend!r}. Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[name](cfg)


def available_backends():
    return sorted(_REGISTRY)


__all__ = [
    "HMRBackend",
    "BackendError",
    "GVHMRBackend",
    "WHAMBackend",
    "TRAMBackend",
    "NoopBackend",
    "SMPLestXBackend",
    "WHACBackend",
    "OSXBackend",
    "Hand4WholePlusBackend",
    "MultiHMRBackend",
    "FusionBackend",
    "get_backend",
    "available_backends",
]
