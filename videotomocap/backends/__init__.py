"""HMR (human mesh recovery) backends.

Each backend wraps one upstream research tool behind a common interface so the
orchestration layer never has to know which one is in use.  Pick with
``PipelineConfig.backend`` -- swapping is one config line.

Availability of the underlying code was verified 2026-07 (see README):

Body-only (fast bulk; hands zero-padded):
    gvhmr -> https://github.com/zju3dv/GVHMR      (SIGGRAPH Asia 2024)  [default]
    wham  -> https://github.com/yohanshin/WHAM     (CVPR 2024)
    tram  -> https://github.com/yufu-wang/tram     (ECCV 2024)
    trace -> https://github.com/Arthur151/ROMP     (CVPR 2023, world-grounded, Apache-2.0)
    hmr2  -> https://github.com/shubham-goel/4D-Humans (ICCV 2023, camera-relative)

Whole-body SMPL-X (recovers articulated hands):
    smplestx   -> https://github.com/SMPLCap/SMPLest-X            (TPAMI 2025)
    camenduru_smplerx -> https://github.com/camenduru/SMPLer-X   (SMPLer-X, runnable repackaging)
    hybrik     -> https://github.com/jeffffffli/HybrIK           (HybrIK-X, TPAMI 2025, MIT)
    sam3dbody  -> https://github.com/facebookresearch/sam-3d-body (Meta 2026, MHR->SMPL-X fit)
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
from .trace import TRACEBackend
from .fourdhumans import FourDHumansBackend
from .hybrik import HybrIKXBackend
from .noop import NoopBackend
from .smplx_frames import (
    SMPLestXBackend,
    CamenduruSMPLerXBackend,
    WHACBackend,
    OSXBackend,
    Hand4WholePlusBackend,
    MultiHMRBackend,
)
from .sam3dbody import Sam3dBodyBackend
from .fusion import FusionBackend

_REGISTRY = {
    # body-only
    "gvhmr": GVHMRBackend,
    "wham": WHAMBackend,
    "tram": TRAMBackend,
    "trace": TRACEBackend,
    "hmr2": FourDHumansBackend,
    "4dhumans": FourDHumansBackend,   # alias
    "fourdhumans": FourDHumansBackend,   # alias
    # whole-body SMPL-X (with hands)
    "smplestx": SMPLestXBackend,
    "smplerx": SMPLestXBackend,   # alias
    "camenduru_smplerx": CamenduruSMPLerXBackend,
    "camenduru": CamenduruSMPLerXBackend,   # alias
    "hybrik": HybrIKXBackend,
    "hybrikx": HybrIKXBackend,   # alias
    "whac": WHACBackend,
    "osx": OSXBackend,
    "hand4whole": Hand4WholePlusBackend,
    "multihmr": MultiHMRBackend,
    "sam3dbody": Sam3dBodyBackend,
    "sam3d": Sam3dBodyBackend,   # alias
    # combined
    "fusion": FusionBackend,
    # testing
    "noop": NoopBackend,
}


def get_backend(cfg) -> HMRBackend:
    """Instantiate the backend named by ``cfg.backend`` (case-insensitive)."""
    name = cfg.backend.lower()
    if name not in _REGISTRY:
        raise BackendError(f"Unknown backend {cfg.backend!r}. Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[name](cfg)


def available_backends():
    """List registered backend names, sorted."""
    return sorted(_REGISTRY)


__all__ = [
    "HMRBackend",
    "BackendError",
    "GVHMRBackend",
    "WHAMBackend",
    "TRAMBackend",
    "TRACEBackend",
    "FourDHumansBackend",
    "HybrIKXBackend",
    "NoopBackend",
    "SMPLestXBackend",
    "CamenduruSMPLerXBackend",
    "WHACBackend",
    "OSXBackend",
    "Hand4WholePlusBackend",
    "MultiHMRBackend",
    "Sam3dBodyBackend",
    "FusionBackend",
    "get_backend",
    "available_backends",
]
