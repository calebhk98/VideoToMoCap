"""Caption + aggregate backends (Pipeline 3).

Two roles, each wrapping one upstream model behind a common interface so the
orchestration layer never has to know which is in use. Pick with
``CaptionConfig.captioner`` / ``.aggregator`` -- swapping is one config line,
exactly like Pipeline 1's ``backend``.

Availability of the underlying models (verified 2026-07, see README.md):

Captioners (Step 3, per-frame VLM):
    joycaption -> fancyfeast/llama-joycaption-beta-one-hf-llava (Llama 3.1 + vision)

Aggregators (Step 4, text -> structured label):
    dolphin    -> cognitivecomputations/Dolphin3.0-Llama3.1-8B (community instruct tune)

Testing:
    noop       -> synthetic captioner + aggregator (no GPU, no weights, no decoder)
"""

from __future__ import annotations

from .base import Aggregator, Captioner, CaptionError, FrameCaption, SegmentLabel
from .dolphin import DolphinBackend
from .joycaption import JoyCaptionBackend
from .noop import NoopAggregator, NoopCaptioner

_CAPTIONERS = {
    "joycaption": JoyCaptionBackend,
    "noop": NoopCaptioner,
}

_AGGREGATORS = {
    "dolphin": DolphinBackend,
    "noop": NoopAggregator,
}


def get_captioner(cfg) -> Captioner:
    """Instantiate the captioner named by ``cfg.captioner`` (case-insensitive)."""
    name = cfg.captioner.lower()
    if name not in _CAPTIONERS:
        raise CaptionError(f"Unknown captioner {cfg.captioner!r}. Choose from {sorted(_CAPTIONERS)}.")
    return _CAPTIONERS[name](cfg)


def get_aggregator(cfg) -> Aggregator:
    """Instantiate the aggregator named by ``cfg.aggregator`` (case-insensitive)."""
    name = cfg.aggregator.lower()
    if name not in _AGGREGATORS:
        raise CaptionError(f"Unknown aggregator {cfg.aggregator!r}. Choose from {sorted(_AGGREGATORS)}.")
    return _AGGREGATORS[name](cfg)


def available_captioners():
    """List registered captioner names, sorted."""
    return sorted(_CAPTIONERS)


def available_aggregators():
    """List registered aggregator names, sorted."""
    return sorted(_AGGREGATORS)


__all__ = [
    "Captioner",
    "Aggregator",
    "CaptionError",
    "FrameCaption",
    "SegmentLabel",
    "JoyCaptionBackend",
    "DolphinBackend",
    "NoopCaptioner",
    "NoopAggregator",
    "get_captioner",
    "get_aggregator",
    "available_captioners",
    "available_aggregators",
]
