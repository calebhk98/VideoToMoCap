"""Synthetic caption + aggregate backends -- no GPU, no weights, no decoder.

Exercise the whole Pipeline 3 flow (segment -> caption -> aggregate -> index ->
window -> fine-tune dataset) end-to-end on any machine, exactly as Pipeline 1's
HMR ``noop`` backend does. The captioner fabricates a deterministic caption per
frame *without touching pixels*; the aggregator folds captions into a
deterministic ``(description, tags)`` label. Never use their output as real data.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import List

from .base import Aggregator, Captioner, FrameCaption, SegmentLabel

# A tiny fixed vocabulary so synthetic captions look caption-ish and are stable.
_SUBJECTS = ["a person", "two people", "a dog", "a room", "a street", "a kitchen", "a garden"]
_ACTIONS = ["walking", "sitting", "cooking", "talking", "reading", "standing", "playing"]
_DETAILS = ["indoors", "outdoors", "in daylight", "at night", "near a window", "by a table"]


def _pick(seed: str, options: List[str]) -> str:
    """Deterministically pick an option from a string seed (no RNG state)."""
    h = int(hashlib.sha1(seed.encode()).hexdigest(), 16)
    return options[h % len(options)]


class NoopCaptioner(Captioner):
    """GPU-free synthetic captioner; fabricates one caption per frame timestamp."""

    name = "noop"

    def caption_frames(self, video_path: Path, times: List[float], out_dir: Path) -> List[FrameCaption]:
        stem = Path(video_path).name
        out: List[FrameCaption] = []
        for t in times:
            key = f"{stem}:{t:.3f}"
            text = f"{_pick(key + 's', _SUBJECTS)} {_pick(key + 'a', _ACTIONS)} {_pick(key + 'd', _DETAILS)}."
            out.append(FrameCaption(time=t, text=text))
        return out


class NoopAggregator(Aggregator):
    """GPU-free synthetic aggregator; folds captions into a deterministic label."""

    name = "noop"

    def aggregate(self, captions: List[FrameCaption], *, num_tags: int) -> SegmentLabel:
        texts = [c.text for c in captions]
        # honour caption_focus so the synthetic path exercises both label styles
        # (and the pairing bridge sees the motion-focused wording it will pass on).
        focus = getattr(self.cfg, "caption_focus", "scene")
        lead = "Body movement:" if focus == "motion" else "Scene:"
        first = texts[0] if texts else "empty"
        description = f"{lead} {first} ({len(texts)} frames observed.)"
        tags = _keyword_tags(texts, num_tags)
        return SegmentLabel(description=description, tags=tags)


_STOPWORDS = {"a", "an", "the", "in", "on", "by", "at", "near", "of", "and", "to", "is", "are"}


def _keyword_tags(texts: List[str], num_tags: int) -> List[str]:
    """Most-frequent content words across the frame captions -- a cheap tag proxy."""
    words = re.findall(r"[a-zA-Z]+", " ".join(texts).lower())
    counts = Counter(w for w in words if w not in _STOPWORDS and len(w) > 2)
    return [w for w, _ in counts.most_common(num_tags)]
