"""Step 1 -- cut a long video into caption segments.

Hybrid strategy from the spec:
  * **scene detection** (PySceneDetect, primary) cuts on natural boundaries so a
    segment's content lines up with its caption;
  * **fixed-interval sub-chunking** (layered on top) caps every segment at
    ``max_segment_seconds`` and merges sub-``min_segment_seconds`` scenes, so
    neither a long static shot nor a rapid-cut burst produces unusable segments.

The chunking math is pure and dependency-free (and unit-tested); only the two
signals that need to read the file -- scene boundaries and duration -- are lazy
shells over PySceneDetect / OpenCV, so ``import videocaption`` never pulls them
in and the GPU-free tests run without any decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

Span = Tuple[float, float]


class SegmentError(RuntimeError):
    """Raised when segmentation cannot read a video (missing decoder / bad file)."""


@dataclass
class Segment:
    """One caption segment: a half-open ``[start, end)`` time span within a video."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _split_fixed(start: float, end: float, max_seconds: float) -> List[Span]:
    """Split ``[start, end)`` into as-even-as-possible pieces each <= max_seconds."""
    dur = end - start
    if dur <= max_seconds:
        return [(start, end)]
    n = math.ceil(dur / max_seconds)
    step = dur / n  # even split reads better than n-1 full pieces + a stub
    return [(start + i * step, start + (i + 1) * step if i < n - 1 else end) for i in range(n)]


def _merge_short_scenes(scenes: List[Span], min_seconds: float) -> List[Span]:
    """Fold scenes shorter than ``min_seconds`` into an adjacent scene.

    Merging happens *before* the fixed-interval split, so the split still
    guarantees the final cap -- a merged-then-long scene just gets re-divided.
    A short leading scene folds into the next; any other short scene folds into
    the previous one already accepted.
    """
    merged: List[Span] = []
    for start, end in scenes:
        if end - start >= min_seconds or not merged:
            merged.append((start, end))
            continue
        prev_start, _ = merged[-1]
        merged[-1] = (prev_start, end)  # extend the previous span to swallow this stub
    # A short leading scene can survive as merged[0]; fold it forward if it exists.
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_seconds:
        first_start, _ = merged.pop(0)
        _, second_end = merged[0]
        merged[0] = (first_start, second_end)
    return merged


def subchunk(scenes: List[Span], max_seconds: float, min_seconds: float) -> List[Span]:
    """Turn raw scene spans into capped, stub-free segment spans (pure)."""
    merged = _merge_short_scenes(scenes, min_seconds)
    spans: List[Span] = []
    for start, end in merged:
        spans.extend(_split_fixed(start, end, max_seconds))
    return spans


def plan_segments(scenes: List[Span], duration: float, max_seconds: float, min_seconds: float) -> List[Segment]:
    """Combine scene spans + duration into the final indexed segment list (pure).

    ``scenes`` empty (or scene detection off) -> the whole video is one span, which
    ``subchunk`` then cuts into fixed ``max_seconds`` intervals.
    """
    spans = scenes or [(0.0, duration)]
    spans = subchunk(spans, max_seconds, min_seconds)
    return [Segment(index=i, start=round(s, 3), end=round(e, 3)) for i, (s, e) in enumerate(spans)]


# ---------------------------------------------------------------------------
# The two file-reading signals -- lazy, isolated, never imported at top level.
# ---------------------------------------------------------------------------

def probe_duration(video_path: Path) -> float:
    """Video length in seconds via OpenCV (frame count / fps). Lazy import."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SegmentError(
            "segmentation needs OpenCV to read video duration. `pip install opencv-python`."
        ) from exc
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or frames <= 0:
            raise SegmentError(f"could not read duration from {video_path}")
        return float(frames / fps)
    finally:
        cap.release()


def detect_scenes(video_path: Path, threshold: float) -> List[Span]:
    """Scene-boundary spans via PySceneDetect. Lazy import; [] if unavailable.

    Returning [] (rather than raising) lets the caller fall back to pure
    fixed-interval cuts when PySceneDetect isn't installed -- scene detection is
    an optimization, not a hard requirement.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError:  # pragma: no cover - environment dependent
        return []
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def segment_video(
    video_path: Path,
    *,
    scene_detect: bool,
    scene_threshold: float,
    max_seconds: float,
    min_seconds: float,
    duration: Optional[float] = None,
    scenes: Optional[List[Span]] = None,
) -> Tuple[List[Segment], float]:
    """Segment one video -> (segments, duration). Reads the file unless both
    ``duration`` and ``scenes`` are supplied (the tests pass them in)."""
    if duration is None:
        duration = probe_duration(video_path)
    if scenes is None:
        scenes = detect_scenes(video_path, scene_threshold) if scene_detect else []
    return plan_segments(scenes, duration, max_seconds, min_seconds), duration
