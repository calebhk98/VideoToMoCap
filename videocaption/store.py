"""On-disk per-video artifacts: the segmentation and the per-segment labels.

Kept out of the manifest so it stays small: the manifest tracks coarse video
state, while the (potentially many) segments and their labels live in one json
per video here. Labels are checkpointed incrementally -- after each segment is
captioned -- so a crash mid-video resumes at the first un-labelled segment rather
than recaptioning the whole thing. All writes are atomic (tmp + ``os.replace``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .backends.base import SegmentLabel
from .config import CaptionConfig
from .manifest import Manifest, Video
from .segment import Segment


@dataclass
class CaptionRow:
    """One search-index / label row: a captioned segment of a video."""

    video_id: str
    rel_path: str
    seg_index: int
    start: float
    end: float
    description: str
    tags: List[str]

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "rel_path": self.rel_path,
            "seg_index": self.seg_index,
            "start": self.start,
            "end": self.end,
            "description": self.description,
            "tags": list(self.tags),
        }


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _segments_path(cfg: CaptionConfig, video_id: str) -> Path:
    return cfg.segments_dir / f"{video_id}.json"


def _labels_path(cfg: CaptionConfig, video_id: str) -> Path:
    return cfg.labels_dir / f"{video_id}.json"


# -- segmentation ------------------------------------------------------------

def save_segments(cfg: CaptionConfig, video_id: str, segments: List[Segment], duration: float) -> None:
    """Persist Step 1's output for a video (written once, then read on resume)."""
    payload = {
        "video_id": video_id,
        "duration": duration,
        "segments": [{"index": s.index, "start": s.start, "end": s.end} for s in segments],
    }
    _atomic_write_json(_segments_path(cfg, video_id), payload)


def load_segments(cfg: CaptionConfig, video_id: str) -> Optional[List[Segment]]:
    """Load a video's segments, or None if it hasn't been segmented yet."""
    path = _segments_path(cfg, video_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return [Segment(index=s["index"], start=s["start"], end=s["end"]) for s in payload["segments"]]


# -- labels (incremental) ----------------------------------------------------

def load_labels(cfg: CaptionConfig, video_id: str) -> Dict[int, SegmentLabel]:
    """Load the labels captioned so far for a video (empty dict if none yet)."""
    path = _labels_path(cfg, video_id)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {int(k): SegmentLabel(description=v["description"], tags=list(v.get("tags", [])))
            for k, v in payload.items()}


def save_label(cfg: CaptionConfig, video_id: str, seg_index: int, label: SegmentLabel) -> None:
    """Checkpoint one segment's label, merging into the video's labels file."""
    labels = load_labels(cfg, video_id)
    labels[seg_index] = label
    payload = {str(k): v.to_dict() for k, v in sorted(labels.items())}
    _atomic_write_json(_labels_path(cfg, video_id), payload)


# -- joins -------------------------------------------------------------------

def video_rows(cfg: CaptionConfig, video: Video) -> List[CaptionRow]:
    """Join a video's segments + labels into rows (only fully-labelled segments)."""
    segments = load_segments(cfg, video.video_id) or []
    labels = load_labels(cfg, video.video_id)
    rows: List[CaptionRow] = []
    for seg in segments:
        label = labels.get(seg.index)
        if label is None:
            continue
        rows.append(CaptionRow(
            video_id=video.video_id, rel_path=video.rel_path, seg_index=seg.index,
            start=seg.start, end=seg.end, description=label.description, tags=label.tags,
        ))
    return rows


def iter_rows(cfg: CaptionConfig, manifest: Manifest) -> Iterator[CaptionRow]:
    """All captioned rows across the manifest, ordered by video then segment."""
    for video in manifest.videos:
        for row in video_rows(cfg, video):
            yield row
