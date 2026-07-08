"""Step 5.5 -- group caption segments into larger training/inference windows.

Two granularities matter (per the spec):
  * **caption segments** (Step 1, ~2 min): the unit captions are generated at.
  * **windows** (~10 min): a larger unit grouping several consecutive segments,
    which is what the Step 6 video-native model actually consumes in one pass and
    the full ``(start, end, description, tags)`` list it emits.

So the fine-grained segmentation only builds the *labels*; the trained model runs
on windows. Grouping is pure consecutive packing (unit-tested) -- no file access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .config import CaptionConfig
from .manifest import Manifest
from .store import CaptionRow, video_rows


@dataclass
class Window:
    """A ~window_seconds span of one video plus the dense label list inside it."""

    video_id: str
    rel_path: str
    start: float
    end: float
    entries: List[CaptionRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "rel_path": self.rel_path,
            "start": self.start,
            "end": self.end,
            "labels": [
                {"start": e.start, "end": e.end, "description": e.description, "tags": list(e.tags)}
                for e in self.entries
            ],
        }


def group_windows(rows: List[CaptionRow], window_seconds: float) -> List[Window]:
    """Pack consecutive segment ``rows`` (one video, in order) into windows (pure).

    A new segment starts a new window whenever adding it would push the window's
    span past ``window_seconds`` -- so each window holds at least one segment and
    stays close to (never far past) the target length.
    """
    ordered = sorted(rows, key=lambda r: r.start)
    windows: List[Window] = []
    for row in ordered:
        cur = windows[-1] if windows else None
        if cur is not None and row.end - cur.start <= window_seconds:
            cur.entries.append(row)
            cur.end = row.end
            continue
        windows.append(Window(video_id=row.video_id, rel_path=row.rel_path,
                              start=row.start, end=row.end, entries=[row]))
    return windows


def build_windows(cfg: CaptionConfig, manifest: Manifest) -> List[Window]:
    """All windows across the archive, grouping each video's captioned rows."""
    windows: List[Window] = []
    for video in manifest.videos:
        windows.extend(group_windows(video_rows(cfg, video), cfg.window_seconds))
    return windows


def write_windows(windows: List[Window], path: Path) -> Path:
    """Persist the window list as json (input to the Step 6 dataset builder)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"windows": [w.to_dict() for w in windows]}, indent=2))
    tmp.replace(path)
    return path
