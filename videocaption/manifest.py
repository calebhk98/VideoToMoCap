"""Ingest the archive into a manifest + the video state machine (Pipeline 3).

Deliberately dumb and reversible, exactly like Pipeline 1's ``ingest``: the
manifest only records *what exists* and *what state each video is in*. Nothing is
copied or transcoded. It is the single source of truth the rest of the pipeline
reads from, and it is checkpointed after every video so a crash mid-backlog just
means re-invoking ``caption``.

Per-video artifacts (segmentation + labels) live in their own json files on disk
(see ``config.segments_dir`` / ``labels_dir``); the manifest only tracks the
coarse state, so it stays small even for a 1,000-hour archive.

Video states
------------
    pending    -- discovered, not yet processed
    excluded   -- manually removed from processing
    segmented  -- Step 1 done: segments written, not yet captioned
    done       -- captioned + labelled (rows ready for the index)
    failed     -- a stage errored on this video (see .error)
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import CaptionConfig

PENDING = "pending"
EXCLUDED = "excluded"
SEGMENTED = "segmented"
DONE = "done"
FAILED = "failed"


@dataclass
class Video:
    """One tracked archive file plus its position in the processing state machine."""

    video_id: str
    """Stable id derived from the relative path (safe for filenames)."""
    rel_path: str
    """Path relative to ``footage_root`` (portable across machines)."""
    status: str = PENDING
    error: Optional[str] = None
    duration: Optional[float] = None
    """Video length in seconds (filled in during segmentation)."""
    n_segments: Optional[int] = None
    """How many caption segments this video was cut into (Step 1)."""

    def is_processable(self) -> bool:
        """True if the pipeline should (re)run on this video."""
        return self.status in (PENDING, SEGMENTED, FAILED)


@dataclass
class Manifest:
    """The on-disk source of truth: every discovered video and its status."""

    footage_root: str
    videos: List[Video] = field(default_factory=list)

    # -- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        """Write to ``path``, replacing it atomically so a crash mid-write can
        never leave a corrupt/partial manifest for the next run to load."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"footage_root": self.footage_root, "videos": [asdict(v) for v in self.videos]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)  # os.replace: atomic overwrite on Linux AND Windows

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load a manifest previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        videos = [Video(**v) for v in payload["videos"]]
        return cls(footage_root=payload["footage_root"], videos=videos)

    # -- queries --------------------------------------------------------
    def by_status(self, *statuses: str) -> List[Video]:
        """Return videos whose status is any of ``statuses``."""
        return [v for v in self.videos if v.status in statuses]

    def get(self, video_id: str) -> Video:
        """Look up a video by id; raises ``KeyError`` if it isn't in the manifest."""
        for v in self.videos:
            if v.video_id == video_id:
                return v
        raise KeyError(video_id)

    def counts(self) -> Dict[str, int]:
        """Tally videos per status, e.g. for the CLI ``status`` command."""
        out: Dict[str, int] = {}
        for v in self.videos:
            out[v.status] = out.get(v.status, 0) + 1
        return out


def _video_id(rel_path: str) -> str:
    """Human-readable-ish id: sanitized stem + short hash to guarantee uniqueness."""
    stem = Path(rel_path).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    digest = hashlib.sha1(rel_path.encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def scan(cfg: CaptionConfig) -> Manifest:
    """Walk ``footage_root`` and build a fresh manifest of all candidate videos."""
    root = cfg.footage_root
    if not root.exists():
        raise FileNotFoundError(f"footage_root does not exist: {root}")

    exts = tuple(e.lower() for e in cfg.video_exts)
    videos: List[Video] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel_str = path.relative_to(root).as_posix()
        videos.append(Video(video_id=_video_id(rel_str), rel_path=rel_str))
    manifest = Manifest(footage_root=str(root), videos=videos)
    exclude(manifest, video_ids=cfg.exclude_ids, patterns=cfg.exclude_patterns, note="excluded via config")
    return manifest


def refresh(cfg: CaptionConfig, manifest: Manifest) -> Manifest:
    """Merge a new scan into an existing manifest, preserving per-video state.

    New files become ``pending``; existing videos keep their status; videos whose
    file has disappeared are dropped.
    """
    fresh = scan(cfg)
    old_by_id = {v.video_id: v for v in manifest.videos}
    merged: List[Video] = []
    for v in fresh.videos:
        prev = old_by_id.get(v.video_id)
        if prev is not None:
            v.status = prev.status
            v.error = prev.error
            v.duration = prev.duration
            v.n_segments = prev.n_segments
        merged.append(v)
    return Manifest(footage_root=fresh.footage_root, videos=merged)


def exclude(
    manifest: Manifest,
    *,
    video_ids: Iterable[str] = (),
    patterns: Iterable[str] = (),
    note: str = "manually excluded",
) -> int:
    """Mark videos as excluded so no downstream stage will touch them.

    ``patterns`` are glob patterns matched against the relative path. Returns the
    number of videos newly excluded.
    """
    video_ids = set(video_ids)
    patterns = list(patterns)
    n = 0
    for v in manifest.videos:
        if v.status == EXCLUDED:
            continue
        # fnmatchcase (not fnmatch) so a pattern matches identically on Windows and
        # Linux -- plain fnmatch is case-insensitive on Windows only.
        hit = v.video_id in video_ids or any(fnmatch.fnmatchcase(v.rel_path, p) for p in patterns)
        if hit:
            v.status = EXCLUDED
            v.error = None
            n += 1
    return n


def include(manifest: Manifest, *, video_ids: Iterable[str] = (), patterns: Iterable[str] = ()) -> int:
    """Reverse an exclusion (bring videos back to ``pending``)."""
    video_ids = set(video_ids)
    patterns = list(patterns)
    n = 0
    for v in manifest.videos:
        if v.status != EXCLUDED:
            continue
        hit = v.video_id in video_ids or any(fnmatch.fnmatchcase(v.rel_path, p) for p in patterns)
        if hit:
            v.status = PENDING
            n += 1
    return n
