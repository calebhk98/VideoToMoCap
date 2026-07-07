"""Step 0 -- ingest raw footage into a manifest, and exclude unwanted clips.

Ingestion is deliberately dumb and reversible: it only records *what exists* and
*what state each clip is in*.  Nothing is copied or transcoded.  The manifest is
the single source of truth the rest of the pipeline reads from.

Clip states
-----------
    pending   -- discovered, not yet processed
    excluded  -- manually removed from processing (e.g. the two family visits)
    hmr_done  -- human-mesh recovery produced motion for this clip
    pose_done -- shape stripped, anonymized pose written
    failed    -- backend errored on this clip (see .error)
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import PipelineConfig

PENDING = "pending"
EXCLUDED = "excluded"
HMR_DONE = "hmr_done"
POSE_DONE = "pose_done"
FAILED = "failed"


@dataclass
class Clip:
    """One tracked footage file plus its position in the processing state machine."""

    clip_id: str
    """Stable id derived from the relative path (safe for filenames)."""
    camera: str
    rel_path: str
    """Path relative to ``footage_root`` (portable across machines)."""
    status: str = PENDING
    error: Optional[str] = None
    n_frames: Optional[int] = None
    note: Optional[str] = None
    partial_body: bool = False
    """True if any body region is out of frame for this clip's camera."""
    suspected_mirrored: bool = False
    """Handedness disagrees with the corpus consensus -> likely a flipped clip."""
    mirrored: bool = False
    """This clip's pose was left/right corrected (``auto_mirror: correct``)."""
    low_quality: bool = False
    """Motion has a hard quality issue (teleport/pose-jump/NaN) -> a drop candidate."""
    quality_issues: List[str] = field(default_factory=list)
    """Auto-detected quality findings from ``videotomocap.quality`` (empty = clean)."""
    unreliable_joints: List[int] = field(default_factory=list)
    """SMPL joint indices whose motion is HMR-inferred (out of frame), not
    observed -- derived from ``cfg.camera_occlusions`` / ``partial_body_cameras``.
    Use to filter or mask these joints downstream."""

    def is_processable(self) -> bool:
        """True if HMR should (re)run on this clip: pending, or previously failed."""
        return self.status in (PENDING, FAILED)


@dataclass
class Manifest:
    """The on-disk source of truth: every discovered clip and its status.

    Nothing about the footage itself is copied here -- only bookkeeping -- so the
    manifest is cheap to rewrite after every clip (see :meth:`save`).
    """

    footage_root: str
    clips: List[Clip] = field(default_factory=list)

    # -- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        """Write to ``path``, replacing it atomically so a crash mid-write can
        never leave a corrupt/partial manifest for the next run to load."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"footage_root": self.footage_root, "clips": [asdict(c) for c in self.clips]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)  # os.replace: atomic overwrite on Linux AND Windows

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load a manifest previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        clips = [Clip(**c) for c in payload["clips"]]
        return cls(footage_root=payload["footage_root"], clips=clips)

    # -- queries --------------------------------------------------------
    def by_status(self, *statuses: str) -> List[Clip]:
        """Return clips whose status is any of ``statuses``."""
        return [c for c in self.clips if c.status in statuses]

    def get(self, clip_id: str) -> Clip:
        """Look up a clip by id; raises ``KeyError`` if it isn't in the manifest."""
        for c in self.clips:
            if c.clip_id == clip_id:
                return c
        raise KeyError(clip_id)

    def counts(self) -> Dict[str, int]:
        """Tally clips per status, e.g. for the CLI ``status`` command."""
        out: Dict[str, int] = {}
        for c in self.clips:
            out[c.status] = out.get(c.status, 0) + 1
        return out


def _clip_id(rel_path: str) -> str:
    """Human-readable-ish id: sanitized stem + short hash to guarantee uniqueness."""
    stem = Path(rel_path).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    digest = hashlib.sha1(rel_path.encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def _camera_of(rel_path: Path, depth: int) -> str:
    """Camera id = the first ``depth`` path components (e.g. depth=1 -> 'cam03').

    Falls back to a single shared 'cam' bucket when the tree is too flat to
    slice, so a camera-less layout doesn't raise.
    """
    parts = rel_path.parts
    if depth <= 0 or len(parts) <= 1:
        return "cam"
    return "/".join(parts[:depth])


def scan(cfg: PipelineConfig) -> Manifest:
    """Walk ``footage_root`` and build a fresh manifest of all candidate clips."""
    root = cfg.footage_root
    if not root.exists():
        raise FileNotFoundError(f"footage_root does not exist: {root}")

    exts = tuple(e.lower() for e in cfg.video_exts)
    clips: List[Clip] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(root)
        rel_str = rel.as_posix()
        camera = _camera_of(rel, cfg.camera_dir_depth)
        unreliable = _unreliable_joints_for(cfg, camera)
        clips.append(
            Clip(
                clip_id=_clip_id(rel_str),
                camera=camera,
                rel_path=rel_str,
                partial_body=bool(unreliable),
                unreliable_joints=unreliable,
            )
        )
    manifest = Manifest(footage_root=str(root), clips=clips)
    # Apply config-defined exclusions so they don't have to be retyped each run.
    exclude(manifest, clip_ids=cfg.exclude_ids, patterns=cfg.exclude_patterns, note="excluded via config")
    return manifest


def _unreliable_joints_for(cfg: PipelineConfig, camera: str) -> List[int]:
    """SMPL joints out of frame for a camera, from occlusion config + the
    ``partial_body_cameras`` (=legs) shorthand."""
    from .regions import joints_for_regions

    regions = list(cfg.camera_occlusions.get(camera, []))
    if camera in set(cfg.partial_body_cameras):
        regions.append("legs")
    return joints_for_regions(regions) if regions else []


def refresh(cfg: PipelineConfig, manifest: Manifest) -> Manifest:
    """Merge a new scan into an existing manifest, preserving per-clip state.

    New files become ``pending``; existing clips keep their status/exclusions;
    clips whose file has disappeared are dropped.
    """
    fresh = scan(cfg)
    old_by_id = {c.clip_id: c for c in manifest.clips}
    merged: List[Clip] = []
    for c in fresh.clips:
        prev = old_by_id.get(c.clip_id)
        if prev is not None:
            c.status = prev.status
            c.error = prev.error
            c.n_frames = prev.n_frames
            c.note = prev.note
        merged.append(c)
    return Manifest(footage_root=fresh.footage_root, clips=merged)


def exclude(
    manifest: Manifest,
    *,
    clip_ids: Iterable[str] = (),
    patterns: Iterable[str] = (),
    note: str = "manually excluded",
) -> int:
    """Mark clips as excluded so no downstream stage will touch them.

    ``patterns`` are glob patterns matched against the relative path -- the
    intended way to pull out the two family-visit date ranges in one shot, e.g.
    ``exclude(m, patterns=["*/2024-12-25/*", "cam07/2025-03-*"])``.
    Returns the number of clips newly excluded.
    """
    clip_ids = set(clip_ids)
    patterns = list(patterns)
    n = 0
    for c in manifest.clips:
        if c.status == EXCLUDED:
            continue
        # fnmatchcase (not fnmatch) so a pattern matches identically on Windows and
        # Linux -- plain fnmatch is case-insensitive on Windows only.
        hit = c.clip_id in clip_ids or any(fnmatch.fnmatchcase(c.rel_path, p) for p in patterns)
        if hit:
            c.status = EXCLUDED
            c.note = note
            n += 1
    return n


def include(manifest: Manifest, *, clip_ids: Iterable[str] = (), patterns: Iterable[str] = ()) -> int:
    """Reverse an exclusion (bring clips back to ``pending``)."""
    clip_ids = set(clip_ids)
    patterns = list(patterns)
    n = 0
    for c in manifest.clips:
        if c.status != EXCLUDED:
            continue
        # fnmatchcase (not fnmatch) so a pattern matches identically on Windows and
        # Linux -- plain fnmatch is case-insensitive on Windows only.
        hit = c.clip_id in clip_ids or any(fnmatch.fnmatchcase(c.rel_path, p) for p in patterns)
        if hit:
            c.status = PENDING
            c.note = None
            n += 1
    return n
