"""Bridge: pair Pipeline 3 captions with Pipeline 1 motion for text-to-motion.

Pipeline 1 recovers **one motion sequence per footage file**; Pipeline 3 cuts each
file into scene **caption segments** and labels them. To train a *text-conditioned*
motion model (Pipeline 2, ``conditioning: text``) you need one ``(motion snippet,
caption)`` pair per segment -- so this module slices each clip's recovered motion
at the caption-segment time spans and re-emits the snippets in the same
AMASS-SMPL-H + ``index.json`` shape ``videotomocap.dataset`` produces, with the
segment's caption carried on each entry.

The join is free: Pipeline 1's ``clip_id`` and Pipeline 3's ``video_id`` use the
identical id formula, so a footage file has the same id in both (given the same
``footage_root``). The sliced motion is already anonymized -- it comes from
Pipeline 1's ``pose/`` npz, which is written *after* ``anonymize()`` -- so no new
identity path is introduced, and ``to_amass_npz`` keeps betas neutral regardless.

``videocaption`` is imported lazily inside the builder so ``import videotomocap``
stays numpy-only and never pulls the caption package in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .dataset import to_amass_npz
from .pose import SmplMotion


def slice_motion(motion: SmplMotion, start: float, end: float) -> Optional[SmplMotion]:
    """Return the ``[start, end)``-second slice of a clip's motion, or None if <2 frames.

    Frame indices come from the clip's own fps; ``[start, end)`` are wall-clock
    seconds (``resample_fps`` preserves duration, so seconds map straight to the
    resampled timeline). Hands/joint_valid are carried through the slice.
    """
    fps = motion.fps
    i0 = max(0, int(round(start * fps)))
    i1 = min(motion.n_frames, int(round(end * fps)))
    if i1 - i0 < 2:
        return None

    def _sl(arr):
        return None if arr is None else arr[i0:i1].copy()

    return SmplMotion(
        poses=motion.poses[i0:i1].copy(),
        trans=motion.trans[i0:i1].copy(),
        fps=fps,
        betas=None,  # already anonymized upstream; keep neutral
        left_hand_pose=_sl(motion.left_hand_pose),
        right_hand_pose=_sl(motion.right_hand_pose),
        joint_valid=None if motion.joint_valid is None else motion.joint_valid.copy(),
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, sliced=[round(start, 3), round(end, 3)]),
    )


def _emit_segment(motion, video, row, amass_dir: Path, min_frames: int, counters: Dict[str, int]) -> Optional[dict]:
    """Slice + export one caption segment's motion snippet; return its index entry."""
    snippet = slice_motion(motion, row.start, row.end)
    if snippet is None or snippet.n_frames < min_frames:
        counters["too_short"] += 1
        return None
    try:
        payload = to_amass_npz(snippet)
    except ValueError:
        # camera-relative motion can't become world-frame AMASS; skip loudly-counted.
        counters["incam_skipped"] += 1
        return None
    seg_id = f"{video.video_id}_seg{row.seg_index:04d}"
    np.savez(amass_dir / f"{seg_id}.npz", **payload)
    counters["segments"] += 1
    return {
        "clip_id": seg_id,
        "n_frames": snippet.n_frames,
        "fps": snippet.fps,
        "caption": row.description,   # reused as-is for text conditioning
        "tags": list(row.tags),
        "source_video": video.video_id,
        "start": row.start,
        "end": row.end,
    }


def _split_by_source(entries: List[dict], val_fraction: float, seed: int) -> None:
    """Assign train/val at the *source-video* level, so segments from one file never
    straddle the split (that would leak near-identical motion between train and val)."""
    videos = sorted({e["source_video"] for e in entries})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    n_val = max(1, int(len(videos) * val_fraction)) if videos else 0
    val_videos = {videos[i] for i in order[:n_val]}
    for e in entries:
        e["split"] = "val" if e["source_video"] in val_videos else "train"


def _write_index_stats(entries: List[dict], out_dir: Path, counters: Dict[str, int]) -> dict:
    (out_dir / "index.json").write_text(json.dumps({"clips": entries}, indent=2))
    total_frames = sum(e["n_frames"] for e in entries)
    fps = float(entries[0]["fps"]) if entries else 0.0
    stats = {
        "n_clips": len(entries),
        "n_frames": total_frames,
        "total_seconds": round(total_frames / fps, 1) if fps else 0.0,
        "total_hours": round(total_frames / fps / 3600.0, 3) if fps else 0.0,
        "fps": fps,
        **counters,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def build_captioned_dataset(
    pose_dir: Path,
    caption_work_root: Path,
    out_dir: Path,
    *,
    min_frames: int = 30,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> dict:
    """Emit a text-conditioned dataset from Pipeline 1 motion + Pipeline 3 captions.

    ``pose_dir``            Pipeline 1's ``work/pose`` (anonymized per-clip npz).
    ``caption_work_root``   Pipeline 3's ``work/caption`` (manifest + labels).
    ``out_dir``             where ``amass/*.npz`` + ``index.json`` + ``stats.json`` land.

    Each captioned segment whose source clip has recovered motion becomes one
    snippet+caption pair. Segments with no motion (clip excluded/failed in
    Pipeline 1) or too few frames are counted and skipped, never fatal. The
    ``index.json`` is exactly ``motion_model``'s input shape, plus a ``caption``
    field its ``conditioning: text`` path now reads.
    """
    from videocaption import store as vstore  # lazy: keep `import videotomocap` numpy-only
    from videocaption.config import CaptionConfig
    from videocaption.manifest import Manifest

    ccfg = CaptionConfig(work_root=Path(caption_work_root))
    if not ccfg.manifest_path.exists():
        raise FileNotFoundError(
            f"No caption manifest at {ccfg.manifest_path}; run `python -m videocaption caption` first."
        )
    manifest = Manifest.load(ccfg.manifest_path)
    pose_dir = Path(pose_dir)
    amass_dir = Path(out_dir) / "amass"
    amass_dir.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = []
    counters = {"with_motion": 0, "missing_motion": 0, "too_short": 0, "incam_skipped": 0, "segments": 0}
    for video in manifest.videos:
        rows = vstore.video_rows(ccfg, video)
        if not rows:
            continue
        pose_path = pose_dir / f"{video.video_id}.npz"
        if not pose_path.exists():
            counters["missing_motion"] += 1
            continue
        counters["with_motion"] += 1
        motion = SmplMotion.load_npz(pose_path)
        for row in rows:
            entry = _emit_segment(motion, video, row, amass_dir, min_frames, counters)
            if entry is not None:
                entries.append(entry)

    _split_by_source(entries, val_fraction, seed)
    return _write_index_stats(entries, Path(out_dir), counters)
