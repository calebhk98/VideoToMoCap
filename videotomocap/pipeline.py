"""Orchestration -- drive the whole video -> motion-dataset flow.

Stages, each resumable via the manifest (re-running only touches clips that are
not yet done):

    scan/exclude  -> HMR (backend)  -> anonymize  -> aggregate dataset

Everything is idempotent: state lives in the manifest and on disk, so a crash
part-way through a 2-year backlog just means re-invoking ``run``.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional

from . import ingest
from .backends import get_backend
from .config import PipelineConfig
from .dataset import build_dataset
from .ingest import Clip, Manifest
from .pose import anonymize, resample_fps


def _pose_path(cfg: PipelineConfig, clip: Clip) -> Path:
    return cfg.pose_dir / f"{clip.clip_id}.npz"


def process_clip(cfg: PipelineConfig, manifest: Manifest, clip: Clip, backend) -> None:
    """Run HMR + anonymization for one clip and update its manifest state."""
    video = cfg.footage_root / clip.rel_path
    hmr_out = cfg.hmr_dir / clip.clip_id
    static = clip.camera in set(cfg.static_cameras)

    motion = backend.run(video, hmr_out, static=static)
    motion = resample_fps(motion, cfg.target_fps)
    anon = anonymize(motion, drop_shape=cfg.drop_shape, keep_translation=cfg.keep_translation)

    cfg.pose_dir.mkdir(parents=True, exist_ok=True)
    anon.save_npz(_pose_path(cfg, clip))

    clip.n_frames = anon.n_frames
    clip.status = ingest.POSE_DONE
    clip.error = None


def run_hmr(cfg: PipelineConfig, manifest: Manifest, *, limit: Optional[int] = None) -> Manifest:
    """Process all processable clips. Failures are recorded, not fatal."""
    backend = get_backend(cfg)
    todo = [c for c in manifest.by_status(ingest.PENDING, ingest.FAILED)]
    if limit is not None:
        todo = todo[:limit]

    for i, clip in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {clip.clip_id}  ({clip.rel_path})")
        try:
            process_clip(cfg, manifest, clip, backend)
        except Exception as exc:  # noqa: BLE001 -- one bad clip must not kill the batch
            clip.status = ingest.FAILED
            clip.error = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {clip.error}")
            traceback.print_exc()
        finally:
            manifest.save(cfg.manifest_path)  # checkpoint after every clip
    return manifest


def build(cfg: PipelineConfig, manifest: Manifest):
    """Aggregate all anonymized clips into the AMASS-format training dataset."""
    done = manifest.by_status(ingest.POSE_DONE)
    pose_paths = [_pose_path(cfg, c) for c in done]
    pose_paths = [p for p in pose_paths if p.exists()]
    stats = build_dataset(
        pose_paths,
        cfg.dataset_dir,
        gender=cfg.gender,
        val_fraction=cfg.val_fraction,
        min_frames=cfg.min_clip_frames,
    )
    return stats


def run_all(cfg: PipelineConfig, *, limit: Optional[int] = None):
    """Full flow from an existing (scanned + excluded) manifest to a dataset."""
    manifest = Manifest.load(cfg.manifest_path)
    run_hmr(cfg, manifest, limit=limit)
    return build(cfg, manifest)
