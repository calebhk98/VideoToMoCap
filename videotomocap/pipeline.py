"""Orchestration -- drive the whole video -> motion-dataset flow.

Stages, each resumable via the manifest (re-running only touches clips that are
not yet done):

    scan/exclude  -> HMR (backend)  -> anonymize  -> aggregate dataset

Everything is idempotent: state lives in the manifest and on disk, so a crash
part-way through a 2-year backlog just means re-invoking ``run``.

Parallelism
-----------
Clips are fully independent, so HMR is embarrassingly parallel: ``run_hmr`` with
``workers > 1`` processes several at once, pinning each worker to a GPU
round-robin (``gpus=['0','1']`` -> one clip per 3090). The heavy work happens in
each backend's own subprocess, which releases the GIL, so a thread pool gives
real speedup here. The manifest is checkpointed under a lock after every clip,
so the run stays crash-resumable even in parallel.
"""

from __future__ import annotations

import copy
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import ingest
from .backends import get_backend
from .config import PipelineConfig
from .dataset import build_dataset
from .gpu import resolve_devices, resolve_workers
from .ingest import Clip, Manifest
from .pose import SMPL_NJOINTS, anonymize, resample_fps


def _pose_path(cfg: PipelineConfig, clip: Clip) -> Path:
    return cfg.pose_dir / f"{clip.clip_id}.npz"


def _compute_and_save(cfg: PipelineConfig, clip: Clip) -> int:
    """Run HMR + anonymize for one clip and write its pose npz. Returns n_frames.

    Builds its own backend from ``cfg`` so it is safe to call from many threads
    concurrently (each gets an independent backend bound to its own GPU clone).
    """
    backend = get_backend(cfg)
    video = cfg.footage_root / clip.rel_path
    hmr_out = cfg.hmr_dir / clip.clip_id
    static = clip.camera in set(cfg.static_cameras)

    motion = backend.run(video, hmr_out, static=static)
    motion = resample_fps(motion, cfg.target_fps)
    if cfg.refine:
        from .refine import refine_motion  # optional post-processing; keep import lazy
        motion = refine_motion(motion)
    anon = anonymize(motion, drop_shape=cfg.drop_shape, keep_translation=cfg.keep_translation)
    anon.joint_valid = _joint_valid_mask(clip)  # label out-of-frame joints (if any)

    cfg.pose_dir.mkdir(parents=True, exist_ok=True)
    anon.save_npz(_pose_path(cfg, clip))
    return anon.n_frames


def _joint_valid_mask(clip: Clip):
    """(24,) bool mask, False where the camera can't see the joint. None if all seen."""
    if not clip.unreliable_joints:
        return None
    mask = np.ones(SMPL_NJOINTS, dtype=bool)
    mask[clip.unreliable_joints] = False
    return mask


def process_clip(cfg: PipelineConfig, manifest: Manifest, clip: Clip, backend=None) -> None:
    """Process one clip and update its manifest state (sequential path)."""
    clip.n_frames = _compute_and_save(cfg, clip)
    clip.status = ingest.POSE_DONE
    clip.error = None


def _todo(manifest: Manifest, limit: Optional[int]) -> List[Clip]:
    todo = manifest.by_status(ingest.PENDING, ingest.FAILED)
    return todo[:limit] if limit is not None else todo


def run_hmr(
    cfg: PipelineConfig,
    manifest: Manifest,
    *,
    limit: Optional[int] = None,
    workers: Optional[int] = None,
    gpus=None,
) -> Manifest:
    """Process all processable clips, in parallel across GPUs. Failures recorded.

    Devices come from ``gpus`` (or ``cfg.gpus``; 'auto' detects them). Worker
    count comes from ``workers`` (or ``cfg.workers``; auto = detected GPUs x
    ``cfg.workers_per_gpu``). One bad clip never kills the batch, and the
    manifest is checkpointed after every clip so the run stays resumable.
    """
    devices = resolve_devices(gpus if gpus is not None else cfg.gpus)
    n_workers = resolve_workers(
        workers if workers is not None else cfg.workers, devices, cfg.workers_per_gpu
    )
    return _run_hmr(cfg, manifest, workers=n_workers, devices=devices, limit=limit)


def _clone_for_device(cfg: PipelineConfig, device: Optional[str]) -> PipelineConfig:
    """Shallow config copy pinned to one GPU (exported to the backend subprocess)."""
    c = copy.copy(cfg)
    c.cuda_device = device
    return c


def _run_hmr(
    cfg: PipelineConfig,
    manifest: Manifest,
    *,
    workers: int,
    devices: List[Optional[str]],
    limit: Optional[int] = None,
) -> Manifest:
    """Fan clips out over a thread pool, round-robin across ``devices``.

    Threads (not processes) suffice: each clip's heavy work is in the backend
    subprocess, which releases the GIL, so N threads drive N concurrent
    subprocesses. ``workers`` > ``len(devices)`` packs multiple clips onto a
    card (overlapping GPU compute with the next clip's decode/IO).
    """
    todo = _todo(manifest, limit)
    total = len(todo)
    lock = threading.Lock()
    counter = {"done": 0}

    def work(idx: int, clip: Clip):
        device = devices[idx % len(devices)]
        clip_cfg = _clone_for_device(cfg, device)
        try:
            n_frames = _compute_and_save(clip_cfg, clip)
            return clip.clip_id, True, None, n_frames, device
        except Exception as exc:  # noqa: BLE001 -- isolate per-clip failure
            return clip.clip_id, False, f"{type(exc).__name__}: {exc}", None, device

    print(f"Processing {total} clips on {workers} worker(s) across devices {devices}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i, clip) for i, clip in enumerate(todo)]
        for fut in as_completed(futures):
            clip_id, ok, error, n_frames, device = fut.result()
            with lock:  # serialize manifest mutation + checkpoint
                clip = manifest.get(clip_id)
                if ok:
                    clip.status = ingest.POSE_DONE
                    clip.n_frames = n_frames
                    clip.error = None
                else:
                    clip.status = ingest.FAILED
                    clip.error = error
                manifest.save(cfg.manifest_path)
                counter["done"] += 1
                tag = f"gpu{device}" if device is not None else "cpu"
                outcome = "ok" if ok else f"FAILED {error}"
                print(f"[{counter['done']}/{total}] {tag} {clip_id}: {outcome}")
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


def run_all(
    cfg: PipelineConfig,
    *,
    limit: Optional[int] = None,
    workers: Optional[int] = None,
    gpus: Optional[List[str]] = None,
) -> "object":
    """Full flow from an existing (scanned + excluded) manifest to a dataset."""
    manifest = Manifest.load(cfg.manifest_path)
    run_hmr(cfg, manifest, limit=limit, workers=workers, gpus=gpus)
    return build(cfg, manifest)
