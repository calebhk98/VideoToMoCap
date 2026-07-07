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

from . import ingest
from .backends import get_backend
from .config import PipelineConfig
from .dataset import build_dataset
from .ingest import Clip, Manifest
from .pose import anonymize, resample_fps


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
    anon = anonymize(motion, drop_shape=cfg.drop_shape, keep_translation=cfg.keep_translation)

    cfg.pose_dir.mkdir(parents=True, exist_ok=True)
    anon.save_npz(_pose_path(cfg, clip))
    return anon.n_frames


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
    gpus: Optional[List[str]] = None,
) -> Manifest:
    """Process all processable clips. Failures are recorded, not fatal.

    ``workers`` (default ``cfg.workers``) > 1 fans clips out concurrently across
    ``gpus`` (default ``cfg.gpus``). One bad clip never kills the batch.
    """
    n_workers = workers if workers is not None else cfg.workers
    if n_workers and n_workers > 1:
        return _run_hmr_parallel(cfg, manifest, workers=n_workers, gpus=gpus, limit=limit)

    todo = _todo(manifest, limit)
    for i, clip in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {clip.clip_id}  ({clip.rel_path})")
        try:
            process_clip(cfg, manifest, clip)
        except Exception as exc:  # noqa: BLE001 -- one bad clip must not kill the batch
            clip.status = ingest.FAILED
            clip.error = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {clip.error}")
            traceback.print_exc()
        finally:
            manifest.save(cfg.manifest_path)  # checkpoint after every clip
    return manifest


def _clone_for_device(cfg: PipelineConfig, device: Optional[str]) -> PipelineConfig:
    """Shallow config copy pinned to one GPU (exported to the backend subprocess)."""
    c = copy.copy(cfg)
    c.cuda_device = device
    return c


def _run_hmr_parallel(
    cfg: PipelineConfig,
    manifest: Manifest,
    *,
    workers: int,
    gpus: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Manifest:
    devices = list(gpus) if gpus else (list(cfg.gpus) if cfg.gpus else [None])
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
