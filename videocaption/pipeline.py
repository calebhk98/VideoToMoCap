"""Orchestration -- drive the whole archive -> captions -> index -> windows flow.

Stages, each resumable via the manifest + on-disk artifacts (re-running only
touches what isn't done):

    scan/exclude -> segment -> caption+aggregate -> index -> windows -> ft-dataset

Everything is idempotent: video state lives in the manifest and labels are
checkpointed per segment, so a crash partway through a 1,000-hour backlog just
means re-invoking ``caption``.

Parallelism
-----------
Videos are independent, so captioning is embarrassingly parallel: ``caption``
with ``workers > 1`` processes several videos at once, pinning each to a GPU
round-robin. With 2x3090 that's one captioner/aggregator instance per card
(roughly doubling throughput). The heavy work is in the model subprocess, which
releases the GIL, so a thread pool gives real speedup. The manifest is
checkpointed under a lock after each video, so the run stays crash-resumable.
"""

from __future__ import annotations

import copy
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from . import manifest as mf
from . import store
from .backends import get_aggregator, get_captioner
from .config import CaptionConfig
from .frames import frame_times
from .index import build_index
from .manifest import Manifest, Video
from .segment import segment_video
from .window import build_windows, write_windows


# ---------------------------------------------------------------------------
# Device resolution (kept local so the package doesn't depend on Pipeline 1).
# ---------------------------------------------------------------------------

def _detect_gpus() -> List[str]:
    """GPU ids from ``nvidia-smi``; [] if it isn't present (CPU-only box)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def resolve_devices(gpus) -> List[Optional[str]]:
    """Normalize the ``gpus`` config into a device list. ``[None]`` = ambient device."""
    if not gpus:
        return [None]
    if isinstance(gpus, str):
        gpus = [gpus]
    gpus = [str(g) for g in gpus]
    if gpus == ["auto"]:
        return _detect_gpus() or [None]
    return gpus


def _expand(devices: List[Optional[str]], workers_per_gpu: int) -> List[Optional[str]]:
    """Repeat each device ``workers_per_gpu`` times (packs N videos onto one card)."""
    per = max(1, int(workers_per_gpu))
    return [d for d in devices for _ in range(per)]


def _clone_for_device(cfg: CaptionConfig, device: Optional[str]) -> CaptionConfig:
    """Shallow config copy pinned to one GPU (exported to the model subprocess)."""
    c = copy.copy(cfg)
    c.cuda_device = device
    return c


# ---------------------------------------------------------------------------
# Per-video work (Steps 1-4), resumable at segment granularity.
# ---------------------------------------------------------------------------

def _segment_dir(cfg: CaptionConfig, video: Video, seg_index: int) -> Path:
    return cfg.frames_dir / video.video_id / f"seg_{seg_index:04d}"


def process_video(cfg: CaptionConfig, video: Video) -> int:
    """Segment (if needed) then caption every un-labelled segment. Returns n_segments.

    Builds its own captioner/aggregator from ``cfg`` so it is safe to call from
    many threads concurrently (each bound to its own GPU clone). Segments already
    present in the labels file are skipped, so a re-run resumes mid-video.
    """
    video_path = cfg.footage_root / video.rel_path
    segments = store.load_segments(cfg, video.video_id)
    if segments is None:
        segments, duration = segment_video(
            video_path, scene_detect=cfg.scene_detect, scene_threshold=cfg.scene_threshold,
            max_seconds=cfg.max_segment_seconds, min_seconds=cfg.min_segment_seconds,
        )
        store.save_segments(cfg, video.video_id, segments, duration)
        video.duration = round(duration, 2)

    done = store.load_labels(cfg, video.video_id)
    todo = [s for s in segments if s.index not in done]
    if todo:
        captioner = get_captioner(cfg)
        aggregator = get_aggregator(cfg)
        for seg in todo:
            _caption_segment(cfg, video, video_path, seg, captioner, aggregator)
    return len(segments)


def _caption_segment(cfg, video, video_path, seg, captioner, aggregator) -> None:
    """Steps 2-4 for one segment: sample -> caption frames -> aggregate -> checkpoint."""
    times = frame_times(seg.start, seg.end, cfg.frame_sample_fps, cfg.max_frames_per_segment)
    out_dir = _segment_dir(cfg, video, seg.index)
    captions = captioner.caption_frames(video_path, times, out_dir)
    label = aggregator.aggregate(captions, num_tags=cfg.num_tags)
    store.save_label(cfg, video.video_id, seg.index, label)


def _todo(manifest: Manifest, limit: Optional[int]) -> List[Video]:
    todo = manifest.by_status(mf.PENDING, mf.SEGMENTED, mf.FAILED)
    return todo[:limit] if limit is not None else todo


def caption(
    cfg: CaptionConfig,
    manifest: Manifest,
    *,
    limit: Optional[int] = None,
    workers: Optional[int] = None,
    gpus=None,
) -> Manifest:
    """Caption all processable videos, in parallel across GPUs. Failures recorded.

    One bad video never kills the batch (its status becomes ``failed`` and it's
    skipped); the manifest is checkpointed after each video.
    """
    devices = _expand(resolve_devices(gpus if gpus is not None else cfg.gpus), cfg.workers_per_gpu)
    n_workers = workers if workers is not None else len(devices)
    todo = _todo(manifest, limit)
    if not todo:
        print("Nothing to caption.")
        return manifest
    return _run_caption(cfg, manifest, todo, workers=max(1, n_workers), devices=devices)


def _run_caption(cfg, manifest, todo, *, workers, devices) -> Manifest:
    """Fan ``todo`` videos out over a thread pool, round-robin across ``devices``."""
    total = len(todo)
    lock = threading.Lock()
    counter = {"done": 0}

    def work(idx: int, video: Video):
        device = devices[idx % len(devices)]
        clip_cfg = _clone_for_device(cfg, device)
        try:
            n_segments = process_video(clip_cfg, video)
            return video.video_id, True, None, n_segments, device
        except Exception as exc:  # noqa: BLE001 -- isolate per-video failure
            return video.video_id, False, f"{type(exc).__name__}: {exc}", None, device

    print(f"Captioning {total} videos on {workers} worker(s) across devices {devices}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i, v) for i, v in enumerate(todo)]
        for fut in as_completed(futures):
            vid, ok, error, n_segments, device = fut.result()
            with lock:  # serialize manifest mutation + checkpoint
                _record(manifest, cfg, vid, ok, error, n_segments)
                counter["done"] += 1
                tag = f"gpu{device}" if device is not None else "cpu"
                print(f"[{counter['done']}/{total}] {tag} {vid}: {'ok' if ok else 'FAILED ' + error}")
    return manifest


def _record(manifest, cfg, video_id, ok, error, n_segments) -> None:
    """Fold one worker's result into the manifest and checkpoint it.

    ``process_video`` mutates the shared ``Video`` object (it sets ``duration``
    during segmentation), so only status/segment-count need setting here.
    """
    video = manifest.get(video_id)
    if ok:
        video.status = mf.DONE
        video.n_segments = n_segments
        video.error = None
    else:
        video.status = mf.FAILED
        video.error = error
    manifest.save(cfg.manifest_path)


# ---------------------------------------------------------------------------
# Downstream aggregation stages (Steps 5 / 5.5 / 6-dataset).
# ---------------------------------------------------------------------------

def index(cfg: CaptionConfig, manifest: Manifest) -> dict:
    """Build the search index (Step 5) from every captioned row."""
    summary = build_index(cfg, store.iter_rows(cfg, manifest))
    print(f"index: {summary['n_rows']} rows across {summary['n_videos']} videos -> {cfg.index_dir}")
    return summary


def windows(cfg: CaptionConfig, manifest: Manifest) -> dict:
    """Group captioned segments into training windows (Step 5.5) and persist them."""
    wins = build_windows(cfg, manifest)
    path = write_windows(wins, cfg.finetune_dir / "windows.json")
    n_labels = sum(len(w.entries) for w in wins)
    print(f"windows: {len(wins)} windows ({n_labels} labels) -> {path}")
    return {"n_windows": len(wins), "n_labels": n_labels, "path": str(path)}


def finetune_data(cfg: CaptionConfig, manifest: Manifest) -> dict:
    """Build the Step 6 bootstrap fine-tune dataset from the windows."""
    from .finetune import build_dataset

    wins = build_windows(cfg, manifest)
    path = build_dataset(cfg, wins)
    print(f"fine-tune dataset: {len(wins)} examples -> {path}")
    return {"n_examples": len(wins), "path": str(path)}


def run_all(
    cfg: CaptionConfig,
    *,
    limit: Optional[int] = None,
    workers: Optional[int] = None,
    gpus=None,
) -> dict:
    """Full flow from a scanned manifest to index + windows + fine-tune dataset."""
    manifest = Manifest.load(cfg.manifest_path)
    caption(cfg, manifest, limit=limit, workers=workers, gpus=gpus)
    idx = index(cfg, manifest)
    win = windows(cfg, manifest)
    ft = finetune_data(cfg, manifest)
    return {"index": idx, "windows": win, "finetune": ft}
