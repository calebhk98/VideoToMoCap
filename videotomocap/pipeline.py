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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import ingest
from .backends import get_backend
from .config import PipelineConfig
from .dataset import build_dataset
from . import gpu
from .gpu import (
    DEFAULT_VRAM_PER_WORKER_MB,
    distribute_workers,
    expand_devices,
    measure_peak_vram,
    plan_workers_per_gpu,
    resolve_devices,
)
from .ingest import Clip, Manifest
from .mirror import decide_mirrored, handedness_score, mirror_motion
from .pose import SMPL_NJOINTS, SmplMotion, anonymize, resample_fps
from .quality import assess_quality, has_hard_issue


# Multi-person plumbing lives in `multiperson` to keep this file focused; alias to
# the historic private names so the single-subject code paths read unchanged.
from .multiperson import (  # noqa: E402
    apply_result as _apply_result,
    build_per_person as _build_per_person,
    consent_gate as _consent_gate,
    iter_units as _iter_units,
    pose_path as _pose_path,
    set_mirror as _set_mirror,
)


def _static_hint(cfg: PipelineConfig, clip: Clip) -> bool:
    """Whether HMR can skip visual odometry: a configured static camera, or one
    auto-detected as static by the video pre-analysis."""
    return clip.camera in set(cfg.static_cameras) or clip.camera_motion == "static"


def analyze_videos(cfg: PipelineConfig, manifest: Manifest, *, analyze_fn=None) -> dict:
    """Pre-HMR raw-video pass: skip empty clips and/or tag camera motion.

    Samples each PENDING clip's pixels (OpenCV). ``skip_empty`` excludes clips
    with no activity before they hit the GPU; ``auto_camera_motion`` tags each as
    static/moving so HMR can skip visual odometry on locked-off cameras. No-op
    unless one is enabled; degrades to a warning (not an error) if OpenCV is
    missing.
    """
    if not cfg.skip_empty and cfg.auto_camera_motion == "off":
        return {"skipped_empty": 0, "static": 0, "moving": 0}

    from .video import VideoError, analyze_video
    probe = analyze_fn or analyze_video
    skipped = static = moving = 0
    for clip in manifest.by_status(ingest.PENDING):
        try:
            info = probe(cfg.footage_root / clip.rel_path)
        except VideoError as exc:
            print(f"video pre-analysis disabled: {exc}")
            return {"skipped_empty": skipped, "static": static, "moving": moving}
        if cfg.skip_empty and info["activity"] < cfg.empty_activity_threshold:
            clip.status = ingest.EXCLUDED
            clip.note = "empty (no activity)"
            skipped += 1
            continue
        if cfg.auto_camera_motion == "flag":
            is_static = info["camera_motion"] < cfg.camera_motion_threshold
            clip.camera_motion = "static" if is_static else "moving"
            static += int(is_static)
            moving += int(not is_static)
    manifest.save(cfg.manifest_path)
    print(f"video pre-analysis: {skipped} empty skipped, {static} static / {moving} moving cameras")
    return {"skipped_empty": skipped, "static": static, "moving": moving}


def _finalize(cfg: PipelineConfig, clip: Clip, motion, video: Path):
    """Resample -> optional refine -> anonymize -> joint-valid mask. Shared path."""
    motion = resample_fps(motion, cfg.target_fps)
    if cfg.refine:
        from .refine import refine_motion  # optional post-processing; keep import lazy
        motion = refine_motion(motion, method=cfg.refine_method, kappa=cfg.confidence_kappa, cfg=cfg, video=video)
    anon = anonymize(motion, drop_shape=cfg.drop_shape, keep_translation=cfg.keep_translation)
    anon.joint_valid = _joint_valid_mask(cfg, clip, anon)  # label unreliable joints (if any)
    return anon


def _compute_and_save(cfg: PipelineConfig, clip: Clip):
    """Run HMR + anonymize for one clip and write its pose npz(s).

    Returns ``n_frames`` (int) in single-subject mode -- the original contract --
    or a ``List[Track]`` in ``multi_person`` mode (one per recovered person).
    Builds its own backend from ``cfg`` so it is safe to call from many threads.
    """
    backend = get_backend(cfg)
    video = cfg.footage_root / clip.rel_path
    hmr_out = cfg.hmr_dir / clip.clip_id
    cfg.pose_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.multi_person:
        motion = backend.run(video, hmr_out, static=_static_hint(cfg, clip))
        anon = _finalize(cfg, clip, motion, video)
        anon.save_npz(_pose_path(cfg, clip))
        return anon.n_frames
    return _save_tracks(cfg, clip, backend, video, hmr_out)


def _save_tracks(cfg: PipelineConfig, clip: Clip, backend, video: Path, hmr_out: Path):
    """Multi-person: recover every person, write one anonymized pose npz per track,
    and retain each track's betas in the consent-gated identity store (for shape
    assignment). The pose npz stays shape-neutral -- betas never go into it."""
    from . import identity
    from .ingest import Track

    tracks = backend.run_tracks(video, hmr_out, static=_static_hint(cfg, clip))
    records = []
    for i, motion in enumerate(tracks):
        track_id = f"p{i}"
        unit_id = f"{clip.clip_id}__{track_id}"
        raw_betas = motion.betas  # capture BEFORE anonymize, for identity clustering
        anon = _finalize(cfg, clip, motion, video)
        pose_rel = f"{unit_id}.npz"
        anon.save_npz(cfg.pose_dir / pose_rel)
        if cfg.person_assignment == "shape":
            identity.save_track_betas(cfg, unit_id, raw_betas)
        records.append(Track(track_id=track_id, n_frames=anon.n_frames, pose_rel=pose_rel))
    return records


def _joint_valid_mask(cfg: PipelineConfig, clip: Clip, motion):
    """(24,) bool mask, False for unreliable joints. None if all reliable.

    Combines the camera occlusion tags (``clip.unreliable_joints``) with, when
    ``auto_occlusion`` is on, joints detected as frozen in this clip's motion.
    """
    unreliable = set(clip.unreliable_joints)
    if cfg.auto_occlusion == "flag":
        from .regions import infer_static_joints
        unreliable.update(infer_static_joints(motion))
    if not unreliable:
        return None
    mask = np.ones(SMPL_NJOINTS, dtype=bool)
    mask[sorted(unreliable)] = False
    return mask


def process_clip(cfg: PipelineConfig, manifest: Manifest, clip: Clip, backend=None) -> None:
    """Process one clip and update its manifest state (sequential path)."""
    _apply_result(clip, _compute_and_save(cfg, clip))
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
    free_mem_fn=None,
) -> Manifest:
    """Process all processable clips, in parallel across GPUs. Failures recorded.

    Devices come from ``gpus`` (or ``cfg.gpus``; 'auto' detects them). The number
    of clips per GPU comes from ``cfg.workers_per_gpu`` -- an int, or ``'auto'``
    to size it from each card's free VRAM (calibrating on the first clip if no
    ``vram_per_worker_mb`` is set). ``workers`` overrides the total directly. One
    bad clip never kills the batch; the manifest is checkpointed after each clip.
    """
    analyze_videos(cfg, manifest)  # skip empties / tag camera motion before the GPU stage
    devices = resolve_devices(gpus if gpus is not None else cfg.gpus)
    todo = _todo(manifest, limit)
    plan, todo = _plan_workers(cfg, manifest, devices, todo, workers, free_mem_fn)
    expanded = expand_devices(devices, plan)
    return _run_hmr(cfg, manifest, todo, workers=max(1, len(expanded)), devices=expanded)


def _plan_workers(cfg, manifest, devices, todo, workers_override, free_mem_fn):
    """Return ({device: worker_count}, remaining_todo).

    Auto mode may consume the first clip to calibrate VRAM, so it also returns the
    (possibly shortened) todo list.
    """
    explicit = workers_override if workers_override is not None else (
        cfg.workers if isinstance(cfg.workers, int) else None
    )
    if explicit:
        return distribute_workers(explicit, devices), todo

    wpg = cfg.workers_per_gpu
    if isinstance(wpg, str) and wpg.lower() == "auto":
        return _auto_plan(cfg, manifest, devices, todo, free_mem_fn)
    return {d: max(1, int(wpg)) for d in devices}, todo


def _auto_plan(cfg, manifest, devices, todo, free_mem_fn):
    """Size workers-per-GPU from free VRAM; calibrate on clip 0 if no estimate."""
    probe = free_mem_fn or gpu.gpu_free_memory
    real = [d for d in devices if d is not None]
    if not real or not probe():
        return {d: 1 for d in devices}, todo  # no GPU info -> one clip per device

    est = cfg.vram_per_worker_mb
    remaining = todo
    if est is None and todo:
        est, remaining = _calibrate(cfg, manifest, real[0], todo, probe)
    est = est or DEFAULT_VRAM_PER_WORKER_MB

    free = probe()  # re-read after the calibration clip released its memory
    fitted = plan_workers_per_gpu(free, est, cfg.vram_headroom_mb, cfg.max_workers_per_gpu)
    plan = {d: (fitted.get(d, 1) if d is not None else 1) for d in devices}
    print(f"Auto workers/GPU from ~{est} MiB/clip, {cfg.vram_headroom_mb} MiB headroom: {plan}")
    return plan, remaining


def _calibrate(cfg, manifest, device, todo, probe):
    """Process clip 0 alone while sampling VRAM; return (peak_mb_or_None, rest)."""
    clip = todo[0]
    est = measure_peak_vram(lambda: _process_and_record(cfg, manifest, clip, device), device, free_mem_fn=probe)
    print(f"Calibrated ~{est} MiB/clip on gpu{device}" if est else "Calibration unavailable; using estimate")
    return (est if est and est > 0 else None), todo[1:]


def _process_and_record(cfg: PipelineConfig, manifest: Manifest, clip: Clip, device: Optional[str]) -> None:
    """Compute + save one clip and update its manifest state (single-threaded)."""
    try:
        _apply_result(clip, _compute_and_save(_clone_for_device(cfg, device), clip))
        clip.status = ingest.POSE_DONE
        clip.error = None
    except Exception as exc:  # noqa: BLE001 -- isolate per-clip failure
        clip.status = ingest.FAILED
        clip.error = f"{type(exc).__name__}: {exc}"
    manifest.save(cfg.manifest_path)


def _clone_for_device(cfg: PipelineConfig, device: Optional[str]) -> PipelineConfig:
    """Shallow config copy pinned to one GPU (exported to the backend subprocess)."""
    c = copy.copy(cfg)
    c.cuda_device = device
    return c


def _run_hmr(
    cfg: PipelineConfig,
    manifest: Manifest,
    todo: List[Clip],
    *,
    workers: int,
    devices: List[Optional[str]],
) -> Manifest:
    """Fan ``todo`` clips out over a thread pool, round-robin across ``devices``.

    Threads (not processes) suffice: each clip's heavy work is in the backend
    subprocess, which releases the GIL, so N threads drive N concurrent
    subprocesses. ``devices`` is already expanded (a card repeated N times packs
    N clips onto it), so len(devices) == workers.
    """
    total = len(todo)
    lock = threading.Lock()
    counter = {"done": 0}

    def work(idx: int, clip: Clip):
        device = devices[idx % len(devices)]
        clip_cfg = _clone_for_device(cfg, device)
        try:
            result = _compute_and_save(clip_cfg, clip)
            return clip.clip_id, True, None, result, device
        except Exception as exc:  # noqa: BLE001 -- isolate per-clip failure
            return clip.clip_id, False, f"{type(exc).__name__}: {exc}", None, device

    print(f"Processing {total} clips on {workers} worker(s) across devices {devices}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i, clip) for i, clip in enumerate(todo)]
        for fut in as_completed(futures):
            clip_id, ok, error, result, device = fut.result()
            with lock:  # serialize manifest mutation + checkpoint
                clip = manifest.get(clip_id)
                if ok:
                    clip.status = ingest.POSE_DONE
                    _apply_result(clip, result)
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


def detect_mirroring(cfg: PipelineConfig, manifest: Manifest) -> dict:
    """Corpus-relative left/right mirror pass over the recovered clips.

    Scores every clip's handedness, takes the corpus consensus as the subject's
    true dominant side, flags the confident disagreements, and -- in 'correct'
    mode -- flips their pose npz in place so the dataset handedness is consistent.
    No-op when ``cfg.auto_mirror == 'off'``.
    """
    if cfg.auto_mirror == "off":
        return {"flagged": 0, "corrected": 0}

    units = list(_iter_units(cfg, manifest))
    scores = {u.unit_id: handedness_score(SmplMotion.load_npz(u.pose_path)) for u in units}
    # Consensus is per-PERSON in multi mode (a mixed-handed family has no single
    # corpus consensus); all units share one corpus group in single mode.
    groups: dict = defaultdict(dict)
    for u in units:
        key = u.person_id if (cfg.multi_person and u.person_id) else "__corpus__"
        groups[key][u.unit_id] = scores[u.unit_id]
    flags: dict = {}
    for grp in groups.values():
        flags.update(decide_mirrored(grp, margin=cfg.mirror_margin))

    corrected = 0
    for u in units:
        flagged = flags.get(u.unit_id, False)
        _set_mirror(u, suspected=flagged)
        if flagged and cfg.auto_mirror == "correct":
            mirror_motion(SmplMotion.load_npz(u.pose_path)).save_npz(u.pose_path)
            _set_mirror(u, corrected=True)
            corrected += 1
    manifest.save(cfg.manifest_path)

    n_flagged = sum(1 for v in flags.values() if v)
    print(f"auto_mirror={cfg.auto_mirror}: {n_flagged} suspected mirrored, {corrected} corrected")
    return {"flagged": n_flagged, "corrected": corrected}


def assess_quality_pass(cfg: PipelineConfig, manifest: Manifest) -> dict:
    """Score each recovered clip's plausibility and record the findings.

    Sets ``quality_issues`` (+ ``low_quality`` for hard failures) on every
    POSE_DONE clip. No-op when ``cfg.quality_filter == 'off'``. Dropping happens
    in ``build`` (exclude mode); this pass only annotates.
    """
    if cfg.quality_filter == "off":
        return {"flagged": 0, "hard": 0}

    # Assess per motion unit and record at clip level (union across a clip's
    # tracks in multi mode), so quality_filter=exclude drops a clip with any hard
    # track. Reset first so re-runs don't accumulate stale issues.
    for clip in manifest.by_status(ingest.POSE_DONE):
        clip.quality_issues = []
        clip.low_quality = False
    for u in _iter_units(cfg, manifest):
        issues = assess_quality(
            SmplMotion.load_npz(u.pose_path),
            max_speed_ms=cfg.quality_max_speed_ms,
            max_joint_step=cfg.quality_max_joint_step,
            min_motion=cfg.quality_min_motion,
        )
        if issues:
            u.clip.quality_issues = sorted(set(u.clip.quality_issues) | set(issues))
        if has_hard_issue(issues):
            u.clip.low_quality = True
    manifest.save(cfg.manifest_path)

    hard = sum(1 for c in manifest.clips if c.low_quality)
    flagged = sum(1 for c in manifest.clips if c.quality_issues)
    print(f"quality_filter={cfg.quality_filter}: {flagged} clips with issues, {hard} hard")
    return {"flagged": flagged, "hard": hard}


def cluster_actions_pass(cfg: PipelineConfig, manifest: Manifest) -> int:
    """Group clips into ``cfg.cluster_actions`` motion clusters (pseudo-labels).

    Sets ``action_cluster`` on each POSE_DONE clip. No-op when the count is <= 0.
    Returns the number of clusters actually assigned.
    """
    if cfg.cluster_actions <= 0:
        return 0
    from .cluster import clip_features, cluster_clips

    done = manifest.by_status(ingest.POSE_DONE)
    feats = {c.clip_id: clip_features(SmplMotion.load_npz(_pose_path(cfg, c)))
             for c in done if _pose_path(cfg, c).exists()}
    labels = cluster_clips(feats, cfg.cluster_actions)
    for clip in done:
        clip.action_cluster = labels.get(clip.clip_id, -1)
    manifest.save(cfg.manifest_path)
    n = len(set(labels.values()))
    print(f"cluster_actions={cfg.cluster_actions}: {len(labels)} clips into {n} clusters")
    return n


def build(cfg: PipelineConfig, manifest: Manifest):
    """Run the analysis passes, then aggregate anonymized clips into the dataset."""
    if cfg.multi_person and cfg.person_assignment != "manual":
        from . import identity
        identity.assign_people(cfg, manifest)  # cluster shapes -> person_id (before mirror/consent)

    detect_mirroring(cfg, manifest)      # no-op unless auto_mirror is set (per-person in multi)
    assess_quality_pass(cfg, manifest)   # no-op unless quality_filter is set
    cluster_actions_pass(cfg, manifest)  # no-op unless cluster_actions > 0

    units = list(_iter_units(cfg, manifest))
    if cfg.quality_filter == "exclude":
        units = [u for u in units if not u.clip.low_quality]  # drop hard-failing
    units = _consent_gate(cfg, units)

    extra = {}
    for u in units:
        entry = {}
        if u.clip.action_cluster >= 0:
            entry["action_cluster"] = u.clip.action_cluster
        if u.person_id is not None:
            entry["person_id"] = u.person_id
        if entry:
            extra[u.unit_id] = entry
    stats = build_dataset(
        [u.pose_path for u in units],
        cfg.dataset_dir,
        gender=cfg.gender,
        val_fraction=cfg.val_fraction,
        min_frames=cfg.min_clip_frames,
        extra_per_clip=extra,
    )
    if cfg.multi_person:
        _build_per_person(cfg, units)  # dataset/by_person/<id>/ for "moves like <person>"
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
    return build(cfg, manifest)  # build runs the mirror + quality passes
