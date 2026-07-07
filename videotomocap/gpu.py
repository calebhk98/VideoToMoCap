"""GPU discovery + worker-count resolution for the parallel HMR runner.

Kept dependency-free: we shell out to ``nvidia-smi`` rather than import torch
(the core package stays light; torch only lives in the backend subprocess envs).
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Callable, Dict, List, Optional

# Conservative fallback per-worker VRAM footprint (MiB) when we can't calibrate
# and none is configured. Batch-1 SMPL(-X)/hand inference is typically 4-8 GB.
DEFAULT_VRAM_PER_WORKER_MB = 6000


def detect_gpus() -> List[str]:
    """Return available GPU ids as strings, e.g. ['0','1'].

    Honors an existing ``CUDA_VISIBLE_DEVICES`` (so scheduler/container limits are
    respected), otherwise asks ``nvidia-smi``. Returns [] when no GPU is found --
    the caller then runs on CPU (or the backend picks its own device).
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        return [d.strip() for d in visible.split(",") if d.strip()]

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def resolve_devices(gpus) -> List[Optional[str]]:
    """Turn a config ``gpus`` value into a concrete device list.

    Accepts: 'auto' / ['auto'] -> detect; an explicit list ['0','1']; or empty ->
    a single ``None`` slot meaning "inherit the ambient device / let the backend
    choose". Never returns an empty list, so round-robin assignment is always safe.
    """
    if isinstance(gpus, str):
        gpus = [gpus]
    if gpus and any(str(g).lower() == "auto" for g in gpus):
        detected = detect_gpus()
        return detected or [None]
    if gpus:
        return [str(g) for g in gpus]
    return [None]


def resolve_workers(workers: Optional[int], devices: List[Optional[str]], workers_per_gpu: int) -> int:
    """Pick a worker count: explicit ``workers`` wins, else devices * per-GPU.

    With real GPUs this fills each card ``workers_per_gpu`` deep (raise it to
    overlap one clip's GPU phase with another's decode/IO and saturate a fast
    card). With no GPU detected it falls back to ``workers_per_gpu`` CPU workers.
    """
    if workers is not None and workers > 0:
        return workers
    n_devices = len([d for d in devices if d is not None]) or 1
    return max(1, n_devices * max(1, workers_per_gpu))


# ---------------------------------------------------------------------------
# Dynamic / VRAM-aware sizing: "how many clips fit on each card"
# ---------------------------------------------------------------------------

def gpu_free_memory() -> Dict[str, int]:
    """Return {gpu_id: free MiB} from nvidia-smi. {} if it can't be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    free: Dict[str, int] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            free[parts[0]] = int(parts[1])
    return free


def plan_workers_per_gpu(free_by_dev: Dict[str, int], est_mb: int, headroom_mb: int, cap: int) -> Dict[str, int]:
    """How many workers fit on each GPU: (free - headroom) // per-worker estimate.

    Always at least 1 per known GPU, never more than ``cap``. ``headroom_mb`` is
    left free as a safety margin so we don't drive a card to the edge of OOM.
    """
    est = max(1, int(est_mb))
    out: Dict[str, int] = {}
    for dev, free in free_by_dev.items():
        usable = max(0, int(free) - int(headroom_mb))
        out[dev] = max(1, min(int(cap), usable // est))
    return out


def distribute_workers(total: int, devices: List[Optional[str]]) -> Dict[Optional[str], int]:
    """Spread ``total`` workers as evenly as possible across ``devices``."""
    plan: Dict[Optional[str], int] = {d: 0 for d in devices}
    for i in range(max(0, total)):
        plan[devices[i % len(devices)]] += 1
    return plan


def expand_devices(devices: List[Optional[str]], plan: Dict) -> List[Optional[str]]:
    """Repeat each device by its planned worker count -> a flat assignment list.

    e.g. devices ['0','1'] + plan {'0':2,'1':1} -> ['0','0','1'] (3 workers, gpu0
    packed 2 deep). Never empty, so the pool always has at least one slot.
    """
    out: List[Optional[str]] = []
    for d in devices:
        out.extend([d] * max(0, int(plan.get(d, 1))))
    return out or list(devices[:1])


def measure_peak_vram(
    run_fn: Callable[[], None],
    device: Optional[str],
    *,
    free_mem_fn: Optional[Callable[[], Dict[str, int]]] = None,
    poll: float = 0.5,
) -> Optional[int]:
    """Run ``run_fn`` while sampling ``device`` free memory; return peak MiB used.

    This is the auto-calibration primitive: process one clip, watch how far free
    VRAM dips, and that dip is roughly one worker's footprint. Returns None if the
    device's memory can't be read (no GPU / CPU run), in which case the caller
    falls back to a configured estimate or the default.
    """
    free_mem_fn = free_mem_fn or gpu_free_memory
    baseline = free_mem_fn().get(device) if device is not None else None
    if baseline is None:
        run_fn()
        return None

    lowest = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            cur = free_mem_fn().get(device)
            if cur is not None and cur < lowest[0]:
                lowest[0] = cur
            stop.wait(poll)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        run_fn()
    finally:
        stop.set()
        sampler.join()
    return max(0, baseline - lowest[0])
