"""GPU discovery + worker-count resolution for the parallel HMR runner.

Kept dependency-free: we shell out to ``nvidia-smi`` rather than import torch
(the core package stays light; torch only lives in the backend subprocess envs).
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional


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
