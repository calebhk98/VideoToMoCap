"""Lightweight raw-video analysis: is anything happening, is the camera moving.

Two pre-HMR signals computed by sampling a handful of frames and comparing them:
  * **activity** -- mean absolute frame-to-frame difference. Near zero = a static
    scene (empty security footage), so we can skip the expensive HMR entirely.
  * **camera_motion** -- median global pixel shift between frames (phase
    correlation). Small = a locked-off camera, so HMR can skip visual odometry.

This is the one place the core touches pixels, so OpenCV is imported lazily and
only when these features are enabled -- the package still installs and the whole
GPU-free path still runs without it. The frame-comparison math is separated from
decoding so it's testable on synthetic frames without a video file or cv2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class VideoError(RuntimeError):
    pass


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise VideoError(
            "skip_empty / auto_camera_motion need OpenCV. `pip install opencv-python` "
            "or turn those config options off."
        ) from exc
    return cv2


def read_frames(path, *, samples: int = 16, max_dim: int = 160) -> List[np.ndarray]:
    """Decode ``samples`` evenly-spaced grayscale frames, downscaled to ``max_dim``."""
    cv2 = _load_cv2()
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            return []
        idxs = np.linspace(0, total - 1, min(samples, total)).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = max_dim / max(gray.shape)
            if scale < 1.0:
                gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
            frames.append(gray.astype(np.float32))
        return frames
    finally:
        cap.release()


def activity_from_frames(frames: List[np.ndarray]) -> float:
    """Mean absolute inter-frame difference, normalized to ~[0,1]. 0 = static."""
    if len(frames) < 2:
        return 0.0
    diffs = [np.abs(frames[i] - frames[i - 1]).mean() for i in range(1, len(frames))]
    return float(np.mean(diffs) / 255.0)


def _phase_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Magnitude (pixels) of the global translation between two frames.

    Cross-power spectrum / phase correlation: the peak of the inverse FFT of the
    normalized cross-spectrum sits at the translation offset. Robust to lighting
    since only phase is used.
    """
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-8
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = a.shape
    dy = peak[0] if peak[0] <= h // 2 else peak[0] - h  # unwrap to signed shift
    dx = peak[1] if peak[1] <= w // 2 else peak[1] - w
    return float(np.hypot(dx, dy))


def camera_motion_from_frames(frames: List[np.ndarray]) -> float:
    """Median global pixel shift between consecutive frames. ~0 = static camera."""
    if len(frames) < 2:
        return 0.0
    shifts = [_phase_shift(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    return float(np.median(shifts))


def analyze_video(path, *, frames: Optional[List[np.ndarray]] = None, samples: int = 16) -> Dict[str, float]:
    """Return {'activity', 'camera_motion', 'n_sampled'} for a clip.

    ``frames`` may be passed in (tests); otherwise the video is decoded.
    """
    frames = frames if frames is not None else read_frames(Path(path), samples=samples)
    return {
        "activity": activity_from_frames(frames),
        "camera_motion": camera_motion_from_frames(frames),
        "n_sampled": len(frames),
    }
