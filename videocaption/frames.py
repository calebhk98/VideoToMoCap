"""Step 2 -- sample frames from a caption segment.

The *which timestamps* decision is pure arithmetic (unit-tested); the *decode
pixels to disk* part is a lazy shell over ffmpeg / decord, so the module imports
without either and the GPU-free tests never touch a decoder.

A 1-2 min segment at 1 fps is 60-120 frames -- small enough to batch through the
captioner efficiently. ``max_frames`` thins the sampling evenly so a long segment
at a high rate can't blow past a safe batch size.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


class FrameError(RuntimeError):
    """Raised when frames cannot be decoded (missing ffmpeg/decord or a bad file)."""


def frame_times(start: float, end: float, sample_fps: float, max_frames: int) -> List[float]:
    """Timestamps (seconds) to sample within ``[start, end)`` -- pure.

    Samples at ``sample_fps`` starting half a step in (so a frame lands in the
    middle of each sampling interval, not on the cut), capped at ``max_frames``
    by widening the step evenly. Always returns at least one timestamp.
    """
    dur = max(0.0, end - start)
    if dur <= 0 or sample_fps <= 0:
        return [start]
    n = min(max_frames, max(1, int(dur * sample_fps)))
    step = dur / n
    return [round(start + (i + 0.5) * step, 3) for i in range(n)]


def extract_frames(video_path: Path, times: List[float], out_dir: Path, *, extractor: str = "ffmpeg") -> List[Path]:
    """Decode the frames at ``times`` into ``out_dir`` as jpgs; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if extractor == "decord":
        return _extract_decord(video_path, times, out_dir)
    return _extract_ffmpeg(video_path, times, out_dir)


def _frame_path(out_dir: Path, i: int) -> Path:
    return out_dir / f"frame_{i:04d}.jpg"


def _extract_ffmpeg(video_path: Path, times: List[float], out_dir: Path) -> List[Path]:
    """One accurate seek + single-frame grab per timestamp (robust across formats)."""
    paths: List[Path] = []
    for i, t in enumerate(times):
        dst = _frame_path(out_dir, i)
        cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", str(video_path),
               "-frames:v", "1", "-q:v", "3", str(dst)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise FrameError("frame extraction needs ffmpeg on PATH (or set frame_extractor: decord).") from exc
        except subprocess.CalledProcessError as exc:
            raise FrameError(f"ffmpeg failed to grab frame at {t:.3f}s of {video_path}") from exc
        paths.append(dst)
    return paths


def _extract_decord(video_path: Path, times: List[float], out_dir: Path) -> List[Path]:
    """Batch-decode via decord + Pillow. Lazy import of both."""
    try:
        import decord
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise FrameError("frame_extractor: decord needs `decord` and `pillow` installed.") from exc
    reader = decord.VideoReader(str(video_path))
    fps = reader.get_avg_fps() or 1.0
    idxs = [min(len(reader) - 1, max(0, int(round(t * fps)))) for t in times]
    batch = reader.get_batch(idxs).asnumpy()
    paths: List[Path] = []
    for i, arr in enumerate(batch):
        dst = _frame_path(out_dir, i)
        Image.fromarray(arr).save(dst, quality=90)
        paths.append(dst)
    return paths
