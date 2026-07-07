"""Backend interface + shared helpers for converting native output to SMPL-72."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import PipelineConfig
from ..pose import SMPL_BODY_POSE_DIM, SMPL_POSE_DIM, SmplMotion


class BackendError(RuntimeError):
    pass


class HMRBackend(ABC):
    """Recover SMPL motion from a single video clip.

    Concrete backends shell out to the upstream tool, then parse its output into
    a :class:`~videotomocap.pose.SmplMotion` (always SMPL-72 axis-angle so the
    rest of the pipeline is uniform).
    """

    name = "base"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        """Process ``video_path`` and return the recovered motion.

        ``out_dir`` is a per-clip scratch directory the backend may fill with its
        native artifacts.  ``static`` hints that the camera is fixed (skip VO).
        """

    # -- helpers shared by subprocess-based backends --------------------
    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
        except FileNotFoundError as exc:
            raise BackendError(
                f"Could not launch {self.name}: {exc}. Is `{self.cfg.backend_python}` on PATH "
                f"and backend_repo set to the cloned repo?"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise BackendError(f"{self.name} exited with status {exc.returncode} on {cmd}") from exc

    def _require_repo(self) -> Path:
        repo = self.cfg.backend_repo
        if repo is None or not Path(repo).exists():
            raise BackendError(
                f"{self.name} needs `backend_repo` pointing at the cloned "
                f"{self.name.upper()} checkout (got {repo!r})."
            )
        return Path(repo)


# ---------------------------------------------------------------------------
# Rotation / SMPL-family conversion helpers
# ---------------------------------------------------------------------------

def _matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle. Pure NumPy."""
    m = np.asarray(mat, dtype=np.float64)
    trace = np.trace(m, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    # axis from the skew-symmetric part
    rx = m[..., 2, 1] - m[..., 1, 2]
    ry = m[..., 0, 2] - m[..., 2, 0]
    rz = m[..., 1, 0] - m[..., 0, 1]
    axis = np.stack([rx, ry, rz], axis=-1)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    small = norm < 1e-8
    axis = np.where(small, np.zeros_like(axis), axis / np.where(small, 1.0, norm))
    return (axis * angle[..., None]).astype(np.float32)


def _rot6d_to_axis_angle(r6: np.ndarray) -> np.ndarray:
    """(..., 6) 6D rotation representation -> (..., 3) axis-angle."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)  # columns
    return _matrix_to_axis_angle(mat)


def to_axis_angle(arr: np.ndarray, njoints: int) -> np.ndarray:
    """Normalize a per-joint rotation block to (T, njoints*3) axis-angle.

    Accepts axis-angle (T, njoints*3), rotation matrices (T, njoints, 3, 3), or
    6D (T, njoints, 6) / (T, njoints*6).
    """
    arr = np.asarray(arr)
    t = arr.shape[0]
    if arr.ndim == 2 and arr.shape[1] == njoints * 3:
        return arr.astype(np.float32)                                   # already axis-angle
    if arr.ndim == 4 and arr.shape[1:] == (njoints, 3, 3):
        return _matrix_to_axis_angle(arr).reshape(t, njoints * 3)
    if arr.ndim == 2 and arr.shape[1] == njoints * 3 * 3:
        return _matrix_to_axis_angle(arr.reshape(t, njoints, 3, 3)).reshape(t, njoints * 3)
    if arr.ndim == 3 and arr.shape[1:] == (njoints, 6):
        return _rot6d_to_axis_angle(arr).reshape(t, njoints * 3)
    if arr.ndim == 2 and arr.shape[1] == njoints * 6:
        return _rot6d_to_axis_angle(arr.reshape(t, njoints, 6)).reshape(t, njoints * 3)
    raise BackendError(f"Cannot interpret rotation block of shape {arr.shape} for {njoints} joints")


def assemble_smpl72(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """Combine global_orient + body_pose into full SMPL-72 axis-angle.

    Handles the SMPL-X case (body_pose 63 = 21 joints) by zero-padding the two
    SMPL hand/wrist joints (joints 22-23) to neutral -- consistent with the known
    limitation that these body-only HMR methods do not recover articulated hands.
    """
    go = to_axis_angle(global_orient, 1)                    # (T,3)
    bp_dim = body_pose.shape[1] if body_pose.ndim == 2 else body_pose.shape[1] * 3
    if bp_dim == SMPL_BODY_POSE_DIM:                         # 69 -> SMPL body directly
        bp = to_axis_angle(body_pose, 23)
    elif bp_dim == 63:                                       # 63 -> SMPL-X body (21 joints)
        bp21 = to_axis_angle(body_pose, 21)
        pad = np.zeros((bp21.shape[0], 6), np.float32)       # neutral hands (joints 22,23)
        bp = np.concatenate([bp21, pad], axis=1)
    else:
        raise BackendError(f"Unexpected body_pose width {bp_dim}; expected 63 (SMPL-X) or 69 (SMPL)")
    poses = np.concatenate([go.reshape(-1, 3), bp], axis=1).astype(np.float32)
    assert poses.shape[1] == SMPL_POSE_DIM, poses.shape
    return poses
