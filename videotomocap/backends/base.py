"""Backend interface + shared helpers for converting native output to SMPL-72."""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import PipelineConfig
from ..pose import SMPL_BODY_POSE_DIM, SMPL_POSE_DIM, SmplMotion


class BackendError(RuntimeError):
    """Raised for backend setup/execution problems (missing repo, subprocess
    failure, unparseable/unexpected output shape) -- always with actionable context."""


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
        env = self._subprocess_env()
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)
        except FileNotFoundError as exc:
            raise BackendError(
                f"Could not launch {self.name}: {exc}. Is `{self.cfg.backend_python}` on PATH "
                f"and backend_repo set to the cloned repo?"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise BackendError(f"{self.name} exited with status {exc.returncode} on {cmd}") from exc

    def _subprocess_env(self) -> Optional[dict]:
        """Env for the backend subprocess -- pins its GPU when the parallel runner
        assigned one (so N workers land on N different cards)."""
        device = getattr(self.cfg, "cuda_device", None)
        if device is None:
            return None  # inherit the parent env unchanged
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        return env

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
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle. Pure NumPy.

    Goes via a quaternion using Shepperd's method (pick the largest of trace /
    the three diagonals for the divisor) rather than reading the skew-symmetric
    part directly.  The naive skew approach silently returns a zero vector for
    exact 180-degree rotations -- there the matrix is symmetric so the skew part
    vanishes -- which corrupts any joint that flips a full half-turn.  This route
    is stable across the whole range [0, pi].
    """
    m = np.asarray(mat, dtype=np.float64)
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22

    # Four numerically-distinct branches; each is well-conditioned in its region.
    # Each b* is the quaternion (w,x,y,z) computed with that branch's divisor.
    s0 = np.sqrt(np.maximum(trace + 1.0, 1e-12)) * 2.0
    b0 = (0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0)
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2.0
    b1 = ((m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1)
    s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2.0
    b2 = ((m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2)
    s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2.0
    b3 = ((m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3)

    cond0 = trace > 0
    cond1 = (~cond0) & (m00 >= m11) & (m00 >= m22)
    cond2 = (~cond0) & (~cond1) & (m11 >= m22)
    conds = [cond0, cond1, cond2]
    quat = np.stack(
        [np.select(conds, [b0[i], b1[i], b2[i]], default=b3[i]) for i in range(4)],
        axis=-1,
    )
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-12
    quat = np.where(quat[..., :1] < 0, -quat, quat)  # canonical hemisphere (w >= 0)

    w = quat[..., 0]
    xyz = quat[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)  # in [0, pi]
    small = sin_half < 1e-8
    axis = np.where(small[..., None], np.zeros_like(xyz), xyz / (sin_half[..., None] + 1e-12))
    return (axis * angle[..., None]).astype(np.float32)


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix (Rodrigues). Pure NumPy."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = np.where(small, 0.0, aa / np.where(small, 1.0, theta))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(aa.shape[:-1] + (3, 3))
    eye = np.eye(3)
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    return eye + s * K + (1.0 - c) * (K @ K)


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


def assemble_hand(hand_pose) -> np.ndarray:
    """Normalize a MANO hand block to (T, 45) axis-angle.

    SMPL-X stores hands as MANO's own per-joint articulation (15 joints x 3),
    which drops straight into our SmplMotion hand fields -- provided the upstream
    method used full pose, not PCA (``use_pca=False``). Accepts axis-angle,
    rotation matrices, or 6D via the generic converter.
    """
    return to_axis_angle(np.asarray(hand_pose), 15)


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
