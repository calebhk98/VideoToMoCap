"""SMPL(-X) motion container + shape/pose separation (the anonymization step).

Why the anonymization step exists
----------------------------------
SMPL-family models factor a human into **shape** (``betas`` -- limb lengths,
build; identity-revealing) and **pose** (per-joint rotations over time -- the
movement).  To use personal footage as motion training data *without* carrying
personal likeness forward, we keep pose and discard shape.  After this step the
motion is expressed on a canonical neutral body: the trajectory and joint angles
are yours, the body dimensions are nobody's.

Caveat we do not hide from you: motion *style* (gait, posture) is itself weakly
identifying -- that is the whole point of the project -- but the metric body
shape, which is the strongest biometric here, is dropped.

Hands
-----
The body core is always SMPL-72 axis-angle so the whole pipeline stays uniform.
Articulated hands are carried *alongside* it as optional MANO pose
(``left_hand_pose``/``right_hand_pose``, 45 = 15 finger joints x 3).  Body-only
backends (GVHMR/WHAM/TRAM) leave them ``None`` -> neutral hands; whole-body and
fusion backends fill them.  Hand pose is movement, not identity, so it is kept
through anonymization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# SMPL: 24 joints * 3 axis-angle params = 72 (global_orient[3] + body_pose[69]).
SMPL_NJOINTS = 24
SMPL_POSE_DIM = SMPL_NJOINTS * 3  # 72
SMPL_BODY_POSE_DIM = SMPL_POSE_DIM - 3  # 69

# MANO articulated hand: 15 finger joints * 3 axis-angle = 45, per hand.
MANO_POSE_DIM = 45


def _as_hand(arr, n_frames: int, name: str) -> Optional[np.ndarray]:
    """Validate/coerce an optional per-frame MANO hand block to (T, 45)."""
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (n_frames, MANO_POSE_DIM):
        raise ValueError(f"{name} must be (T,{MANO_POSE_DIM}) matching poses; got {arr.shape}")
    return arr


@dataclass
class SmplMotion:
    """A normalized motion clip: SMPL-72 body + optional MANO hands.

    All backends convert their native output into this container so the rest of
    the pipeline is backend-agnostic.
    """

    poses: np.ndarray            # (T, 72) axis-angle: [global_orient(3), body_pose(69)]
    trans: np.ndarray            # (T, 3) root translation in metres
    fps: float
    betas: Optional[np.ndarray] = None            # (10,) or (16,) shape; dropped by anonymize()
    left_hand_pose: Optional[np.ndarray] = None   # (T, 45) MANO; None -> neutral hand
    right_hand_pose: Optional[np.ndarray] = None   # (T, 45) MANO; None -> neutral hand
    frame: str = "global"        # 'global' (world-grounded) or 'incam'
    source_clip: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.poses = np.asarray(self.poses, dtype=np.float32)
        self.trans = np.asarray(self.trans, dtype=np.float32)
        if self.poses.ndim != 2 or self.poses.shape[1] != SMPL_POSE_DIM:
            raise ValueError(
                f"poses must be (T,{SMPL_POSE_DIM}); got {self.poses.shape}. "
                "Convert body_pose to full SMPL axis-angle in the backend adapter."
            )
        if self.trans.shape != (self.poses.shape[0], 3):
            raise ValueError(f"trans must be (T,3) matching poses; got {self.trans.shape}")
        if self.betas is not None:
            self.betas = np.asarray(self.betas, dtype=np.float32).reshape(-1)
        self.left_hand_pose = _as_hand(self.left_hand_pose, self.poses.shape[0], "left_hand_pose")
        self.right_hand_pose = _as_hand(self.right_hand_pose, self.poses.shape[0], "right_hand_pose")

    @property
    def n_frames(self) -> int:
        return int(self.poses.shape[0])

    @property
    def has_hands(self) -> bool:
        return self.left_hand_pose is not None or self.right_hand_pose is not None

    @property
    def global_orient(self) -> np.ndarray:
        return self.poses[:, :3]

    @property
    def body_pose(self) -> np.ndarray:
        return self.poses[:, 3:]

    # -- persistence ----------------------------------------------------
    def save_npz(self, path) -> None:
        """Write this clip to a single npz, hand slots included even when absent."""
        zeros_hand = np.zeros((self.n_frames, MANO_POSE_DIM), np.float32)
        np.savez(
            path,
            poses=self.poses,
            trans=self.trans,
            betas=self.betas if self.betas is not None else np.zeros(16, np.float32),
            left_hand_pose=self.left_hand_pose if self.left_hand_pose is not None else zeros_hand,
            right_hand_pose=self.right_hand_pose if self.right_hand_pose is not None else zeros_hand,
            # per-hand flags so a left-only clip round-trips to left-only (not a
            # spurious neutral right hand)
            has_left=np.bool_(self.left_hand_pose is not None),
            has_right=np.bool_(self.right_hand_pose is not None),
            fps=np.float32(self.fps),
            frame=self.frame,
            source_clip=self.source_clip or "",
        )

    @classmethod
    def load_npz(cls, path) -> "SmplMotion":
        """Inverse of :meth:`save_npz`; ``has_left``/``has_right`` decide whether
        the (possibly neutral-zero) stored hand block is restored or dropped to
        ``None``, so a hand that was never estimated doesn't reappear as neutral."""
        d = np.load(path, allow_pickle=False)
        has_left = bool(d["has_left"]) if "has_left" in d else False
        has_right = bool(d["has_right"]) if "has_right" in d else False
        return cls(
            poses=d["poses"],
            trans=d["trans"],
            betas=d["betas"],
            left_hand_pose=d["left_hand_pose"] if has_left else None,
            right_hand_pose=d["right_hand_pose"] if has_right else None,
            fps=float(d["fps"]),
            frame=str(d["frame"]),
            source_clip=str(d["source_clip"]) or None,
        )


def anonymize(motion: SmplMotion, *, drop_shape: bool = True, keep_translation: bool = True) -> SmplMotion:
    """Strip identity from a motion clip.

    * ``drop_shape``   -- discard ``betas`` (replace with the neutral zero shape).
      This is the core privacy guarantee: no body-shape identity leaves here.
    * ``keep_translation`` -- if False, zero the root translation too (keep only
      joint rotations), for a purely pose-relative dataset.

    Hand articulation is movement, not identity -- it is passed through untouched.
    """
    betas = None if drop_shape else motion.betas
    trans = motion.trans if keep_translation else np.zeros_like(motion.trans)
    return SmplMotion(
        poses=motion.poses.copy(),
        trans=trans.copy(),
        fps=motion.fps,
        betas=betas,
        left_hand_pose=None if motion.left_hand_pose is None else motion.left_hand_pose.copy(),
        right_hand_pose=None if motion.right_hand_pose is None else motion.right_hand_pose.copy(),
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, anonymized=True, drop_shape=drop_shape),
    )


def resample_fps(motion: SmplMotion, target_fps: float) -> SmplMotion:
    """Linearly resample a clip to ``target_fps``.

    Axis-angle is interpolated component-wise: fine for the small frame-to-frame
    deltas of adjacent video frames; for large gaps prefer slerp on rotations.
    """
    if abs(motion.fps - target_fps) < 1e-6 or motion.n_frames < 2:
        return motion
    t_old = np.arange(motion.n_frames) / motion.fps
    duration = t_old[-1]
    n_new = max(2, int(round(duration * target_fps)) + 1)
    t_new = np.linspace(0.0, duration, n_new)

    def interp(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return np.stack([np.interp(t_new, t_old, arr[:, i]) for i in range(arr.shape[1])], axis=1).astype(np.float32)

    return SmplMotion(
        poses=interp(motion.poses),
        trans=interp(motion.trans),
        fps=target_fps,
        betas=None if motion.betas is None else motion.betas.copy(),
        left_hand_pose=interp(motion.left_hand_pose),
        right_hand_pose=interp(motion.right_hand_pose),
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, resampled_from=motion.fps),
    )
