"""Automatic per-clip quality assessment from the recovered motion.

HMR on hours of footage will produce some garbage: frames where tracking teleports
the root across the scene, a limb snaps 180 degrees between frames, NaNs from a
failed solve, or a clip where nothing actually moves. Left in, these pollute the
motion dataset. This module scores each clip's plausibility from the SMPL motion
alone -- no video decode, no detector, no weights, just NumPy -- so bad clips can
be flagged (or dropped) automatically instead of hand-reviewed.

Two tiers of finding:
  * **hard** (``non_finite``, ``root_teleport``, ``pose_jump``) -- physically
    impossible; the clip is genuinely broken. These are what ``quality_filter:
    exclude`` drops.
  * **soft** (``static_low_motion``) -- valid but barely moving, so low training
    value. Flagged for your awareness, never auto-dropped (a still clip is real).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .pose import SmplMotion

HARD_ISSUES = frozenset({"non_finite", "root_teleport", "pose_jump"})


def assess_quality(
    motion: SmplMotion,
    *,
    max_speed_ms: float = 12.0,
    max_joint_step: float = 1.5,
    min_motion: float = 1.0e-3,
) -> List[str]:
    """Return a list of quality issues for a clip ([] = clean).

    * ``non_finite``       -- NaN/Inf in pose or translation (failed solve).
    * ``root_teleport``    -- root moves faster than ``max_speed_ms`` (m/s):
      tracking jumped, not human locomotion.
    * ``pose_jump``        -- a joint rotates more than ``max_joint_step`` rad
      between adjacent frames: a tracking glitch.
    * ``static_low_motion``-- mean per-frame pose change below ``min_motion``:
      valid but barely moving (soft; low training value).
    """
    issues: List[str] = []
    poses, trans = motion.poses, motion.trans

    if not (np.isfinite(poses).all() and np.isfinite(trans).all()):
        issues.append("non_finite")
        return issues  # further metrics are meaningless once values are non-finite

    if motion.n_frames < 2:
        return issues  # single frame -> no velocity/jitter to judge

    root_speed = np.linalg.norm(np.diff(trans, axis=0), axis=1) * motion.fps
    if float(root_speed.max()) > max_speed_ms:
        issues.append("root_teleport")

    pose_step = np.abs(np.diff(poses, axis=0))
    if float(pose_step.max()) > max_joint_step:
        issues.append("pose_jump")

    mean_motion = float(np.abs(np.diff(poses, axis=0)).mean())
    if mean_motion < min_motion:
        issues.append("static_low_motion")

    return issues


def has_hard_issue(issues: List[str]) -> bool:
    """True if any issue is physically impossible (a drop candidate)."""
    return any(i in HARD_ISSUES for i in issues)
