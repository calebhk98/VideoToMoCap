"""Automatic left/right mirror detection + correction.

Some footage is horizontally flipped (front-camera "selfie" videos especially),
which swaps the subject's left and right in the recovered motion -- corrupting
handedness (writing, eating, throwing with the dominant hand). There is no
metadata flag for this and a mirrored person still looks like a valid person, so
per-video detection from pixels alone is unreliable.

But across a *corpus of one person* there is a usable signal: your handedness is
consistent, so genuinely-oriented clips agree on which hand is dominant, and
mirrored clips are the minority whose handedness is flipped. We score each clip's
handedness from the recovered motion, take the corpus consensus as your true
dominant side, and flag (or correct) the clips that confidently disagree.

Two pure, testable pieces:
  * ``mirror_motion``   -- the exact fix: the standard SMPL left/right mirror
    (swap L/R joints, negate the y/z axis-angle components, flip root-x). Applying
    it in motion space is equivalent to having flipped the video, so no HMR re-run.
  * ``handedness_score`` + ``decide_mirrored`` -- the automatic, corpus-relative
    detector (no per-video tags).

Honest limits: the detector is a heuristic. It can't tell a mirrored clip from
one where you genuinely used your non-dominant hand a lot, so it only flags
*confident* disagreements and defaults to flagging (not silently flipping). Gross
motion (walking, sitting) is roughly symmetric and barely affected either way.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .pose import MANO_POSE_DIM, SMPL_NJOINTS, SmplMotion

# SMPL left/right joint pairs (index, index). Central joints (pelvis, spine*,
# neck, head) have no partner and are only mirrored, not swapped.
_LR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]
# Arm chains used for the handedness signal (collar, shoulder, elbow, wrist).
_LEFT_ARM = [13, 16, 18, 20]
_RIGHT_ARM = [14, 17, 19, 21]


def _swap_permutation() -> List[int]:
    perm = list(range(SMPL_NJOINTS))
    for left, right in _LR_PAIRS:
        perm[left], perm[right] = right, left
    return perm


_SWAP = _swap_permutation()


def _mirror_axis_angle(block: np.ndarray, njoints: int) -> np.ndarray:
    """Negate the y,z components of each joint's axis-angle (reflection across the
    sagittal plane). Rotation vectors are pseudovectors, so under an x-reflection
    (rx,ry,rz) -> (rx,-ry,-rz). This is the standard SMPL mirror convention."""
    a = block.reshape(-1, njoints, 3).copy()
    a[..., 1] *= -1.0
    a[..., 2] *= -1.0
    return a


def mirror_motion(motion: SmplMotion) -> SmplMotion:
    """Return a left/right mirrored copy (equivalent to flipping the source video).

    Involution: ``mirror_motion(mirror_motion(m))`` reproduces ``m``.
    """
    t = motion.n_frames
    poses = _mirror_axis_angle(motion.poses, SMPL_NJOINTS)   # mirror each joint...
    poses = poses[:, _SWAP, :].reshape(t, SMPL_NJOINTS * 3)  # ...then swap L<->R

    trans = motion.trans.copy()
    trans[:, 0] *= -1.0  # reflect the left/right (x) axis of the trajectory

    def mirror_hand(hand: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if hand is None:
            return None
        return _mirror_axis_angle(hand, 15).reshape(t, MANO_POSE_DIM)

    # a mirrored left hand IS a right hand -> swap sides as well as mirror
    left = mirror_hand(motion.right_hand_pose)
    right = mirror_hand(motion.left_hand_pose)

    joint_valid = None if motion.joint_valid is None else motion.joint_valid[_SWAP].copy()

    return SmplMotion(
        poses=poses.astype(np.float32),
        trans=trans,
        fps=motion.fps,
        betas=motion.betas,
        left_hand_pose=left,
        right_hand_pose=right,
        joint_valid=joint_valid,
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, mirrored=not motion.meta.get("mirrored", False)),
    )


def _motion_energy(block: Optional[np.ndarray]) -> float:
    """Sum of squared frame-to-frame deltas -- how much this block moves."""
    if block is None or block.shape[0] < 2:
        return 0.0
    return float((np.diff(block, axis=0) ** 2).sum())


def handedness_score(motion: SmplMotion) -> float:
    """Signed handedness in [-1, 1]: >0 = right side more active, <0 = left.

    Aggregated over the whole clip from arm (and, when present, hand) motion
    energy. Under ``mirror_motion`` the sign flips, which is what the detector
    keys on.
    """
    poses = motion.poses.reshape(motion.n_frames, SMPL_NJOINTS, 3)
    right = _motion_energy(poses[:, _RIGHT_ARM, :]) + _motion_energy(motion.right_hand_pose)
    left = _motion_energy(poses[:, _LEFT_ARM, :]) + _motion_energy(motion.left_hand_pose)
    total = right + left
    return 0.0 if total == 0.0 else (right - left) / total


def decide_mirrored(scores: Dict[str, float], *, margin: float = 0.15, min_clips: int = 3) -> Dict[str, bool]:
    """Flag clips whose handedness confidently opposes the corpus consensus.

    ``consensus`` = sign of the median score (your true dominant side, learned
    from the whole corpus -- so a left-handed user isn't mass-flagged). A clip is
    flagged only if it leans the *other* way by at least ``margin``. With fewer
    than ``min_clips`` there isn't enough to form a consensus -> flag nothing.
    """
    if len(scores) < min_clips:
        return {clip_id: False for clip_id in scores}
    consensus = 1.0 if float(np.median(list(scores.values()))) >= 0.0 else -1.0
    return {
        clip_id: bool(np.sign(score) == -consensus and abs(score) >= margin)  # plain bool -> JSON-safe
        for clip_id, score in scores.items()
    }
