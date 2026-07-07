"""Pure-NumPy post-processing refinements for a recovered :class:`SmplMotion`.

Two ideas from the research watchlist (``RESEARCH_WATCHLIST.md``) are tractable
*today*, without training anything or touching a GPU, because they are just
signal-processing passes over a rotation/translation time series:

1. **Temporal de-jitter** (``temporal_dejitter``) -- inspired by **HTD-Refine**
   (arXiv:2605.26879): that paper trains a network to refine any HMR estimate
   by penalizing high-order temporal dynamics (velocity/acceleration) of joint
   rotations. We do not have their network or training data, but the *idea* --
   suppress acceleration-scale jitter while preserving the underlying motion --
   is exactly what a Savitzky-Golay smoother does: locally fit a low-order
   polynomial per channel and evaluate it at the center frame. No learning
   required, and it is trivially invertible in strength (``strength=0`` is a
   no-op, ``strength=1`` is fully smoothed).

2. **Trajectory-level anti-drift** (``anti_drift_stationary``) -- a deliberately
   scoped-down stand-in for true foot-contact / foot-skating cleanup. Real
   foot-skate detection needs the *world position* of the ankle/toe joints,
   which requires running the SMPL joint regressor against ``betas`` (bone
   lengths) -- this module has neither the mesh model nor (post-anonymization)
   the shape parameters to do that. What we *can* do without any of that: the
   root trajectory (``trans``) is directly available, and HMR/SLAM trajectory
   drift shows up as slow positional creep during periods where the person is
   actually planted (root velocity near zero). We detect those stationary
   spans from root velocity alone and damp the drift within them. This fixes
   "the whole body slides while standing still" but cannot fix a single foot
   sliding while the other leg is mid-stride and the root keeps moving -- that
   needs real per-joint contact, which is out of scope here.

Rotation caveat (shared with ``pose.resample_fps``): axis-angle channels are
smoothed component-wise, not via slerp/SLERP-style geodesic averaging. This is
fine for the small frame-to-frame deltas typical of video-rate motion capture,
but is not a proper rotation-manifold average -- do not feed this a sequence
with large per-frame rotation jumps and expect a faithful in-between.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .pose import SmplMotion


# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing (the HTD-Refine-inspired de-jitter core)
# ---------------------------------------------------------------------------

def _savgol_coeffs(window: int, polyorder: int) -> np.ndarray:
    """Convolution kernel for a centered Savitzky-Golay filter.

    Fits a degree-``polyorder`` polynomial to ``window`` samples centered on
    each point (least squares) and evaluates the fit at the center. That
    evaluation is linear in the samples, so it reduces to one fixed kernel,
    computed once here: the first row of the Vandermonde pseudo-inverse (the
    constant term of the fitted polynomial *is* its value at offset 0).
    """
    if window % 2 == 0 or window < polyorder + 2:
        raise ValueError(
            f"window must be odd and >= polyorder+2; got window={window}, polyorder={polyorder}"
        )
    half = window // 2
    offsets = np.arange(-half, half + 1)
    vander = np.vander(offsets, polyorder + 1, increasing=True)  # (window, polyorder+1)
    return np.linalg.pinv(vander)[0]  # (window,) -- constant-term row


def _savgol_apply(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply a centered SG kernel along axis 0 of a (T,) or (T, D) array.

    Reflect-pads the boundary by half the window so the output stays length
    T; that is an approximation at the very first/last few frames (there is
    no true "future" to fit against there), which is the standard tradeoff
    for any centered/non-causal filter.
    """
    half = coeffs.shape[0] // 2
    was_1d = x.ndim == 1
    x2 = x[:, None] if was_1d else x
    padded = np.pad(x2, ((half, half), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, coeffs.shape[0], axis=0)  # (T, D, window)
    smoothed = np.einsum("tdw,w->td", windows, coeffs)
    return smoothed[:, 0] if was_1d else smoothed


def jitter_metric(x: np.ndarray) -> float:
    """Variance of the second temporal difference -- a proxy for the
    acceleration-scale "high-order dynamics" HTD-Refine-style passes target.
    Lower is smoother; used by the tests to assert de-jitter actually helps."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 3:
        return 0.0
    return float(np.var(np.diff(x, n=2, axis=0)))


def _hand_copy(hand: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if hand is None else hand.copy()


def _copy(motion: SmplMotion) -> SmplMotion:
    """Cheap full copy, used so every refine pass returns a fresh object and
    never mutates the caller's motion (even when a pass is a no-op)."""
    return SmplMotion(
        poses=motion.poses.copy(), trans=motion.trans.copy(), fps=motion.fps, betas=motion.betas,
        left_hand_pose=_hand_copy(motion.left_hand_pose), right_hand_pose=_hand_copy(motion.right_hand_pose),
        frame=motion.frame, source_clip=motion.source_clip, meta=dict(motion.meta),
    )


def temporal_dejitter(
    motion: SmplMotion,
    *,
    window: int = 9,
    polyorder: int = 2,
    strength: float = 1.0,
    smooth_hands: bool = True,
) -> SmplMotion:
    """Savitzky-Golay de-jitter over pose (axis-angle), trans, and hands.

    ``strength`` in [0, 1] blends toward the smoothed signal (0 = unchanged,
    1 = fully replaced by the SG fit); anything in between is a cheap way to
    trade jitter reduction against fidelity to fast, intentional motion.
    Clips that are too short for ``window`` are returned unchanged (a real
    fit needs at least ``window`` frames of context).
    """
    if motion.n_frames <= window:
        return _copy(motion)

    coeffs = _savgol_coeffs(window, polyorder)

    def blend(x: np.ndarray) -> np.ndarray:
        smoothed = _savgol_apply(x, coeffs)
        return ((1.0 - strength) * x + strength * smoothed).astype(np.float32)

    def blend_hand(hand: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if hand is None:
            return None
        return blend(hand) if smooth_hands else hand.copy()

    return SmplMotion(
        poses=blend(motion.poses),
        trans=blend(motion.trans),
        fps=motion.fps,
        betas=motion.betas,
        left_hand_pose=blend_hand(motion.left_hand_pose),
        right_hand_pose=blend_hand(motion.right_hand_pose),
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, dejitter_window=window, dejitter_polyorder=polyorder, dejitter_strength=strength),
    )


# ---------------------------------------------------------------------------
# Trajectory-level anti-drift (the honest, scoped-down foot-skate cleanup)
# ---------------------------------------------------------------------------

def detect_stationary_frames(trans: np.ndarray, fps: float, *, vel_thresh: float = 0.02) -> np.ndarray:
    """Per-frame boolean mask: root speed (m/s) below ``vel_thresh``.

    Root speed is the only "is this person planted" signal available without
    a mesh/joint model, so this is necessarily coarser than real foot contact
    -- see the module docstring.
    """
    n = trans.shape[0]
    if n < 2:
        return np.zeros(n, dtype=bool)
    step_speed = np.linalg.norm(np.diff(trans, axis=0), axis=1) * fps  # (T-1,)
    speed = np.concatenate([step_speed[:1], step_speed])  # first frame borrows frame-1's speed
    return speed < vel_thresh


def _segments_from_mask(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Contiguous ``True`` runs of ``mask`` as (start, end) half-open pairs,
    keeping only runs at least ``min_len`` frames long."""
    if not mask.any():
        return []
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def anti_drift_stationary(
    motion: SmplMotion,
    *,
    vel_thresh: float = 0.02,
    min_duration: float = 0.2,
    damping: float = 0.7,
) -> SmplMotion:
    """Damp root-translation drift during detected stationary spans.

    Within each qualifying span (root speed below ``vel_thresh`` for at least
    ``min_duration`` seconds), pull ``trans`` toward the span's median
    position by ``damping`` (0 = untouched, 1 = fully pinned) -- the drift a
    world-grounded HMR/SLAM stage accumulates while a person is genuinely
    standing still. Poses and everything else pass through unchanged.
    """
    stationary = detect_stationary_frames(motion.trans, motion.fps, vel_thresh=vel_thresh)
    min_len = max(2, int(round(min_duration * motion.fps)))
    segments = _segments_from_mask(stationary, min_len)

    trans = motion.trans.copy()
    for start, end in segments:
        anchor = np.median(trans[start:end], axis=0)
        trans[start:end] = (1.0 - damping) * trans[start:end] + damping * anchor

    return SmplMotion(
        poses=motion.poses.copy(),
        trans=trans.astype(np.float32),
        fps=motion.fps,
        betas=motion.betas,
        left_hand_pose=_hand_copy(motion.left_hand_pose),
        right_hand_pose=_hand_copy(motion.right_hand_pose),
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, anti_drift_segments=len(segments), anti_drift_damping=damping),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def refine_motion(
    motion: SmplMotion,
    *,
    smooth: bool = True,
    window: int = 9,
    polyorder: int = 2,
    strength: float = 1.0,
    smooth_hands: bool = True,
    anti_drift: bool = True,
    vel_thresh: float = 0.02,
    min_duration: float = 0.2,
    damping: float = 0.7,
) -> SmplMotion:
    """Apply the enabled refinement passes and return a new :class:`SmplMotion`.

    Order matters: de-jitter first (so drift detection below sees clean
    velocities, not smoothing artifacts), then anti-drift. Either pass can be
    disabled independently. Always returns a fresh object -- the input
    ``motion`` is never mutated, even when both passes are disabled.
    """
    out = motion
    if smooth:
        out = temporal_dejitter(out, window=window, polyorder=polyorder, strength=strength, smooth_hands=smooth_hands)
    if anti_drift:
        out = anti_drift_stationary(out, vel_thresh=vel_thresh, min_duration=min_duration, damping=damping)
    if out is motion:
        out = _copy(motion)
    out.meta["refined"] = True
    return out
