"""Post-processing refinements for a recovered :class:`SmplMotion`.

Everything in *this* module is pure NumPy signal processing -- no weights, no GPU
-- which is why it can run inline in the pipeline on any machine. That's a
property of these particular passes, not a rule that refinement must avoid
learned models: an optional, off-by-default learned pass (DPoser-X pose prior)
lives in ``refine_learned.py`` and shells out to its own env, selected via
``refine_method: dposer``. Keep the heavy path over there so this file stays
laptop-readable.

Two ideas from the research watchlist (``RESEARCH_WATCHLIST.md``) are tractable
*today*, without training anything or touching a GPU, because they are just
signal-processing passes over a rotation/translation time series:

1. **Temporal de-jitter** -- inspired by **HTD-Refine** (arXiv:2605.26879): that
   paper trains a network to refine any HMR estimate by penalizing high-order
   temporal dynamics (velocity/acceleration) of joint rotations. We have neither
   the network nor its training data, but the *objective* is reimplementable two
   ways here: ``temporal_dejitter`` (a Savitzky-Golay local polynomial fit) and
   ``variational_smooth`` (a global least-squares smoother that directly
   minimizes ``||x-y||^2 + lam*||accel(x)||^2`` -- literally the paper's
   penalize-acceleration objective, solved rather than learned). No learning
   required; both are no-ops at zero strength/lambda.

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
# Variational de-jitter: HTD-Refine's objective, solved instead of learned
# ---------------------------------------------------------------------------
#
# HTD-Refine trains a network to refine motion by penalizing its high-order
# temporal dynamics. Without the network we can still optimize that *objective*
# directly: find the signal x that minimizes  ||x - y||^2 + lam*||D2 x||^2, where
# D2 is the second difference (acceleration). That is a global least-squares
# smoother (Whittaker / Hodrick-Prescott) -- a principled step up from the local
# Savitzky-Golay fit: it trades data fidelity against acceleration energy over
# the whole window at once. Closed form: (I + lam*D2^T D2) x = y. We solve it in
# overlapping windows (Hann-tapered overlap-add) so it scales to long clips.


def _second_diff_matrix(w: int) -> np.ndarray:
    """(w-2, w) second-difference operator: rows [1, -2, 1] (discrete accel.)."""
    d = np.zeros((w - 2, w))
    idx = np.arange(w - 2)
    d[idx, idx], d[idx, idx + 1], d[idx, idx + 2] = 1.0, -2.0, 1.0
    return d


def _variational_inverse(w: int, lam: float) -> np.ndarray:
    """Inverse of the SPD operator (I + lam D2^T D2); applied to each window."""
    d2 = _second_diff_matrix(w)
    return np.linalg.inv(np.eye(w) + lam * (d2.T @ d2))


def _windowed_solve(x: np.ndarray, inv: np.ndarray, w: int) -> np.ndarray:
    """Overlap-add the per-window smoother across a (T, D) sequence.

    A Hann taper weights each window so overlapping solutions blend smoothly;
    where only one window covers a frame the taper cancels in the normalization,
    so the endpoints are exactly that window's solution (no edge darkening).
    """
    t, d = x.shape
    step = max(1, w // 2)
    starts = list(range(0, max(1, t - w + 1), step))
    if starts[-1] != t - w:
        starts.append(max(0, t - w))
    taper = (np.hanning(w) + 1e-6)[:, None]
    out = np.zeros((t, d))
    weight = np.zeros((t, 1))
    for s in starts:
        seg = x[s:s + w]
        out[s:s + w] += taper * (inv @ seg)
        weight[s:s + w] += taper
    return (out / np.maximum(weight, 1e-8)).astype(np.float32)


def variational_smooth(
    motion: SmplMotion,
    *,
    lam: float = 10.0,
    window: int = 128,
    smooth_hands: bool = True,
) -> SmplMotion:
    """De-jitter by minimizing ||x-y||^2 + lam*||accel(x)||^2 (HTD-Refine objective).

    ``lam`` sets the smoothing strength (0 = no-op). A clean constant-acceleration
    signal (zero second difference) is returned unchanged, so genuine motion is
    preserved while acceleration-scale jitter is suppressed.
    """
    t = motion.n_frames
    w = min(window, t)
    if t < 5 or lam <= 0 or w < 5:
        return _copy(motion)
    inv = _variational_inverse(w, lam)

    def smooth(x: np.ndarray) -> np.ndarray:
        return _windowed_solve(x, inv, w)

    def smooth_hand(hand: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if hand is None:
            return None
        return smooth(hand) if smooth_hands else hand.copy()

    return SmplMotion(
        poses=smooth(motion.poses),
        trans=smooth(motion.trans),
        fps=motion.fps,
        betas=motion.betas,
        left_hand_pose=smooth_hand(motion.left_hand_pose),
        right_hand_pose=smooth_hand(motion.right_hand_pose),
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, dejitter="variational", dejitter_lambda=lam, dejitter_window=w),
    )


# ---------------------------------------------------------------------------
# Confidence-weighted de-jitter: smooth each joint by how noisy it looks
# ---------------------------------------------------------------------------
#
# temporal_dejitter/variational_smooth apply ONE strength to every joint. But a
# clip's noise is rarely uniform -- an occluded or truncated joint (which HMR
# infers, not observes) jitters far more than a well-seen one. The idea here,
# from confidence-weighted fusion (Optimal-state Dynamics, NeurIPS 2024 -- treat
# the estimate as a noisy sensor and lean on the smooth prediction where it's
# unreliable) and partial-label masking (PedGen, ICLR 2025 -- down-weight
# untrustworthy labels rather than trust or drop them wholesale), is to make the
# smoothing strength *per-joint* and drive it from each joint's own local
# acceleration: trust clean joints, denoise noisy ones. Still pure NumPy, still a
# no-op at zero strength -- just a spatially-varying version of the SG blend.


def _confidence_weights(x: np.ndarray, kappa: float) -> np.ndarray:
    """Per-frame, per-joint smoothing weight in [0, 1) from local acceleration.

    ``x`` is ``(T, njoint*3)`` axis-angle. For each 3-vector joint we take the
    magnitude of its second temporal difference (acceleration = the jitter scale)
    and map it through ``a / (a + kappa)`` -- a smooth, bounded, monotonic knee:
    ~0 for still/clean joints, →1 for very jerky ones. ``kappa`` is the
    acceleration (rad/frame^2) at which a joint gets half its max smoothing.
    """
    t = x.shape[0]
    nj = x.shape[1] // 3
    xb = x.reshape(t, nj, 3)
    acc = np.zeros((t, nj), dtype=np.float64)
    if t >= 3:
        mag = np.linalg.norm(np.diff(xb, n=2, axis=0), axis=2)  # (T-2, nj), centered on middle sample
        acc[1:-1] = mag
        acc[0], acc[-1] = mag[0], mag[-1]                       # borrow neighbours at the ends
    return acc / (acc + kappa)


def confidence_dejitter(
    motion: SmplMotion,
    *,
    window: int = 9,
    polyorder: int = 2,
    kappa: float = 0.02,
    max_strength: float = 1.0,
    smooth_hands: bool = True,
) -> SmplMotion:
    """Adaptive Savitzky-Golay de-jitter: per-joint strength from local jitter.

    Like :func:`temporal_dejitter` but the blend weight varies per joint and per
    frame -- a joint is pulled toward its SG fit in proportion to how much it is
    accelerating (see ``_confidence_weights``), capped at ``max_strength``. This
    denoises inferred/occluded joints hard while leaving cleanly-tracked joints
    nearly untouched. Rotational channels (body + hands, all radians) get the
    adaptive treatment; ``trans`` (metres, not comparable to ``kappa``) gets a
    uniform SG blend at ``max_strength``. No-op for clips shorter than ``window``.
    """
    if motion.n_frames <= window:
        return _copy(motion)
    coeffs = _savgol_coeffs(window, polyorder)

    def adaptive(x: np.ndarray) -> np.ndarray:
        smoothed = _savgol_apply(x, coeffs)
        w = _confidence_weights(x, kappa) * max_strength         # (T, nj)
        wcol = np.repeat(w, 3, axis=1)                           # (T, nj*3)
        return ((1.0 - wcol) * x + wcol * smoothed).astype(np.float32)

    def adaptive_hand(hand: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if hand is None:
            return None
        return adaptive(hand) if smooth_hands else hand.copy()

    trans_sm = _savgol_apply(motion.trans, coeffs)
    trans_out = ((1.0 - max_strength) * motion.trans + max_strength * trans_sm).astype(np.float32)

    return SmplMotion(
        poses=adaptive(motion.poses),
        trans=trans_out,
        fps=motion.fps,
        betas=motion.betas,
        left_hand_pose=adaptive_hand(motion.left_hand_pose),
        right_hand_pose=adaptive_hand(motion.right_hand_pose),
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, dejitter="confidence", dejitter_kappa=kappa, dejitter_max_strength=max_strength),
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
    method: str = "savgol",
    window: int = 9,
    polyorder: int = 2,
    strength: float = 1.0,
    lam: float = 10.0,
    kappa: float = 0.02,
    smooth_hands: bool = True,
    anti_drift: bool = True,
    vel_thresh: float = 0.02,
    min_duration: float = 0.2,
    damping: float = 0.7,
    dposer: object = None,
) -> SmplMotion:
    """Apply the enabled refinement passes and return a new :class:`SmplMotion`.

    ``method`` picks the de-jitter: 'savgol' (uniform local polynomial fit),
    'variational' (global acceleration-penalized least squares -- the HTD-Refine
    objective), 'confidence' (per-joint adaptive SG, smoothing each joint by its
    own local jitter; ``kappa`` sets the accel knee, ``strength`` caps it), or
    'dposer' (a learned DPoser-X pose prior; heavy, opt-in -- pass ``dposer=cfg``,
    which supplies ``dposer_repo``/``dposer_python``/etc. See ``refine_learned``).
    Order matters: de-jitter first (so drift detection sees clean velocities),
    then anti-drift. Either pass can be disabled; always returns a fresh object
    (the input is never mutated, even when both passes are off).
    """
    if method not in ("savgol", "variational", "confidence", "dposer"):
        raise ValueError(f"refine method must be 'savgol', 'variational', 'confidence', or 'dposer', got {method!r}")
    out = motion
    if smooth and method == "dposer":
        if dposer is None:
            raise ValueError("refine_method='dposer' needs the pipeline config passed as dposer=cfg")
        from .refine_learned import dposer_refine  # heavy path: import only when selected
        out = dposer_refine(out, dposer)
    elif smooth and method == "variational":
        out = variational_smooth(out, lam=lam, window=max(window, 32), smooth_hands=smooth_hands)
    elif smooth and method == "confidence":
        out = confidence_dejitter(out, window=window, polyorder=polyorder, kappa=kappa, max_strength=strength, smooth_hands=smooth_hands)
    elif smooth:
        out = temporal_dejitter(out, window=window, polyorder=polyorder, strength=strength, smooth_hands=smooth_hands)
    if anti_drift:
        out = anti_drift_stationary(out, vel_thresh=vel_thresh, min_duration=min_duration, damping=damping)
    if out is motion:
        out = _copy(motion)
    out.meta["refined"] = True
    return out
