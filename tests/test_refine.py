"""Tests for the pure-NumPy motion refinement passes (de-jitter, anti-drift).

Runs under pytest OR directly: `python tests/test_refine.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap.pose import MANO_POSE_DIM, SmplMotion
from videotomocap.refine import (
    _savgol_coeffs,
    _segments_from_mask,
    anti_drift_stationary,
    confidence_dejitter,
    detect_stationary_frames,
    jitter_metric,
    refine_motion,
    temporal_dejitter,
    variational_smooth,
)


def assert_raises(exc, fn):
    """Tiny pytest.raises stand-in so tests run under plain `python` too."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _body(n=40, seed=0, hands=False):
    rng = np.random.default_rng(seed)
    m = SmplMotion(
        poses=rng.normal(0, 0.2, size=(n, 72)).astype(np.float32),
        trans=rng.normal(0, 0.1, size=(n, 3)).astype(np.float32),
        fps=30.0,
    )
    if not hands:
        return m
    return SmplMotion(
        poses=m.poses, trans=m.trans, fps=m.fps,
        left_hand_pose=rng.normal(0, 0.2, size=(n, MANO_POSE_DIM)).astype(np.float32),
        right_hand_pose=rng.normal(0, 0.2, size=(n, MANO_POSE_DIM)).astype(np.float32),
    )


def _linear_motion(n=40, fps=30.0):
    """A clean, noise-free motion: pose channels and trans all ramp linearly.
    Second difference is exactly zero -- the "nothing to fix" baseline."""
    t = np.arange(n, dtype=np.float32)
    poses = (0.01 * t)[:, None] * np.ones((1, 72), np.float32)
    trans = (0.02 * t)[:, None] * np.array([[1.0, 0.5, -0.3]], np.float32)
    return SmplMotion(poses=poses.astype(np.float32), trans=trans.astype(np.float32), fps=fps)


# -- Savitzky-Golay core ------------------------------------------------------

def test_savgol_coeffs_sum_to_one():
    # A constant signal must be reproduced exactly -> coefficients partition unity.
    coeffs = _savgol_coeffs(9, 2)
    assert coeffs.shape == (9,)
    assert abs(coeffs.sum() - 1.0) < 1e-8


def test_savgol_coeffs_rejects_bad_window():
    assert_raises(ValueError, lambda: _savgol_coeffs(8, 2))   # even window
    assert_raises(ValueError, lambda: _savgol_coeffs(3, 4))   # window too small for polyorder


# -- temporal_dejitter ---------------------------------------------------------

def test_temporal_dejitter_reduces_jitter():
    rng = np.random.default_rng(1)
    clean = _linear_motion(60)
    noisy = SmplMotion(
        poses=clean.poses + rng.normal(0, 0.05, clean.poses.shape).astype(np.float32),
        trans=clean.trans + rng.normal(0, 0.02, clean.trans.shape).astype(np.float32),
        fps=clean.fps,
    )
    out = temporal_dejitter(noisy, window=9, polyorder=2, strength=1.0)
    assert jitter_metric(out.poses) < jitter_metric(noisy.poses)
    assert jitter_metric(out.trans) < jitter_metric(noisy.trans)
    assert out.poses.shape == noisy.poses.shape
    assert out.meta["dejitter_window"] == 9


def test_temporal_dejitter_preserves_clean_linear_motion_interior():
    # Interior frames (far enough from the reflect-padded boundary) of a
    # perfectly linear signal have zero residual under a >=1st-order SG fit,
    # so de-jitter should leave them essentially untouched.
    clean = _linear_motion(60)
    out = temporal_dejitter(clean, window=9, polyorder=2, strength=1.0)
    half = 4
    assert np.allclose(out.poses[half:-half], clean.poses[half:-half], atol=1e-4)
    assert np.allclose(out.trans[half:-half], clean.trans[half:-half], atol=1e-4)


def test_temporal_dejitter_strength_zero_is_noop():
    m = _body(30, seed=2)
    out = temporal_dejitter(m, window=9, polyorder=2, strength=0.0)
    assert np.allclose(out.poses, m.poses)
    assert np.allclose(out.trans, m.trans)


def test_temporal_dejitter_short_clip_is_noop_copy():
    m = _body(5, seed=3)  # n_frames <= window(9)
    out = temporal_dejitter(m, window=9, polyorder=2)
    assert out is not m
    assert np.allclose(out.poses, m.poses) and np.allclose(out.trans, m.trans)


def test_temporal_dejitter_preserves_hand_shapes_and_respects_smooth_hands_flag():
    m = _body(40, seed=4, hands=True)
    smoothed = temporal_dejitter(m, smooth_hands=True)
    untouched = temporal_dejitter(m, smooth_hands=False)
    assert smoothed.left_hand_pose.shape == m.left_hand_pose.shape
    assert smoothed.right_hand_pose.shape == m.right_hand_pose.shape
    assert np.allclose(untouched.left_hand_pose, m.left_hand_pose)
    assert not np.allclose(smoothed.left_hand_pose, m.left_hand_pose)


# -- stationary detection / anti-drift -----------------------------------------

def test_detect_stationary_frames_flags_low_velocity():
    fps = 30.0
    still = np.tile([1.0, 2.0, 0.0], (20, 1)).astype(np.float32)  # zero velocity
    walking = np.cumsum(np.full((20, 3), 0.05, np.float32), axis=0) + still[-1]  # ~1.5 m/s
    trans = np.concatenate([still, walking], axis=0)
    mask = detect_stationary_frames(trans, fps, vel_thresh=0.02)
    assert mask[:20].all()
    assert not mask[20:].any()


def test_segments_from_mask_filters_short_runs():
    mask = np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    segs = _segments_from_mask(mask, min_len=3)
    assert segs == [(0, 3), (7, 11)]  # the lone True at index 4 is too short


def test_anti_drift_stationary_damps_slow_drift_leaves_motion_alone():
    fps = 30.0
    n_still, n_move = 60, 40
    # "stationary" segment: real-world planted foot, but the estimator has
    # accumulated slow positional drift (per-frame step well under vel_thresh).
    drift = np.arange(n_still, dtype=np.float32)[:, None] * np.array([[0.0002, -0.0001, 0.0]], np.float32)
    still = np.array([1.0, 0.5, 0.2], np.float32) + drift
    # a fast-moving segment right after -- must be left untouched.
    steps = np.full((n_move, 3), 0.05, np.float32)
    moving = still[-1] + np.cumsum(steps, axis=0)
    trans = np.concatenate([still, moving], axis=0)
    poses = np.zeros((n_still + n_move, 72), np.float32)
    motion = SmplMotion(poses=poses, trans=trans, fps=fps)

    out = anti_drift_stationary(motion, vel_thresh=0.02, min_duration=0.2, damping=1.0)

    assert out.meta["anti_drift_segments"] >= 1
    # damping=1.0 fully pins the stationary span to its median -> zero spread.
    assert np.var(out.trans[:n_still], axis=0).max() < 1e-10
    assert np.var(still, axis=0).max() > 0  # sanity: original really did drift
    # the moving segment never qualifies as stationary -> untouched exactly.
    assert np.allclose(out.trans[n_still:], trans[n_still:])
    assert np.allclose(out.poses, poses)  # anti-drift never touches poses


def test_anti_drift_stationary_partial_damping_reduces_but_does_not_zero_variance():
    fps = 30.0
    drift = np.arange(50, dtype=np.float32)[:, None] * np.array([[0.0002, 0.0, 0.0]], np.float32)
    trans = np.array([0.0, 0.0, 0.0], np.float32) + drift
    motion = SmplMotion(poses=np.zeros((50, 72), np.float32), trans=trans, fps=fps)
    out = anti_drift_stationary(motion, vel_thresh=0.02, min_duration=0.2, damping=0.5)
    assert 0 < np.var(out.trans, axis=0).max() < np.var(trans, axis=0).max()


# -- orchestrator ----------------------------------------------------------

def test_refine_motion_marks_meta_and_preserves_hands():
    m = _body(40, seed=5, hands=True)
    out = refine_motion(m)
    assert out.meta["refined"] is True
    assert out.poses.shape == m.poses.shape
    assert out.has_hands
    assert out.left_hand_pose.shape == m.left_hand_pose.shape
    assert out.fps == m.fps


def test_refine_motion_never_mutates_input_even_when_all_passes_disabled():
    m = _body(20, seed=6)
    out = refine_motion(m, smooth=False, anti_drift=False)
    assert out is not m
    assert "refined" not in m.meta          # original object untouched
    assert out.meta["refined"] is True
    assert np.allclose(out.poses, m.poses) and np.allclose(out.trans, m.trans)


def test_refine_motion_combines_dejitter_and_antidrift():
    rng = np.random.default_rng(7)
    fps = 30.0
    n_still, n_move = 60, 30
    drift = np.arange(n_still, dtype=np.float32)[:, None] * np.array([[0.0002, 0.0, 0.0]], np.float32)
    still = np.array([0.3, 0.1, 0.0], np.float32) + drift
    moving = still[-1] + np.cumsum(np.full((n_move, 3), 0.05, np.float32), axis=0)
    trans = np.concatenate([still, moving], axis=0) + rng.normal(0, 0.001, (n_still + n_move, 3)).astype(np.float32)
    poses = rng.normal(0, 0.2, (n_still + n_move, 72)).astype(np.float32)
    motion = SmplMotion(poses=poses, trans=trans, fps=fps)

    out = refine_motion(motion, window=9, polyorder=2, strength=1.0, damping=0.8, vel_thresh=0.02, min_duration=0.2)
    assert jitter_metric(out.poses) < jitter_metric(motion.poses)
    assert out.meta["refined"] is True
    assert "dejitter_window" in out.meta and "anti_drift_segments" in out.meta


# -- jitter_metric -----------------------------------------------------------

def test_jitter_metric_zero_for_linear_motion():
    clean = _linear_motion(20)
    assert jitter_metric(clean.poses) < 1e-10
    assert jitter_metric(clean.trans) < 1e-10


def test_jitter_metric_zero_for_short_sequence():
    assert jitter_metric(np.zeros((2, 3), np.float32)) == 0.0


# -- variational (HTD-Refine-objective) smoother -----------------------------

def test_variational_smooth_preserves_linear_motion():
    # zero-acceleration signal -> the penalty is 0, so the smoother is identity
    t = 200
    lin = np.outer(np.linspace(0, 1, t), np.ones(72)).astype(np.float32)
    m = SmplMotion(poses=lin, trans=np.zeros((t, 3), np.float32), fps=30.0)
    out = variational_smooth(m, lam=20.0, window=64)
    assert np.abs(out.poses - lin).max() < 1e-4


def test_variational_smooth_reduces_jitter_and_lam_zero_is_noop():
    rng = np.random.default_rng(3)
    t = 200
    clean = np.cumsum(rng.normal(0, 0.02, (t, 72)), axis=0).astype(np.float32)
    noisy = clean + rng.normal(0, 0.05, (t, 72)).astype(np.float32)
    m = SmplMotion(poses=noisy, trans=np.zeros((t, 3), np.float32), fps=30.0)
    smoothed = variational_smooth(m, lam=30.0, window=64)
    assert jitter_metric(smoothed.poses) < jitter_metric(noisy)
    assert np.allclose(variational_smooth(m, lam=0.0).poses, noisy)  # lam=0 -> unchanged
    assert variational_smooth(m, lam=10.0).n_frames == t             # length preserved


def test_variational_smooth_preserves_hands_and_flag():
    m = _body(60, hands=True)
    out = variational_smooth(m, lam=10.0, window=32)
    assert out.left_hand_pose.shape == m.left_hand_pose.shape
    assert not np.array_equal(out.left_hand_pose, m.left_hand_pose)   # smoothed by default
    off = variational_smooth(m, lam=10.0, window=32, smooth_hands=False)
    assert np.array_equal(off.left_hand_pose, m.left_hand_pose)       # left alone


def test_refine_motion_method_variational_and_rejects_bad():
    m = _body(60, seed=2)
    out = refine_motion(m, method="variational", anti_drift=False)
    assert out.meta["dejitter"] == "variational" and out.meta["refined"] is True
    assert_raises(ValueError, lambda: refine_motion(m, method="bogus"))


def _clean_plus_one_noisy_joint(n=60, noisy=5, seed=3):
    """A motion where one joint (index `noisy`) is heavily jittered and every
    other joint ramps cleanly -- the setup that separates adaptive from uniform."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    poses = (0.01 * t)[:, None] * np.ones((1, 72), np.float32)
    poses[:, noisy * 3:noisy * 3 + 3] += rng.normal(0, 0.3, size=(n, 3)).astype(np.float32)
    return SmplMotion(poses=poses.astype(np.float32),
                      trans=(0.02 * t)[:, None] * np.ones((1, 3), np.float32), fps=30.0)


def test_confidence_dejitter_smooths_noisy_joint_more_than_clean():
    noisy = 5
    m = _clean_plus_one_noisy_joint(noisy=noisy)
    out = confidence_dejitter(m, kappa=0.02)

    def joint_change(a, b, j):
        return float(np.abs(a[:, j * 3:j * 3 + 3] - b[:, j * 3:j * 3 + 3]).mean())

    moved_noisy = joint_change(m.poses, out.poses, noisy)
    moved_clean = joint_change(m.poses, out.poses, 10)  # an untouched clean joint
    # the jittery joint is pulled toward its fit; the clean one is left alone
    assert moved_noisy > 10 * moved_clean
    assert jitter_metric(out.poses[:, noisy * 3:noisy * 3 + 3]) < jitter_metric(m.poses[:, noisy * 3:noisy * 3 + 3])
    assert out.meta["dejitter"] == "confidence"


def test_confidence_dejitter_noop_and_hands():
    short = _body(5, seed=1)   # <= default window → returned unchanged
    assert np.array_equal(confidence_dejitter(short).poses, short.poses)
    m = _body(40, seed=1, hands=True)
    off = confidence_dejitter(m, smooth_hands=False)
    assert np.array_equal(off.left_hand_pose, m.left_hand_pose)
    on = confidence_dejitter(m, smooth_hands=True)
    assert on.left_hand_pose.shape == m.left_hand_pose.shape


def test_refine_motion_method_confidence():
    m = _clean_plus_one_noisy_joint()
    out = refine_motion(m, method="confidence", anti_drift=False)
    assert out.meta["dejitter"] == "confidence" and out.meta["refined"] is True


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} refine tests passed")


if __name__ == "__main__":
    _run_all()
