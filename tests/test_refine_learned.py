"""Tests for the learned-refine plumbing (videotomocap/refine_learned.py).

The DPoser-X model itself needs a GPU + weights, but everything *around* it --
writing our npz contract, invoking a runner, reading the result back, blending,
no-op guards, error handling -- is pure Python and must stay GPU-free. We inject a
fake runner (a stand-in for the subprocess) so the whole round-trip is exercised
without any weights, mirroring how the `noop` backend covers the backend flow.

Runs under pytest OR directly: `python tests/test_refine_learned.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap.pose import MANO_POSE_DIM, SmplMotion
from videotomocap.refine import refine_motion
from videotomocap.refine_learned import LearnedRefineError, dposer_refine


def assert_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


class _Cfg:
    """Minimal duck-typed stand-in for PipelineConfig's dposer_* fields."""
    def __init__(self, **kw):
        self.dposer_strength = kw.get("strength", 1.0)
        self.dposer_repo = kw.get("repo", "/tmp")
        self.dposer_python = None
        self.dposer_config = "configs/body/subvp/timefc.py"
        self.cuda_device = None


def _motion(n=20, seed=0, hands=False):
    rng = np.random.default_rng(seed)
    kw = {}
    if hands:
        kw["left_hand_pose"] = rng.normal(0, 0.2, (n, MANO_POSE_DIM)).astype(np.float32)
        kw["right_hand_pose"] = rng.normal(0, 0.2, (n, MANO_POSE_DIM)).astype(np.float32)
    return SmplMotion(
        poses=rng.normal(0, 0.2, (n, 72)).astype(np.float32),
        trans=rng.normal(0, 0.1, (n, 3)).astype(np.float32),
        fps=30.0, **kw,
    )


def _fake_runner(payload_transform):
    """Build a runner that loads our input npz, transforms it, writes our output."""
    def runner(cfg, in_path, out_path):
        data = dict(np.load(in_path))
        np.savez(out_path, **payload_transform(data))
    return runner


def test_dposer_refine_round_trip_and_blend():
    m = _motion(hands=True)
    # Fake model: return all-zeros. With strength=0.5 the output is halfway there.
    runner = _fake_runner(lambda d: {k: np.zeros_like(v) for k, v in d.items()})
    out = dposer_refine(m, _Cfg(strength=0.5), runner=runner)
    assert np.allclose(out.poses, 0.5 * m.poses, atol=1e-6)
    assert np.allclose(out.left_hand_pose, 0.5 * m.left_hand_pose, atol=1e-6)
    assert np.array_equal(out.trans, m.trans)          # trans is never touched
    assert out.meta["refine_learned"] == "dposer" and out.meta["dposer_strength"] == 0.5


def test_dposer_refine_noop_when_strength_zero_or_too_short():
    m = _motion()
    called = []
    runner = _fake_runner(lambda d: called.append(1) or d)
    same = dposer_refine(m, _Cfg(strength=0.0), runner=runner)
    assert np.array_equal(same.poses, m.poses) and not called   # runner never invoked
    short = SmplMotion(poses=m.poses[:1], trans=m.trans[:1], fps=30.0)
    assert np.array_equal(dposer_refine(short, _Cfg(), runner=runner).poses, short.poses[:1])


def test_dposer_refine_errors_on_bad_output():
    m = _motion()
    no_out = lambda cfg, i, o: None                                    # writes nothing
    assert_raises(LearnedRefineError, lambda: dposer_refine(m, _Cfg(), runner=no_out))
    wrong = _fake_runner(lambda d: {"poses": d["poses"][:, :30]})      # wrong shape
    assert_raises(LearnedRefineError, lambda: dposer_refine(m, _Cfg(), runner=wrong))
    missing = _fake_runner(lambda d: {"trans": np.zeros((5, 3))})      # no 'poses'
    assert_raises(LearnedRefineError, lambda: dposer_refine(m, _Cfg(), runner=missing))


def test_dposer_refine_missing_repo_raises():
    cfg = _Cfg()
    cfg.dposer_repo = None
    assert_raises(LearnedRefineError, lambda: dposer_refine(_motion(), cfg))  # real runner, no repo


def test_refine_motion_dposer_requires_cfg():
    m = _motion()
    assert_raises(ValueError, lambda: refine_motion(m, method="dposer"))       # dposer=None
    # with a cfg + injected-free path it would run; here just prove the dispatch guard.


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} learned-refine tests passed")


if __name__ == "__main__":
    _run_all()
