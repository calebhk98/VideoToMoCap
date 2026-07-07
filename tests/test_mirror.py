"""Tests for automatic left/right mirror detection + correction.

Runs under pytest OR directly: `python tests/test_mirror.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import ingest, pipeline
from videotomocap.config import PipelineConfig
from videotomocap.mirror import decide_mirrored, handedness_score, mirror_motion
from videotomocap.pose import SmplMotion


def _motion(n=20, seed=0, hands=True):
    rng = np.random.default_rng(seed)
    return SmplMotion(
        poses=rng.normal(0, 0.3, (n, 72)).astype(np.float32),
        trans=rng.normal(0, 0.1, (n, 3)).astype(np.float32),
        fps=30.0,
        left_hand_pose=rng.normal(0, 0.2, (n, 45)).astype(np.float32) if hands else None,
        right_hand_pose=rng.normal(0, 0.2, (n, 45)).astype(np.float32) if hands else None,
    )


def _right_dominant(n=30, arm_joint=17):
    """Motion where one shoulder (17=right, 16=left) sweeps and the rest is still."""
    poses = np.zeros((n, 72), np.float32)
    poses[:, arm_joint * 3] = np.linspace(0.0, 1.0, n)  # x-rotation of that shoulder
    return SmplMotion(poses=poses, trans=np.zeros((n, 3), np.float32), fps=30.0)


def test_mirror_is_an_involution():
    m = _motion(seed=1)
    m.joint_valid = np.ones(24, bool)
    m.joint_valid[16] = False  # left shoulder unreliable
    back = mirror_motion(mirror_motion(m))
    assert np.allclose(back.poses, m.poses, atol=1e-5)
    assert np.allclose(back.trans, m.trans, atol=1e-5)
    assert np.allclose(back.left_hand_pose, m.left_hand_pose, atol=1e-5)
    assert np.array_equal(back.joint_valid, m.joint_valid)


def test_mirror_swaps_left_and_right_joints():
    poses = np.zeros((4, 72), np.float32)
    poses[:, 16 * 3:16 * 3 + 3] = [0.5, 0.3, -0.2]  # left shoulder
    mm = mirror_motion(SmplMotion(poses=poses, trans=np.zeros((4, 3), np.float32), fps=30.0))
    # value moves to the RIGHT shoulder, with y,z negated; left shoulder now zero
    assert np.allclose(mm.poses[:, 17 * 3:17 * 3 + 3], [0.5, -0.3, 0.2])
    assert np.allclose(mm.poses[:, 16 * 3:16 * 3 + 3], 0.0)


def test_handedness_sign_flips_under_mirror():
    right = _right_dominant(arm_joint=17)
    assert handedness_score(right) > 0.5                    # right-dominant
    assert handedness_score(mirror_motion(right)) < -0.5    # mirror -> left-dominant


def test_decide_mirrored_flags_minority_and_adapts_to_lefties():
    # right-handed corpus: the one negative clip is the odd one out
    right_corpus = {"a": 0.4, "b": 0.5, "c": 0.45, "d": -0.4}
    flags = decide_mirrored(right_corpus, margin=0.15)
    assert flags == {"a": False, "b": False, "c": False, "d": True}
    assert all(isinstance(v, bool) for v in flags.values())   # JSON-safe python bools
    # left-handed corpus: consensus flips, now the positive clip is flagged
    left_corpus = {"a": -0.4, "b": -0.5, "c": -0.45, "d": 0.4}
    assert decide_mirrored(left_corpus, margin=0.15)["d"] is True
    # too few clips -> no consensus, flag nothing
    assert decide_mirrored({"a": 0.9, "b": -0.9}, min_clips=3) == {"a": False, "b": False}


def test_detect_mirroring_correct_flips_the_outlier():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(footage_root=Path(tmp), work_root=Path(tmp) / "w",
                             backend="noop", auto_mirror="correct")
        cfg.pose_dir.mkdir(parents=True)
        clips = []
        for i in range(4):
            arm = 17 if i < 3 else 16  # 3 right-dominant, 1 left-dominant (mirrored)
            _right_dominant(arm_joint=arm).save_npz(cfg.pose_dir / f"c{i}.npz")
            clips.append(ingest.Clip(clip_id=f"c{i}", camera="cam", rel_path=f"c{i}.mp4",
                                     status=ingest.POSE_DONE, n_frames=30))
        m = ingest.Manifest(footage_root="x", clips=clips)
        m.save(cfg.manifest_path)

        result = pipeline.detect_mirroring(cfg, m)
        assert result == {"flagged": 1, "corrected": 1}
        assert m.get("c3").suspected_mirrored and m.get("c3").mirrored
        assert not m.get("c0").suspected_mirrored
        # the corrected clip is now right-dominant like the rest
        assert handedness_score(SmplMotion.load_npz(cfg.pose_dir / "c3.npz")) > 0


def test_auto_mirror_off_and_flag_do_not_touch_pose():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(footage_root=Path(tmp), work_root=Path(tmp) / "w", backend="noop")
        cfg.pose_dir.mkdir(parents=True)
        clips = []
        for i in range(4):
            _right_dominant(arm_joint=17 if i < 3 else 16).save_npz(cfg.pose_dir / f"c{i}.npz")
            clips.append(ingest.Clip(clip_id=f"c{i}", camera="cam", rel_path=f"c{i}.mp4",
                                     status=ingest.POSE_DONE, n_frames=30))
        m = ingest.Manifest(footage_root="x", clips=clips)
        m.save(cfg.manifest_path)
        before = SmplMotion.load_npz(cfg.pose_dir / "c3.npz").poses.copy()

        cfg.auto_mirror = "off"
        assert pipeline.detect_mirroring(cfg, m) == {"flagged": 0, "corrected": 0}
        cfg.auto_mirror = "flag"
        assert pipeline.detect_mirroring(cfg, m)["corrected"] == 0     # flags only
        assert m.get("c3").suspected_mirrored and not m.get("c3").mirrored
        assert np.array_equal(SmplMotion.load_npz(cfg.pose_dir / "c3.npz").poses, before)  # unchanged


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} mirror tests passed")


if __name__ == "__main__":
    _run_all()
