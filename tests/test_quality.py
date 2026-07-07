"""Tests for automatic motion-quality assessment + its pipeline pass.

Runs under pytest OR directly: `python tests/test_quality.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import ingest, pipeline
from videotomocap.config import PipelineConfig
from videotomocap.pose import SmplMotion
from videotomocap.quality import assess_quality, has_hard_issue


def _motion(poses, trans=None, fps=30.0):
    t = poses.shape[0]
    return SmplMotion(
        poses=poses.astype(np.float32),
        trans=(np.zeros((t, 3)) if trans is None else trans).astype(np.float32),
        fps=fps,
    )


def _clean(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return _motion(np.cumsum(rng.normal(0, 0.02, (n, 72)), axis=0),
                   np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0))


def test_assess_quality_catches_each_failure():
    assert assess_quality(_clean()) == []                       # smooth motion -> clean

    tp = np.zeros((10, 3), np.float32); tp[5:] += 5.0            # 5 m in 1/30 s
    assert "root_teleport" in assess_quality(_motion(np.zeros((10, 72), np.float32), tp))

    pj = np.zeros((10, 72), np.float32); pj[5:, 9] = 3.0        # 3 rad joint flip
    assert "pose_jump" in assess_quality(_motion(pj))

    assert "static_low_motion" in assess_quality(_motion(np.zeros((30, 72), np.float32)))

    nan = np.zeros((5, 72), np.float32); nan[2, 0] = np.nan
    issues = assess_quality(_motion(nan))
    assert issues == ["non_finite"] and has_hard_issue(issues)


def test_hard_vs_soft():
    assert has_hard_issue(["root_teleport"])
    assert not has_hard_issue(["static_low_motion"])            # soft: real, just still
    assert not has_hard_issue([])


def _pose_manifest(cfg, specs):
    """specs: list of (clip_id, motion) -> write npz + POSE_DONE clip."""
    cfg.pose_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for clip_id, motion in specs:
        motion.save_npz(cfg.pose_dir / f"{clip_id}.npz")
        clips.append(ingest.Clip(clip_id=clip_id, camera="cam", rel_path=f"{clip_id}.mp4",
                                 status=ingest.POSE_DONE, n_frames=motion.n_frames))
    m = ingest.Manifest(footage_root="x", clips=clips)
    m.save(cfg.manifest_path)
    return m


def test_quality_pass_flags_but_does_not_drop():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(footage_root=Path(tmp), work_root=Path(tmp) / "w",
                             backend="noop", quality_filter="flag", min_clip_frames=5)
        bad = _motion(np.zeros((40, 72), np.float32),
                      np.concatenate([np.zeros((20, 3)), np.full((20, 3), 5.0)]))  # teleport
        m = _pose_manifest(cfg, [("good", _clean()), ("bad", bad)])
        res = pipeline.assess_quality_pass(cfg, m)
        assert res["hard"] == 1
        assert m.get("bad").low_quality and "root_teleport" in m.get("bad").quality_issues
        assert not m.get("good").low_quality
        # flag mode keeps everything in the dataset
        stats = pipeline.build(cfg, m)
        assert stats.n_clips == 2


def test_quality_exclude_drops_hard_failures():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(footage_root=Path(tmp), work_root=Path(tmp) / "w",
                             backend="noop", quality_filter="exclude", min_clip_frames=5)
        bad = _motion(np.zeros((40, 72), np.float32),
                      np.concatenate([np.zeros((20, 3)), np.full((20, 3), 5.0)]))
        m = _pose_manifest(cfg, [("good", _clean()), ("bad", bad)])
        stats = pipeline.build(cfg, m)
        assert stats.n_clips == 1                                # the teleport clip dropped
        assert m.get("bad").low_quality


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} quality tests passed")


if __name__ == "__main__":
    _run_all()
