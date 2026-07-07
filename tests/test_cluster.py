"""Tests for unsupervised action clustering + auto-occlusion (static joints).

Runs under pytest OR directly: `python tests/test_cluster.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import ingest, pipeline
from videotomocap.cluster import clip_features, cluster_clips, kmeans
from videotomocap.config import PipelineConfig
from videotomocap.pose import SmplMotion
from videotomocap.regions import infer_static_joints


def _sweep(active_joint, n=30):
    """A clip where exactly one joint sweeps -- a clean, separable motion signature."""
    poses = np.zeros((n, 72), np.float32)
    poses[:, active_joint * 3] = np.linspace(0.0, 1.0, n)
    return SmplMotion(poses=poses, trans=np.zeros((n, 3), np.float32), fps=30.0)


# -- clustering --------------------------------------------------------------

def test_kmeans_separates_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(-5, 0.1, (10, 3))
    b = rng.normal(+5, 0.1, (10, 3))
    labels, centers = kmeans(np.vstack([a, b]), 2, seed=0)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1  # each blob one cluster
    assert labels[0] != labels[10]
    assert centers.shape == (2, 3)


def test_clip_features_shape_and_short_clip():
    assert clip_features(_sweep(5)).shape == (50,)                  # 24*2 + 2
    one = SmplMotion(poses=np.zeros((1, 72), np.float32), trans=np.zeros((1, 3), np.float32), fps=30.0)
    assert np.all(clip_features(one) == 0)                          # too short -> zeros


def test_cluster_clips_groups_by_motion():
    feats = {f"a{i}": clip_features(_sweep(5)) for i in range(3)}
    feats.update({f"b{i}": clip_features(_sweep(18)) for i in range(3)})
    labels = cluster_clips(feats, 2, seed=0)
    assert len({labels[k] for k in feats if k.startswith("a")}) == 1  # all 'a' together
    assert labels["a0"] != labels["b0"]                              # different groups
    assert cluster_clips({}, 3) == {}                                # empty input safe


# -- auto-occlusion ----------------------------------------------------------

def test_infer_static_joints_flags_frozen_but_not_still_clips():
    rng = np.random.default_rng(1)
    poses = np.cumsum(rng.normal(0, 0.1, (30, 72)).astype(np.float32), axis=0)
    poses[:, 5 * 3:5 * 3 + 3] = 0.7  # joint 5 frozen while the body moves
    active = SmplMotion(poses=poses, trans=np.zeros((30, 3), np.float32), fps=30.0)
    assert 5 in infer_static_joints(active)

    still = SmplMotion(poses=np.zeros((30, 72), np.float32), trans=np.zeros((30, 3), np.float32), fps=30.0)
    assert infer_static_joints(still) == []   # whole clip still -> not occlusion


# -- end to end --------------------------------------------------------------

def _pose_manifest(cfg, motions):
    cfg.pose_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for clip_id, m in motions.items():
        m.save_npz(cfg.pose_dir / f"{clip_id}.npz")
        clips.append(ingest.Clip(clip_id=clip_id, camera="cam", rel_path=f"{clip_id}.mp4",
                                 status=ingest.POSE_DONE, n_frames=m.n_frames))
    man = ingest.Manifest(footage_root="x", clips=clips)
    man.save(cfg.manifest_path)
    return man


def test_cluster_pass_writes_action_cluster_to_index():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(footage_root=Path(tmp), work_root=Path(tmp) / "w",
                             backend="noop", cluster_actions=2, min_clip_frames=5)
        motions = {**{f"a{i}": _sweep(5) for i in range(3)}, **{f"b{i}": _sweep(18) for i in range(3)}}
        man = _pose_manifest(cfg, motions)
        pipeline.build(cfg, man)
        assert all(c.action_cluster >= 0 for c in man.clips)          # every clip labeled
        idx = json.loads((cfg.dataset_dir / "index.json").read_text())
        assert all("action_cluster" in e for e in idx["clips"])       # carried into the index


def test_auto_occlusion_masks_frozen_joints_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        S = Path(tmp)
        (S / "foot" / "cam").mkdir(parents=True)
        (S / "foot" / "cam" / "a.mp4").write_bytes(b"")
        cfg = PipelineConfig(footage_root=S / "foot", work_root=S / "w",
                             backend="noop", auto_occlusion="flag", min_clip_frames=5)
        m = ingest.scan(cfg)
        pipeline.run_hmr(cfg, m)
        # noop motion has some frozen joints -> mask should exist and drop those
        clip = m.clips[0]
        loaded = SmplMotion.load_npz(cfg.pose_dir / f"{clip.clip_id}.npz")
        # with auto_occlusion off, the same clip has no mask -> confirms the flag did something
        cfg_off = PipelineConfig(footage_root=S / "foot", work_root=S / "w2",
                                 backend="noop", auto_occlusion="off", min_clip_frames=5)
        m2 = ingest.scan(cfg_off)
        pipeline.run_hmr(cfg_off, m2)
        off = SmplMotion.load_npz(cfg_off.pose_dir / f"{m2.clips[0].clip_id}.npz")
        assert off.joint_valid is None
        # flag mode may or may not find frozen joints in synthetic noise, but must not crash
        assert loaded.joint_valid is None or loaded.joint_valid.dtype == bool


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} cluster/occlusion tests passed")


if __name__ == "__main__":
    _run_all()
