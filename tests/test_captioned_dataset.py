"""Tests for the caption<->mocap pairing bridge (videotomocap.captioned_dataset).

Verifies the seam that makes text-to-motion possible: Pipeline 1 motion sliced at
Pipeline 3 caption-segment boundaries -> AMASS + index.json with captions ->
Pipeline 2 (`conditioning: text`) writing real per-clip caption files. Uses tiny
hand-built motion npz + a seeded caption store, so it needs numpy but no GPU/models.
Runs under pytest OR directly: `python tests/test_captioned_dataset.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videocaption import manifest as vmf, store as vstore
from videocaption.backends.base import SegmentLabel
from videocaption.config import CaptionConfig
from videocaption.segment import Segment
from videotomocap.captioned_dataset import build_captioned_dataset, slice_motion
from videotomocap.dataset import AMASS_SMPLH_POSE_DIM
from videotomocap.pose import SMPL_POSE_DIM, SmplMotion


def assert_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _make_motion(n=200, fps=20.0) -> SmplMotion:
    """A deterministic global-frame motion clip with a fake identity to be dropped."""
    rng = np.random.default_rng(0)
    poses = rng.normal(0, 0.01, size=(n, SMPL_POSE_DIM)).astype(np.float32)
    trans = np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0).astype(np.float32)
    return SmplMotion(poses=poses, trans=trans, fps=fps, betas=rng.normal(0, 1, 16).astype(np.float32),
                      frame="global", source_clip="v.mp4")


def _seed_caption_store(cfg, video_id, rel_path, spans, descriptions):
    """Write a Pipeline 3 manifest + one video's segments + labels."""
    segs = [Segment(i, s, e) for i, (s, e) in enumerate(spans)]
    vstore.save_segments(cfg, video_id, segs, duration=spans[-1][1])
    for i, desc in enumerate(descriptions):
        vstore.save_label(cfg, video_id, i, SegmentLabel(desc, tags=[f"t{i}"]))
    return vmf.Video(video_id=video_id, rel_path=rel_path, status=vmf.DONE)


def _setup(tmp: Path, *, with_motion=True):
    """A pose dir (Pipeline 1) + a caption work root (Pipeline 3) sharing an id."""
    pose_dir = tmp / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    ccfg = CaptionConfig(work_root=tmp / "caption")
    vid = "clip_deadbeef"
    if with_motion:
        _make_motion().save_npz(pose_dir / f"{vid}.npz")  # 200 frames @ 20fps = 10s
    video = _seed_caption_store(ccfg, vid, "a/clip.mp4",
                                spans=[(0, 3), (3, 6), (6, 9)],
                                descriptions=["a person cooking", "a person walking", "a dog resting"])
    vmf.Manifest(footage_root="x", videos=[video]).save(ccfg.manifest_path)
    return pose_dir, ccfg


# -- slice_motion (pure-ish) ------------------------------------------------

def test_slice_motion_frame_math_and_hands():
    m = _make_motion(n=200, fps=20.0)
    m.left_hand_pose = np.zeros((200, 45), np.float32)
    snip = slice_motion(m, 3.0, 6.0)
    assert snip.n_frames == 60 and snip.left_hand_pose.shape == (60, 45)
    assert slice_motion(m, 9.99, 10.0) is None                 # < 2 frames -> None
    assert slice_motion(m, 8.0, 100.0).n_frames == 40          # end clamped to the clip


# -- build_captioned_dataset ------------------------------------------------

def test_build_pairs_motion_with_captions():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pose_dir, ccfg = _setup(tmp)
        out = tmp / "ds"
        stats = build_captioned_dataset(pose_dir, ccfg.work_root, out, min_frames=30, val_fraction=0.5)
        assert stats["segments"] == 3 and stats["with_motion"] == 1
        index = json.loads((out / "index.json").read_text())["clips"]
        assert {c["clip_id"] for c in index} == {"clip_deadbeef_seg0000", "clip_deadbeef_seg0001", "clip_deadbeef_seg0002"}
        first = next(c for c in index if c["clip_id"].endswith("seg0000"))
        assert first["caption"] == "a person cooking" and first["n_frames"] == 60
        # each snippet is a real AMASS-SMPL-H file with the sliced length
        amass = np.load(out / "amass" / "clip_deadbeef_seg0000.npz")
        assert amass["poses"].shape == (60, AMASS_SMPLH_POSE_DIM)
        assert np.allclose(amass["betas"], 0.0)                # privacy: identity stays dropped


def test_split_is_by_source_video():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pose_dir, ccfg = _setup(tmp)
        out = tmp / "ds"
        build_captioned_dataset(pose_dir, ccfg.work_root, out, min_frames=30, val_fraction=0.5)
        index = json.loads((out / "index.json").read_text())["clips"]
        # all 3 segments share one source video -> all land in the SAME split (no leakage)
        assert len({c["split"] for c in index}) == 1


def test_missing_motion_is_counted_not_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pose_dir, ccfg = _setup(tmp, with_motion=False)   # captions exist, no pose npz
        stats = build_captioned_dataset(pose_dir, ccfg.work_root, tmp / "ds", min_frames=30)
        assert stats["segments"] == 0 and stats["missing_motion"] == 1


def test_too_short_segments_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pose_dir, ccfg = _setup(tmp)
        # min_frames above a 3s (60-frame) segment -> everything skipped
        stats = build_captioned_dataset(pose_dir, ccfg.work_root, tmp / "ds", min_frames=100)
        assert stats["segments"] == 0 and stats["too_short"] == 3


def test_missing_caption_manifest_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert_raises(FileNotFoundError,
                      lambda: build_captioned_dataset(tmp / "pose", tmp / "nocap", tmp / "ds"))


# -- end-to-end into Pipeline 2's text conditioning -------------------------

def test_captions_flow_into_motion_model_text_conditioning():
    from motion_model.config import MotionModelConfig
    from motion_model.trainers import get_trainer
    from motion_model.data import humanml3d_line, PLACEHOLDER_CAPTION

    # the line formatter itself
    line = humanml3d_line("A person waves")
    assert line.startswith("A person waves#") and line.rstrip().endswith("#0.0#0.0")
    assert humanml3d_line("") == PLACEHOLDER_CAPTION

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pose_dir, ccfg = _setup(tmp)
        out = tmp / "ds"
        build_captioned_dataset(pose_dir, ccfg.work_root, out, min_frames=30, val_fraction=0.5)
        mm = MotionModelConfig(dataset_dir=out, work_root=tmp / "mm", method="noop", conditioning="text")
        prepared = get_trainer(mm).prepare()
        caption_file = prepared / "texts" / "clip_deadbeef_seg0000.txt"
        assert caption_file.read_text().startswith("a person cooking#")   # real caption, not placeholder


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} captioned-dataset tests passed")


if __name__ == "__main__":
    _run_all()
