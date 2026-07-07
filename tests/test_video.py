"""Tests for raw-video pre-analysis (skip-empty + static/moving camera).

The frame-comparison math is tested on synthetic frames; the pipeline pass is
tested with an injected analyze function, so neither needs OpenCV or a video
file. Runs under pytest OR directly: `python tests/test_video.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import ingest, pipeline
from videotomocap.config import PipelineConfig
from videotomocap.video import (
    VideoError,
    activity_from_frames,
    camera_motion_from_frames,
    _phase_shift,
)


def _img(seed=0):
    return np.random.default_rng(seed).normal(128, 40, (48, 48)).astype(np.float32)


# -- frame math --------------------------------------------------------------

def test_activity_static_vs_moving():
    img = _img()
    assert activity_from_frames([img, img, img]) == 0.0            # identical -> no activity
    assert activity_from_frames([img, _img(1)]) > 0.05            # different -> activity
    assert activity_from_frames([img]) == 0.0                      # single frame


def test_phase_shift_and_camera_motion():
    img = _img()
    assert abs(_phase_shift(img, np.roll(img, 3, axis=1)) - 3.0) < 0.5   # detects 3px pan
    assert camera_motion_from_frames([img, img]) == 0.0                 # static camera


# -- pipeline pass (injected analyzer, no cv2) -------------------------------

def _footage(root, names):
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def test_analyze_videos_skips_empty_and_tags_camera_motion():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "foot"
        _footage(root, ["cam/empty.mp4", "cam/busy.mp4", "cam/panning.mp4"])
        cfg = PipelineConfig(footage_root=root, work_root=Path(tmp) / "w", backend="noop",
                             skip_empty=True, auto_camera_motion="flag",
                             empty_activity_threshold=0.01, camera_motion_threshold=2.0)
        m = ingest.scan(cfg)

        def fake(video):
            name = Path(video).name
            if name == "empty.mp4":
                return {"activity": 0.001, "camera_motion": 0.0, "n_sampled": 8}
            if name == "panning.mp4":
                return {"activity": 0.2, "camera_motion": 9.0, "n_sampled": 8}
            return {"activity": 0.2, "camera_motion": 0.2, "n_sampled": 8}

        res = pipeline.analyze_videos(cfg, m, analyze_fn=fake)
        assert res == {"skipped_empty": 1, "static": 1, "moving": 1}
        by = {c.rel_path.split("/")[-1]: c for c in m.clips}
        assert by["empty.mp4"].status == ingest.EXCLUDED and "empty" in by["empty.mp4"].note
        assert by["busy.mp4"].camera_motion == "static"
        assert by["panning.mp4"].camera_motion == "moving"


def test_analyze_videos_noop_when_disabled_and_graceful_without_cv2():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "foot"
        _footage(root, ["cam/a.mp4"])
        cfg = PipelineConfig(footage_root=root, work_root=Path(tmp) / "w", backend="noop")
        assert pipeline.analyze_videos(cfg, ingest.scan(cfg)) == {"skipped_empty": 0, "static": 0, "moving": 0}

        cfg.skip_empty = True
        m = ingest.scan(cfg)

        def missing(video):
            raise VideoError("no cv2")

        # OpenCV missing -> warns and bails, leaves clips untouched (no crash)
        pipeline.analyze_videos(cfg, m, analyze_fn=missing)
        assert all(c.status == ingest.PENDING for c in m.clips)


def test_static_hint_and_config_coercion():
    cfg = PipelineConfig(backend="noop", static_cameras=["cam_fixed"])
    fixed = ingest.Clip(clip_id="x", camera="cam_fixed", rel_path="cam_fixed/a.mp4")
    moving = ingest.Clip(clip_id="y", camera="cam_pan", rel_path="cam_pan/b.mp4")
    detected = ingest.Clip(clip_id="z", camera="cam_pan", rel_path="cam_pan/c.mp4", camera_motion="static")
    assert pipeline._static_hint(cfg, fixed)          # listed static camera
    assert not pipeline._static_hint(cfg, moving)     # unknown -> not static
    assert pipeline._static_hint(cfg, detected)       # auto-detected static
    # YAML off-boolean coercion for the new mode field
    assert PipelineConfig(auto_camera_motion=False).auto_camera_motion == "off"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} video tests passed")


if __name__ == "__main__":
    _run_all()
