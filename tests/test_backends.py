"""Unit tests for backend selection + the mockable adapter internals.

The WHAM/TRAM/GVHMR parse helpers are exercised with synthetic tensors/dicts, so
no external tool or GPU is needed. Runs under pytest OR directly:
`python tests/test_backends.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap.backends import BackendError, get_backend
from videotomocap.config import PipelineConfig


def assert_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def test_get_backend_unknown_raises():
    assert_raises(BackendError, lambda: get_backend(PipelineConfig(backend="bogus")))


def test_require_repo_and_run_cmd_errors():
    from videotomocap.backends.gvhmr import GVHMRBackend
    be = GVHMRBackend(PipelineConfig(backend="gvhmr", backend_repo=None))
    assert_raises(BackendError, be._require_repo)
    be2 = GVHMRBackend(PipelineConfig(backend="gvhmr", backend_repo="/no/such/repo"))
    assert_raises(BackendError, be2._require_repo)
    # missing interpreter and nonzero exit both surface as BackendError
    assert_raises(BackendError, lambda: be._run_cmd(["/no/such/interpreter"]))
    assert_raises(BackendError, lambda: be._run_cmd([sys.executable, "-c", "import sys; sys.exit(3)"]))


def test_wham_parse_track_selects_world_and_raises():
    from videotomocap.backends.wham import WHAMBackend
    be = WHAMBackend(PipelineConfig(backend="wham", use_frame="global", target_fps=30.0))
    aa = np.zeros((5, 72), np.float32)
    world = np.ones((5, 72), np.float32) * 0.1
    tr = {"pose": aa, "trans": np.zeros((5, 3), np.float32),
          "pose_world": world, "trans_world": np.ones((5, 3), np.float32)}
    m = be._parse_track(tr, Path("clip.mp4"), Path("x.pkl"))
    assert m.poses.shape == (5, 72) and np.allclose(m.trans, 1.0)  # picked world frame
    assert_raises(BackendError, lambda: be._parse_track({"betas": np.zeros(10)}, Path("c"), Path("p")))


def test_wham_longest_track():
    from videotomocap.backends.wham import WHAMBackend
    data = {"a": {"pose": np.zeros((3, 72))}, "b": {"pose": np.zeros((9, 72))}}
    assert len(WHAMBackend._longest_track(data)["pose"]) == 9
    assert_raises(BackendError, lambda: WHAMBackend._longest_track({}))


def test_tram_parse_selects_world_and_raises():
    from videotomocap.backends.base import axis_angle_to_matrix
    from videotomocap.backends.tram import TRAMBackend
    be = TRAMBackend(PipelineConfig(backend="tram", use_frame="global", target_fps=30.0))
    with tempfile.TemporaryDirectory() as tmp:
        rotmat = axis_angle_to_matrix(np.zeros((6, 24, 3)))  # (6,24,3,3) identity
        d = {"pred_rotmat": rotmat, "pred_trans": np.zeros((6, 3), np.float32),
             "pred_trans_world": np.ones((6, 3), np.float32), "pred_shape": np.zeros(10, np.float32)}
        f = Path(tmp) / "hps_track_1.npy"
        np.save(f, d, allow_pickle=True)
        m = be._parse(f, Path("clip.mp4"))
        assert m.poses.shape == (6, 72) and np.allclose(m.trans, 1.0)  # world translation
        bad = Path(tmp) / "bad.npy"
        np.save(bad, {"pred_trans": np.zeros((2, 3))}, allow_pickle=True)
        assert_raises(BackendError, lambda: be._parse(bad, Path("c")))


def test_gvhmr_find_results_layouts_and_missing():
    from videotomocap.backends.gvhmr import GVHMRBackend
    be = GVHMRBackend(PipelineConfig(backend="gvhmr"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "clip").mkdir()
        (out / "clip" / "hmr4d_results.pt").write_bytes(b"x")
        assert be._find_results(out, "clip").name == "hmr4d_results.pt"
    with tempfile.TemporaryDirectory() as tmp:
        assert_raises(BackendError, lambda: be._find_results(Path(tmp), "clip"))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} backend tests passed")


if __name__ == "__main__":
    _run_all()
