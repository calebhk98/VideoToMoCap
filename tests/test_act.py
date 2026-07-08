"""Unit tests for the text-to-motion `act` flow: joints2smpl recovery + the CLI.

GPU-free: the `noop` trainer samples a synthetic SMPL-72 npz, which flows through
to_smpl_motion straight to a saved motion (no torch fit needed). The pure-NumPy
HumanML3D recovery (263 -> joints) is tested directly. Runs under pytest OR:
`python tests/test_act.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model import cli, joints2smpl
from motion_model.config import MotionModelConfig


def _cfg(tmp: Path, **kw) -> MotionModelConfig:
    return MotionModelConfig(work_root=tmp / "w", method=kw.pop("method", "noop"), **kw)


# -- pure-NumPy HumanML3D recovery ------------------------------------------

def test_recover_from_ric_shape_and_zero():
    joints = joints2smpl.recover_from_ric(np.zeros((5, 263)), 22)
    assert joints.shape == (5, 22, 3) and np.allclose(joints, 0.0)


def test_recover_from_ric_places_first_joint():
    data = np.zeros((3, 263))
    data[:, 4] = 1.0                       # first ric joint's x = 1 (SMPL joint index 1)
    joints = joints2smpl.recover_from_ric(data, 22)
    assert np.allclose(joints[:, 0], 0.0)                 # root at origin
    assert np.allclose(joints[:, 1], [1.0, 0.0, 0.0])     # joint 1 placed


def test_recover_from_ric_rejects_short_vector():
    try:
        joints2smpl.recover_from_ric(np.zeros((2, 10)), 22)
        raise AssertionError("expected ValueError for too-short vector")
    except ValueError:
        pass


# -- to_smpl_motion passthrough (SMPL npz) ----------------------------------

def test_to_smpl_motion_passthrough_smpl_npz():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = tmp / "sample.npz"
        np.savez(p, poses=np.zeros((10, 72), np.float32), trans=np.zeros((10, 3), np.float32))
        motion = joints2smpl.to_smpl_motion(p, _cfg(tmp), fps=20)
        assert motion.poses.shape == (10, 72) and motion.fps == 20


def test_to_smpl_motion_263_without_repo_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = tmp / "gen.npz"
        np.savez(p, motion=np.zeros((8, 263), np.float32))   # HumanML3D -> needs joints2smpl fit
        try:
            joints2smpl.to_smpl_motion(p, _cfg(tmp), fps=20)   # no cfg.repo -> fit unavailable
            raise AssertionError("expected RuntimeError without a joints2smpl repo")
        except RuntimeError as exc:
            assert "joints2smpl" in str(exc)


# -- act CLI end-to-end (noop) ----------------------------------------------

def test_cli_act_with_noop_writes_motion():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rc = cli.main(["--work-root", str(tmp / "w"), "--method", "noop", "act", "walk regally"])
        assert rc == 0
        assert (tmp / "w" / "act" / "motion.npz").exists()


def test_act_unsupported_method_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        # protomotions has no text-to-motion sample() -> the base default raises.
        rc = None
        try:
            rc = cli.main(["--work-root", str(Path(tmp) / "w"), "--method", "protomotions", "act", "jump"])
        except Exception as exc:  # noqa: BLE001 - CLI surfaces the TrainerError
            assert "sample" in str(exc)
            return
        assert rc != 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} act tests passed")


if __name__ == "__main__":
    _run_all()
