"""Unit tests for the rotation / SMPL-family conversion helpers.

Runs under pytest OR directly: `python tests/test_conversions.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap.backends.base import (
    _matrix_to_axis_angle,
    assemble_smpl72,
    to_axis_angle,
)
from videotomocap.pose import SMPL_POSE_DIM, SmplMotion, anonymize, resample_fps


def _aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: (...,3) axis-angle -> (...,3,3). Reference impl for tests."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(theta < 1e-8, 0.0, aa / np.where(theta < 1e-8, 1.0, theta))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(aa.shape[:-1] + (3, 3))
    I = np.eye(3)
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def test_matrix_axis_angle_roundtrip():
    rng = np.random.default_rng(0)
    aa = rng.normal(0, 0.8, size=(50, 3))  # moderate angles
    mat = _aa_to_matrix(aa)
    back = _matrix_to_axis_angle(mat)
    assert np.allclose(back, aa, atol=1e-4), np.abs(back - aa).max()


def test_to_axis_angle_accepts_matrix_and_passthrough():
    rng = np.random.default_rng(1)
    aa = rng.normal(0, 0.5, size=(10, 24, 3))
    mat = _aa_to_matrix(aa)  # (10,24,3,3)
    conv = to_axis_angle(mat, 24)
    assert conv.shape == (10, 72)
    assert np.allclose(conv, aa.reshape(10, 72), atol=1e-4)
    # already axis-angle -> unchanged
    flat = aa.reshape(10, 72).astype(np.float32)
    assert np.allclose(to_axis_angle(flat, 24), flat)


def test_assemble_smpl72_from_smplx_63():
    # SMPL-X body (21 joints, 63) should pad two neutral hand joints -> 72.
    go = np.zeros((5, 3), np.float32)
    bp = np.ones((5, 63), np.float32)
    poses = assemble_smpl72(go, bp)
    assert poses.shape == (5, SMPL_POSE_DIM)
    assert np.allclose(poses[:, 3:66], 1.0)   # 21 body joints preserved
    assert np.allclose(poses[:, 66:], 0.0)    # hand joints neutral


def test_assemble_smpl72_from_smpl_69():
    go = np.zeros((3, 3), np.float32)
    bp = np.arange(3 * 69, dtype=np.float32).reshape(3, 69)
    poses = assemble_smpl72(go, bp)
    assert poses.shape == (3, 72)
    assert np.allclose(poses[:, 3:], bp)


def test_anonymize_drops_shape():
    m = SmplMotion(poses=np.zeros((10, 72), np.float32), trans=np.zeros((10, 3), np.float32),
                   fps=30.0, betas=np.arange(10, dtype=np.float32))
    a = anonymize(m, drop_shape=True)
    assert a.betas is None
    assert a.meta["anonymized"] is True
    # pose is preserved exactly
    assert np.allclose(a.poses, m.poses)


def test_resample_changes_length():
    m = SmplMotion(poses=np.zeros((60, 72), np.float32), trans=np.zeros((60, 3), np.float32), fps=60.0)
    r = resample_fps(m, 30.0)
    assert abs(r.n_frames - 30) <= 2
    assert r.fps == 30.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} conversion tests passed")


if __name__ == "__main__":
    _run_all()
