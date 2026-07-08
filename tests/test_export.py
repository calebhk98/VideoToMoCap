"""Tests for the SMPL->BVH step-3 export bridge (videotomocap.export).

GPU-free and model-free: a synthetic rest skeleton stands in for the gated SMPL
model (the noop pattern), so we exercise the BVH structure + rotation math without
any weights. The load-bearing checks are that every MOTION row has exactly the
channel count the HIERARCHY declares, and that the Euler decomposition round-trips
(a wrong convention would silently corrupt every exported animation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import export
from videotomocap.pose import SMPL_NJOINTS, SmplMotion


def _rest_skeleton() -> np.ndarray:
    """A synthetic but tree-consistent rest pose: each joint offset from its parent."""
    rng = np.random.default_rng(0)
    rest = np.zeros((SMPL_NJOINTS, 3))
    for j, parent in enumerate(export.SMPL_PARENTS):
        if parent >= 0:
            rest[j] = rest[parent] + rng.normal(scale=0.2, size=3)
    return rest


def _motion(t: int = 5) -> SmplMotion:
    rng = np.random.default_rng(1)
    poses = rng.normal(scale=0.3, size=(t, 72)).astype(np.float32)
    trans = rng.normal(size=(t, 3)).astype(np.float32)
    return SmplMotion(poses=poses, trans=trans, fps=30.0)


def _rz_ry_rx(z, y, x) -> np.ndarray:
    cz, sz, cy, sy, cx, sx = np.cos(z), np.sin(z), np.cos(y), np.sin(y), np.cos(x), np.sin(x)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def test_euler_zyx_round_trips():
    z0, y0, x0 = np.radians([10.0, 20.0, 30.0])
    z, y, x = export._euler_zyx_deg(_rz_ry_rx(z0, y0, x0))
    assert np.allclose([z, y, x], [10.0, 20.0, 30.0], atol=1e-6)


def test_bvh_has_hierarchy_and_motion_sections():
    bvh = export.motion_to_bvh(_motion(4), _rest_skeleton())
    assert bvh.startswith("HIERARCHY")
    assert "ROOT Pelvis" in bvh and "MOTION" in bvh
    assert "Frames: 4" in bvh and "Frame Time: 0.03333333" in bvh


def test_every_motion_row_matches_declared_channels():
    # 24 joints: root has 6 channels, the other 23 have 3 -> 6 + 23*3 = 75 per frame.
    motion = _motion(6)
    bvh = export.motion_to_bvh(motion, _rest_skeleton())
    _, _, tail = bvh.partition("Frame Time:")
    rows = tail.strip().splitlines()[1:]              # drop the Frame-Time value line
    assert len(rows) == motion.n_frames
    for row in rows:
        assert len(row.split()) == 6 + (SMPL_NJOINTS - 1) * 3


def test_declares_all_joints_and_leaf_end_sites():
    bvh = export.motion_to_bvh(_motion(2), _rest_skeleton())
    joint_decls = bvh.count("ROOT ") + bvh.count("JOINT ")
    assert joint_decls == SMPL_NJOINTS                 # every SMPL joint present once
    assert bvh.count("End Site") == 5                  # L_Foot, R_Foot, Head, L_Hand, R_Hand


def test_root_position_is_translation_scaled():
    motion = _motion(3)
    bvh = export.motion_to_bvh(motion, _rest_skeleton(), scale=100.0)
    first_row = bvh.strip().splitlines()[-motion.n_frames].split()
    expected = motion.trans[0] * 100.0
    assert np.allclose([float(v) for v in first_row[:3]], expected, atol=1e-3)


def test_zero_pose_gives_zero_rotations():
    motion = SmplMotion(poses=np.zeros((1, 72), np.float32), trans=np.zeros((1, 3), np.float32), fps=30.0)
    bvh = export.motion_to_bvh(motion, _rest_skeleton())
    row = [float(v) for v in bvh.strip().splitlines()[-1].split()]
    assert np.allclose(row, 0.0, atol=1e-6)            # identity rotations -> all-zero Euler


def test_dfs_order_covers_every_joint_root_first():
    _, dfs = export._hierarchy_lines(np.zeros((SMPL_NJOINTS, 3)))
    assert dfs[0] == 0 and sorted(dfs) == list(range(SMPL_NJOINTS))


def test_write_bvh_requires_a_skeleton_source(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "a.bvh"
        export.write_bvh(_motion(2), out, rest_joints=_rest_skeleton())
        assert out.exists() and out.read_text().startswith("HIERARCHY")
        try:
            export.write_bvh(_motion(2), Path(tmp) / "b.bvh")   # neither rest_joints nor model
            raise AssertionError("expected ValueError with no skeleton source")
        except ValueError:
            pass


def test_smpl_rest_joints_from_model_npz():
    import tempfile

    # A tiny fake SMPL model: J_regressor @ v_template -> known joints.
    with tempfile.TemporaryDirectory() as tmp:
        n_verts = 40
        j_reg = np.zeros((SMPL_NJOINTS, n_verts))
        for j in range(SMPL_NJOINTS):
            j_reg[j, j] = 1.0                          # joint j = vertex j
        v_template = np.arange(n_verts * 3).reshape(n_verts, 3).astype(float)
        p = Path(tmp) / "model.npz"
        np.savez(p, J_regressor=j_reg, v_template=v_template)
        joints = export.smpl_rest_joints(p)
        assert joints.shape == (SMPL_NJOINTS, 3)
        assert np.allclose(joints, v_template[:SMPL_NJOINTS])


def test_cli_export_bvh_end_to_end():
    import tempfile

    from videotomocap import cli

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _motion(4).save_npz(tmp / "clip0.npz")                 # a recovered pose clip
        n_verts = 40
        j_reg = np.eye(SMPL_NJOINTS, n_verts)
        np.savez(tmp / "model.npz", J_regressor=j_reg,
                 v_template=np.arange(n_verts * 3).reshape(n_verts, 3).astype(float))
        out = tmp / "clip0.bvh"
        rc = cli.main(["export-bvh", "--pose", str(tmp / "clip0.npz"),
                       "--model", str(tmp / "model.npz"), "--out", str(out)])
        assert rc == 0 and out.exists() and out.read_text().startswith("HIERARCHY")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} export tests passed")


if __name__ == "__main__":
    _run_all()
