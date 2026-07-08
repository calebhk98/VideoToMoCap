"""Unit tests for the SMPL-24 -> target-rig retarget-config generator (videotomocap.retarget).

Pure-NumPy + model-free (a synthetic rest skeleton stands in for the gated SMPL model),
so it runs GPU-free. Verifies the bone maps that let the non-SMPL character tools be
driven by SMPL motion. Runs under pytest OR: `python tests/test_retarget.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import retarget
from videotomocap.export import SMPL_JOINT_NAMES, SMPL_PARENTS


def _rest_skeleton() -> np.ndarray:
    rng = np.random.default_rng(0)
    rest = np.zeros((len(SMPL_JOINT_NAMES), 3))
    for j, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            rest[j] = rest[parent] + rng.normal(scale=0.2, size=3)
    return rest


def test_mixamo_bone_map():
    cfg = retarget.build_retarget_config(_rest_skeleton(), "mixamo")
    assert cfg["bone_map"]["Pelvis"] == "mixamorig:Hips"
    assert cfg["bone_map"]["L_Knee"] == "mixamorig:LeftLeg"
    assert cfg["bone_map"]["L_Wrist"] == "mixamorig:LeftHand"
    assert cfg["bone_map"]["L_Hand"] is None            # no Mixamo counterpart
    assert "L_Hand" in cfg["unmapped"] and "R_Hand" in cfg["unmapped"]


def test_ue5_bone_map():
    cfg = retarget.build_retarget_config(_rest_skeleton(), "ue5")
    assert cfg["bone_map"]["Pelvis"] == "pelvis"
    assert cfg["bone_map"]["L_Elbow"] == "lowerarm_l"
    assert cfg["bone_map"]["R_Ankle"] == "foot_r"


def test_smpl_identity_map_and_parents():
    cfg = retarget.build_retarget_config(_rest_skeleton(), "smpl")
    assert cfg["bone_map"]["Pelvis"] == "Pelvis"        # identity
    assert cfg["parents"]["L_Knee"] == "L_Hip"          # kinematic parent preserved
    assert cfg["parents"]["Pelvis"] is None             # root


def test_rest_lengths_match_offsets():
    rest = _rest_skeleton()
    cfg = retarget.build_retarget_config(rest, "mixamo")
    # L_Knee's rest length == |rest[L_Knee] - rest[L_Hip]|
    li = SMPL_JOINT_NAMES.index("L_Knee")
    expected = float(np.linalg.norm(rest[li] - rest[SMPL_PARENTS[li]]))
    assert abs(cfg["rest_lengths_m"]["L_Knee"] - round(expected, 6)) < 1e-6


def test_bad_target_and_shape_raise():
    try:
        retarget.build_retarget_config(_rest_skeleton(), "unity")
        raise AssertionError("expected ValueError for unknown target")
    except ValueError:
        pass
    try:
        retarget.build_retarget_config(np.zeros((10, 3)), "mixamo")
        raise AssertionError("expected ValueError for bad rest shape")
    except ValueError:
        pass


def test_write_retarget_config_json():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "map.json"
        retarget.write_retarget_config(out, "mixamo", rest_joints=_rest_skeleton())
        data = json.loads(out.read_text())
        assert data["target"] == "mixamo" and data["bone_map"]["Head"] == "mixamorig:Head"
        try:
            retarget.write_retarget_config(Path(tmp) / "x.json", "mixamo")   # no source
            raise AssertionError("expected ValueError with no rest_joints/model")
        except ValueError:
            pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} retarget tests passed")


if __name__ == "__main__":
    _run_all()
