"""Tests for hand support: fusion glue (FK, wrist relocalization, smoothing),
hand-carrying SmplMotion, AMASS export with hands, and the SMPL-X frame parser.

Runs under pytest OR directly: `python tests/test_fusion.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap.backends.base import axis_angle_to_matrix, _matrix_to_axis_angle
from videotomocap.backends.fusion import (
    SMPL_PARENTS,
    L_WRIST,
    body_global_rotations,
    graft_hands,
    one_euro_smooth,
    _relocalize_wrist,
    _fill_invalid,
)
from videotomocap.dataset import SMPLH_LHAND, SMPLH_RHAND, to_amass_npz
from videotomocap.pose import MANO_POSE_DIM, SmplMotion, anonymize, resample_fps


def _body(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return SmplMotion(
        poses=rng.normal(0, 0.2, size=(n, 72)).astype(np.float32),
        trans=rng.normal(0, 0.1, size=(n, 3)).astype(np.float32),
        fps=30.0,
    )


# -- forward kinematics ------------------------------------------------------

def test_body_global_rotations_identity_pose():
    poses = np.zeros((3, 72), np.float32)
    glob = body_global_rotations(poses)
    assert glob.shape == (3, 24, 3, 3)
    assert np.allclose(glob, np.eye(3), atol=1e-6)  # all identity when pose is zero


def test_body_global_rotations_chains_parent():
    # rotate only the root; every joint's global rotation should equal the root's
    poses = np.zeros((1, 72), np.float32)
    poses[0, :3] = [0.0, 0.7, 0.0]
    glob = body_global_rotations(poses)
    root = axis_angle_to_matrix(np.array([0.0, 0.7, 0.0]))
    for j in range(24):
        assert np.allclose(glob[0, j], root, atol=1e-6), j


def test_relocalize_wrist_inverts_parent():
    # If the desired world wrist equals parent_global, local wrist must be identity.
    poses = _body(4, seed=1).poses
    glob = body_global_rotations(poses)
    parent = SMPL_PARENTS[L_WRIST]
    wrist_world_aa = np.stack([_matrix_to_axis_angle(glob[t, parent][None])[0] for t in range(4)])
    local_aa = _relocalize_wrist(glob, L_WRIST, wrist_world_aa)
    assert np.allclose(axis_angle_to_matrix(local_aa), np.eye(3), atol=1e-5)


# -- smoothing / gap fill ----------------------------------------------------

def test_one_euro_smooth_reduces_jitter():
    rng = np.random.default_rng(2)
    clean = np.linspace(0, 1, 50)[:, None] * np.ones((1, 3))
    noisy = clean + rng.normal(0, 0.1, size=clean.shape)
    smoothed = one_euro_smooth(noisy, alpha=0.2)
    assert np.var(np.diff(smoothed, axis=0)) < np.var(np.diff(noisy, axis=0))


def test_fill_invalid_holds_last_good():
    hand = np.arange(5 * 45, dtype=np.float32).reshape(5, 45)
    valid = np.array([True, False, False, True, False])
    out = _fill_invalid(hand, valid)
    assert np.allclose(out[1], hand[0]) and np.allclose(out[2], hand[0])  # held from frame 0
    assert np.allclose(out[3], hand[3])                                   # fresh detection
    assert np.allclose(out[4], hand[3])                                   # held from frame 3


# -- graft -------------------------------------------------------------------

def test_graft_hands_copies_fingers_and_keeps_body():
    body = _body(8)
    lh = np.ones((8, MANO_POSE_DIM), np.float32) * 0.3
    rh = np.ones((8, MANO_POSE_DIM), np.float32) * -0.2
    merged = graft_hands(body, lh, rh)
    assert merged.has_hands
    assert merged.left_hand_pose.shape == (8, MANO_POSE_DIM)
    # body pose untouched when graft_wrist is off
    assert np.allclose(merged.poses, body.poses)


def test_graft_wrist_changes_only_wrist_slots():
    body = _body(6, seed=7)
    lh = np.zeros((6, MANO_POSE_DIM), np.float32)
    wrist = np.tile([0.5, 0.0, 0.0], (6, 1)).astype(np.float32)
    merged = graft_hands(body, lh, None, left_wrist_aa=wrist, graft_wrist=True)
    changed = ~np.isclose(merged.poses, body.poses).all(axis=0)
    changed_idx = set(np.where(changed)[0])
    assert changed_idx <= {L_WRIST * 3, L_WRIST * 3 + 1, L_WRIST * 3 + 2}


# -- hand-carrying motion container -----------------------------------------

def test_smpl_motion_rejects_bad_hand_shape():
    try:
        SmplMotion(poses=np.zeros((5, 72), np.float32), trans=np.zeros((5, 3), np.float32),
                   fps=30.0, left_hand_pose=np.zeros((5, 30), np.float32))
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad hand shape")


def test_hands_survive_anonymize_and_resample_and_roundtrip():
    body = _body(20, seed=9)
    lh = np.linspace(0, 1, 20 * 45).reshape(20, 45).astype(np.float32)
    m = graft_hands(body, lh, None)
    # anonymize keeps hands, drops betas
    a = anonymize(SmplMotion(poses=m.poses, trans=m.trans, fps=m.fps,
                             betas=np.ones(10, np.float32), left_hand_pose=lh))
    assert a.betas is None and a.left_hand_pose is not None
    # resample keeps hands and rescales frame count
    r = resample_fps(m, 15.0)
    assert r.left_hand_pose is not None and abs(r.n_frames - 10) <= 2
    # npz round-trip preserves hands
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.npz"
        m.save_npz(p)
        back = SmplMotion.load_npz(p)
        assert back.has_hands and np.allclose(back.left_hand_pose, m.left_hand_pose, atol=1e-5)
        assert back.right_hand_pose is None  # was never set -> stays None on reload


def test_amass_export_places_hands_in_smplh_slots():
    body = _body(4)
    lh = np.full((4, 45), 0.11, np.float32)
    rh = np.full((4, 45), 0.22, np.float32)
    m = graft_hands(body, lh, rh)
    payload = to_amass_npz(m)
    assert payload["poses"].shape[1] == 156
    assert np.allclose(payload["poses"][:, SMPLH_LHAND], 0.11)
    assert np.allclose(payload["poses"][:, SMPLH_RHAND], 0.22)


def test_amass_export_neutral_hands_when_absent():
    payload = to_amass_npz(_body(4))
    assert np.allclose(payload["poses"][:, 66:], 0.0)  # all hand/face slots neutral


# -- SMPL-X frame parser (mockable without any external tool) -----------------

def test_smplx_frames_stack_parses_hands():
    from videotomocap.backends.smplx_frames import SMPLestXBackend
    from videotomocap.config import PipelineConfig

    cfg = PipelineConfig(backend="smplestx", target_fps=30.0)
    be = SMPLestXBackend(cfg)
    # three synthetic per-frame SMPL-X dicts
    dicts = []
    for i in range(3):
        dicts.append({
            "global_orient": np.zeros(3, np.float32),
            "body_pose": np.full(63, 0.01 * i, np.float32),
            "left_hand_pose": np.full(45, 0.02 * i, np.float32),
            "right_hand_pose": np.full(45, 0.03 * i, np.float32),
            "betas": np.ones(10, np.float32),
            "transl": np.array([i, 0, 0], np.float32),
        })
    motion = be._stack(dicts, Path("clip.mp4"), Path("/tmp/out"))
    assert motion.poses.shape == (3, 72)
    assert motion.has_hands
    assert np.allclose(motion.trans[:, 0], [0, 1, 2])
    assert np.allclose(motion.left_hand_pose[2], 0.04)  # 0.02 * 2


def test_hand_estimator_parse_frames_splits_left_right():
    from videotomocap.backends.fusion import WiLoRHandEstimator
    from videotomocap.config import PipelineConfig

    est = WiLoRHandEstimator(PipelineConfig(backend="fusion"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        files = []
        for i in range(3):
            # frame i: a right hand always, a left hand only on frame 0
            hands = [{"hand_pose": np.full(45, 0.1 * (i + 1), np.float32),
                      "global_orient": np.zeros(3, np.float32), "is_right": 1}]
            if i == 0:
                hands.append({"hand_pose": np.full(45, 0.9, np.float32),
                              "global_orient": np.zeros(3, np.float32), "is_right": 0})
            f = tmp / f"frame_{i:03d}.npz"
            np.savez(f, hands=np.array(hands, dtype=object))
            files.append(f)
        out = est._parse_frames(sorted(files))
        assert out["right_valid"].all()                       # right seen every frame
        assert out["left_valid"].tolist() == [True, False, False]
        assert np.allclose(out["right_hand_pose"][2], 0.3)    # 0.1 * 3


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} fusion/hand tests passed")


if __name__ == "__main__":
    _run_all()
