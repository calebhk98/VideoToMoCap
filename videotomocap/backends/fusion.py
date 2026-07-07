"""Fusion backend -- graft a dedicated hand estimator onto a body estimate.

This is the practical path to high-detail hands on today's tools: run a body
backend (GVHMR/WHAM/TRAM/SMPLest-X) for the body + camera, run a hand
specialist (WiLoR / HaMeR) for articulated fingers, then merge. It exists
because body-only video methods do not recover fingers, and the one native
video whole-body+hands method (DanceHMR) has no public code as of 2026-07.

What the merge does (see ``graft_hands``):
  * **Fingers (easy, high value):** SMPL-X's ``*_hand_pose`` *is* MANO's finger
    articulation, so the hand net's finger pose drops in directly. This is the
    detail the user cares about ("pick up an apple and eat it").
  * **Wrist (hard, optional):** the hand net predicts the wrist in its own crop
    camera, which is NOT the body's kinematic-chain-local wrist rotation.
    Naively overwriting it twists the wrist (documented upstream: hamer#26,
    smplx#124). We therefore keep the body's wrist by default and only compose
    the hand-net wrist via forward kinematics when ``graft_wrist`` is on AND the
    hand estimate is known to share the body's frame.
  * **Smoothing:** per-frame hand nets jitter; a One-Euro filter tames it and we
    hold the last good pose over dropped frames instead of snapping to neutral.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pose import MANO_POSE_DIM, SmplMotion
from .base import (
    BackendError,
    HMRBackend,
    assemble_hand,
    axis_angle_to_matrix,
    to_axis_angle,
    _matrix_to_axis_angle,
)

# SMPL kinematic tree (parent of each of the 24 joints; -1 = root).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
L_WRIST, R_WRIST = 20, 21  # SMPL wrist joint indices (parents: L elbow 18, R elbow 19)


# ---------------------------------------------------------------------------
# Forward kinematics + smoothing (pure NumPy, fully testable)
# ---------------------------------------------------------------------------

def body_global_rotations(poses72: np.ndarray) -> np.ndarray:
    """(T,72) SMPL axis-angle -> (T,24,3,3) global rotation of every joint.

    Chains each joint's local rotation onto its parent's global rotation, which
    is what we need to express a world-frame wrist rotation back in the body's
    kinematic-local frame.
    """
    poses72 = np.asarray(poses72, dtype=np.float64)
    t = poses72.shape[0]
    local = axis_angle_to_matrix(poses72.reshape(t, 24, 3))  # (T,24,3,3)
    glob: List[np.ndarray] = [local[:, 0]]
    for j in range(1, 24):
        glob.append(glob[SMPL_PARENTS[j]] @ local[:, j])
    return np.stack(glob, axis=1)


def one_euro_smooth(seq: np.ndarray, *, alpha: float = 0.4) -> np.ndarray:
    """Cheap causal EMA smoother over a (T, D) sequence.

    Not the full One-Euro filter (no adaptive cutoff), but the same intent:
    suppress per-frame hand-net jitter while preserving intentional motion.
    ``alpha`` in (0,1]; lower = smoother.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[0] < 2:
        return seq
    out = seq.copy()
    for i in range(1, seq.shape[0]):
        out[i] = alpha * seq[i] + (1.0 - alpha) * out[i - 1]
    return out


def _fill_invalid(hand: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
    """Hold the last valid frame over dropped detections (else leave neutral)."""
    if valid is None:
        return hand
    out = hand.copy()
    last = None
    for i in range(out.shape[0]):
        if valid[i]:
            last = out[i]
        elif last is not None:
            out[i] = last
    return out


def graft_hands(
    body: SmplMotion,
    left_hand_pose: Optional[np.ndarray],
    right_hand_pose: Optional[np.ndarray],
    *,
    left_wrist_aa: Optional[np.ndarray] = None,
    right_wrist_aa: Optional[np.ndarray] = None,
    left_valid: Optional[np.ndarray] = None,
    right_valid: Optional[np.ndarray] = None,
    graft_wrist: bool = False,
    smooth_alpha: float = 0.4,
) -> SmplMotion:
    """Merge hand-net fingers (and optionally wrists) into a body-only motion."""
    poses = body.poses.copy()

    def prep(hand: Optional[np.ndarray], valid) -> Optional[np.ndarray]:
        if hand is None:
            return None
        hand = _fill_invalid(np.asarray(hand, np.float32), valid)
        return one_euro_smooth(hand, alpha=smooth_alpha)

    lh = prep(left_hand_pose, left_valid)
    rh = prep(right_hand_pose, right_valid)

    # Optional wrist composition: convert the hand-net world wrist rotation into
    # the body's kinematic-local frame via inv(parent_global) @ wrist_world.
    if graft_wrist:
        glob = body_global_rotations(poses)
        if left_wrist_aa is not None:
            poses[:, L_WRIST * 3:L_WRIST * 3 + 3] = _relocalize_wrist(glob, L_WRIST, left_wrist_aa)
        if right_wrist_aa is not None:
            poses[:, R_WRIST * 3:R_WRIST * 3 + 3] = _relocalize_wrist(glob, R_WRIST, right_wrist_aa)

    return SmplMotion(
        poses=poses,
        trans=body.trans.copy(),
        fps=body.fps,
        betas=body.betas,
        left_hand_pose=lh,
        right_hand_pose=rh,
        frame=body.frame,
        source_clip=body.source_clip,
        meta=dict(body.meta, hands_grafted=True, graft_wrist=graft_wrist),
    )


def _relocalize_wrist(glob: np.ndarray, joint: int, wrist_world_aa: np.ndarray) -> np.ndarray:
    """local_wrist = inv(parent_global) @ wrist_world  -> axis-angle (T,3)."""
    parent = SMPL_PARENTS[joint]
    parent_glob = glob[:, parent]                              # (T,3,3)
    wrist_world = axis_angle_to_matrix(np.asarray(wrist_world_aa, np.float64))
    local = np.transpose(parent_glob, (0, 2, 1)) @ wrist_world  # inv = transpose for rotations
    return _matrix_to_axis_angle(local)


# ---------------------------------------------------------------------------
# Hand estimators (subprocess wrappers around the hand specialists)
# ---------------------------------------------------------------------------

class HandEstimator(ABC):
    """Recover per-frame MANO hands from a video, split into left/right."""

    name = "hand"

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def estimate(self, video_path: Path, out_dir: Path) -> Dict[str, Optional[np.ndarray]]:
        """Return dict with left/right_hand_pose (T,45), *_wrist_aa (T,3), *_valid (T,)."""

    def _parse_frames(self, files: List[Path]) -> Dict[str, Optional[np.ndarray]]:
        """Shared parser: one file per frame, each holding up to two hands.

        Expects per-frame dicts with a handedness flag (``is_right``/``right``)
        plus MANO ``hand_pose`` (45) and ``global_orient`` (3). Frames missing a
        given hand are marked invalid so smoothing can hold the last good pose.
        """
        n = len(files)
        lh = np.zeros((n, MANO_POSE_DIM), np.float32)
        rh = np.zeros((n, MANO_POSE_DIM), np.float32)
        lw = np.zeros((n, 3), np.float32)
        rw = np.zeros((n, 3), np.float32)
        lv = np.zeros(n, bool)
        rv = np.zeros(n, bool)
        for i, f in enumerate(files):
            for hand in self._detections(f):
                pose = assemble_hand(hand["hand_pose"].reshape(1, -1))[0]
                wrist = to_axis_angle(np.asarray(hand["global_orient"]).reshape(1, 3), 1)[0]
                if hand["is_right"]:
                    rh[i], rw[i], rv[i] = pose, wrist, True
                else:
                    lh[i], lw[i], lv[i] = pose, wrist, True
        return {
            "left_hand_pose": lh, "right_hand_pose": rh,
            "left_wrist_aa": lw, "right_wrist_aa": rw,
            "left_valid": lv, "right_valid": rv,
        }

    def _detections(self, path: Path) -> List[dict]:
        d = np.load(path, allow_pickle=True)
        payload = {k: d[k] for k in d.files} if isinstance(d, np.lib.npyio.NpzFile) else d.item()
        dets = payload.get("hands", [payload])  # some tools nest per-hand dicts under 'hands'
        out = []
        for det in dets:
            is_right = bool(np.asarray(det.get("is_right", det.get("right", 1))).reshape(-1)[0])
            out.append({"hand_pose": det["hand_pose"], "global_orient": det["global_orient"], "is_right": is_right})
        return out


class WiLoRHandEstimator(HandEstimator):
    """WiLoR -- fast, robust per-frame MANO with built-in handedness (CVPR 2025).

    https://github.com/rolpotamias/WiLoR. Default hand estimator: real-time-ish,
    good jitter characteristics. Weights are CC-BY-NC-ND (personal/research use).
    """

    name = "wilor"

    def estimate(self, video_path, out_dir):
        repo = self.cfg.hand_repo or self.cfg.backend_repo
        if not repo or not Path(repo).exists():
            raise BackendError("WiLoR needs `hand_repo` set to the cloned WiLoR checkout.")
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            self.cfg.hand_python or self.cfg.backend_python, "demo.py",
            "--video", str(Path(video_path).resolve()),
            "--out_folder", str(out_dir.resolve()),
            "--save_params",
        ]
        _run(argv, Path(repo), self.name, self.cfg)
        files = sorted(out_dir.rglob("*.npz")) or sorted(out_dir.rglob("*.npy"))
        if not files:
            raise BackendError(f"WiLoR produced no per-frame params under {out_dir}")
        return self._parse_frames(files)


class HaMeRHandEstimator(HandEstimator):
    """HaMeR -- ViT-H MANO hand reconstruction, MIT code (CVPR 2024).

    https://github.com/geopavlakos/hamer. Alternative to WiLoR.
    """

    name = "hamer"

    def estimate(self, video_path, out_dir):
        repo = self.cfg.hand_repo or self.cfg.backend_repo
        if not repo or not Path(repo).exists():
            raise BackendError("HaMeR needs `hand_repo` set to the cloned HaMeR checkout.")
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            self.cfg.hand_python or self.cfg.backend_python, "demo.py",
            "--video", str(Path(video_path).resolve()),
            "--out_folder", str(out_dir.resolve()),
            "--save_params", "--full_frame",
        ]
        _run(argv, Path(repo), self.name, self.cfg)
        files = sorted(out_dir.rglob("*.npz")) or sorted(out_dir.rglob("*.npy"))
        if not files:
            raise BackendError(f"HaMeR produced no per-frame params under {out_dir}")
        return self._parse_frames(files)


_HAND_ESTIMATORS = {"wilor": WiLoRHandEstimator, "hamer": HaMeRHandEstimator}


def _run(argv, cwd, name, cfg):
    import subprocess
    try:
        subprocess.run(argv, cwd=str(cwd), check=True)
    except FileNotFoundError as exc:
        raise BackendError(f"Could not launch {name}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise BackendError(f"{name} exited with status {exc.returncode}") from exc


# ---------------------------------------------------------------------------
# The fusion backend itself
# ---------------------------------------------------------------------------

class FusionBackend(HMRBackend):
    """Body backend + hand estimator -> whole-body SMPL-X motion with real hands.

    Config:
        backend: fusion
        body_backend: gvhmr        # any registered body/whole-body backend
        hand_backend: wilor        # 'wilor' | 'hamer'
        backend_repo / backend_python  -> the BODY tool
        hand_repo / hand_python        -> the HAND tool
        graft_wrist: false         # compose hand-net wrist into the body (advanced)
    """

    name = "fusion"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        from . import get_backend  # lazy: avoid circular import at module load

        body_name = getattr(self.cfg, "body_backend", "gvhmr")
        if body_name == "fusion":
            raise BackendError("fusion.body_backend cannot itself be 'fusion'")
        # Run the body method through a config clone that points `backend` at it.
        body_cfg = _clone_with(self.cfg, backend=body_name)
        body_motion = get_backend(body_cfg).run(video_path, out_dir / "body", static=static)

        hand_name = getattr(self.cfg, "hand_backend", "wilor")
        if hand_name not in _HAND_ESTIMATORS:
            raise BackendError(f"Unknown hand_backend {hand_name!r}; choose from {sorted(_HAND_ESTIMATORS)}")
        hands = _HAND_ESTIMATORS[hand_name](self.cfg).estimate(video_path, out_dir / "hands")

        merged = graft_hands(
            body_motion,
            hands["left_hand_pose"], hands["right_hand_pose"],
            left_wrist_aa=hands["left_wrist_aa"], right_wrist_aa=hands["right_wrist_aa"],
            left_valid=hands["left_valid"], right_valid=hands["right_valid"],
            graft_wrist=getattr(self.cfg, "graft_wrist", False),
        )
        merged.meta.update(body_backend=body_name, hand_backend=hand_name)
        return merged


def _clone_with(cfg, **overrides):
    """Shallow copy of a PipelineConfig with a few fields overridden."""
    import copy
    c = copy.copy(cfg)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c
