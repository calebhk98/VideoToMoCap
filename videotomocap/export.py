"""Step 3 on-ramp: turn recovered SMPL motion into a riggable BVH skeleton.

Pipelines 1-2 give you SMPL motion (the mocap, and what the motion model emits).
Step 3 -- making a character that *looks* real move that way -- is off-the-shelf DCC
work (UE5 MetaHuman, Blender), but it needs the motion in a format those tools
ingest. BVH is that lingua franca: it imports into Blender and Unreal, and drives
MetaHuman through UE5's IK Retargeter. See docs/STEP3_RENDER.md for the full path
(and the 3D-Gaussian-avatar alternative that consumes SMPL pose directly).

This is the one piece of step 3 that belongs in-repo: a pure-NumPy exporter from our
`SmplMotion` to BVH. SMPL body_pose is already per-joint rotation *relative to the
parent* -- exactly BVH's local-rotation model -- so no forward kinematics is needed
for the animation; we only need the rest skeleton (bone offsets) once, from the
neutral SMPL model you already registered. Hands (MANO) are movement, but BVH here
carries the 24-joint SMPL body only; articulated fingers are a documented follow-on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backends.base import axis_angle_to_matrix
from .pose import SMPL_NJOINTS, SmplMotion

# SMPL 24-joint kinematic tree (parent index per joint; -1 = root). Fixed + public.
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
SMPL_JOINT_NAMES = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
)


def _children() -> Dict[int, List[int]]:
    """child lists per joint, index-ordered (deterministic DFS)."""
    kids: Dict[int, List[int]] = {i: [] for i in range(SMPL_NJOINTS)}
    for j, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            kids[parent].append(j)
    return kids


def smpl_rest_joints(model_path: Path) -> np.ndarray:
    """Neutral (betas=0) SMPL joint locations J = J_regressor @ v_template, (24,3).

    Loads the SMPL model you already registered for the HMR backends -- no second
    registration. Expects an SMPL (24-joint) neutral model with ``J_regressor`` and
    ``v_template``; SMPL-X works for the 22 body joints (hands approximate). Fails
    loud with what was missing so it's fixable.
    """
    data = _load_model(Path(model_path))
    try:
        j_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        v_template = np.asarray(data["v_template"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(
            f"{model_path} has no {exc} -- need a neutral SMPL model with J_regressor + "
            f"v_template (the one your HMR backend uses), or pass rest_joints explicitly."
        ) from exc
    joints = j_regressor @ v_template          # (njoints, 3)
    return np.asarray(joints[:SMPL_NJOINTS], dtype=np.float64)


def _load_model(path: Path) -> Dict:
    """Load an SMPL model from .npz or a pickled .pkl into a key->array dict."""
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    import pickle

    with open(path, "rb") as fh:
        return {k: v for k, v in pickle.load(fh, encoding="latin1").items()}


def _euler_zyx_deg(rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose (...,3,3) into intrinsic Z*Y*X Euler angles (degrees) matching BVH's
    ``Zrotation Yrotation Xrotation`` channel order."""
    r00, r10, r20 = rot[..., 0, 0], rot[..., 1, 0], rot[..., 2, 0]
    r21, r22 = rot[..., 2, 1], rot[..., 2, 2]
    y = np.arctan2(-r20, np.sqrt(r00 * r00 + r10 * r10))
    z = np.arctan2(r10, r00)
    x = np.arctan2(r21, r22)
    return np.degrees(z), np.degrees(y), np.degrees(x)


def motion_to_bvh(motion: SmplMotion, rest_joints: np.ndarray, *,
                  fps: Optional[float] = None, scale: float = 100.0) -> str:
    """Render `motion` (SMPL-72) as BVH text against a rest skeleton.

    ``scale`` converts metres to the target's unit (default 100 -> centimetres, what
    Blender/UE mocap import expects). ``rest_joints`` is (24,3) neutral joint
    locations (see :func:`smpl_rest_joints`).
    """
    rest = np.asarray(rest_joints, dtype=np.float64)
    if rest.shape != (SMPL_NJOINTS, 3):
        raise ValueError(f"rest_joints must be ({SMPL_NJOINTS},3); got {rest.shape}")
    offsets = _bone_offsets(rest) * scale
    hierarchy, dfs_order = _hierarchy_lines(offsets)
    fps = float(fps or motion.fps)
    motion_lines = _motion_lines(motion, dfs_order, scale)
    header = ["HIERARCHY", *hierarchy, "MOTION",
              f"Frames: {motion.n_frames}", f"Frame Time: {1.0 / fps:.8f}"]
    return "\n".join(header + motion_lines) + "\n"


def _bone_offsets(rest: np.ndarray) -> np.ndarray:
    """Rest bone vector per joint (child - parent); root offset is the origin."""
    offsets = np.zeros_like(rest)
    for j, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            offsets[j] = rest[j] - rest[parent]
    return offsets


def _hierarchy_lines(offsets: np.ndarray) -> Tuple[List[str], List[int]]:
    """Build the HIERARCHY block and the DFS joint order the MOTION rows must follow."""
    kids = _children()
    dfs: List[int] = []
    lines = _joint_block(0, "", True, offsets, kids, dfs)
    return lines, dfs


def _joint_block(idx: int, indent: str, is_root: bool, offsets: np.ndarray,
                 kids: Dict[int, List[int]], dfs: List[int]) -> List[str]:
    """Recursively emit one joint's HIERARCHY lines (root has 6 channels, else 3)."""
    off = offsets[idx]
    tag = "ROOT" if is_root else "JOINT"
    inner = indent + "\t"
    channels = ("CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
                if is_root else "CHANNELS 3 Zrotation Yrotation Xrotation")
    lines = [f"{indent}{tag} {SMPL_JOINT_NAMES[idx]}", f"{indent}{{",
             f"{inner}OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}", f"{inner}{channels}"]
    dfs.append(idx)
    if not kids[idx]:
        lines += _end_site(off, inner)
    for child in kids[idx]:
        lines += _joint_block(child, inner, False, offsets, kids, dfs)
    lines.append(f"{indent}}}")
    return lines


def _end_site(off: np.ndarray, indent: str) -> List[str]:
    """A leaf's End Site, extended along its incoming bone so it isn't degenerate."""
    return [f"{indent}End Site", f"{indent}{{",
            f"{indent}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}", f"{indent}}}"]


def _motion_lines(motion: SmplMotion, dfs_order: List[int], scale: float) -> List[str]:
    """One channel row per frame: root position then Z,Y,X Euler per joint in DFS order."""
    poses = motion.poses.reshape(motion.n_frames, SMPL_NJOINTS, 3)
    rot = axis_angle_to_matrix(poses)               # (T, 24, 3, 3)
    z, y, x = _euler_zyx_deg(rot)                    # each (T, 24)
    trans = np.asarray(motion.trans, dtype=np.float64) * scale
    rows = []
    for f in range(motion.n_frames):
        vals = [trans[f, 0], trans[f, 1], trans[f, 2]]
        for j in dfs_order:                          # root first -> its rot follows its position
            vals += [z[f, j], y[f, j], x[f, j]]
        rows.append(" ".join(f"{v:.6f}" for v in vals))
    return rows


def write_bvh(motion: SmplMotion, out_path: Path, *, rest_joints: Optional[np.ndarray] = None,
              model_path: Optional[Path] = None, scale: float = 100.0) -> Path:
    """Write `motion` to a BVH file. Provide ``rest_joints`` or ``model_path`` (one is
    required to build the rest skeleton). Returns the path written."""
    if rest_joints is None:
        if model_path is None:
            raise ValueError("write_bvh needs rest_joints or model_path to build the rest skeleton")
        rest_joints = smpl_rest_joints(model_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(motion_to_bvh(motion, rest_joints, scale=scale))
    return out_path
