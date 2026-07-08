"""Retarget-config generator: SMPL-24 -> a target humanoid rig's bone map + proportions.

The character tools that aren't SMPL-X-native (mpfb2/MakeHuman -> a Mixamo rig, or a
MetaHuman/UE5 mannequin) can't be driven by SMPL motion without a bone-name/hierarchy
remap -- the single fiddliest manual step of step 3. This emits that remap as JSON the
Blender/UE retarget scripts consume: which SMPL joint maps to which target bone, the
SMPL kinematic parents, and each bone's rest length (so a retargeter can rescale to the
character's proportions). Pure NumPy; the rest skeleton comes from
``export.smpl_rest_joints`` (the neutral SMPL model you already registered) or is passed in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .export import SMPL_JOINT_NAMES, SMPL_PARENTS, smpl_rest_joints

# SMPL-24 joint name -> target rig bone (without prefix). None = no counterpart on the
# target (SMPL's coarse hand joints have no Mixamo/UE bone). Verified against the standard
# Mixamo and UE5 mannequin hierarchies; re-verify against your target rig's bone names.
_MIXAMO = {
    "Pelvis": "Hips", "L_Hip": "LeftUpLeg", "R_Hip": "RightUpLeg", "Spine1": "Spine",
    "L_Knee": "LeftLeg", "R_Knee": "RightLeg", "Spine2": "Spine1", "L_Ankle": "LeftFoot",
    "R_Ankle": "RightFoot", "Spine3": "Spine2", "L_Foot": "LeftToeBase", "R_Foot": "RightToeBase",
    "Neck": "Neck", "L_Collar": "LeftShoulder", "R_Collar": "RightShoulder", "Head": "Head",
    "L_Shoulder": "LeftArm", "R_Shoulder": "RightArm", "L_Elbow": "LeftForeArm",
    "R_Elbow": "RightForeArm", "L_Wrist": "LeftHand", "R_Wrist": "RightHand",
    "L_Hand": None, "R_Hand": None,
}
_UE5 = {
    "Pelvis": "pelvis", "Spine1": "spine_01", "Spine2": "spine_02", "Spine3": "spine_03",
    "Neck": "neck_01", "Head": "head",
    "L_Collar": "clavicle_l", "L_Shoulder": "upperarm_l", "L_Elbow": "lowerarm_l", "L_Wrist": "hand_l",
    "R_Collar": "clavicle_r", "R_Shoulder": "upperarm_r", "R_Elbow": "lowerarm_r", "R_Wrist": "hand_r",
    "L_Hip": "thigh_l", "L_Knee": "calf_l", "L_Ankle": "foot_l", "L_Foot": "ball_l",
    "R_Hip": "thigh_r", "R_Knee": "calf_r", "R_Ankle": "foot_r", "R_Foot": "ball_r",
    "L_Hand": None, "R_Hand": None,
}
# target -> (bone-name prefix, name table). 'smpl' maps every joint to itself (identity).
TARGETS = {
    "mixamo": ("mixamorig:", _MIXAMO),
    "ue5": ("", _UE5),
    "smpl": ("", {n: n for n in SMPL_JOINT_NAMES}),
}


def build_retarget_config(rest_joints: np.ndarray, target: str = "mixamo") -> Dict:
    """SMPL-24 -> target rig config (bone map + parents + rest offsets/lengths)."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}, got {target!r}")
    rest = np.asarray(rest_joints, dtype=np.float64)
    if rest.shape != (len(SMPL_JOINT_NAMES), 3):
        raise ValueError(f"rest_joints must be ({len(SMPL_JOINT_NAMES)},3); got {rest.shape}")
    prefix, table = TARGETS[target]

    bone_map, parents, offsets, lengths = {}, {}, {}, {}
    for j, name in enumerate(SMPL_JOINT_NAMES):
        tgt = table.get(name)
        bone_map[name] = (prefix + tgt) if tgt else None
        parent = SMPL_PARENTS[j]
        parents[name] = SMPL_JOINT_NAMES[parent] if parent >= 0 else None
        vec = rest[j] - rest[parent] if parent >= 0 else np.zeros(3)
        offsets[name] = [round(float(v), 6) for v in vec]
        lengths[name] = round(float(np.linalg.norm(vec)), 6)

    unmapped = sorted(n for n, b in bone_map.items() if b is None)
    return {
        "source": "smpl24", "target": target,
        "bone_map": bone_map, "parents": parents,
        "rest_offsets_m": offsets, "rest_lengths_m": lengths,
        "unmapped": unmapped,
        "notes": ("SMPL-24 -> " + target + " bone map for an IK retargeter (UE5 IK Retargeter / "
                  "Blender Auto-Rig Pro Remap / Rigify). rest_lengths let the target rescale to the "
                  "character's proportions. Unmapped joints have no target bone (drop them)."),
    }


def write_retarget_config(out_path: Path, target: str = "mixamo", *,
                          rest_joints: Optional[np.ndarray] = None,
                          model_path: Optional[Path] = None) -> Path:
    """Write the retarget config JSON. Provide ``rest_joints`` or ``model_path``."""
    if rest_joints is None:
        if model_path is None:
            raise ValueError("write_retarget_config needs rest_joints or model_path")
        rest_joints = smpl_rest_joints(model_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_retarget_config(rest_joints, target), indent=2))
    return out_path
