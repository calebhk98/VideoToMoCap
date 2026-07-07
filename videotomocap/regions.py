"""Body regions -> SMPL joint indices, for flagging out-of-frame / occluded parts.

When a camera never sees part of the body (legs below a desk, a cropped head, a
person half out of frame), the HMR method still regresses a *full* SMPL body --
it infers the unseen joints rather than observing them. We can't stop it doing
that, but we can record *which* joints are guesses so they can be filtered or
masked out of training. This module is the joint bookkeeping for that.

SMPL's 24 joints:
  0 pelvis      6 spine2      12 neck        18 left_elbow
  1 left_hip    7 left_ankle  13 left_collar 19 right_elbow
  2 right_hip   8 right_ankle 14 right_collar 20 left_wrist
  3 spine1      9 spine3      15 head         21 right_wrist
  4 left_knee   10 left_foot  16 left_shoulder 22 left_hand
  5 right_knee  11 right_foot 17 right_shoulder 23 right_hand
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

# Region name -> the SMPL joint indices that become unreliable if that region is
# out of frame. Composable: 'legs' = both legs; 'left_arm' just the left chain.
BODY_REGIONS: Dict[str, List[int]] = {
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
    "feet": [10, 11],
    "arms": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
    "left_arm": [13, 16, 18, 20, 22],
    "right_arm": [14, 17, 19, 21, 23],
    "hands": [22, 23],
    "head": [12, 15],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],  # alias for 'legs' (waist-up framing)
}

N_SMPL_JOINTS = 24


def joints_for_regions(regions: Iterable[str]) -> List[int]:
    """Union of SMPL joint indices covered by the named regions (sorted, unique).

    Raises ValueError on an unknown region name so a typo in config fails loud.
    """
    out = set()
    for name in regions:
        key = name.strip().lower()
        if key not in BODY_REGIONS:
            raise ValueError(f"Unknown body region {name!r}; choose from {sorted(BODY_REGIONS)}")
        out.update(BODY_REGIONS[key])
    return sorted(out)


def infer_static_joints(motion, *, energy_threshold: float = 1e-4, min_active: int = 4) -> List[int]:
    """Automatically flag joints that never move across a clip (auto-occlusion).

    When a joint is out of frame, HMR can't observe it and typically freezes it at
    a default pose -- so a joint with essentially zero motion over the whole clip
    is a good proxy for occluded/out-of-frame, *provided the body is otherwise
    active* (if nothing moves it's a still clip, not occlusion, so we bail). This
    also catches genuinely-still limbs, which is fine for the intended use:
    masking non-informative joints out of training. Pure motion signal -- no
    camera intrinsics or body model needed.
    """
    poses = np.asarray(motion.poses)
    if poses.shape[0] < 3:
        return []
    per_joint = poses.reshape(poses.shape[0], N_SMPL_JOINTS, 3)
    energy = (np.diff(per_joint, axis=0) ** 2).sum(axis=(0, 2))  # (24,) total motion per joint
    if int((energy >= energy_threshold).sum()) < min_active:
        return []  # whole body is still -> a static clip, not an occlusion signal
    return [int(j) for j in range(N_SMPL_JOINTS) if energy[j] < energy_threshold]
