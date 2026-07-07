"""Step 4 -- aggregate anonymized clips into a motion dataset.

Output is written in **AMASS-SMPL npz** form, because that is the lingua franca
the downstream motion-diffusion world already speaks: HumanML3D's preprocessing
consumes AMASS npz, and HumanML3D features are what MDM / priorMDM train on.  By
emitting AMASS-shaped files we plug into that ecosystem instead of inventing a
private format.

Each AMASS npz carries:
    poses            (T, 156)  SMPL-H layout; our SMPL-72 sits in the first 72,
                               the remaining hand/face slots are zero (neutral).
    trans            (T, 3)    root translation (metres)
    betas            (16,)     zeros -> neutral body (identity already stripped)
    gender           str
    mocap_framerate  float
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .pose import SMPL_POSE_DIM, SmplMotion

AMASS_SMPLH_POSE_DIM = 156  # 52 joints * 3


def to_amass_npz(motion: SmplMotion, gender: str = "neutral") -> Dict[str, np.ndarray]:
    """Convert a (already anonymized) SmplMotion into an AMASS-SMPL npz payload."""
    t = motion.n_frames
    poses = np.zeros((t, AMASS_SMPLH_POSE_DIM), np.float32)
    poses[:, :SMPL_POSE_DIM] = motion.poses  # global_orient + body_pose; hands/face left neutral
    return {
        "poses": poses,
        "trans": motion.trans.astype(np.float32),
        "betas": np.zeros(16, np.float32),
        "gender": np.array(gender),
        "mocap_framerate": np.array(float(motion.fps), np.float32),
    }


@dataclass
class DatasetStats:
    n_clips: int
    n_frames: int
    total_seconds: float
    fps: float

    def as_dict(self) -> Dict:
        return {
            "n_clips": self.n_clips,
            "n_frames": self.n_frames,
            "total_seconds": round(self.total_seconds, 1),
            "total_hours": round(self.total_seconds / 3600.0, 3),
            "fps": self.fps,
        }


def build_dataset(
    pose_npz_paths: List[Path],
    out_dir: Path,
    *,
    gender: str = "neutral",
    val_fraction: float = 0.05,
    min_frames: int = 30,
    seed: int = 0,
) -> DatasetStats:
    """Read anonymized pose clips and write an AMASS-format dataset + split.

    Layout produced under ``out_dir``:
        amass/<clip_id>.npz     one AMASS-SMPL file per clip
        index.json              clip list with frame counts + train/val split
        stats.json              dataset totals (hours of motion, fps, ...)
    """
    amass_dir = out_dir / "amass"
    amass_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict] = []
    total_frames = 0
    fps_seen = None
    for p in sorted(pose_npz_paths):
        motion = SmplMotion.load_npz(p)
        if motion.n_frames < min_frames:
            continue
        clip_id = Path(p).stem
        payload = to_amass_npz(motion, gender=gender)
        np.savez(amass_dir / f"{clip_id}.npz", **payload)
        entries.append({"clip_id": clip_id, "n_frames": motion.n_frames, "fps": motion.fps})
        total_frames += motion.n_frames
        fps_seen = motion.fps

    # Deterministic train/val split at the clip level (no frame leakage).
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entries))
    n_val = max(1, int(len(entries) * val_fraction)) if entries else 0
    val_ids = {entries[i]["clip_id"] for i in order[:n_val]}
    for e in entries:
        e["split"] = "val" if e["clip_id"] in val_ids else "train"

    (out_dir / "index.json").write_text(json.dumps({"clips": entries}, indent=2))

    fps = float(fps_seen) if fps_seen else 0.0
    stats = DatasetStats(
        n_clips=len(entries),
        n_frames=total_frames,
        total_seconds=total_frames / fps if fps else 0.0,
        fps=fps,
    )
    (out_dir / "stats.json").write_text(json.dumps(stats.as_dict(), indent=2))
    return stats
