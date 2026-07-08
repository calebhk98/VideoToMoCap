"""Bridge the motion model's output back to SMPL, so generated motion can drive a
character (or export to BVH) -- the inverse of ``features.py`` (SMPL -> 263).

MDM/MoMask emit HumanML3D **263-d** vectors, not SMPL params; ``export.write_bvh`` and
the SMPL-X-native characters need SMPL-72 axis-angle. So this module:

  * :func:`recover_from_ric` -- pure-NumPy HumanML3D recovery: a (T,263) vector -> (T,22,3)
    joint POSITIONS. Deterministic linear algebra + quaternion integration, GPU-free.
  * :func:`fit_smpl` -- joint positions -> SMPL-72 axis-angle. This is an optimization
    (torch), so it's a subprocess seam to the standard ``joints2smpl`` optimizer that
    MDM ships; kept out of the light core exactly like the HMR backends.
  * :func:`to_smpl_motion` -- the glue the ``act`` CLI uses: read the model's output
    (SMPL npz -> pass through; 263 -> recover joints -> fit).

protomotions already emits SMPL (``amass_smplh``), so its output passes straight through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from videotomocap.pose import SmplMotion  # SmplMotion lives in Pipeline 1's package
from .config import MotionModelConfig

HML3D_DIM = 263


def _qinv(q: np.ndarray) -> np.ndarray:
    """Conjugate of a (w,x,y,z) quaternion (unit-norm assumed)."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (...,3) by quaternions ``q`` (...,4) in (w,x,y,z) form."""
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def _recover_root(data: np.ndarray):
    """Root rotation quaternion (T,4) + root position (T,3) from the 263 vector's head."""
    rot_vel = data[..., 0]
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)          # integrate angular velocity about Y
    r_rot_quat = np.zeros(data.shape[:-1] + (4,))
    r_rot_quat[..., 0] = np.cos(r_rot_ang)
    r_rot_quat[..., 2] = np.sin(r_rot_ang)

    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, 0] = data[..., :-1, 1]              # linear velocity (x,z) in root frame
    r_pos[..., 1:, 2] = data[..., :-1, 2]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)            # -> world frame
    r_pos = np.cumsum(r_pos, axis=-2)                  # integrate to a trajectory
    r_pos[..., 1] = data[..., 3]                       # absolute root height
    return r_rot_quat, r_pos


def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """HumanML3D 263-d (denormalized) -> (T, joints_num, 3) joint positions. Pure NumPy."""
    data = np.asarray(data, dtype=np.float64)
    if data.shape[-1] < 4 + (joints_num - 1) * 3:
        raise ValueError(f"expected >= {4 + (joints_num - 1) * 3} dims for {joints_num} joints; got {data.shape[-1]}")
    r_rot_quat, r_pos = _recover_root(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4].reshape(data.shape[:-1] + (joints_num - 1, 3))
    # local (root-frame) joint positions -> world, then add the root trajectory.
    q = np.broadcast_to(_qinv(r_rot_quat)[..., None, :], positions.shape[:-1] + (4,))
    positions = _qrot(q, positions)
    positions[..., 0] += r_pos[..., None, 0]
    positions[..., 2] += r_pos[..., None, 2]
    return np.concatenate([r_pos[..., None, :], positions], axis=-2)


def fit_smpl(joints: np.ndarray, cfg: MotionModelConfig, out_dir: Path) -> Path:
    """Fit SMPL-72 axis-angle to (T,22,3) joint positions via the joints2smpl optimizer.

    Optimization (torch) -> subprocess in the trainer env, like the HMR backends. Returns
    the path to an npz with ``poses (T,72)`` + ``trans (T,3)``. Wire ``cfg.repo`` to an
    MDM checkout (its ``visualize/joints2smpl`` is the reference); verify the command.
    """
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    joints_path = out_dir / "joints.npy"
    np.save(joints_path, np.asarray(joints, np.float32))
    out_npz = out_dir / "smpl_fit.npz"
    if cfg.repo is None or not Path(cfg.repo).exists():
        raise RuntimeError(
            "joints2smpl needs cfg.repo pointing at an MDM checkout (its visualize/joints2smpl "
            "optimizer turns joint positions into SMPL-72). Set repo, or use a method that emits "
            "SMPL directly (protomotions). Recovered joints were written to " + str(joints_path))
    cmd = [cfg.trainer_python, "-m", "visualize.joints2smpl.fit_seq",
           "--input", str(joints_path), "--output", str(out_npz)]
    subprocess.run([str(c) for c in cmd], cwd=str(cfg.repo), check=True)
    return out_npz


def to_smpl_motion(output_path: Path, cfg: MotionModelConfig, fps: Optional[float] = None) -> SmplMotion:
    """Read the model's output into a SmplMotion. SMPL npz passes through; a 263 array is
    recovered to joints then fit to SMPL (via :func:`fit_smpl`)."""
    output_path = Path(output_path)
    fps = float(fps or cfg.target_fps)
    data = np.load(output_path, allow_pickle=True)
    keys = set(data.files) if hasattr(data, "files") else set()

    if "poses" in keys:                                # already SMPL-72 (noop / protomotions)
        trans = data["trans"] if "trans" in keys else np.zeros((len(data["poses"]), 3), np.float32)
        return SmplMotion(poses=data["poses"], trans=trans, fps=fps)

    vec = data["motion"] if "motion" in keys else (data[data.files[0]] if keys else np.asarray(data))
    vec = np.asarray(vec)
    if vec.ndim == 3 and vec.shape[-2:] == (22, 3):    # already joint positions
        joints = vec
    elif vec.shape[-1] == HML3D_DIM:                   # HumanML3D 263 -> joints
        joints = recover_from_ric(vec, 22)
    else:
        raise ValueError(f"unrecognized model output {output_path} (shape {vec.shape}); expected "
                         f"SMPL 'poses', 22x3 joints, or a {HML3D_DIM}-d HumanML3D vector.")
    fit = fit_smpl(joints, cfg, output_path.parent / "fit")
    fd = np.load(fit)
    return SmplMotion(poses=fd["poses"], trans=fd["trans"], fps=fps)
