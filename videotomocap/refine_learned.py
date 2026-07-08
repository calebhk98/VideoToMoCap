"""Optional *learned* refinement -- opt-in, off by default, runs out-of-process.

`refine.py` is signal processing you can read on a laptop. This module is the
other kind: it leans on a pretrained neural **pose prior** to pull recovered
motion toward a plausible-pose manifold. It is deliberately kept separate so the
light modules stay light.

It is never on the default path -- ``refine`` is off by default and this is
reached only via ``refine_method: dposer`` -- so the GPU-free core and tests never
touch it. And like the neural backends, the heavy model runs in **its own env**:
DPoser-X pins ``torch 1.12.1 / CUDA 11.3``, incompatible with this package's
environment, so we do NOT import it here. We write poses to an npz, shell out to a
bridging driver (``scripts/dposer_refine.py``) in the DPoser-X env, and read the
denoised poses back. Nothing in this file imports torch.

DPoser-X (moonbow721/DPoser-X, ICCV 2025) is a diffusion whole-body pose prior;
its ``run.tester.body.motion_denoising`` frames de-noising as prior-guided
sampling. Config::

    refine: true
    refine_method: dposer
    dposer_repo: /opt/DPoser-X
    dposer_python: /opt/miniconda3/envs/dposer/bin/python
    dposer_config: configs/body/subvp/timefc.py
    dposer_strength: 1.0          # blend denoised vs original (0 = off, 1 = full)

The npz contract (``poses`` (T,72) axis-angle + optional MANO hands) is ours and
fully under our control; the driver maps it to/from DPoser-X's SMPL-X tensors.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .pose import SmplMotion
from .refine import _copy   # reuse the fresh-copy helper (never mutate the input)

# Ships alongside the package: videotomocap/refine_learned.py -> repo/scripts/.
_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "dposer_refine.py"


class LearnedRefineError(RuntimeError):
    """Raised when the learned refine pass is mis-configured or its tool fails."""


def _write_input(motion: SmplMotion, path: Path) -> None:
    """Write the motion in our own npz contract for the driver to consume."""
    payload = {"poses": motion.poses.astype(np.float32)}
    if motion.left_hand_pose is not None:
        payload["left_hand_pose"] = motion.left_hand_pose.astype(np.float32)
    if motion.right_hand_pose is not None:
        payload["right_hand_pose"] = motion.right_hand_pose.astype(np.float32)
    np.savez(path, **payload)


def _default_runner(cfg, in_path: Path, out_path: Path) -> None:
    """Shell out to the DPoser-X bridging driver in its own env (cwd = repo)."""
    repo = getattr(cfg, "dposer_repo", None)
    if repo is None or not Path(repo).exists():
        raise LearnedRefineError(
            "refine_method='dposer' needs `dposer_repo` pointing at a cloned "
            f"DPoser-X checkout (got {repo!r})."
        )
    cmd = [
        getattr(cfg, "dposer_python", None) or "python", str(_DRIVER),
        "--in", str(in_path),
        "--out", str(out_path),
        "--config", getattr(cfg, "dposer_config", "configs/body/subvp/timefc.py"),
    ]
    device = getattr(cfg, "cuda_device", None)
    env = None
    if device is not None:
        import os
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    try:
        subprocess.run(cmd, cwd=str(repo), check=True, env=env)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LearnedRefineError(f"DPoser-X driver failed: {exc}") from exc


def _blend(orig: np.ndarray, refined: np.ndarray, strength: float, label: str) -> np.ndarray:
    if refined.shape != orig.shape:
        raise LearnedRefineError(f"DPoser-X returned {label} of shape {refined.shape}, expected {orig.shape}")
    return ((1.0 - strength) * orig + strength * refined).astype(np.float32)


def _blend_hand(orig, out, key, strength):
    if orig is None or key not in out.files:
        return None if orig is None else orig.copy()
    return _blend(orig, out[key], strength, key)


def dposer_refine(motion: SmplMotion, cfg, *, runner: Callable = _default_runner) -> SmplMotion:
    """Refine ``motion`` with the DPoser-X pose prior; blend by ``dposer_strength``.

    ``runner(cfg, in_path, out_path)`` runs the model and must write the denoised
    npz to ``out_path`` (injectable so the plumbing is testable without weights).
    No-op for strength <= 0 or clips too short to be worth a prior pass.
    """
    strength = float(getattr(cfg, "dposer_strength", 1.0))
    if strength <= 0.0 or motion.n_frames < 2:
        return _copy(motion)

    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        _write_input(motion, in_path)
        runner(cfg, in_path, out_path)
        if not out_path.exists():
            raise LearnedRefineError(f"DPoser-X driver wrote no output at {out_path}")
        with np.load(out_path) as out:
            if "poses" not in out.files:
                raise LearnedRefineError(f"DPoser-X output {out_path} lacks 'poses'; keys: {list(out.files)}")
            poses = _blend(motion.poses, out["poses"], strength, "poses")
            left = _blend_hand(motion.left_hand_pose, out, "left_hand_pose", strength)
            right = _blend_hand(motion.right_hand_pose, out, "right_hand_pose", strength)

    return SmplMotion(
        poses=poses,
        trans=motion.trans.copy(),
        fps=motion.fps,
        betas=motion.betas,
        left_hand_pose=left,
        right_hand_pose=right,
        frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, refine_learned="dposer", dposer_strength=strength),
    )
