"""Optional *learned* refinement -- opt-in, off by default, runs out-of-process.

`refine.py` is signal processing you can read on a laptop. This module is the
other kind: it leans on a pretrained neural model to pull recovered motion toward
a plausible-pose manifold. It is kept separate so the light modules stay light.

Learned refiners are never on the default path -- ``refine`` is off by default and
these are reached only by name via ``refine_method`` -- so the GPU-free core and
tests never touch them. And like the neural backends, each heavy model runs in
**its own env** as a subprocess (their deps are mutually incompatible with this
package and with each other), so nothing here imports torch: we write poses to an
npz, shell out to a bridging driver, and read the refined poses back.

Two are wired, selectable via ``refine_method``:

- ``dposer`` -- **DPoser-X** (moonbow721/DPoser-X, ICCV 2025), a diffusion
  whole-body pose prior. Motion-only: it denoises the recovered pose with no
  access to pixels, so it works after any backend.
- ``scorehmr`` -- **ScoreHMR** (statho/ScoreHMR, CVPR 2024, MIT), diffusion
  *image-guided* refinement. It needs the **source video** (reprojection
  guidance), so it only runs where the clip is available (the ``hmr`` stage).

Config (only the block for your chosen method matters)::

    refine: true
    refine_method: dposer            # or: scorehmr
    dposer_repo: /opt/DPoser-X
    dposer_python: /opt/miniconda3/envs/dposer/bin/python
    dposer_config: configs/body/subvp/timefc.py
    dposer_strength: 1.0
    scorehmr_repo: /opt/ScoreHMR
    scorehmr_python: /opt/miniconda3/envs/scorehmr/bin/python
    scorehmr_strength: 1.0

The npz contract (``poses`` (T,72) axis-angle + optional MANO hands) is ours and
under our control; each driver maps it to/from that tool's tensors.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .pose import SmplMotion
from .refine import _copy   # reuse the fresh-copy helper (never mutate the input)

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_DRIVERS = {
    "dposer": _SCRIPTS / "dposer_refine.py",
    "scorehmr": _SCRIPTS / "scorehmr_refine.py",
}


class LearnedRefineError(RuntimeError):
    """Raised when a learned refine pass is mis-configured or its tool fails."""


# ---------------------------------------------------------------------------
# Shared npz round-trip (identical for every learned refiner; only the runner
# -- which tool, which env, which flags -- differs)
# ---------------------------------------------------------------------------

def _write_input(motion: SmplMotion, path: Path) -> None:
    """Write the motion in our own npz contract for a driver to consume."""
    payload = {"poses": motion.poses.astype(np.float32)}
    if motion.left_hand_pose is not None:
        payload["left_hand_pose"] = motion.left_hand_pose.astype(np.float32)
    if motion.right_hand_pose is not None:
        payload["right_hand_pose"] = motion.right_hand_pose.astype(np.float32)
    np.savez(path, **payload)


def _blend(orig: np.ndarray, refined: np.ndarray, strength: float, label: str) -> np.ndarray:
    if refined.shape != orig.shape:
        raise LearnedRefineError(f"learned refiner returned {label} of shape {refined.shape}, expected {orig.shape}")
    return ((1.0 - strength) * orig + strength * refined).astype(np.float32)


def _blend_hand(orig, out, key, strength):
    if orig is None or key not in out.files:
        return None if orig is None else orig.copy()
    return _blend(orig, out[key], strength, key)


def _apply(motion: SmplMotion, strength: float, runner: Callable, tag: str) -> SmplMotion:
    """Write -> ``runner(in_path, out_path)`` -> read -> blend by ``strength``.

    ``runner`` runs the model (a subprocess in production, a stub in tests) and
    must write the refined npz to ``out_path``. No-op for strength <= 0 or clips
    too short to be worth a learned pass.
    """
    if strength <= 0.0 or motion.n_frames < 2:
        return _copy(motion)
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        _write_input(motion, in_path)
        runner(in_path, out_path)
        if not out_path.exists():
            raise LearnedRefineError(f"{tag} driver wrote no output at {out_path}")
        with np.load(out_path) as out:
            if "poses" not in out.files:
                raise LearnedRefineError(f"{tag} output {out_path} lacks 'poses'; keys: {list(out.files)}")
            poses = _blend(motion.poses, out["poses"], strength, "poses")
            left = _blend_hand(motion.left_hand_pose, out, "left_hand_pose", strength)
            right = _blend_hand(motion.right_hand_pose, out, "right_hand_pose", strength)

    return SmplMotion(
        poses=poses, trans=motion.trans.copy(), fps=motion.fps, betas=motion.betas,
        left_hand_pose=left, right_hand_pose=right, frame=motion.frame,
        source_clip=motion.source_clip,
        meta=dict(motion.meta, refine_learned=tag, refine_strength=strength),
    )


def _require_repo(repo, method: str) -> Path:
    if repo is None or not Path(repo).exists():
        raise LearnedRefineError(
            f"refine_method={method!r} needs its repo pointing at a cloned checkout (got {repo!r})."
        )
    return Path(repo)


def _subprocess_runner(python, driver: Path, repo: Path, extra, cuda_device) -> Callable:
    """A runner that shells out to ``driver`` in the tool's env (cwd = repo)."""
    def runner(in_path: Path, out_path: Path) -> None:
        cmd = [python or "python", str(driver), "--in", str(in_path), "--out", str(out_path), *extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cuda_device)) if cuda_device is not None else None
        try:
            subprocess.run(cmd, cwd=str(repo), check=True, env=env)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise LearnedRefineError(f"{driver.name} failed: {exc}") from exc
    return runner


# ---------------------------------------------------------------------------
# The refiners
# ---------------------------------------------------------------------------

def dposer_refine(motion: SmplMotion, cfg, *, video: Optional[Path] = None, runner: Optional[Callable] = None) -> SmplMotion:
    """DPoser-X pose-prior denoising (motion-only; pixels not needed)."""
    strength = float(getattr(cfg, "dposer_strength", 1.0))
    if runner is None and strength > 0.0 and motion.n_frames >= 2:
        repo = _require_repo(getattr(cfg, "dposer_repo", None), "dposer")
        runner = _subprocess_runner(
            getattr(cfg, "dposer_python", None), _DRIVERS["dposer"], repo,
            ["--config", getattr(cfg, "dposer_config", "configs/body/subvp/timefc.py")],
            getattr(cfg, "cuda_device", None),
        )
    return _apply(motion, strength, runner, "dposer")


def scorehmr_refine(motion: SmplMotion, cfg, *, video: Optional[Path] = None, runner: Optional[Callable] = None) -> SmplMotion:
    """ScoreHMR image-guided refinement. Needs the source video (reprojection)."""
    strength = float(getattr(cfg, "scorehmr_strength", 1.0))
    if runner is None and strength > 0.0 and motion.n_frames >= 2:
        if video is None:
            raise LearnedRefineError(
                "refine_method='scorehmr' is image-guided and needs the source video, "
                "which is only available during the `hmr` stage."
            )
        repo = _require_repo(getattr(cfg, "scorehmr_repo", None), "scorehmr")
        runner = _subprocess_runner(
            getattr(cfg, "scorehmr_python", None), _DRIVERS["scorehmr"], repo,
            ["--video", str(video)], getattr(cfg, "cuda_device", None),
        )
    return _apply(motion, strength, runner, "scorehmr")


_REFINERS = {"dposer": dposer_refine, "scorehmr": scorehmr_refine}


def available_refiners():
    """Learned refine methods selectable via ``refine_method``."""
    return sorted(_REFINERS)


def learned_refine(motion: SmplMotion, cfg, method: str, *, video: Optional[Path] = None,
                   runner: Optional[Callable] = None) -> SmplMotion:
    """Dispatch to the named learned refiner (``dposer`` | ``scorehmr``)."""
    fn = _REFINERS.get(method)
    if fn is None:
        raise LearnedRefineError(f"unknown learned refine method {method!r}; choose from {available_refiners()}")
    return fn(motion, cfg, video=video, runner=runner)
