#!/usr/bin/env python
"""Driver: ScoreHMR image-guided refinement, bridging our npz contract <-> ScoreHMR.

Runs INSIDE the ScoreHMR conda env (statho/ScoreHMR, MIT; torch + its HMR2.0 /
detectron2 deps) -- NOT importable by the videotomocap core. ``refine_learned.py``
shells out to it.

ScoreHMR differs from the ``dposer`` refiner: it is **image-guided**. It runs a
diffusion model over SMPL parameters and steers sampling with *reprojection*
guidance against the actual video frames, so it needs the source ``--video`` (not
just our poses). It effectively re-detects + refines, then we align its output to
our clip's frames and hand back refined SMPL-72.

Contract:
  input : ``--video`` (source clip) and ``--in`` npz (our ``poses`` (T,72), used
          to know the target frame count / as init if the API supports it).
  output: ``--out`` npz with ``poses`` (T,72) axis-angle, SAME T as the input.

The ScoreHMR call is the one place to adjust for your checkout -- upstream demo
APIs drift. It's isolated in ``run_scorehmr`` below; everything else is plain
array surgery on our own contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ScoreHMR image-guided refinement bridge")
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--video", required=True, type=Path, help="Source clip (reprojection guidance)")
    p.add_argument("--scorehmr-repo", type=Path, help="Cloned statho/ScoreHMR (added to sys.path)")
    return p.parse_args(argv)


def run_scorehmr(video_path, n_frames):
    """Return refined SMPL-72 axis-angle poses (n_frames, 72) for the primary subject.

    ADJUST FOR YOUR CHECKOUT: wire this to ScoreHMR's video pipeline. Its
    ``demo_video.py`` runs detection + HMR2.0 init + score-guided refinement over
    the frames; construct the model here, run it on ``video_path``, follow the
    dominant track, and return that track's per-frame SMPL pose as (T, 72)
    axis-angle (global_orient(3) + body_pose(69)). Align/resample to ``n_frames``
    so the result blends 1:1 with our input.
    """
    from score_hmr.utils import setup_model_and_data  # noqa: from --scorehmr-repo

    raise NotImplementedError(
        "Wire run_scorehmr() to your ScoreHMR checkout's video refinement entry "
        "point; it must return (n_frames, 72) axis-angle SMPL poses."
    )


def main(argv=None):
    import sys

    args = parse_args(argv)
    if args.scorehmr_repo is not None:
        sys.path.insert(0, str(Path(args.scorehmr_repo).resolve()))

    data = dict(np.load(args.in_path))
    n = len(data["poses"])
    poses = np.asarray(run_scorehmr(args.video, n), dtype=np.float32)
    if poses.shape != (n, 72):
        raise SystemExit(f"ScoreHMR returned {poses.shape}, expected {(n, 72)} to blend with the input")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_path, poses=poses)
    print(f"wrote ScoreHMR-refined poses to {args.out_path}")


if __name__ == "__main__":
    main()
