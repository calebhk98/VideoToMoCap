#!/usr/bin/env python
"""Driver: SAM 3D Body (MHR) -> per-frame SMPL-X npz, for the ``sam3dbody`` backend.

This runs INSIDE the SAM-3D-Body conda env -- it imports torch, ``sam_3d_body``,
Meta's ``mhr_smpl_conversion`` tool, and ``smplx``, none of which belong in the
GPU-free videotomocap core. ``backends/sam3dbody.py`` shells out to it; it is NOT
imported by the package.

Two stages per frame, then a save:

  1. **MHR inference** -- ``estimator.process_one_image(frame)`` returns one dict
     per detected person; we keep the primary subject (largest bbox), matching how
     the other backends pick a single subject from multi-person output.
  2. **MHR -> SMPL-X fit** -- Meta's ``convert_sam3d_output_to_smpl`` (Apache-2.0)
     runs an optimization fit against an ``smplx.SMPLX(use_pca=False)`` target, so
     the result is real axis-angle: ``global_orient`` (T,3), ``body_pose`` (T,63),
     ``left_hand_pose``/``right_hand_pose`` (T,45 MANO), ``betas``, ``transl``.
  3. **Save** -- one ``<out>/smplx/{frame:06d}.npz`` per frame with those keys,
     which :class:`videotomocap.backends.smplx_frames.SmplXFramesBackend` parses
     into SMPL-72 + hands like any whole-body method.

Upstream APIs drift between revisions -- treat the import paths and the two
upstream calls below as the one place to adjust for your checkout (mirrors the
"verify the demo command against your checkout" note the neural backends carry).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM 3D Body -> per-frame SMPL-X npz")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--sam3d-repo", type=Path, help="Cloned facebookresearch/sam-3d-body (added to sys.path)")
    p.add_argument("--mhr-repo", type=Path, help="Cloned facebookresearch/MHR (holds the conversion tool)")
    p.add_argument("--hf-repo", default="facebook/sam-3d-body-dinov3", help="HF repo id for the estimator")
    p.add_argument("--checkpoint", type=Path, help="model.ckpt (else resolved from --hf-repo)")
    p.add_argument("--mhr-model", type=Path, help="assets/mhr_model.pt")
    p.add_argument("--smplx-dir", type=Path, required=True, help="Dir with the official SMPL-X model files")
    p.add_argument("--bbox-thr", type=float, default=0.8, help="Detector confidence threshold")
    p.add_argument("--batch", type=int, default=256, help="Frames per MHR->SMPL-X fit batch")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame (1 = all)")
    return p.parse_args(argv)


def _add_repo_to_path(repo):
    if repo is not None:
        sys.path.insert(0, str(Path(repo).resolve()))


def iter_frames(video_path, stride):
    """Yield (index, RGB uint8 frame) from the video. Lazy cv2 import (heavy dep)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video {video_path}")
    idx = kept = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield kept, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            kept += 1
        idx += 1
    cap.release()


def primary_person(people):
    """Largest-bbox detection -- the subject in personal single-person footage.

    ``people`` is ``process_one_image``'s list of per-person dicts; None if empty
    (frames with no detection are skipped so the sequence stays contiguous)."""
    if not people:
        return None
    return max(people, key=lambda d: _bbox_area(d.get("bbox")))


def _bbox_area(bbox):
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = np.asarray(bbox).reshape(-1)[:4]
    return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def run_mhr_inference(args):
    """Stage 1: per-frame MHR params for the primary subject. Returns a list of
    (frame_index, mhr_output_dict)."""
    from sam_3d_body import setup_sam_3d_body  # noqa: import from --sam3d-repo

    estimator = setup_sam_3d_body(hf_repo_id=args.hf_repo)
    collected = []
    for idx, rgb in iter_frames(args.video, args.stride):
        person = primary_person(estimator.process_one_image(rgb, bbox_thr=args.bbox_thr))
        if person is not None:
            collected.append((idx, person))
    if not collected:
        raise SystemExit("SAM 3D Body detected no person in any frame")
    return collected


def convert_to_smplx(args, mhr_outputs):
    """Stage 2: fit MHR -> SMPL-X (use_pca=False so hands come out as MANO axis-angle).
    Returns a dict of stacked (T, *) arrays with the standard SMPL-X keys."""
    import torch
    from mhr_smpl_conversion.conversion import Conversion  # from --mhr-repo/tools
    import smplx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smplx_model = smplx.SMPLX(str(args.smplx_dir), use_pca=False, flat_hand_mean=True,
                              create_expression=False, batch_size=1).to(device)
    conversion = Conversion(mhr_model_path=str(args.mhr_model), smpl_model=smplx_model, method="pytorch")

    # convert_sam3d_output_to_smpl consumes process_one_image dicts directly and
    # returns SMPL-X axis-angle params. Batched to bound the per-fit memory.
    people = [d for _, d in mhr_outputs]
    result = conversion.convert_sam3d_output_to_smpl(
        sam3d_outputs=people, return_smpl_parameters=True, is_tracking=False, batch_size=args.batch,
    )
    params = result.result_parameters
    return {k: _to_numpy(params[k]) for k in
            ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "betas", "transl")
            if k in params}


def _to_numpy(v):
    return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)


def save_per_frame(out_dir, params):
    """Write <out>/smplx/{frame:06d}.npz -- one file per frame, keys the backend reads."""
    smplx_dir = Path(out_dir) / "smplx"
    smplx_dir.mkdir(parents=True, exist_ok=True)
    n = len(params["global_orient"])
    betas = params.get("betas")
    for t in range(n):
        frame = {k: params[k][t] for k in params if k != "betas"}
        if betas is not None:
            frame["betas"] = betas[t] if len(betas) == n else betas[0]
        np.savez(smplx_dir / f"{t:06d}.npz", **frame)
    print(f"wrote {n} SMPL-X frames to {smplx_dir}")


def main(argv=None):
    args = parse_args(argv)
    _add_repo_to_path(args.sam3d_repo)
    _add_repo_to_path(args.mhr_repo)
    mhr_outputs = run_mhr_inference(args)
    params = convert_to_smplx(args, mhr_outputs)
    save_per_frame(args.out_dir, params)


if __name__ == "__main__":
    main()
