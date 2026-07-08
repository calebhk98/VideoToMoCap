#!/usr/bin/env python
"""Driver: SAM-Body4D -> per-frame SMPL-X npz, for the ``sambody4d`` backend.

SAM-Body4D (arXiv:2512.08406, https://github.com/gaomingqi/sam-body4d, MIT) is a
*training-free* video pipeline, not a new HMR model: SAM-3 promptable video
segmentation -> identity-consistent masklets -> Diffusion-VAS amodal/occlusion
completion -> **Meta's SAM 3D Body run per-frame, guided by the stabilized
masks**. Its temporal consistency and moving-camera/occlusion robustness come
from the *mask tracking*, so its per-frame body params are exactly the same MHR
outputs the plain ``sam3dbody`` backend already fits to SMPL-X. That is why this
driver reuses ``sam3d_to_smplx``'s conversion verbatim -- SAM-Body4D vendors the
same ``facebookresearch/sam-3d-body`` model.

Pipeline here (runs INSIDE the SAM-Body4D conda env -- torch, ``sam_body4d``,
``sam_3d_body``, Meta's ``mhr_smpl_conversion``, ``smplx`` -- none of which
belong in the GPU-free videotomocap core):

  1. **SAM-Body4D tracking + per-frame HMR** -- run its offline pipeline over the
     clip and collect, per frame, the primary subject's raw SAM 3D Body output
     dict (the occlusion-completed, mask-guided one). This is the ONE upstream
     touch-point; see ``run_sambody4d`` -- adapt it to your checkout.
  2. **MHR -> SMPL-X fit** -- Meta's ``convert_sam3d_output_to_smpl`` (Apache-2.0),
     reused from ``sam3d_to_smplx`` unchanged, so hands come out as MANO
     axis-angle (``use_pca=False``).
  3. **Save** -- one ``<out>/smplx/{frame:06d}.npz`` per frame, the standard keys
     :class:`videotomocap.backends.smplx_frames.SmplXFramesBackend` parses into
     SMPL-72 + hands.

Why not parse SAM-Body4D's own on-disk output? Its released tags (<=0.2.0) write
only per-frame ``.ply`` meshes + a camera ``.json``; the SMPL-X param export in
``app.py`` is present but commented out. Capturing the in-memory MHR dict and
running the same fit the ``sam3dbody`` backend uses is the robust path and needs
no upstream source edit.

Upstream APIs drift between revisions -- treat ``run_sambody4d``'s import paths
and the two calls it makes as the one place to adjust for your checkout (mirrors
the "verify the demo command against your checkout" note the neural backends
carry).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the SAM 3D Body driver's MHR->SMPL-X fit + per-frame writer unchanged --
# SAM-Body4D emits the same MHR outputs, so only stage 1 (tracking) is new here.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from sam3d_to_smplx import (  # noqa: E402  (path set above)
    _add_repo_to_path,
    convert_to_smplx,
    primary_person,
    save_per_frame,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM-Body4D -> per-frame SMPL-X npz")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--sambody4d-repo", type=Path, required=True,
                   help="Cloned gaomingqi/sam-body4d (added to sys.path, run dir)")
    p.add_argument("--sam3d-repo", type=Path,
                   help="Cloned facebookresearch/sam-3d-body (SAM-Body4D vendors it; on sys.path)")
    p.add_argument("--mhr-repo", type=Path, help="Cloned facebookresearch/MHR (holds the conversion tool)")
    p.add_argument("--config", type=Path, help="SAM-Body4D config (default: <repo>/configs/body4d.yaml)")
    p.add_argument("--mhr-model", type=Path, required=True, help="assets/mhr_model.pt")
    p.add_argument("--smplx-dir", type=Path, required=True, help="Dir with the official SMPL-X model files")
    p.add_argument("--occlusion", action="store_true",
                   help="Enable Diffusion-VAS amodal completion (occlusion-robust but ~10x slower)")
    p.add_argument("--batch", type=int, default=256, help="Frames per MHR->SMPL-X fit batch")
    return p.parse_args(argv)


def run_sambody4d(args):
    """Stage 1: SAM-Body4D tracking + per-frame HMR -> list of (frame_index, sam3d_output_dict).

    THE upstream touch-point. SAM-Body4D's released entry point is
    ``scripts/offline_app.py``; its pipeline builds SAM-3 masklets, optionally
    runs Diffusion-VAS occlusion completion, then calls SAM 3D Body per frame
    (``process_frames``). We drive that pipeline in-process and keep, per frame,
    the primary subject's raw SAM 3D Body output dict -- the same structure
    ``estimator.process_one_image`` yields, which stage 2 already knows how to
    fit. Adapt the import + the two calls below to your checkout.
    """
    from sam_body4d.pipeline import Body4DPipeline  # noqa: import from --sambody4d-repo

    config = args.config or (Path(args.sambody4d_repo).resolve() / "configs" / "body4d.yaml")
    pipeline = Body4DPipeline.from_config(str(config), occlusion=args.occlusion)

    # Per-frame, per-person SAM 3D Body outputs, mask-guided + occlusion-completed.
    # `results` is expected to be an ordered iterable of per-frame lists of person
    # dicts (each with `bbox` + the SAM 3D Body params) -- the moving parts are
    # the mask tracking upstream of it, not this shape.
    results = pipeline.run(str(args.video))
    collected = []
    for idx, people in enumerate(results):
        person = primary_person(people)
        if person is not None:
            collected.append((idx, person))
    if not collected:
        raise SystemExit("SAM-Body4D tracked no person in the clip")
    return collected


def main(argv=None):
    args = parse_args(argv)
    _add_repo_to_path(args.sambody4d_repo)
    _add_repo_to_path(args.sam3d_repo)
    _add_repo_to_path(args.mhr_repo)
    mhr_outputs = run_sambody4d(args)
    params = convert_to_smplx(args, mhr_outputs)   # reused verbatim from sam3d_to_smplx
    save_per_frame(args.out_dir, params)


if __name__ == "__main__":
    main()
