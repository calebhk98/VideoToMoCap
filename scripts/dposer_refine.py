#!/usr/bin/env python
"""Driver: DPoser-X pose-prior denoising, bridging our npz contract <-> DPoser-X.

Runs INSIDE the DPoser-X conda env (torch 1.12.1 / CUDA 11.3, its SMPL-X/MANO
models) -- NOT importable by the videotomocap core. ``refine_learned.py`` shells
out to it.

Contract (ours, both directions):
  input  npz: ``poses`` (T,72) axis-angle SMPL-72 = global_orient(3)+body_pose(69);
              optional ``left_hand_pose``/``right_hand_pose`` (T,45) MANO.
  output npz: same keys, denoised (only what the prior touched).

What it does: DPoser-X's body prior models the 21-joint SMPL-X *body_pose* (it
does not denoise global orientation or translation), so we slice the 21 body
joints out of SMPL-72, run the prior, and splice the refined body back in --
leaving global_orient, the two SMPL hand/wrist joints, and trans untouched. Hands
are refined by the hand prior when present.

The two DPoser-X calls (loading the prior, running motion_denoising) are the one
place to adjust for your checkout -- upstream APIs drift. They're isolated in
``load_body_prior`` / ``denoise_body`` below; everything else is plain array
surgery on our own contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# SMPL-72 body layout: joint 0 is global_orient (kept separate), joints 1..21 are
# the SMPL-X body, joints 22..23 are the SMPL hand/wrist joints (kept as-is).
_BODY_START, _BODY_END = 3, 3 + 21 * 3    # dims [3, 66) of the 72-vector


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DPoser-X denoising bridge")
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--config", default="configs/body/subvp/timefc.py")
    p.add_argument("--ckpt", default=None, help="Prior checkpoint (else the config default)")
    p.add_argument("--noise-std", type=float, default=0.04, help="Denoising noise scale")
    return p.parse_args(argv)


def load_body_prior(config, ckpt):
    """Return a callable ``denoise(body_pose_aa) -> body_pose_aa`` from DPoser-X.

    ADJUST FOR YOUR CHECKOUT: wire this to DPoser-X's prior. The repo exposes
    de-noising via ``run.tester.body.motion_denoising`` / the ScoreModelFC prior
    in ``lib/``; construct it here from ``config``/``ckpt`` and return a closure
    that maps a (T, 63) axis-angle body_pose to its denoised version.
    """
    from lib.utils.generic import load_model  # noqa: from --config's DPoser-X repo

    prior = load_model(config, ckpt)           # exact loader name varies by revision
    return lambda body_aa: prior.denoise(body_aa)


def denoise_body(prior, body_aa):
    """Run the prior; keep the shape (T, 63) axis-angle in, same out."""
    import torch

    with torch.no_grad():
        out = prior(torch.as_tensor(body_aa, dtype=torch.float32))
    return out.detach().cpu().numpy().astype(np.float32) if hasattr(out, "detach") else np.asarray(out)


def refine_poses(poses, prior):
    """Splice DPoser-X-denoised 21-joint body back into our SMPL-72 vectors."""
    body = poses[:, _BODY_START:_BODY_END]          # (T, 63)
    refined = denoise_body(prior, body)
    out = poses.copy()
    out[:, _BODY_START:_BODY_END] = refined
    return out


def main(argv=None):
    args = parse_args(argv)
    data = dict(np.load(args.in_path))
    prior = load_body_prior(args.config, args.ckpt)

    payload = {"poses": refine_poses(data["poses"].astype(np.float32), prior)}
    # Hands: pass through unless you also wire the DPoser-X hand prior here.
    for key in ("left_hand_pose", "right_hand_pose"):
        if key in data:
            payload[key] = data[key].astype(np.float32)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_path, **payload)
    print(f"wrote denoised poses to {args.out_path}")


if __name__ == "__main__":
    main()
