"""Synthetic character backend -- no GPU, no weights, no upstream repo.

Exercises the whole Pipeline 3 flow (generate -> render -> critique -> refine) on
any machine so the self-test and unit tests stay GPU-free, exactly like the HMR and
motion-model ``noop`` backends. It writes a tiny placeholder asset + a fake SMPL-X
param file so the critic and downstream steps have real paths to point at. Never use
its output as a real character.
"""

from __future__ import annotations

import numpy as np

from .base import Character, CharacterBackend


class NoopBackend(CharacterBackend):
    """GPU-free synthetic generator that powers the tests; see module docstring."""

    name = "noop"
    input_kind = "either"
    rig = "smplx"
    native_smplx = True
    role = "synthetic generator (tests only)"

    def generate(self) -> Character:
        out = self.cfg.asset_dir
        out.mkdir(parents=True, exist_ok=True)
        asset = out / "character.glb"
        asset.write_bytes(b"glTF-NOOP")                 # a stand-in 3D asset
        params = out / "smplx_params.npz"
        # Neutral SMPL-X: betas zeroed (privacy-neutral), a rest pose -> drivable.
        np.savez(params, betas=np.zeros(10, np.float32), body_pose=np.zeros(63, np.float32))
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=asset, asset_format="glb",
            rig="smplx", native_smplx=True, smplx_params=params,
            meta={"synthetic": True, "license": "n/a", "source_image": str(self.cfg.input_image)},
        )
