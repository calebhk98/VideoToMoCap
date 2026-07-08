"""Clothed single-image human reconstruction backends: ICON / ECON / SIFU.

All three share the ICON ``apps.infer`` driver: one RGB image in, a fitted SMPL-X
body (``obj/*_smpl.npy`` with betas/pose) + a clothed mesh out. They differ only in
config file, output subdir name, and whether ``-hps_type pixie`` (SMPL-X) must be
forced. So one base class builds the command and harvests the outputs; the three
subclasses are just those constants -- exactly the pattern the HMR backends use.

Drivability nuance: the *body* fit is SMPL-X, but the clothed mesh isn't guaranteed
topology-consistent with SMPL-X, so it's not ``native_smplx`` (can't be LBS-driven
as-is). ECON ships ``apps.avatarizer``/``apps.animation`` to bind + drive it -- which
dovetails with this repo's own HybrIK-X backend producing the SMPL-X motion. We
record the SMPL-X params and flag the binding step in ``meta``.

Commands verified against each repo 2026-07; re-verify against your checkout (upstream
demos drift). ICON/ECON are MPI non-commercial; SIFU's code is MIT (but the SMPL-X /
PIXIE assets it downloads keep their own non-commercial terms).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from .base import BackendError, Character, CharacterBackend

# Clothed mesh filename preference (best final surface first) across the family.
_MESH_PREFERENCE = ("_full.obj", "_refine.obj", "_recon.obj", "_remesh.obj")


class _ClothedReconBackend(CharacterBackend):
    """ICON-family driver. Subclasses set the config + output-subdir constants."""

    input_kind = "image"
    rig = "smplx"
    native_smplx = False   # body fit is SMPL-X; clothed mesh needs avatarizer binding
    cfg_file = "./configs/icon-filter.yaml"
    force_pixie = True     # force SMPL-X (hands+expression) rather than SMPL body-only
    license_note = "non-commercial research (MPI-IS)"

    def generate(self) -> Character:
        repo = self._require_repo()
        img = self._require_image()
        in_dir, out_dir = self._stage(img)
        self._run_cmd(self._build_cmd(in_dir, out_dir), cwd=repo)
        params = self._find(out_dir, ["_smpl.npy"])
        mesh = self._find(out_dir, list(_MESH_PREFERENCE))
        if params is None or mesh is None:
            raise BackendError(
                f"{self.name}: expected obj/*_smpl.npy + a clothed .obj under {out_dir}; "
                f"found none. Verify the {self.name} command against your checkout.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=mesh, asset_format="obj",
            rig="smplx", native_smplx=False, smplx_params=params,
            meta={"license": self.license_note, "source_image": str(img),
                  "drive_via": "run apps.avatarizer then apps.animation to bind+drive (SMPL-X)"},
        )

    def _stage(self, img: Path):
        """Copy the concept image into a fresh in_dir; return (in_dir, out_dir)."""
        in_dir = self.cfg.asset_dir / "in"
        out_dir = self.cfg.asset_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, in_dir / img.name)
        return in_dir, out_dir

    def _build_cmd(self, in_dir: Path, out_dir: Path) -> List[str]:
        cmd = [self.cfg.backend_python, "-m", "apps.infer", "-cfg", self.cfg_file,
               "-in_dir", str(in_dir), "-out_dir", str(out_dir)]
        if self.force_pixie:
            cmd += ["-hps_type", "pixie"]     # SMPL-X, not SMPL body-only
        cmd += self.cfg.extra_args
        return cmd

    def _find(self, out_dir: Path, suffixes: List[str]) -> Optional[Path]:
        """First output file matching any suffix (searched recursively, preference order)."""
        for suffix in suffixes:
            hits = sorted(out_dir.rglob(f"*{suffix}"))
            if hits:
                return hits[0]
        return None


class IconBackend(_ClothedReconBackend):
    name = "icon"
    role = "clothed single-image human (ICON, MPI non-commercial)"
    cfg_file = "./configs/icon-filter.yaml"


class EconBackend(_ClothedReconBackend):
    name = "econ"
    role = "clothed single-image human + avatarizer/animation (ECON, MPI non-commercial)"
    cfg_file = "./configs/econ.yaml"
    force_pixie = False     # econ.yaml already selects the SMPL-X estimator


class SifuBackend(_ClothedReconBackend):
    name = "sifu"
    role = "clothed single-image human, side-view detail (SIFU, MIT code)"
    cfg_file = "./configs/sifu.yaml"
    license_note = "MIT code (SMPL-X/PIXIE assets stay non-commercial)"


BACKENDS = {"icon": IconBackend, "econ": EconBackend, "sifu": SifuBackend}
