"""Feed-forward single-image -> 3D human backends: LHM / IDOL / PSHuman / Human3Diffusion.

These lift ONE concept image into a 3D avatar. Two are SMPL-X-native (LHM, IDOL) --
their output is driven by this project's SMPL motion with NO retargeting, which is the
whole point. Two are reconstruction-only (PSHuman, Human3Diffusion): a textured mesh
with no rig, so driving them needs a separate SMPL-X fit.

Honest, verified-2026-07 caveats baked into each adapter (re-verify against your
checkout -- upstream demos drift):
  * LHM     -- cleanest: `bash inference.sh` / `LHM.launch`, SMPL-X `.json` motion in,
               3D-Gaussian `.ply` out. Fits 24GB at MINI/500M. Apache-2.0.
  * IDOL    -- run_demo.py HARDCODES the input/motion paths in main(); the released
               demo exports VIDEO, not a mesh. Adapter builds the command but fails
               loud pointing you to patch run_demo (input path + gaussian export).
  * PSHuman -- >40GB VRAM as released (does NOT fit a 24GB card); reconstruction-only.
  * Human3Diffusion -- reconstruction-only (Gaussian/TSDF .ply), no SMPL-X, VRAM unstated.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from .base import BackendError, Character, CharacterBackend


def _stage_image(cfg) -> Path:
    """Copy the concept image into a fresh per-method in dir; return its path."""
    in_dir = cfg.asset_dir / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    src = Path(cfg.input_image)
    dst = in_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _first(root: Path, patterns: List[str]) -> Optional[Path]:
    """First file under ``root`` matching any glob pattern (preference order)."""
    for pat in patterns:
        hits = sorted(Path(root).rglob(pat))
        if hits:
            return hits[0]
    return None


class LhmBackend(CharacterBackend):
    """LHM (aigc3d/LHM, ICCV'25, Apache-2.0). SMPL-X-native Gaussian avatar."""

    name = "lhm"
    input_kind = "image"
    rig = "smplx"
    native_smplx = True
    role = "single-image SMPL-X Gaussian avatar (LHM, Apache-2.0, drivable)"

    def generate(self) -> Character:
        repo = self._require_repo()
        img = self._require_image()
        staged = _stage_image(self.cfg)
        model = self.cfg.weights and Path(self.cfg.weights).name or "LHM-500M-HF"
        # export_mesh=True -> a 3D-Gaussian .ply under exps/meshs/. Verify flags vs checkout.
        cmd = [self.cfg.backend_python, "-m", "LHM.launch", "infer.human_lrm",
               f"model_name={model}", f"image_input={staged}", "export_mesh=True",
               "motion_seqs_dir=None", *self.cfg.extra_args]
        self._run_cmd(cmd, cwd=repo)
        asset = _first(repo / "exps" / "meshs", ["*.ply"]) or _first(repo / "exps", ["*.ply"])
        if asset is None:
            raise BackendError(f"lhm: no .ply under {repo}/exps/meshs; verify the command vs your checkout.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=asset, asset_format="gaussian_ply",
            rig="smplx", native_smplx=True,
            meta={"license": "Apache-2.0", "model": model, "source_image": str(img),
                  "drive_via": "motion_seqs_dir = per-frame SMPL-X json (map SmplMotion -> LHM schema)"},
        )


class IdolBackend(CharacterBackend):
    """IDOL (yiyuzhuang/IDOL, CVPR'25). SMPL-X-native, but the released demo hardcodes
    inputs and exports video -- adapter flags what to patch."""

    name = "idol"
    input_kind = "image"
    rig = "smplx"
    native_smplx = True
    role = "single-image SMPL-X Gaussian avatar (IDOL; released demo is hardcoded)"

    def generate(self) -> Character:
        repo = self._require_repo()
        self._require_image()
        # run_demo.py wires only --render_mode; input/motion paths are hardcoded lists in
        # main(). Build the command, but harvest a mesh if a patched build produced one.
        cmd = [self.cfg.backend_python, "run_demo.py", "--render_mode", "reconstruct",
               *self.cfg.extra_args]
        self._run_cmd(cmd, cwd=repo)
        asset = _first(repo / "outputs", ["*.ply", "*.glb", "*.obj"])
        if asset is None:
            raise BackendError(
                "idol: released run_demo.py hardcodes its input image + motion in main() and "
                "exports only video (no mesh). Patch run_demo.py to read your input path and "
                "export the Gaussian, then re-run. See character/README.md.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=asset, asset_format=asset.suffix.lstrip("."),
            rig="smplx", native_smplx=True,
            meta={"license": "MIT (no LICENSE file in repo)", "source_image": str(self.cfg.input_image),
                  "caveat": "run_demo.py hardcodes inputs; patch to accept --input_path + mesh export"},
        )


class PsHumanBackend(CharacterBackend):
    """PSHuman (pengHTYX/PSHuman, MIT). High-detail textured mesh, reconstruction-only,
    and >40GB VRAM as released -- does NOT fit a 24GB card."""

    name = "pshuman"
    input_kind = "image"
    rig = "none"
    native_smplx = False
    role = "single-image textured mesh (PSHuman; >40GB VRAM, no rig)"

    def generate(self) -> Character:
        repo = self._require_repo()
        img = self._require_image()
        staged = _stage_image(self.cfg)
        # Background removal then 6-view diffusion + remeshing. save_glb=true for .glb.
        self._run_cmd([self.cfg.backend_python, "utils/remove_bg.py", "--path", str(staged.parent)], cwd=repo)
        cmd = [self.cfg.backend_python, "inference.py", "--config", "configs/inference-768-6view.yaml",
               "pretrained_model_name_or_path=pengHTYX/PSHuman_Unclip_768_6views",
               f"validation_dataset.root_dir={staged.parent}", "with_smpl=false",
               "recon_opt.save_glb=true", *self.cfg.extra_args]
        self._run_cmd(cmd, cwd=repo)
        asset = _first(repo / "out", ["*.glb", "*.obj"])
        if asset is None:
            raise BackendError("pshuman: no mesh under repo/out; verify the command + >40GB VRAM.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=asset, asset_format=asset.suffix.lstrip("."),
            rig="none", native_smplx=False,
            meta={"license": "MIT", "source_image": str(img), "vram": ">40GB (won't fit 24GB)",
                  "drive_via": "reconstruction-only: fit SMPL-X to the mesh to rig it"},
        )


class Human3DiffusionBackend(CharacterBackend):
    """Human3Diffusion (YuxuanSnow/Human3Diffusion, MIT). Gaussian + TSDF mesh,
    reconstruction-only (no SMPL-X)."""

    name = "human3diffusion"
    input_kind = "image"
    rig = "none"
    native_smplx = False
    role = "single-image Gaussian + TSDF mesh (Human3Diffusion; no rig)"

    def generate(self) -> Character:
        repo = self._require_repo()
        img = self._require_image()
        staged = _stage_image(self.cfg)
        cmd = [self.cfg.backend_python, "infer_mesh.py", "--test_imgs", str(staged.parent),
               "--output", str(self.cfg.asset_dir / "out"), "--checkpoints", "checkpoints",
               "--mesh_quality", "high", *self.cfg.extra_args]
        self._run_cmd(cmd, cwd=repo)
        asset = _first(self.cfg.asset_dir / "out", ["*tsdf*.ply", "*.ply"])
        if asset is None:
            raise BackendError("human3diffusion: no .ply under out; verify the command vs your checkout.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=asset, asset_format="ply",
            rig="none", native_smplx=False,
            meta={"license": "MIT", "source_image": str(img),
                  "drive_via": "reconstruction-only: no SMPL-X; fit + skin to rig"},
        )


BACKENDS = {
    "lhm": LhmBackend, "idol": IdolBackend,
    "pshuman": PsHumanBackend, "human3diffusion": Human3DiffusionBackend,
}
