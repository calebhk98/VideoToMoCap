"""Generative + parametric character backends: En3D / SO-SMPL / MPFB2 (+ MakeHuman,
MB-Lab guidance stubs).

Two families here:
  * Generative (text/seed/image -> textured human): En3D (Apache-2.0, SMPL-24-aligned
    rig, so plausibly drivable directly) and SO-SMPL (native SMPL-X and uniquely
    disentangles body vs. clothes, but non-commercial + slow SDS training).
  * Parametric (slider-driven): MPFB2 -- the one MakeHuman-family tool with a clean
    HEADLESS `bpy` API (`blender --background --python`). Its rig is MakeHuman/Mixamo
    topology, NOT SMPL, so its motion needs a retarget step (the seam for a future
    retarget-config generator). MakeHuman (GUI-only) and MB-Lab (end-of-life) can't be
    driven headlessly, so they're registered as loud stubs pointing at MPFB2.

Commands verified against each repo 2026-07; re-verify against your checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from .base import BackendError, Character, CharacterBackend
from .lifters import _first, _stage_image

# Headless Blender script that drives MPFB2's HumanService (centralized, shipped
# alongside; verify against your Blender + MPFB2 version).
_MPFB2_SCRIPT = "character/scripts/mpfb2_gen.py"


class En3dBackend(CharacterBackend):
    """En3D (menyifang/En3D, Apache-2.0). Generative textured human on a SMPL-24-aligned
    rig -> plausibly drivable by this repo's SMPL motion (verify empirically)."""

    name = "en3d"
    input_kind = "either"
    rig = "smpl"
    native_smplx = True   # SMPL-24-joint rig per code inspection; confirm on your data
    role = "generative textured human, SMPL-24 rig (En3D, Apache-2.0)"

    def generate(self) -> Character:
        repo = self._require_repo()
        weights = self._require(self.cfg.weights, "weights (En3D models/ dir w/ model_human.pkl)")
        out = self.cfg.asset_dir
        out.mkdir(parents=True, exist_ok=True)
        # demo.py renders a seed; the textured MESH comes from the full img/text synthesis
        # chain (run_img_synthesis.sh / run_text_synthesis.sh). We invoke the seed demo and
        # harvest a mesh if the chain produced one -- else point the user at the chain.
        cmd = [self.cfg.backend_python, "lib/3DGEN/demo.py", f"--outdir={out}", "--trunc=0.7",
               f"--seeds={self.cfg.seed}", f"--network={weights}/model_human.pkl",
               f"--camera={weights}/camera.json", "--planes=False", "--type=human", *self.cfg.extra_args]
        self._run_cmd(cmd, cwd=repo)
        asset = _first(out, ["*refine_tex.obj", "*.glb", "*.obj"])
        if asset is None:
            raise BackendError(
                "en3d: demo.py renders a seed; the textured mesh comes from En3D's multi-stage "
                "synthesis chain (run_img_synthesis.sh / run_text_synthesis.sh, needs an external "
                "ICON clone). Run that chain, then this adapter harvests the .obj. See character/README.md.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=asset, asset_format=asset.suffix.lstrip("."),
            rig="smpl", native_smplx=True,
            meta={"license": "Apache-2.0", "rig_note": "SMPL-24 (inferred from code) -- verify driving",
                  "animate": "lib/animation/animate.py -> rigged .glb (bundled headless Blender)"},
        )


class SoSmplBackend(CharacterBackend):
    """SO-SMPL (shanemankiw/SO-SMPL, non-commercial). Native SMPL-X, disentangles body
    vs. clothes -> a dressable body with skin beneath. SDS-based: slow, per-subject."""

    name = "so_smpl"
    input_kind = "text"
    rig = "smplx"
    native_smplx = True
    role = "text->SMPL-X body+clothes, disentangled (SO-SMPL, non-commercial, slow SDS)"

    def generate(self) -> Character:
        repo = self._require_repo()
        if not self.cfg.prompt:
            raise BackendError("so_smpl is text-only: set `prompt`.")
        tag = f"char_{self.cfg.seed}"
        train = [self.cfg.backend_python, "launch.py", "--config", "configs/smplplus.yaml", "--train",
                 "--gpu", "0", f"seed={self.cfg.seed}", "exp_root_dir=outputs", "name=Stage1", f"tag={tag}",
                 f"system.prompt_processor.prompt={self.cfg.prompt}", "system.geometry.model_type=smplx",
                 *self.cfg.extra_args]
        self._run_cmd(train, cwd=repo)   # SDS optimization -- hours per subject
        export = [self.cfg.backend_python, "launch.py", "--config", "configs/smplplus.yaml", "--export",
                  "--gpu", "0", "name=Stage1", "exp_root_dir=outputs", f"tag={tag}_export",
                  f"resume=outputs/Stage1/{tag}/ckpts/last.ckpt", "system.geometry.model_type=smplx",
                  "system.prompt_processor.prompt=exporting"]
        self._run_cmd(export, cwd=repo)
        asset = _first(repo / "outputs", ["*.obj", "*.glb"])
        if asset is None:
            raise BackendError(f"so_smpl: no exported mesh under {repo}/outputs; verify the --export step.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=asset, asset_format=asset.suffix.lstrip("."),
            rig="smplx", native_smplx=True,
            meta={"license": "non-commercial (+ SMPL-X non-commercial)", "disentangled": "body + clothes",
                  "note": "SDS-based, ~hours/subject; Stage 2 scripts add more garments"},
        )


class Mpfb2Backend(CharacterBackend):
    """MPFB2 (makehumancommunity/mpfb2, GPLv3/CC0). Headless parametric human via
    Blender `--background --python`. Own rig (mixamo/openpose/...), NOT SMPL -> retarget."""

    name = "mpfb2"
    input_kind = "params"
    rig = "mixamo"
    native_smplx = False
    role = "headless parametric human, MakeHuman model (MPFB2; own rig -> retarget)"

    def generate(self) -> Character:
        out = self.cfg.asset_dir / "character.fbx"
        out.parent.mkdir(parents=True, exist_ok=True)
        blender = str(self.cfg.blender or "blender")
        cmd = [blender, "--background", "--python", _MPFB2_SCRIPT, "--",
               "--rig", "mixamo", "--out", str(out), "--prompt", self.cfg.prompt, *self.cfg.extra_args]
        self._run_blender(cmd)
        if not out.exists():
            raise BackendError(f"mpfb2: no FBX at {out}; verify {_MPFB2_SCRIPT} + MPFB2 enabled in your Blender.")
        return Character(
            method=self.name, prompt=self.cfg.prompt, asset_path=out, asset_format="fbx",
            rig="mixamo", native_smplx=False,
            meta={"license": "GPLv3 code / CC0 assets",
                  "drive_via": "own rig (mixamo) -> retarget SMPL motion (export-retarget-config)"},
        )

    def _run_blender(self, cmd: List[str]) -> None:
        try:
            subprocess.run([str(c) for c in cmd], check=True, env=self._subprocess_env())
        except FileNotFoundError as exc:
            raise BackendError(f"mpfb2: Blender not found (set cfg.blender): {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise BackendError(f"mpfb2: Blender exited {exc.returncode}; verify {_MPFB2_SCRIPT}") from exc


class _HeadlessUnsupported(CharacterBackend):
    """A real tool that can't be driven headlessly -- fail loud with the local alternative."""

    reason = ""
    alternative = "method: mpfb2"

    def generate(self) -> Character:
        raise BackendError(f"{self.name}: {self.reason} Use {self.alternative} for a headless parametric human.")


class MakeHumanBackend(_HeadlessUnsupported):
    name = "makehuman"
    role = "parametric human (MakeHuman -- GUI-ONLY, no headless CLI)"
    reason = "MakeHuman has no working headless/CLI export path in the current codebase (GUI only)."


class MbLabBackend(_HeadlessUnsupported):
    name = "mblab"
    role = "parametric human (MB-Lab -- END-OF-LIFE, successor: Charmorph)"
    reason = "MB-Lab is end-of-life (successor Charmorph) with no documented headless path."


BACKENDS = {
    "en3d": En3dBackend, "so_smpl": SoSmplBackend, "mpfb2": Mpfb2Backend,
    "makehuman": MakeHumanBackend, "mblab": MbLabBackend,
}
