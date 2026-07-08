"""Backend interface + the generated-character contract for Pipeline 3.

Each backend wraps one open character-generation tool behind a common interface,
exactly as Pipeline 1's ``HMRBackend`` wraps an HMR tool. The orchestration/critic
layer never needs to know which tool ran -- pick with ``CharacterConfig.method``.

The contract every backend returns is :class:`Character`. The load-bearing field is
``native_smplx``: when true, the avatar rides on the SMPL-X body (IDOL/LHM/PSHuman/…)
and is drivable by this project's SMPL motion with **no retargeting** -- that is the
seam that connects Pipeline 2's motion model to the invented character. Parametric
tools (MakeHuman/MPFB2) emit their own skeleton and need a retarget step instead.

Heavy deps (torch, the upstream repos, Blender) are only ever invoked via subprocess,
so this package stays importable with just numpy+stdlib -- the ``noop`` backend
exercises the whole flow GPU-free, like Pipeline 1's ``noop`` HMR backend.
"""

from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import CharacterConfig


class BackendError(RuntimeError):
    """Backend setup/execution problem -- always raised with actionable context on
    what was expected (a repo, a weight, an input image) and how to fix it."""


@dataclass
class Character:
    """A generated character. ``asset_path`` is the 3D file; ``native_smplx`` says
    whether it can be driven by SMPL motion directly (no retarget)."""

    method: str
    prompt: str
    asset_path: Path
    asset_format: str = "glb"            # 'glb'|'obj'|'ply'|'gaussian_ply'|'fbx'|...
    rig: str = "none"                    # 'smplx'|'smpl'|'makehuman'|'none'
    native_smplx: bool = False           # True -> drivable by SMPL(-X) motion, no retarget
    smplx_params: Optional[Path] = None  # npz with betas (+ canonical pose) when SMPL-X-native
    renders: List[Path] = field(default_factory=list)  # multi-view renders (filled by the critic)
    meta: Dict = field(default_factory=dict)           # method, license, source_image, notes, ...

    def save_manifest(self, path: Path) -> None:
        """Write a JSON sidecar describing this character (provenance + drivability)."""
        payload = {
            "method": self.method, "prompt": self.prompt,
            "asset_path": Path(self.asset_path).as_posix(), "asset_format": self.asset_format,
            "rig": self.rig, "native_smplx": self.native_smplx,
            "smplx_params": self.smplx_params.as_posix() if self.smplx_params else None,
            "renders": [Path(r).as_posix() for r in self.renders], "meta": self.meta,
        }
        Path(path).write_text(json.dumps(payload, indent=2))


class CharacterBackend(ABC):
    """Generate one invented character. See module docstring for the contract."""

    name = "base"
    #: what the backend consumes: 'image' (needs input_image), 'text', or 'either'.
    input_kind = "image"
    #: rig the output carries: 'smplx' (drivable directly), 'makehuman', 'none'.
    rig = "none"
    #: True when the result is SMPL-X-native -> Pipeline 2 motion drives it, no retarget.
    native_smplx = False
    #: one-line description for the `info`/`methods` CLI.
    role = "character generator"

    def __init__(self, cfg: CharacterConfig):
        self.cfg = cfg

    @abstractmethod
    def generate(self) -> Character:
        """Produce a character from the config (prompt and/or input_image)."""

    def describe(self) -> str:
        drive = "SMPL-X-native (no retarget)" if self.native_smplx else f"rig: {self.rig}"
        return f"{self.name}: {self.role} [{self.input_kind}-in, {drive}]"

    # -- helpers shared by subprocess-based backends --------------------
    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        """Run an upstream command, pinning the GPU when the config assigns one."""
        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
        try:
            subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                           check=True, env=self._subprocess_env())
        except FileNotFoundError as exc:
            raise BackendError(
                f"Could not launch {self.name}: {exc}. Is `{self.cfg.backend_python}` on PATH "
                f"and `repo` set to the cloned upstream checkout?") from exc
        except subprocess.CalledProcessError as exc:
            raise BackendError(f"{self.name} exited with status {exc.returncode} on: {printable}") from exc

    def _subprocess_env(self) -> Optional[dict]:
        device = self.cfg.cuda_device
        if device is None:
            return None
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        return env

    def _require_repo(self) -> Path:
        """The cloned upstream repo must exist; fail loud with a hint."""
        repo = self.cfg.repo
        if repo is None or not Path(repo).exists():
            raise BackendError(
                f"{self.name} needs `repo` pointing at the cloned {self.name.upper()} "
                f"checkout (got {repo!r}). See character/README.md for the repo URL.")
        return Path(repo)

    def _require(self, path: Optional[Path], what: str) -> Path:
        if path is None or not Path(path).exists():
            raise BackendError(f"{self.name} needs {what} (got {path!r}). Set it in the config.")
        return Path(path)

    def _require_image(self) -> Path:
        """Image-native backends need ``input_image``; point the user at how to get one."""
        img = self.cfg.input_image
        if img is None or not Path(img).exists():
            raise BackendError(
                f"{self.name} is image-native but input_image is {img!r}. Generate a concept "
                f"image from the prompt first (an image backend / FLUX), then set input_image.")
        return Path(img)
