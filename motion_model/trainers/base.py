"""Trainer interface + shared helpers for Pipeline 2.

Each trainer wraps one upstream training tool behind a common interface, exactly
as Pipeline 1's backends wrap HMR tools. The orchestration layer never needs to
know which one is in use -- pick with ``MotionModelConfig.method``.

Two-phase contract, mirroring the data-flow the whole project is built around:
    prepare()  AMASS dataset -> the features/layout THIS method trains on
    train()    shell out to the upstream trainer -> checkpoints

Heavy deps (torch, Isaac Sim, ...) live in the upstream repos and are only ever
invoked via subprocess -- this package stays importable with just numpy+stdlib.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from ..config import MotionModelConfig


class TrainerError(RuntimeError):
    """Raised for trainer setup/execution problems (missing repo/asset, subprocess
    failure) -- always with actionable context on what was expected and how to fix it."""


class MotionTrainer(ABC):
    """Train a motion model from the anonymized dataset. See module docstring."""

    name = "base"
    #: what prepare() produces: 'humanml3d_263', 'amass_smplh', or 'synthetic'.
    feature_format = "amass_smplh"
    #: one-line description of the method's role, shown by the `info`/`methods` CLI.
    role = "motion model"

    def __init__(self, cfg: MotionModelConfig):
        self.cfg = cfg

    @abstractmethod
    def prepare(self) -> Path:
        """Convert the AMASS dataset into this method's training data; return its dir."""

    @abstractmethod
    def train(self) -> Path:
        """Shell out to the upstream trainer; return the checkpoint dir. Assumes
        :meth:`prepare` has run (the CLI's ``train`` runs it first)."""

    def describe(self) -> str:
        """Human-readable summary of what running this method will do (for `info`)."""
        return f"{self.name}: {self.role} (features: {self.feature_format})"

    # -- helpers shared by subprocess-based trainers --------------------
    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        """Run an upstream command, pinning the GPU when the config assigns one."""
        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
        env = self._subprocess_env()
        try:
            subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True, env=env)
        except FileNotFoundError as exc:
            raise TrainerError(
                f"Could not launch {self.name}: {exc}. Is `{self.cfg.python}` on PATH "
                f"and `repo` set to the cloned upstream checkout?"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise TrainerError(f"{self.name} exited with status {exc.returncode} on: {printable}") from exc

    def _subprocess_env(self) -> Optional[dict]:
        """Env for the training subprocess -- pins its GPU when a device is set."""
        device = self.cfg.cuda_device
        if device is None:
            return None  # inherit the parent env unchanged
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        return env

    def _require_repo(self) -> Path:
        """The cloned upstream training repo must exist; fail loud with a hint."""
        repo = self.cfg.repo
        if repo is None or not Path(repo).exists():
            raise TrainerError(
                f"{self.name} needs `repo` pointing at the cloned {self.name.upper()} "
                f"checkout (got {repo!r}). See motion_model/README.md for the repo URL."
            )
        return Path(repo)

    def _require(self, path: Optional[Path], what: str) -> Path:
        """A required asset (checkpoint, body model, ...) must exist; else fail loud."""
        if path is None or not Path(path).exists():
            raise TrainerError(f"{self.name} needs {what} (got {path!r}). Set it in the config.")
        return Path(path)
