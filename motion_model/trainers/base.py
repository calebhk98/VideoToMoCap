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

    def sample(self, prompt: str, out_dir: Path) -> Path:
        """Generate motion from a text prompt -> path to the model's output file (SMPL
        npz, 22x3 joints, or a HumanML3D-263 array). The ``act`` CLI converts it to SMPL
        via joints2smpl. Default: unsupported -- generator trainers override it."""
        raise TrainerError(
            f"{self.name} has no sample()/text-to-motion path. Use a generator method "
            f"(momask/mdm) whose model produces motion from a prompt.")

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
                f"Could not launch {self.name}: {exc}. Is `{self.cfg.trainer_python}` on PATH "
                f"and `repo` set to the cloned upstream checkout?"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise TrainerError(f"{self.name} exited with status {exc.returncode} on: {printable}") from exc

    def _run_train(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        """Launch the training command, under the early-stop monitor when enabled.

        When ``early_stop`` is off this is exactly ``_run_cmd`` (so nothing changes and
        the command-construction tests still capture here). When on, the run is
        monitored and terminated at the overfitting onset -- see earlystop.py. The val
        curve is located per trainer by metrics.py, so this works for any method that
        has an adapter there.
        """
        if not getattr(self.cfg, "early_stop", False):
            self._run_cmd(cmd, cwd=cwd)
            return
        from .. import earlystop, metrics

        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable}   (early-stop monitored)")
        print(f"  {metrics.describe_source(self.cfg)}")
        reader = metrics.curve_reader(self.cfg, self.cfg.early_stop_patience)
        result = earlystop.run_with_monitor(
            cmd, read_verdict=reader, cwd=cwd, env=self._subprocess_env(),
            poll_interval=self.cfg.early_stop_poll_s)
        if result.stopped:
            print(f"  early-stopped at overfitting onset; keep the checkpoint near step {result.best_step}")
        elif result.returncode not in (0, None):
            raise TrainerError(f"{self.name} exited with status {result.returncode} on: {printable}")

    def _mdm_eval_flags(self) -> List[str]:
        """Overfitting-guard flags for the MDM-family trainers (mdm, closd).

        Off by default. ``save_every`` gives you frequent checkpoints to fall back
        to; ``eval_every`` turns on evaluation against our held-out split (written to
        test.txt) so `overfit-report` has a val curve. Emitted before ``extra_args``
        so a user override still wins. Only these upstream flags are used because
        they're the ones MDM/priorMDM/CLoSD actually expose -- other methods document
        their own cadence knobs rather than have us guess flag names.
        """
        interval = self.cfg.save_every or self.cfg.eval_every
        flags: List[str] = []
        if interval > 0:
            flags += ["--save_interval", str(interval)]
        if self.cfg.eval_every > 0:
            flags += ["--eval_during_training", "--eval_split", "test"]
        return flags

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

    def _stage_humanml3d(self, repo: Path) -> Path:
        """Symlink the prepared data to ``<repo>/dataset/HumanML3D`` where the loader looks.

        MoMask and the MDM family hardcode the HumanML3D dataset path inside their
        own config loading -- neither wires a --data_root/--data_dir flag through for
        the humanml/t2m path -- so the prepared features have to physically sit at
        ``<repo>/dataset/HumanML3D``. Non-destructive: if a *different* dataset is
        already there (e.g. the real HumanML3D used to pretrain the prior), refuse
        rather than clobber it, and tell the user how to resolve it.
        """
        src = self.cfg.prepared_dir.resolve()
        dst = Path(repo) / "dataset" / "HumanML3D"
        if dst.is_symlink():
            if Path(os.readlink(dst)) == src:
                return dst
            dst.unlink()
        elif dst.exists():
            raise TrainerError(
                f"{dst} already exists and is not our prepared data. Move/remove it, or "
                f"symlink it to {src} yourself, so the trainer reads your motion."
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError as exc:  # e.g. Windows without privilege
            raise TrainerError(f"could not link {dst} -> {src} ({exc}); copy the data there manually.") from exc
        return dst
