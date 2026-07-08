"""CLoSD trainer -- https://github.com/GuyTevet/CLoSD (ICLR 2025, MIT).

The closed-loop unification of a generator (A) and a physics controller (B): an
MDM-family diffusion planner (DiP) proposes motion and a PHC tracker realizes it
in sim, with the simulated state re-conditioning the next plan. Because DiP is
MDM-lineage, YOUR personal generator is the right thing to retrain into the
planner slot -- but it is not a config swap: you retrain DiP on your data (in
HumanML3D 263-d), while the identity-agnostic PHC tracker is reused.

Sharp edges (see README.md): the repo is dormant, Isaac-Gym-only (deprecated),
and its closed-loop training stage wants ~50 GB VRAM (exceeds one 24 GB card).
Prefer ProtoMotions unless you specifically want your own generator in the loop.
"""

from __future__ import annotations

from pathlib import Path

from .. import data
from .base import MotionTrainer


class CLoSDTrainer(MotionTrainer):
    """Config-selected closed-loop A+B; retrains DiP on your data, reuses PHC."""

    name = "closd"
    feature_format = "humanml3d_263"
    role = "closed-loop planner+controller (A+B)"

    def prepare(self) -> Path:
        # CLoSD's DiP trains on the same HumanML3D 263-d features as the generators.
        out = data.prepare_humanml3d(self.cfg)
        warn = data.fps_warning(self.cfg.target_fps)
        if warn:
            print(f"  WARNING: {warn}")
        print(f"  CLoSD DiP training data at {out}")
        print(data.humanml3d_handoff(self.cfg, out))
        return out

    def train(self) -> Path:
        """Retrain the DiP planner on your motion; the PHC tracker is reused as-is.

        Stage 3 (closed-loop fine-tune of the tracker against your DiP) is the
        ~50 GB Isaac-Gym step -- run it from the repo once DiP has converged.
        """
        repo = self._require_repo()
        self._stage_humanml3d(repo)   # CLoSD forks MDM, which reads ./dataset/HumanML3D
        save = self.cfg.checkpoint_dir
        save.mkdir(parents=True, exist_ok=True)

        # A bare command trains a vanilla MDM-humanml model; the actual DiP recipe
        # (autoregressive, 10 diffusion steps, BERT text encoder, ...) is added via
        # extra_args -- see the CLoSD README's "Train your own DiP" command.
        cmd = [
            self.cfg.trainer_python, "-m", "closd.diffusion_planner.train.train_mdm",
            "--save_dir", str(save),
            "--dataset", "humanml",
            "--num_steps", str(self.cfg.num_steps),
            "--batch_size", str(self.cfg.batch_size),
            "--lr", str(self.cfg.lr),
        ]
        if self.cfg.resume_checkpoint is not None:
            ckpt = self._require(self.cfg.resume_checkpoint, "resume_checkpoint")
            cmd.extend(["--resume_checkpoint", str(ckpt)])
        cmd += self._mdm_eval_flags()   # save_every/eval_every -> the overfitting guard
        cmd.extend(self.cfg.extra_args)
        self._run_train(cmd, repo)   # early-stop when enabled
        print(
            "  DiP planner training launched. Then close the loop from the repo:\n"
            "    fine-tune the PHC tracker with DiP in-the-loop (Isaac Gym, ~50 GB VRAM);\n"
            "    see the CLoSD README's stage-3 command and env.dip.model_path override."
        )
        return save
