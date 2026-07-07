"""MDM trainer -- https://github.com/GuyTevet/motion-diffusion-model.

The original small (~35M) motion-diffusion model. Kept because the personalization
ecosystem (priorMDM, LoRA-MDM, CLoSD) is built on it. Two modes:

- ``personalization: full`` -> full fine-tune from the pretrained HumanML3D prior.
  ``train.train_mdm --resume_checkpoint <ckpt>`` is a working full-fine-tune path
  in both MDM and priorMDM (training_loop loads the checkpoint's weights). Set
  ``repo`` to an MDM or priorMDM checkout.
- ``personalization: lora`` -> LoRA-MDM adapters (https://github.com/haimsaw/LoRA-MDM),
  which keep the text vocabulary and shift only style -- the lighter, safer choice
  for hours of one person. Uses ``--lora_finetune --starting_checkpoint``; set
  ``repo`` to the LoRA-MDM checkout.
"""

from __future__ import annotations

from pathlib import Path

from .. import data
from .base import MotionTrainer


class MDMTrainer(MotionTrainer):
    """Config-selected generator; shells out to MDM / priorMDM / LoRA-MDM."""

    name = "mdm"
    feature_format = "humanml3d_263"
    role = "motion generator (diffusion transformer, ~35M)"

    def prepare(self) -> Path:
        out = data.prepare_humanml3d(self.cfg)
        warn = data.fps_warning(self.cfg.target_fps)
        if warn:
            print(f"  WARNING: {warn}")
        if self.cfg.conditioning == "none":
            print(
                "  NOTE: MDM has no official unconditional-on-HumanML3D mode; prepared "
                "captions are loader-valid placeholders. For real control use "
                "conditioning: text|action, or personalization: lora (keeps text control)."
            )
        print(f"  MDM training data at {out}")
        print(data.humanml3d_handoff(self.cfg, out))
        return out

    def train(self) -> Path:
        repo = self._require_repo()
        ckpt = self._require(self.cfg.resume_checkpoint, "resume_checkpoint (pretrained HumanML3D prior)")
        self._stage_humanml3d(repo)   # MDM reads ./dataset/HumanML3D; --data_dir is dead for humanml
        save = self.cfg.checkpoint_dir
        save.mkdir(parents=True, exist_ok=True)

        # train.train_mdm is the entry for both; LoRA seeds a *new* fine-tune with
        # --starting_checkpoint + --lora_finetune, a full fine-tune *resumes* the prior.
        cmd = [
            self.cfg.trainer_python, "-m", "train.train_mdm",
            "--save_dir", str(save),
            "--dataset", "humanml",
            "--num_steps", str(self.cfg.num_steps),
            "--batch_size", str(self.cfg.batch_size),
            "--lr", str(self.cfg.lr),
        ]
        if self.cfg.personalization == "lora":
            cmd += ["--lora_finetune", "--starting_checkpoint", str(ckpt)]
        else:
            cmd += ["--resume_checkpoint", str(ckpt)]
        cmd.extend(self.cfg.extra_args)
        self._run_cmd(cmd, cwd=repo)
        return save
