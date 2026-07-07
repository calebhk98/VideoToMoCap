"""MDM trainer -- https://github.com/GuyTevet/motion-diffusion-model.

The original small (~35M) motion-diffusion model. Kept because the personalization
ecosystem (priorMDM, LoRA-MDM, CLoSD) is built on it. Two modes:

- ``personalization: full`` -> full fine-tune from the pretrained HumanML3D prior.
  Vanilla ``train_mdm`` has no working resume, so this uses priorMDM's
  ``train_mdm_motion_control`` mechanics; set ``repo`` to a priorMDM checkout.
- ``personalization: lora`` -> LoRA-MDM adapters (https://github.com/haimsaw/LoRA-MDM),
  which keep the text vocabulary and shift only style -- the lighter, safer choice
  for hours of one person. Set ``repo`` to the LoRA-MDM checkout.
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
        save = self.cfg.checkpoint_dir
        save.mkdir(parents=True, exist_ok=True)

        script = "train.train_mdm" if self.cfg.personalization == "lora" else "train.train_mdm_motion_control"
        cmd = [
            self.cfg.python, "-m", script,
            "--save_dir", str(save),
            "--dataset", "humanml",
            "--data_dir", str(self.cfg.prepared_dir),
            "--resume_checkpoint", str(ckpt),
            "--num_steps", str(self.cfg.num_steps),
            "--batch_size", str(self.cfg.batch_size),
            "--lr", str(self.cfg.lr),
            "--guidance_param", str(self.cfg.guidance_param),
        ]
        if self.cfg.conditioning == "none":
            cmd.append("--unconstrained")
        cmd.extend(self.cfg.extra_args)
        self._run_cmd(cmd, cwd=repo)
        return save
