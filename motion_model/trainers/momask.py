"""MoMask trainer -- https://github.com/EricGuo5513/momask-codes (CVPR 2024, MIT).

The recommended generator: ~44M params, native HumanML3D 263-d features (so
Pipeline 1's AMASS export feeds it via the standard conversion), much higher
fidelity than MDM, and full training is documented (train the RVQ tokenizer, then
the masked + residual transformers). Warm-start from the released checkpoint and
full-fine-tune on your data rather than training from scratch.
"""

from __future__ import annotations

from pathlib import Path

from .. import data
from .base import MotionTrainer


class MoMaskTrainer(MotionTrainer):
    """Config-selected generator; shells out to momask-codes' training scripts."""

    name = "momask"
    feature_format = "humanml3d_263"
    role = "motion generator (RVQ + masked transformer, ~44M)"

    def prepare(self) -> Path:
        out = data.prepare_humanml3d(self.cfg)
        warn = data.fps_warning(self.cfg.target_fps)
        if warn:
            print(f"  WARNING: {warn}")
        print(f"  MoMask training skeleton at {out}")
        print(data.humanml3d_handoff(self.cfg, out))
        return out

    def train(self) -> Path:
        """Train the RVQ tokenizer then the masked+residual transformers.

        MoMask has no single train command -- it's two stages. We launch the RVQ
        stage; run the transformer stages the same way once RVQ has converged
        (they need the trained RVQ name). Keeping them explicit mirrors upstream.
        MoMask also needs its pretrained evaluator bundle present in the repo
        (``bash prepare/download_evaluator.sh``) -- train_vq loads it unconditionally.
        """
        repo = self._require_repo()
        self._stage_humanml3d(repo)   # train_vq reads ./dataset/HumanML3D, not a flag
        save = self.cfg.checkpoint_dir
        save.mkdir(parents=True, exist_ok=True)
        name = "mymotion"

        cmd = [
            self.cfg.trainer_python, "train_vq.py",
            "--name", f"{name}_rvq",
            "--dataset_name", "t2m",       # HumanML3D; fixes data_root to ./dataset/HumanML3D
            "--checkpoints_dir", str(save),
            "--batch_size", str(self.cfg.batch_size),
            "--max_epoch", str(max(1, self.cfg.num_steps // 1000)),
            *self.cfg.extra_args,
        ]
        self._run_cmd(cmd, cwd=repo)
        print(
            "  RVQ stage launched. When it converges, train the generator stages:\n"
            f"    python train_t2m_transformer.py --name {name}_trans --vq_name {name}_rvq ...\n"
            f"    python train_res_transformer.py --name {name}_res  --vq_name {name}_rvq ...\n"
            "  (see momask-codes README; both read the same ./dataset/HumanML3D + --checkpoints_dir)."
        )
        return save
