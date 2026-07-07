"""ProtoMotions trainer -- https://github.com/NVlabs/ProtoMotions (Apache-2.0).

The recommended physics-controller path: actively maintained, permissive licence,
multi-simulator (Isaac Lab / Newton / MuJoCo, not only the deprecated Isaac Gym),
and it unifies AMP / ASE / MaskedMimic / motion-tracking in one codebase.

Crucially its converter consumes exactly Pipeline 1's npz keys (``poses`` +
``trans`` + ``mocap_framerate``), so the AMASS export feeds it with minimal glue
-- NO HumanML3D step. The controller is identity-agnostic (it tracks poses), which
fits the betas-stripped privacy invariant. See ARCHITECTURE.md for how this (B)
layer composes with a generator (A) and renders onto MetaHuman.
"""

from __future__ import annotations

from pathlib import Path

from .. import data
from ..config import PROTOMOTIONS_EXPERIMENTS
from .base import MotionTrainer


class ProtoMotionsTrainer(MotionTrainer):
    """Config-selected physics controller; shells out to ProtoMotions."""

    name = "protomotions"
    feature_format = "amass_smplh"
    role = "physics controller (tracks generated motion in sim)"

    @property
    def motion_file(self) -> Path:
        """The packaged MotionLib the trainer consumes."""
        return self.cfg.prepared_dir / "motions.pt"

    @property
    def _amass_root(self) -> Path:
        """Staging dir for the AMASS npz. The converter writes .motion files *next
        to* its inputs, so we copy Pipeline 1's dataset here rather than mutate it."""
        return self.cfg.prepared_dir / "amass_root"

    def prepare(self) -> Path:
        """Convert the AMASS dataset into a ProtoMotions MotionLib ``.pt`` file."""
        repo = self._require_repo()
        self._require(self.cfg.smpl_model, "smpl_model (SMPL body model for the sim humanoid)")
        out = self.cfg.prepared_dir
        # Copy npz under a named subset so the converter sees an AMASS-root/<subset>/ tree.
        staged = self._amass_root / "personal"
        staged.mkdir(parents=True, exist_ok=True)
        data.copy_amass(self.cfg.dataset_dir, staged, data.load_index(self.cfg.dataset_dir)["clips"])

        # 1) AMASS npz -> .motion files, written in place under the staging tree
        #    (needs poses/trans/mocap_framerate, which dataset.py already writes).
        self._run_cmd([
            self.cfg.trainer_python, "data/scripts/convert_amass_to_proto.py", str(self._amass_root),
            "--humanoid-type", self.cfg.robot,
            "--output-fps", "30",
        ], cwd=repo)
        # 2) Package every .motion under the tree into a single MotionLib .pt.
        self._run_cmd([
            self.cfg.trainer_python, "protomotions/components/motion_lib.py",
            "--motion-path", str(self._amass_root),
            "--output-file", str(self.motion_file),
            "--device", "cpu",
        ], cwd=repo)
        print(f"  ProtoMotions MotionLib at {self.motion_file}")
        return out

    def train(self) -> Path:
        repo = self._require_repo()
        motion = self._require(self.motion_file, "the packaged MotionLib (run `prepare` first)")
        save = self.cfg.checkpoint_dir
        save.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cfg.trainer_python, "protomotions/train_agent.py",
            "--robot-name", self.cfg.robot,
            "--simulator", self.cfg.simulator,
            "--experiment-path", PROTOMOTIONS_EXPERIMENTS[self.cfg.algorithm],
            "--experiment-name", f"{self.cfg.robot}_{self.cfg.algorithm}",
            "--motion-file", str(motion),
            "--num-envs", str(self.cfg.num_envs),
            "--batch-size", str(self.cfg.batch_size),   # required by train_agent.py
            "--ngpu", str(self.cfg.ngpu),
            *self.cfg.extra_args,
        ]
        self._run_cmd(cmd, cwd=repo)
        return save
