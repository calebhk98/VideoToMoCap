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

from .base import MotionTrainer, TrainerError

# ProtoMotions ships one experiment folder per algorithm.
_EXPERIMENT = {
    "mimic": "examples/experiments/mimic/mlp.py",
    "amp": "examples/experiments/amp/mlp.py",
    "ase": "examples/experiments/ase/mlp.py",
    "masked_mimic": "examples/experiments/masked_mimic/mlp.py",
}


class ProtoMotionsTrainer(MotionTrainer):
    """Config-selected physics controller; shells out to ProtoMotions."""

    name = "protomotions"
    feature_format = "amass_smplh"
    role = "physics controller (tracks generated motion in sim)"

    def prepare(self) -> Path:
        """Convert the AMASS dataset into a ProtoMotions MotionLib ``.pt`` file."""
        repo = self._require_repo()
        smpl = self._require(self.cfg.smpl_model, "smpl_model (SMPL body model for the sim humanoid)")
        out = self.cfg.prepared_dir
        out.mkdir(parents=True, exist_ok=True)
        converted = out / "converted"

        # 1) AMASS npz -> ProtoMotions-format motions (needs poses/trans/mocap_framerate,
        #    which dataset.py already writes).
        self._run_cmd([
            self.cfg.python, "data/scripts/convert_amass_to_proto.py", str(self.cfg.amass_dir),
            "--humanoid-type", self.cfg.robot,
            "--output-fps", "30",
            "--output", str(converted),
        ], cwd=repo)
        # 2) Package into a single MotionLib .pt the trainer loads.
        self._run_cmd([
            self.cfg.python, "protomotions/components/motion_lib.py",
            "--motion-path", str(converted),
            "--output-file", str(self.motion_file),
        ], cwd=repo)
        print(f"  ProtoMotions MotionLib at {self.motion_file}")
        return out

    @property
    def motion_file(self) -> Path:
        """The packaged MotionLib the trainer consumes."""
        return self.cfg.prepared_dir / "motions.pt"

    def train(self) -> Path:
        repo = self._require_repo()
        if self.cfg.algorithm not in _EXPERIMENT:
            raise TrainerError(f"algorithm {self.cfg.algorithm!r} has no ProtoMotions experiment path")
        motion = self._require(self.motion_file, "the packaged MotionLib (run `prepare` first)")
        save = self.cfg.checkpoint_dir
        save.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cfg.python, "protomotions/train_agent.py",
            "--robot-name", self.cfg.robot,
            "--simulator", self.cfg.simulator,
            "--experiment-path", _EXPERIMENT[self.cfg.algorithm],
            "--experiment-name", f"{self.cfg.robot}_{self.cfg.algorithm}",
            "--motion-file", str(motion),
            "--num-envs", str(self.cfg.num_envs),
            "--ngpu", str(self.cfg.ngpu),
            *self.cfg.extra_args,
        ]
        self._run_cmd(cmd, cwd=repo)
        return save
