"""Synthetic trainer -- no GPU, no weights, no upstream repo.

Exercises the whole Pipeline 2 flow (prepare -> train) end-to-end on any machine
so the self-test and unit tests stay GPU-free, exactly like the HMR ``noop``
backend does for Pipeline 1. ``prepare`` lays out a real skeleton from the AMASS
dataset; ``train`` fabricates a tiny checkpoint + manifest instead of shelling
out. Never use its output as a real model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import data
from .base import MotionTrainer


class NoopTrainer(MotionTrainer):
    """GPU-free synthetic trainer that powers the tests; see module docstring."""

    name = "noop"
    feature_format = "synthetic"
    role = "synthetic trainer (tests only)"

    def prepare(self) -> Path:
        out = data.prepare_humanml3d(self.cfg)
        print(f"  noop prepared skeleton at {out}")
        return out

    def train(self) -> Path:
        save = self.cfg.checkpoint_dir
        save.mkdir(parents=True, exist_ok=True)
        index = data.load_index(self.cfg.dataset_dir)
        # Fabricate a checkpoint + run manifest so downstream steps have something real
        # to point at, without any heavy dependency.
        (save / "model_final.pt").write_bytes(b"NOOP-CHECKPOINT")
        manifest = {
            "method": self.cfg.method,
            "feature_format": self.feature_format,
            "n_clips": len(index["clips"]),
            "num_steps": self.cfg.num_steps,
            "synthetic": True,
        }
        (save / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
        self._write_synthetic_curve(save)
        print(f"  noop wrote a synthetic checkpoint to {save}")
        return save

    def sample(self, prompt: str, out_dir: Path) -> Path:
        """Fabricate a synthetic SMPL-72 motion npz so the `act` flow runs GPU-free."""
        out_dir.mkdir(parents=True, exist_ok=True)
        t = 40
        out = out_dir / "sample.npz"
        # A gentle synthetic wiggle so the exported BVH isn't all-identity.
        poses = np.zeros((t, 72), np.float32)
        poses[:, 0] = 0.05 * np.sin(np.linspace(0, 6.28, t))   # slight root sway
        np.savez(out, poses=poses, trans=np.zeros((t, 3), np.float32))
        print(f"  noop sampled synthetic motion for {prompt!r} -> {out}")
        return out

    def _write_synthetic_curve(self, save: Path) -> None:
        """Emit a metrics.jsonl with a deliberate overfitting shape (val dips then
        rises) so `overfit-report` has a real curve to read in the GPU-free tests."""
        val = [1.0, 0.7, 0.5, 0.45, 0.5, 0.6, 0.72]  # bottoms at step 300, then climbs
        rows = [{"step": i * 100, "train_loss": round(1.0 - 0.12 * i, 3), "val_loss": v}
                for i, v in enumerate(val)]
        (save / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
