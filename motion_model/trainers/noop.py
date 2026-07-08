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
        print(f"  noop wrote a synthetic checkpoint to {save}")
        return save
