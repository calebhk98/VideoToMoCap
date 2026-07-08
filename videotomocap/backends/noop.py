"""Synthetic backend -- no GPU, no weights, no model download.

Generates smooth, plausible SMPL-72 motion so the *rest* of the pipeline
(exclusion -> anonymization -> dataset export -> motion-model prep) can be run
and tested end-to-end on any machine.  It does not look at pixels; it fabricates
a low-frequency random walk in pose space seeded by the clip name so runs are
reproducible.  Never use its output as real training data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..pose import SMPL_POSE_DIM, SmplMotion
from .base import HMRBackend


class NoopBackend(HMRBackend):
    """GPU-free synthetic backend that powers the tests; see module docstring."""

    name = "noop"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        return self._synth(video_path, person=0, static=static)

    def run_tracks(self, video_path: Path, out_dir: Path, *, static: bool = False) -> List[SmplMotion]:
        """Fabricate ``cfg.synthetic_people`` distinct people so the multi-person +
        identity flow runs GPU-free. Each person gets a stable per-identity shape
        (betas) so cross-clip shape-clustering can actually recover them."""
        n_people = max(1, int(getattr(self.cfg, "synthetic_people", 1)))
        return [self._synth(video_path, person=p, static=static) for p in range(n_people)]

    def _synth(self, video_path: Path, *, person: int, static: bool) -> SmplMotion:
        video_path = Path(video_path)
        seed = int.from_bytes(video_path.name.encode()[:8].ljust(8, b"0"), "little") % (2**32)
        rng = np.random.default_rng(seed + person)

        fps = float(self.cfg.target_fps)
        n = int(fps * rng.integers(4, 12))  # 4-12 s clip
        n = max(n, self.cfg.min_clip_frames + 5)

        # Low-frequency smooth motion: integrate small random accelerations, then
        # keep the whole thing near a neutral standing pose.
        accel = rng.normal(0, 0.002, size=(n, SMPL_POSE_DIM))
        poses = np.cumsum(np.cumsum(accel, axis=0), axis=0)
        poses -= poses.mean(axis=0, keepdims=True)
        poses[:, :3] = 0.0  # keep global orientation upright for readability

        trans = np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0).astype(np.float32)
        # A per-identity shape: seeded by person id (NOT the clip), so the same
        # synthetic person has the same betas across clips -> shape-clustering finds them.
        betas = np.random.default_rng(1000 + person).normal(0, 1.0, size=10).astype(np.float32)

        return SmplMotion(
            poses=poses.astype(np.float32),
            trans=trans,
            fps=fps,
            betas=betas,
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": "noop", "synthetic": True, "static": static, "person": person},
        )
