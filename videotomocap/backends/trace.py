"""TRACE backend -- https://github.com/Arthur151/ROMP (CVPR 2023), Apache-2.0.

World-grounded body-only SMPL from the ``simple_romp`` package (bundled with
ROMP/BEV). TRACE recovers a global trajectory under a moving camera -- the same
family as GVHMR/WHAM/TRAM -- but ships under a genuinely permissive **Apache-2.0**
license, which is why it's worth having alongside the others. Body-only: the two
SMPL hand joints are zero-padded to neutral.

TRACE installs as a console script, not a cloned repo::

    pip install simple-romp            # provides the `trace2` command + weights
    backend: trace
    backend_python: python             # `trace2` must be on PATH
    # backend_repo is unused for this backend

TRACE writes a per-sequence ``.npz`` holding an ``outputs`` dict of stacked
per-detection arrays (SMPL params + world trajectory) plus a ``track_ids`` /
frame-index mapping. Exact key names have drifted between simple_romp revisions,
so the parser below reads them defensively (mirrors wham.py/tram.py). Verify
against your installed version if a field is missing.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..pose import SmplMotion
from .base import BackendError, HMRBackend, assemble_smpl72, _matrix_to_axis_angle


class TRACEBackend(HMRBackend):
    """World-grounded body-only backend (Apache-2.0); see module docstring."""

    name = "trace"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        video_path = Path(video_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # `trace2` is a console entry point from simple_romp; no repo/cwd needed.
        cmd = [
            "trace2",
            "-i", str(video_path),
            "--subject_num=1",
            "--save_path", str(out_dir.resolve()),
            *self.cfg.backend_extra_args,
        ]
        self._run_cmd(cmd)

        results = self._find_results(out_dir)
        return self._parse(results, video_path)

    def _find_results(self, out_dir: Path) -> Path:
        # The sequence-results npz (not the *_tracking.npz companion) carries the
        # `outputs` dict we parse; prefer the larger file if names are ambiguous.
        found = [p for p in out_dir.rglob("*.npz") if "tracking" not in p.stem]
        if not found:
            raise BackendError(f"TRACE produced no results .npz under {out_dir}")
        return max(found, key=lambda p: p.stat().st_size)

    def _parse(self, results_npz: Path, video_path: Path) -> SmplMotion:
        loaded = np.load(results_npz, allow_pickle=True)
        outputs = loaded["outputs"].item() if "outputs" in loaded.files else {k: loaded[k] for k in loaded.files}

        thetas = _pick(outputs, "smpl_thetas", "poses", "pose")          # (N, 72) axis-angle
        if thetas is None:
            raise BackendError(f"TRACE output {results_npz} lacks smpl_thetas; keys: {list(outputs)}")
        thetas = np.asarray(thetas)

        track_ids = _pick(outputs, "track_ids", "track_id", "subject_ids")
        frame_idx = _pick(outputs, "reorganize_idx", "frame_ids", "frame_id")
        order = _subject_order(thetas.shape[0], track_ids, frame_idx)

        poses = self._world_grounded_poses(outputs, thetas[order])
        trans = self._translation(outputs, order)
        betas = _pick(outputs, "smpl_betas", "betas")

        return SmplMotion(
            poses=poses,
            trans=trans,
            fps=float(self.cfg.target_fps),
            betas=None if betas is None else np.asarray(betas)[order][0].reshape(-1).astype(np.float32),
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": "trace", "results": str(results_npz), "n_frames": len(order)},
        )

    def _world_grounded_poses(self, outputs: Dict, thetas: np.ndarray) -> np.ndarray:
        """Full SMPL-72. For use_frame='global', swap the camera-frame root for
        TRACE's world global rotation (its whole point is world grounding)."""
        body = thetas[:, 3:]
        if self.cfg.use_frame != "global":
            return assemble_smpl72(thetas[:, :3], body)
        world_root = _pick(outputs, "world_global_rotmat", "world_global_rots", "global_rotmat")
        if world_root is None:
            return assemble_smpl72(thetas[:, :3], body)   # fall back to camera-frame root
        world_root = np.asarray(world_root)[_subject_slice(outputs, thetas.shape[0])]
        root_aa = _matrix_to_axis_angle(world_root.reshape(-1, 3, 3)).reshape(-1, 3)
        return assemble_smpl72(root_aa, body)

    def _translation(self, outputs: Dict, order: np.ndarray) -> np.ndarray:
        want_world = self.cfg.use_frame == "global"
        key_order = ("world_trans", "cam_trans") if want_world else ("cam_trans", "world_trans")
        trans = _pick(outputs, *key_order)
        if trans is None:
            return np.zeros((len(order), 3), np.float32)
        return np.asarray(trans)[order].reshape(-1, 3).astype(np.float32)


def _pick(d: Dict, *names: str) -> Optional[np.ndarray]:
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


def _subject_slice(outputs: Dict, n: int) -> np.ndarray:
    """Recompute the dominant-track ordering for an auxiliary array of length N."""
    track_ids = _pick(outputs, "track_ids", "track_id", "subject_ids")
    frame_idx = _pick(outputs, "reorganize_idx", "frame_ids", "frame_id")
    return _subject_order(n, track_ids, frame_idx)


def _subject_order(n: int, track_ids: Optional[np.ndarray], frame_idx: Optional[np.ndarray]) -> np.ndarray:
    """Indices of the dominant track's detections, in frame order.

    Personal footage has one subject, but TRACE stacks every detection across all
    frames; follow the most-frequent track id and sort it by frame."""
    if track_ids is None:
        return np.arange(n)
    track_ids = np.asarray(track_ids).reshape(-1)
    subject = Counter(track_ids.tolist()).most_common(1)[0][0]
    idx = np.flatnonzero(track_ids == subject)
    if frame_idx is not None:
        idx = idx[np.argsort(np.asarray(frame_idx).reshape(-1)[idx])]
    return idx
