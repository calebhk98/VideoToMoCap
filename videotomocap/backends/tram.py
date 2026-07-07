"""TRAM backend -- https://github.com/yufu-wang/tram (ECCV 2024).

SLAM-based, world-grounded; part of the same family as GVHMR/WHAM.  TRAM is a
multi-stage pipeline (masks -> DROID-SLAM camera -> VIMO motion) rather than a
single command, so this adapter runs the stages in order and reads the per-track
HPS output.  Output layout has shifted between TRAM revisions -- verify the
paths below against your checkout (they are centralized in ``_STAGES`` /
``_find_hps``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..pose import SmplMotion
from .base import BackendError, HMRBackend, assemble_smpl72, to_axis_angle

# TRAM's scripts, run in order against a single video sequence.
_STAGES = (
    ("scripts/estimate_camera.py", ["--video"]),
    ("scripts/estimate_humans.py", ["--video"]),
)


class TRAMBackend(HMRBackend):
    name = "tram"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        repo = self._require_repo()
        video_path = Path(video_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for script, flag in _STAGES:
            cmd = [self.cfg.backend_python, script, flag[0], str(video_path)]
            cmd.extend(self.cfg.backend_extra_args)
            self._run_cmd(cmd, cwd=repo)

        hps = self._find_hps(repo, out_dir, video_path.stem)
        return self._parse(hps, video_path)

    def _find_hps(self, repo: Path, out_dir: Path, stem: str) -> Path:
        # TRAM writes results under results/<seq>/hps/hps_track_*.npy by default.
        search_roots = [out_dir, repo / "results" / stem, repo / "results"]
        found = []
        for r in search_roots:
            if r.exists():
                found.extend(r.rglob("hps_track_*.npy"))
        if not found:
            raise BackendError(f"TRAM produced no hps_track_*.npy (looked under {search_roots})")
        # longest track = the subject
        return max(found, key=lambda p: len(np.load(p, allow_pickle=True).item().get("pred_trans", [])))

    def _parse(self, hps_npy: Path, video_path: Path) -> SmplMotion:
        d = np.load(hps_npy, allow_pickle=True).item()
        want_world = self.cfg.use_frame == "global"

        # TRAM stores rotations as rotation matrices (pred_rotmat: T,24,3,3) and a
        # separate global/world trajectory. Field names vary by revision.
        rot = d.get("pred_rotmat", d.get("rotmat"))
        if rot is None:
            raise BackendError(f"TRAM output {hps_npy} lacks pred_rotmat; keys: {list(d)}")
        aa = to_axis_angle(np.asarray(rot), 24)
        poses = assemble_smpl72(aa[:, :3], aa[:, 3:])

        trans_key = "pred_trans_world" if want_world else "pred_trans"
        trans = d.get(trans_key, d.get("pred_trans"))
        if trans is None:
            raise BackendError(f"TRAM output {hps_npy} lacks translation; keys: {list(d)}")
        betas = np.asarray(d["pred_shape"]).reshape(-1) if "pred_shape" in d else None

        return SmplMotion(
            poses=poses,
            trans=np.asarray(trans, np.float32).reshape(-1, 3),
            fps=float(self.cfg.target_fps),
            betas=betas,
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": "tram", "results": str(hps_npy)},
        )
