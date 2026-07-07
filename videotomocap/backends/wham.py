"""WHAM backend -- https://github.com/yohanshin/WHAM (CVPR 2024).

Slower full-pipeline throughput than GVHMR on a large backlog, but a strong
world-grounded alternative.  Wraps `demo.py`, which writes ``wham_output.pkl``:
a dict keyed by track id, each holding per-frame SMPL params.  We select the
longest track (personal footage has a single subject) and prefer the
world-grounded fields when present.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..pose import SmplMotion
from .base import BackendError, HMRBackend, assemble_smpl72, to_axis_angle


class WHAMBackend(HMRBackend):
    name = "wham"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        repo = self._require_repo()
        video_path = Path(video_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cfg.backend_python,
            "demo.py",
            "--video", str(video_path),
            "--output_pth", str(out_dir.resolve()),
        ]
        cmd.extend(self.cfg.backend_extra_args)
        self._run_cmd(cmd, cwd=repo)

        pkl = self._find_output(out_dir)
        with open(pkl, "rb") as fh:
            data = pickle.load(fh)
        track = self._longest_track(data)
        return self._parse_track(track, video_path, pkl)

    def _find_output(self, out_dir: Path) -> Path:
        found = list(out_dir.rglob("wham_output.pkl"))
        if not found:
            raise BackendError(f"WHAM produced no wham_output.pkl under {out_dir}")
        return found[0]

    @staticmethod
    def _longest_track(data: dict) -> dict:
        if not data:
            raise BackendError("WHAM output has no tracks")

        def length(v: dict) -> int:
            arr = v.get("pose_world", v.get("pose"))
            return 0 if arr is None else len(arr)

        return max(data.values(), key=length)

    def _parse_track(self, tr: dict, video_path: Path, pkl: Path) -> SmplMotion:
        want_world = self.cfg.use_frame == "global"
        pose = tr.get("pose_world" if want_world else "pose", tr.get("pose"))
        trans = tr.get("trans_world" if want_world else "trans", tr.get("trans"))
        if pose is None or trans is None:
            raise BackendError(f"WHAM track missing pose/trans; keys: {list(tr)}")
        pose = np.asarray(pose)
        # WHAM 'pose' is (T,72) SMPL axis-angle: split into global_orient + body_pose.
        aa = to_axis_angle(pose, 24)
        poses = assemble_smpl72(aa[:, :3], aa[:, 3:])
        betas = np.asarray(tr["betas"]).reshape(len(pose), -1)[0] if "betas" in tr else None

        return SmplMotion(
            poses=poses,
            trans=np.asarray(trans, np.float32).reshape(-1, 3),
            fps=float(self.cfg.target_fps),
            betas=betas,
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": "wham", "results": str(pkl)},
        )
