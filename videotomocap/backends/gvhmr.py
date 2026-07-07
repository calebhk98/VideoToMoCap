"""GVHMR backend -- the default, chosen for throughput on a large backlog.

Wraps `tools/demo/demo.py` from https://github.com/zju3dv/GVHMR and parses the
``hmr4d_results.pt`` it writes.  GVHMR predicts SMPL-X body params
(``body_pose`` is 63-dim axis-angle); we convert to SMPL-72 in the base helper.

Install (per upstream INSTALL.md) into a dedicated env, then point the config at
it:

    backend: gvhmr
    backend_repo: /opt/GVHMR
    backend_python: /opt/miniconda3/envs/gvhmr/bin/python
    static_cameras: [cam01_corner, cam04_corner]   # skip visual odometry (-s)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..pose import SmplMotion
from .base import BackendError, HMRBackend, assemble_smpl72


class GVHMRBackend(HMRBackend):
    """Default body-only backend; see module docstring for install/config."""

    name = "gvhmr"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        repo = self._require_repo()
        video_path = Path(video_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # GVHMR writes to outputs/demo/<video_stem>/hmr4d_results.pt inside the
        # repo. We set --output_root to our per-clip scratch dir to keep runs
        # isolated and resumable.
        cmd = [
            self.cfg.backend_python,
            "tools/demo/demo.py",
            f"--video={video_path}",
            f"--output_root={out_dir.resolve()}",
        ]
        if static:
            cmd.append("-s")  # skip visual odometry for known-static cameras
        cmd.extend(self.cfg.backend_extra_args)
        self._run_cmd(cmd, cwd=repo)

        results = self._find_results(out_dir, video_path.stem)
        return self._parse(results, video_path)

    def _find_results(self, out_dir: Path, stem: str) -> Path:
        candidates = [
            out_dir / stem / "hmr4d_results.pt",
            out_dir / "demo" / stem / "hmr4d_results.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        found = list(out_dir.rglob("hmr4d_results.pt"))
        if found:
            return found[0]
        raise BackendError(f"GVHMR produced no hmr4d_results.pt under {out_dir}")

    def _parse(self, results_pt: Path, video_path: Path) -> SmplMotion:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendError("Parsing GVHMR output requires torch (present in the GVHMR env).") from exc

        pred = torch.load(results_pt, map_location="cpu")
        key = f"smpl_params_{self.cfg.use_frame}"  # 'smpl_params_global' | 'smpl_params_incam'
        if key not in pred:
            raise BackendError(f"{results_pt} missing {key}; keys present: {list(pred)}")
        p = pred[key]

        def np_of(name: str) -> np.ndarray:
            v = p[name]
            return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

        poses = assemble_smpl72(np_of("global_orient"), np_of("body_pose"))
        transl = np_of("transl").reshape(-1, 3).astype(np.float32)
        betas = np_of("betas").reshape(-1).astype(np.float32) if "betas" in p else None
        fps = float(pred.get("fps", self.cfg.target_fps))

        return SmplMotion(
            poses=poses,
            trans=transl,
            fps=fps,
            betas=betas,
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": "gvhmr", "results": str(results_pt)},
        )
