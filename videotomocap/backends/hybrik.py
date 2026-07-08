"""HybrIK-X backend -- https://github.com/jeffffffli/HybrIK (TPAMI 2025), MIT.

Whole-body SMPL-X via analytical-neural inverse kinematics. Same whole-body
capability as SMPLest-X/OSX (body + articulated hands + face), but under a plain
**MIT** license rather than the S-Lab / Max-Planck non-commercial terms most of
the SMPL-X family carries -- the reason to have it.

The ``demo_video_x.py`` script writes a single pickle (``--save-pt``) whose
``pred_thetas`` holds the full SMPL-X pose as per-frame rotation matrices, plus
``pred_betas`` and ``transl``. We slice the standard 55-joint SMPL-X layout back
into our SMPL-72 body + MANO hand fields.

Config::

    backend: hybrik
    backend_repo: /opt/HybrIK
    backend_python: /opt/miniconda3/envs/hybrik/bin/python

NOTE: pose layout/joint count can differ by revision -- the slicing constants
below assume the canonical SMPL-X ordering (global, 21 body, jaw+2 eyes, 15+15
hands). The parser fails loud with the observed shape if it doesn't match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from ..pose import SmplMotion
from .base import BackendError, HMRBackend, assemble_hand, assemble_smpl72

# Canonical SMPL-X full-pose joint layout (smplx `full_pose` order).
_SMPLX_NJOINTS = 55
_BODY = slice(1, 22)         # 21 body joints (after global_orient at 0)
_LEFT_HAND = slice(25, 40)   # 15 MANO joints (jaw=22, leye=23, reye=24 skipped)
_RIGHT_HAND = slice(40, 55)  # 15 MANO joints


class HybrIKXBackend(HMRBackend):
    """Whole-body SMPL-X backend (MIT-licensed); see module docstring."""

    name = "hybrik"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        repo = self._require_repo()
        video_path = Path(video_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cfg.backend_python, "scripts/demo_video_x.py",
            "--video-name", str(video_path),
            "--out-dir", str(out_dir.resolve()),
            "--save-pt",
            *self.cfg.backend_extra_args,
        ]
        self._run_cmd(cmd, cwd=repo)

        pk = self._find_results(out_dir)
        return self._parse(pk, video_path)

    def _find_results(self, out_dir: Path) -> Path:
        found = list(out_dir.rglob("*.pk")) + list(out_dir.rglob("*.pkl"))
        if not found:
            raise BackendError(f"HybrIK-X produced no .pk results under {out_dir} (did you pass --save-pt?)")
        return max(found, key=lambda p: p.stat().st_size)

    def _parse(self, pk_path: Path, video_path: Path) -> SmplMotion:
        import pickle
        with open(pk_path, "rb") as fh:
            res: Dict = pickle.load(fh)

        mats = self._pose_matrices(res, pk_path)          # (T, 55, 3, 3)
        poses = assemble_smpl72(mats[:, 0:1], mats[:, _BODY])   # global + 21 body → SMPL-72
        lh = assemble_hand(mats[:, _LEFT_HAND])
        rh = assemble_hand(mats[:, _RIGHT_HAND])

        transl = np.asarray(res["transl"]).reshape(-1, 3).astype(np.float32) if "transl" in res else None
        betas = np.asarray(res["pred_betas"]).reshape(-1).astype(np.float32) if "pred_betas" in res else None

        return SmplMotion(
            poses=poses,
            trans=transl if transl is not None else np.zeros((len(mats), 3), np.float32),
            fps=float(self.cfg.target_fps),
            betas=betas,
            left_hand_pose=lh,
            right_hand_pose=rh,
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": "hybrik", "results": str(pk_path), "n_frames": len(mats)},
        )

    def _pose_matrices(self, res: Dict, pk_path: Path) -> np.ndarray:
        thetas = res.get("pred_thetas", res.get("pred_theta_mats"))
        if thetas is None:
            raise BackendError(f"HybrIK-X output {pk_path} lacks pred_thetas; keys: {list(res)}")
        thetas = np.asarray(thetas)
        # Accept (T,55,3,3) or a flattened (T, 55*9) rotation-matrix block.
        if thetas.ndim == 2 and thetas.shape[1] == _SMPLX_NJOINTS * 9:
            thetas = thetas.reshape(-1, _SMPLX_NJOINTS, 3, 3)
        if thetas.ndim != 4 or thetas.shape[1:] != (_SMPLX_NJOINTS, 3, 3):
            raise BackendError(
                f"HybrIK-X pred_thetas has shape {thetas.shape}; expected (T,{_SMPLX_NJOINTS},3,3) "
                "SMPL-X rotation matrices. Check the joint layout for your HybrIK checkout."
            )
        return thetas
