"""SAM-Body4D backend -- training-free, occlusion-robust 4D body -> SMPL-X.

SAM-Body4D (arXiv:2512.08406, https://github.com/gaomingqi/sam-body4d, MIT) is a
*pipeline*, not a new model: SAM-3 promptable video segmentation -> identity
masklets -> Diffusion-VAS amodal completion -> **Meta's SAM 3D Body per frame,
guided by the stabilized masks**. Think of it as a temporally-stabilized,
occlusion-robust ``sam3dbody``: the same MHR body params, but recovered through
tracking + amodal completion so clutter, occlusion and camera motion don't break
the per-frame fit. Its "4D" is mask-driven temporal consistency, **not** a
world-frame root trajectory -- like ``sam3dbody``/``hmr2`` it stays
camera-relative (no SLAM), so prefer a world-grounded backend (gvhmr/wham/tram/
whac) when you need global trajectory and this when occlusion is the problem.

Because SAM-Body4D vendors the very same ``facebookresearch/sam-3d-body`` model
``sam3dbody`` wraps, this backend reuses that path end to end: the bundled driver
(``scripts/sambody4d_to_smplx.py``) runs SAM-Body4D's tracking + per-frame HMR
and then Meta's ``mhr_smpl_conversion`` fit (shared with ``sam3d_to_smplx.py``),
writing one SMPL-X ``.npz`` per frame under ``<out>/smplx/``. From there it is an
ordinary whole-body SMPL-X method parsed by :class:`SmplXFramesBackend` into
SMPL-72 body + MANO hands.

**Heavy + gated** -- torch 2.7.1, 14-53 GB VRAM, ~26 min / 90 frames with
occlusion completion on (~2-3 min off, ``--occlusion`` omitted). Needs
HuggingFace access approval for ``facebook/sam3`` and
``facebook/sam-3d-body-dinov3`` (accept the gate + ``huggingface-cli login``).
Licensing: SAM-Body4D code MIT; the SAM 3D Body weights carry Meta's custom SAM
License; the conversion tool Apache-2.0; the SMPL-X model Max-Planck's terms.

Config::

    backend: sambody4d
    backend_python: /opt/miniconda3/envs/body4d/bin/python   # the SAM-Body4D env
    backend_repo: /opt/sam-body4d                            # gaomingqi/sam-body4d (cwd + PYTHONPATH)
    backend_extra_args:                                      # forwarded verbatim to the driver
      - --sam3d-repo=/opt/sam-3d-body
      - --mhr-repo=/opt/MHR
      - --mhr-model=/opt/ckpts/sam-3d-body-dinov3/assets/mhr_model.pt
      - --smplx-dir=/opt/models/smplx
      - --occlusion                                          # drop to skip Diffusion-VAS (much faster)

The exact upstream APIs drift; the driver centralizes them and is meant to be
adapted to your checkout (see its header + the ``smplx_frames`` module NOTE).
"""

from __future__ import annotations

from pathlib import Path

from .smplx_frames import SmplXFramesBackend

# Ships alongside the package: videotomocap/backends/sambody4d.py -> repo/scripts/.
_DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "sambody4d_to_smplx.py"


class SamBody4DBackend(SmplXFramesBackend):
    """SAM-Body4D -> SMPL-X (tracking + occlusion completion + MHR fit); see module docstring."""

    name = "sambody4d"
    frame_glob = "smplx/*.npz"     # the driver nests per-frame SMPL-X npz here

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, str(_DRIVER),
            "--video", str(video_path),
            "--out-dir", str(out_dir.resolve()),
            "--sambody4d-repo", str(repo),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo
