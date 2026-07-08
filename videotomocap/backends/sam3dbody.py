"""SAM 3D Body backend -- Meta's SAM 3D Body (2026) -> SMPL-X via MHR fitting.

**Fit-with-adapter.** SAM 3D Body is a single-image foundation model that emits
MHR (Momentum Human Rig) parameters, not SMPL. Meta's Apache-2.0
``mhr_smpl_conversion`` tool fits those to genuine SMPL-X *axis-angle* (body 63 +
MANO hands 45 when the target ``smplx.SMPLX`` is built with ``use_pca=False`` +
``transl`` + ``betas``), which is exactly what the rest of this pipeline consumes.

There is no upstream video command, so this backend shells out to a bundled
driver (``scripts/sam3d_to_smplx.py``) that runs both stages per frame -- MHR
inference then the fit -- and writes one SMPL-X ``.npz`` per frame under
``<out>/smplx/``. From there it is an ordinary whole-body SMPL-X method: the
:class:`SmplXFramesBackend` base parses those per-frame files (same keys as
SMPLest-X/OSX) into SMPL-72 body + MANO hands.

**Heavy to stand up** -- closest in cost to the ``fusion`` backend: two GPU
stages per clip (a feedforward pass + a ~150-400-iteration Adam fit), a gated
~2.8 GB weight download, and an official SMPL-X model file. Licensing: the SAM 3D
Body weights are under the custom **SAM License** (commercial-ish but with
patent/IP-litigation termination clauses -- read it); the conversion tool is
Apache-2.0; the SMPL-X model carries Max-Planck's own terms.

Config::

    backend: sam3dbody
    backend_python: /opt/miniconda3/envs/sam3d/bin/python   # the SAM-3D env
    backend_repo: /opt/sam-3d-body                          # the model repo (cwd + on PYTHONPATH)
    backend_extra_args:                                     # forwarded verbatim to the driver
      - --mhr-repo=/opt/MHR
      - --checkpoint=/opt/ckpts/sam-3d-body-dinov3/model.ckpt
      - --mhr-model=/opt/ckpts/sam-3d-body-dinov3/assets/mhr_model.pt
      - --smplx-dir=/opt/models/smplx

The exact upstream APIs drift; the driver centralizes them and is meant to be
adapted to your checkout (see its header + backends module NOTE).
"""

from __future__ import annotations

from pathlib import Path

from .smplx_frames import SmplXFramesBackend

# Ships alongside the package: videotomocap/backends/sam3dbody.py -> repo/scripts/.
_DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "sam3d_to_smplx.py"


class Sam3dBodyBackend(SmplXFramesBackend):
    """SAM 3D Body -> SMPL-X (via MHR fit); see module docstring for the two stages."""

    name = "sam3dbody"
    frame_glob = "smplx/*.npz"     # the driver nests per-frame SMPL-X npz here

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, str(_DRIVER),
            "--video", str(video_path),
            "--out-dir", str(out_dir.resolve()),
            "--sam3d-repo", str(repo),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo
