"""Whole-body SMPL-X backends that emit one file per frame.

A large family of whole-body methods (SMPLest-X, SMPLer-X, WHAC, OSX,
Hand4Whole++, Multi-HMR, ...) run per-frame inference and dump a SMPL-X ``.npz``/``.npy`` per
frame with the same standard key names (``global_orient``, ``body_pose``,
``left_hand_pose``, ``right_hand_pose``, ``betas``, ``transl``). Rather than
write five near-identical adapters, ``SmplXFramesBackend`` does the shared
work -- run a command, collect the per-frame files in order, stack them, and
build a hand-carrying :class:`SmplMotion`. Each concrete backend only declares
how to invoke its tool and where its outputs land.

These are the reason the pipeline can produce *high-detail hands*: unlike the
body-only backends (GVHMR/WHAM/TRAM), the SMPL-X ``*_hand_pose`` fields are
parsed and kept.

NOTE: exact demo commands and output paths vary by upstream revision -- they are
centralized in each subclass's ``_command`` / ``frame_glob`` so you can adjust
them to your checkout in one place (mirrors how wham.py / tram.py are written).
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pose import SmplMotion
from .base import BackendError, HMRBackend, assemble_hand, assemble_smpl72


def _squeeze_leading(a: np.ndarray) -> np.ndarray:
    """Per-frame files often store (1, K); drop a leading singleton batch dim."""
    a = np.asarray(a)
    return a[0] if a.ndim >= 2 and a.shape[0] == 1 else a


class SmplXFramesBackend(HMRBackend):
    """Base for whole-body methods that write one SMPL-X file per frame."""

    frame_glob = "*.npz"          # pattern for per-frame outputs, in frame order
    produces_hands = True

    @abstractmethod
    def _command(self, video_path: Path, out_dir: Path, static: bool) -> Tuple[List[str], Path]:
        """Return (argv, cwd) that runs the tool and writes per-frame files."""

    def _load_frame(self, path: Path) -> Dict[str, np.ndarray]:
        """Load one per-frame file into a plain dict of arrays. Override for .npy."""
        d = np.load(path, allow_pickle=True)
        if isinstance(d, np.lib.npyio.NpzFile):
            return {k: d[k] for k in d.files}
        return d.item()  # a pickled dict saved as .npy

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        self._require_repo()
        video_path = Path(video_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        argv, cwd = self._command(video_path, out_dir, static)
        self._run_cmd(argv, cwd=cwd)

        frames = sorted(out_dir.rglob(self.frame_glob))
        if not frames:
            raise BackendError(f"{self.name} produced no '{self.frame_glob}' files under {out_dir}")

        dicts = [self._load_frame(f) for f in frames]
        return self._stack(dicts, video_path, out_dir)

    def _pick(self, d: Dict[str, np.ndarray], *names: str) -> Optional[np.ndarray]:
        for n in names:
            if n in d:
                return _squeeze_leading(d[n])
        return None

    def _stack(self, dicts: List[Dict[str, np.ndarray]], video_path: Path, out_dir: Path) -> SmplMotion:
        """Stack per-frame SMPL-X dicts into arrays and assemble the SmplMotion,
        carrying hands through when the per-frame files provide them."""
        go = np.stack([self._pick(d, "global_orient", "root_pose").reshape(3) for d in dicts])
        bp = np.stack([self._pick(d, "body_pose").reshape(-1) for d in dicts])
        poses = assemble_smpl72(go, bp)

        transl = self._pick(dicts[0], "transl", "trans", "cam_trans")
        if transl is None:
            trans = np.zeros((len(dicts), 3), np.float32)
        else:
            trans = np.stack([self._pick(d, "transl", "trans", "cam_trans").reshape(3) for d in dicts])

        lh = self._stack_hand(dicts, "left_hand_pose", "lhand_pose")
        rh = self._stack_hand(dicts, "right_hand_pose", "rhand_pose")

        b0 = self._pick(dicts[0], "betas", "shape")
        betas = None if b0 is None else b0.reshape(-1)

        return SmplMotion(
            poses=poses,
            trans=trans.astype(np.float32),
            fps=float(self.cfg.target_fps),
            betas=betas,
            left_hand_pose=lh,
            right_hand_pose=rh,
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": self.name, "n_frames": len(dicts), "out_dir": str(out_dir)},
        )

    def _stack_hand(self, dicts, *names: str) -> Optional[np.ndarray]:
        if not self.produces_hands or self._pick(dicts[0], *names) is None:
            return None
        raw = np.stack([self._pick(d, *names).reshape(-1) for d in dicts])
        return assemble_hand(raw)


# --------------------------------------------------------------------------
# Concrete whole-body backends. `backend_repo`/`backend_python` point at each
# upstream checkout; extra flags go through `backend_extra_args`.
# --------------------------------------------------------------------------

class SMPLestXBackend(SmplXFramesBackend):
    """SMPLest-X / SMPLer-X -- the recommended general whole-body SMPL-X regressor.

    https://github.com/SMPLCap/SMPLest-X (also SMPLer-X). Writes one SMPL-X .npz
    per frame via np.savez(**smplx_pred). Non-commercial (S-Lab) license.
    """

    name = "smplestx"

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, "main/inference.py",
            "--video", str(video_path),
            "--output_folder", str(out_dir.resolve()),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo


class CamenduruSMPLerXBackend(SmplXFramesBackend):
    """SMPLer-X via camenduru's runnable repackaging of the model.

    https://github.com/camenduru/SMPLer-X -- the same SMPLer-X regressor as
    ``smplestx`` above, but wrapped for one-shot video inference. Its
    ``main/slurm_inference.sh {VIDEO} {FORMAT} {FPS} {CKPT}`` extracts frames
    with ffmpeg and runs ``main/inference.py`` per frame, writing one SMPL-X npz
    per detection under ``<output_folder>/smplx/{frame:05}_{bbox}.npz`` with the
    standard keys (``global_orient``, ``body_pose``, ``left_hand_pose``,
    ``right_hand_pose``, ``betas``, ``transl``). Non-commercial (S-Lab) license.

    Distinct from ``smplestx`` only in invocation and output layout: camenduru's
    ``inference.py`` runs on a folder of frames (``--img_path``) rather than a
    video, so point ``backend_extra_args`` at your checkpoint (e.g.
    ``--pretrained_model smpler_x_h32``). Single-subject footage yields one bbox
    per frame; if you enable ``--multi_person`` upstream, filter to one detection.
    """

    name = "camenduru_smplerx"
    frame_glob = "smplx/*.npz"      # camenduru nests the per-frame SMPL-X npz here

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, "main/inference.py",
            "--img_path", str(video_path),
            "--output_folder", str(out_dir.resolve()),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo


class WHACBackend(SmplXFramesBackend):
    """WHAC -- world-grounded whole-body SMPL-X *with* camera trajectory.

    https://github.com/SMPLCap/WHAC. The moving-camera counterpart to GVHMR/WHAM
    but whole-body (built on SMPLest-X + DPVO). Prefer for handheld/rotating
    cameras when you also want hands.
    """

    name = "whac"

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, "demo.py",
            "--video", str(video_path),
            "--output_dir", str(out_dir.resolve()),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo


class OSXBackend(SmplXFramesBackend):
    """OSX -- one-stage whole-body SMPL-X, MIT-licensed (code). Lighter fallback.

    https://github.com/IDEA-Research/OSX
    """

    name = "osx"

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, "demo/demo.py",
            "--video", str(video_path),
            "--output", str(out_dir.resolve()),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo


class Hand4WholePlusBackend(SmplXFramesBackend):
    """Hand4Whole++ (CVPR 2026) -- MIT-licensed, hand-specialist whole-body model.

    https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE. Best hands of the
    single-model family (its CHAM module wraps WiLoR internally).
    """

    name = "hand4whole"

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, "demo/demo.py",
            "--video", str(video_path),
            "--output", str(out_dir.resolve()),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo


class MultiHMRBackend(SmplXFramesBackend):
    """Multi-HMR -- fast single-image multi-person whole-body SMPL-X.

    https://github.com/naver/multi-hmr. Single-image model: point it at a folder
    of pre-extracted frames; it writes one .npy per detection. We take the first
    detection per frame (personal footage has a single subject).
    """

    name = "multihmr"
    frame_glob = "*.npy"

    def _command(self, video_path, out_dir, static):
        repo = Path(self.cfg.backend_repo)
        argv = [
            self.cfg.backend_python, "demo.py",
            "--video", str(video_path),
            "--out_folder", str(out_dir.resolve()),
            *self.cfg.backend_extra_args,
        ]
        return argv, repo
