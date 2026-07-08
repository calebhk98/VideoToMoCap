"""4D-Humans / HMR2 backend -- https://github.com/shubham-goel/4D-Humans (ICCV 2023).

Per-frame single-image SMPL regressor (HMR2.0) tracked through video with PHALP.
Unlike the GVHMR/WHAM/TRAM family this is *not* world-grounded -- it recovers
camera-relative SMPL per frame -- and it is body-only (SMPL, no articulated
hands, so the two SMPL hand joints are zero-padded to neutral like the other
body-only backends).

``track.py`` is Hydra-configured and writes a results pickle keyed by frame, each
frame holding one SMPL entry per tracked person (rotations as rotation matrices).
We follow the single dominant track (personal footage has one subject).

Weights: the HMR2 checkpoint + SMPL body model are bundled under the
``3_4DHumans`` / ``4_SMPLhub`` folders of https://huggingface.co/lithiumice/models_hub
-- a one-stop mirror if you'd rather not register on each model's site.

Config (adjust the Hydra keys / output path to your checkout):

    backend: hmr2            # or '4dhumans'
    backend_repo: /opt/4D-Humans
    backend_python: /opt/miniconda3/envs/4D-humans/bin/python
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..pose import SmplMotion
from .base import BackendError, HMRBackend, assemble_smpl72


class FourDHumansBackend(HMRBackend):
    """HMR2 (4D-Humans) body-only backend; see module docstring for install/config."""

    name = "hmr2"

    def run(self, video_path: Path, out_dir: Path, *, static: bool = False) -> SmplMotion:
        repo = self._require_repo()
        video_path = Path(video_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Hydra override syntax (key=value), not argparse flags. output_dir keeps
        # each clip's PHALP artifacts in its own scratch dir so runs stay isolated.
        cmd = [
            self.cfg.backend_python, "track.py",
            f"video.source={video_path}",
            f"video.output_dir={out_dir.resolve()}",
            *self.cfg.backend_extra_args,
        ]
        self._run_cmd(cmd, cwd=repo)

        pkl = self._find_results(repo, out_dir, video_path.stem)
        return self._parse(pkl, video_path)

    def _find_results(self, repo: Path, out_dir: Path, stem: str) -> Path:
        # PHALP writes <output_dir>/results/demo_<seq>.pkl; the exact name has
        # shifted between revisions, so fall back to any results pickle.
        search_roots = [out_dir, repo / "outputs"]
        found: List[Path] = []
        for root in search_roots:
            if root.exists():
                found.extend(root.rglob("*.pkl"))
        results = [p for p in found if "results" in p.parts or p.stem.startswith("demo")]
        if results:
            return max(results, key=lambda p: p.stat().st_size)
        raise BackendError(f"4D-Humans produced no results pickle (looked under {search_roots})")

    def _parse(self, pkl_path: Path, video_path: Path) -> SmplMotion:
        data = _load_pickle(pkl_path)
        frames = sorted(data)
        subject = _dominant_track(data, frames)

        go, bp, cam = [], [], []
        for name in frames:
            picked = _pick_person(data[name], subject)
            if picked is None:
                continue
            smpl, camera = picked
            go.append(np.asarray(smpl["global_orient"]).reshape(1, 3, 3))
            bp.append(np.asarray(smpl["body_pose"]).reshape(23, 3, 3))
            cam.append(np.asarray(camera).reshape(3) if camera is not None else np.zeros(3))

        if not go:
            raise BackendError(f"4D-Humans track {subject!r} yielded no SMPL frames in {pkl_path}")

        poses = assemble_smpl72(np.stack(go), np.stack(bp))
        betas = _first_betas(data, frames, subject)
        return SmplMotion(
            poses=poses,
            trans=np.stack(cam).astype(np.float32),
            fps=float(self.cfg.target_fps),
            betas=betas,
            frame=self.cfg.use_frame,
            source_clip=video_path.name,
            meta={"backend": self.name, "results": str(pkl_path), "track": int(subject)},
        )


def _load_pickle(path: Path):
    """PHALP dumps with joblib; fall back to stdlib pickle for plain dicts."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _track_ids(entry: Dict) -> List:
    """Track ids present in a frame, tolerating the tracked_ids/tid naming drift."""
    return list(entry.get("tracked_ids", entry.get("tid", [])))


def _dominant_track(data: Dict, frames: List[str]):
    """The track id present in the most frames -- the subject in single-person footage."""
    counts: Counter = Counter()
    for name in frames:
        counts.update(_track_ids(data[name]))
    if not counts:
        raise BackendError("4D-Humans results carry no track ids to follow")
    return counts.most_common(1)[0][0]


def _pick_person(entry: Dict, subject):
    """(smpl_dict, camera) for `subject` in this frame, or None if absent."""
    ids = _track_ids(entry)
    if subject not in ids:
        return None
    i = ids.index(subject)
    smpl = entry["smpl"][i]
    camera = entry["camera"][i] if entry.get("camera") is not None else None
    return smpl, camera


def _first_betas(data: Dict, frames: List[str], subject) -> Optional[np.ndarray]:
    for name in frames:
        picked = _pick_person(data[name], subject)
        if picked and "betas" in picked[0]:
            return np.asarray(picked[0]["betas"]).reshape(-1).astype(np.float32)
    return None
