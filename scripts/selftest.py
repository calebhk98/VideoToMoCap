#!/usr/bin/env python
"""End-to-end self-test of Pipeline 1 with zero GPU / weights / video decoders.

Fabricates a small camera tree, then drives the real pipeline with the synthetic
'noop' backend:  scan -> exclude (fake family visit) -> HMR -> anonymize ->
build AMASS dataset -> prepare MDM skeleton.  Asserts the privacy guarantee
(betas dropped) and the AMASS shapes hold.  Run:  python scripts/selftest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import ingest, pipeline
from videotomocap.config import PipelineConfig
from videotomocap.dataset import AMASS_SMPLH_POSE_DIM
from videotomocap.ingest import Manifest
from videotomocap.pose import SMPL_POSE_DIM, SmplMotion
from motion_model import MotionModelConfig, get_trainer


def make_fake_footage(root: Path) -> None:
    """Create empty .mp4 placeholders in a realistic per-camera / per-date tree."""
    layout = {
        "cam01_corner/2024-05-01": ["0800.mp4", "0900.mp4"],
        "cam03_eyelevel/2024-05-01": ["1200.mp4"],
        "cam07_corner/2024-12-24": ["family_visit_1.mp4"],  # to be excluded
        "cam07_corner/2025-01-02": ["1500.mp4", "1600.mp4"],
    }
    for rel, files in layout.items():
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_bytes(b"")  # noop backend ignores contents


def check(cond: bool, msg: str) -> None:
    """Assert-and-print: raise with ``msg`` on failure, else echo it as an ok line."""
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main() -> int:
    """Drive the full pipeline in a scratch temp dir and assert every invariant holds."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        footage = tmp / "footage"
        make_fake_footage(footage)

        cfg = PipelineConfig(
            footage_root=footage,
            work_root=tmp / "work",
            backend="noop",
            static_cameras=["cam01_corner"],
            target_fps=30.0,
        )

        print("1) scan")
        manifest = ingest.scan(cfg)
        manifest.save(cfg.manifest_path)
        check(len(manifest.clips) == 6, f"discovered 6 clips (got {len(manifest.clips)})")
        cams = sorted(set(c.camera for c in manifest.clips))
        check(cams == ["cam01_corner", "cam03_eyelevel", "cam07_corner"], f"cameras: {cams}")

        print("2) exclude the family visit")
        n = ingest.exclude(manifest, patterns=["*/2024-12-24/*"])
        manifest.save(cfg.manifest_path)
        check(n == 1, f"excluded exactly 1 clip (got {n})")
        check(len(manifest.by_status(ingest.EXCLUDED)) == 1, "one clip marked excluded")

        print("3) run HMR (synthetic) + anonymize")
        pipeline.run_hmr(cfg, manifest)
        done = manifest.by_status(ingest.POSE_DONE)
        check(len(done) == 5, f"5 clips processed, family visit skipped (got {len(done)})")
        check(len(manifest.by_status(ingest.FAILED)) == 0, "no failures")

        print("4) privacy guarantee: shape stripped, pose kept")
        sample = SmplMotion.load_npz(cfg.pose_dir / f"{done[0].clip_id}.npz")
        check(sample.poses.shape[1] == SMPL_POSE_DIM, "pose is SMPL-72")
        check(np.allclose(sample.betas, 0.0), "betas (body identity) are zeroed")
        check(sample.n_frames >= cfg.min_clip_frames, "clip long enough to keep")

        print("5) build AMASS dataset")
        stats = pipeline.build(cfg, manifest)
        check(stats.n_clips == 5, f"dataset has 5 clips (got {stats.n_clips})")
        check(stats.total_seconds > 0, "dataset has non-zero duration")
        any_npz = next((cfg.dataset_dir / "amass").glob("*.npz"))
        amass = np.load(any_npz)
        check(amass["poses"].shape[1] == AMASS_SMPLH_POSE_DIM, "AMASS poses are 156-d (SMPL-H layout)")
        check(np.allclose(amass["poses"][:, SMPL_POSE_DIM:], 0.0), "hand/face slots neutral (zero)")

        print("6) reload manifest from disk (resumability)")
        reloaded = Manifest.load(cfg.manifest_path)
        check(reloaded.counts().get(ingest.POSE_DONE) == 5, "state survives round-trip to disk")

        print("7) prepare MDM data skeleton")
        out = tmp / "mdm_data"
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "motion_model" / "prepare_mdm_data.py"),
             "--dataset", str(cfg.dataset_dir), "--out", str(out)],
            capture_output=True, text=True,
        )
        check(rc.returncode == 0, f"prepare_mdm_data ran (stderr: {rc.stderr[-200:]})")
        check((out / "train.txt").exists(), "MDM train split written")
        check(len(list((out / "texts").glob("*.txt"))) == 5, "one caption scaffold per clip")

        print("8) Pipeline 2: prepare + train with the synthetic 'noop' method")
        mm_cfg = MotionModelConfig(dataset_dir=cfg.dataset_dir, work_root=tmp / "mm", method="noop")
        trainer = get_trainer(mm_cfg)
        trainer.prepare()
        save = trainer.train()
        check((save / "model_final.pt").exists(), "motion-model checkpoint written")
        mm_manifest = json.loads((save / "train_manifest.json").read_text())
        check(mm_manifest["n_clips"] == 5, f"trained on all 5 clips (got {mm_manifest['n_clips']})")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
