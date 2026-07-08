#!/usr/bin/env python
"""End-to-end self-test of MULTI-PERSON support with zero GPU / weights.

Drives the real pipeline with the synthetic 'noop' backend in multi_person mode:
scan -> HMR (N people/clip) -> shape-cluster into identities -> consent gate ->
per-person AMASS datasets.  Asserts the fail-closed consent default, correct
identity recovery, per-person export, and that the privacy invariant still holds
(exported betas neutral).  Run:  python scripts/multiperson_selftest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import ingest, people, pipeline
from videotomocap.config import PipelineConfig


def make_fake_footage(root: Path) -> None:
    for rel in ("livingroom/2024-05-01", "kitchen/2024-05-02"):
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        for f in ("morning.mp4", "evening.mp4"):
            (d / f).write_bytes(b"")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        footage = tmp / "footage"
        make_fake_footage(footage)

        # a household of 2 consenting people; each appears (synthetically) in every clip
        cfg = PipelineConfig(
            footage_root=footage, work_root=tmp / "work", backend="noop", target_fps=20.0,
            multi_person=True, synthetic_people=2, max_people=2,
            person_assignment="shape", consent_required=True,
        )

        print("1) scan + multi-person HMR")
        manifest = ingest.scan(cfg)
        manifest.save(cfg.manifest_path)
        pipeline.run_hmr(cfg, manifest)
        done = manifest.by_status(ingest.POSE_DONE)
        check(len(done) == 4, f"4 clips recovered (got {len(done)})")
        check(all(len(c.tracks) == 2 for c in done), "each clip carries 2 person tracks")
        check(len(list(cfg.identity_dir.glob("*.npz"))) == 8, "betas retained per track (identity store)")

        print("2) consent is fail-closed by default")
        stats = pipeline.build(cfg, manifest)  # assigns people, then gates
        check(stats.n_clips == 0, "nothing exports before consent is granted")
        registry = people.load_registry(cfg)
        check(set(registry) == {"person_00", "person_01"}, f"shape-clustered into 2 people: {sorted(registry)}")

        print("3) grant consent -> per-person datasets")
        for pid in registry:
            people.set_consent(cfg, pid, granted=True)
        stats = pipeline.build(cfg, manifest)
        check(stats.n_clips == 8, f"all 8 person-tracks export once consented (got {stats.n_clips})")
        by_person = sorted(p.name for p in (cfg.dataset_dir / "by_person").iterdir())
        check(by_person == ["person_00", "person_01"], "one dataset per person for 'moves like <person>'")

        print("4) privacy: exported motion is shape-neutral")
        any_npz = next((cfg.dataset_dir / "amass").glob("*.npz"))
        check(np.allclose(np.load(any_npz)["betas"], 0.0), "betas zeroed in the export (identity never leaves)")
        index = json.loads((cfg.dataset_dir / "index.json").read_text())["clips"]
        check(all("person_id" in c for c in index), "each dataset entry is tagged with its person")

        print("5) revoke -> that person drops out on rebuild")
        people.set_consent(cfg, "person_01", granted=False)
        stats = pipeline.build(cfg, manifest)
        check(stats.n_clips == 4, f"revoking person_01 removes their tracks (got {stats.n_clips})")
        check(len(cfg.consent_log_path.read_text().splitlines()) >= 3, "consent changes are in the audit log")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
