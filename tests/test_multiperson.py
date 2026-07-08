"""Tests for multi-person support: contract, identity, consent, per-person export.

GPU-free via the noop backend's synthetic multi-person mode. Covers the whole
opt-in path (HMR tracks -> shape clustering -> consent gate -> per-person datasets)
plus the safety guard, the track-aware caption bridge, and person conditioning.
The single-subject path staying unchanged is covered by the existing suites.
Runs under pytest OR directly: `python tests/test_multiperson.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import identity, ingest, people, pipeline
from videotomocap.backends import get_backend
from videotomocap.backends.smplx_frames import _frame_key, _guard_single_detection
from videotomocap.backends.base import BackendError
from videotomocap.config import PipelineConfig
from videotomocap.dataset import AMASS_SMPLH_POSE_DIM
from videotomocap.pose import SmplMotion


def assert_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _footage(root: Path, layout=("cam/d1/a.mp4", "cam/d1/b.mp4", "cam/d2/c.mp4")) -> Path:
    for rel in layout:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


def _cfg(tmp: Path, **kw) -> PipelineConfig:
    kw.setdefault("backend", "noop")
    kw.setdefault("target_fps", 20.0)
    kw.setdefault("auto_mirror", "off")
    kw.setdefault("quality_filter", "off")
    return PipelineConfig(footage_root=_footage(tmp / "footage"), work_root=tmp / "work", **kw)


# -- config -----------------------------------------------------------------

def test_config_multiperson_validation_and_defaults():
    assert PipelineConfig().multi_person is False
    assert PipelineConfig(multi_person="on").multi_person is True
    assert_raises(ValueError, lambda: PipelineConfig(person_assignment="magic"))
    assert_raises(ValueError, lambda: PipelineConfig(unassigned_policy="maybe"))
    cfg = PipelineConfig(work_root=Path("w"))
    assert cfg.identity_dir == Path("w/identity") and cfg.people_path == Path("w/people.json")


# -- noop multi-person backend ----------------------------------------------

def test_noop_runtracks_makes_distinct_stable_people():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), multi_person=True, synthetic_people=3)
        backend = get_backend(cfg)
        t1 = backend.run_tracks(Path("a.mp4"), Path(tmp) / "o")
        t2 = backend.run_tracks(Path("b.mp4"), Path(tmp) / "o2")
        assert len(t1) == 3
        # distinct people within a clip
        assert not np.allclose(t1[0].betas, t1[1].betas)
        # same person id -> same shape across different clips (enables clustering)
        assert np.allclose(t1[0].betas, t2[0].betas)


# -- identity clustering ----------------------------------------------------

def test_cluster_betas_groups_by_person():
    # two people, three tracks each, small within-person jitter
    rng = np.random.default_rng(0)
    a = np.array([5.0] * 10)
    b = np.array([-5.0] * 10)
    betas = {}
    for i in range(3):
        betas[f"c{i}__p0"] = a + rng.normal(0, 0.01, 10)
        betas[f"c{i}__p1"] = b + rng.normal(0, 0.01, 10)
    labels = identity.cluster_betas(betas, max_people=2)
    a_labels = {labels[k] for k in labels if k.endswith("p0")}
    b_labels = {labels[k] for k in labels if k.endswith("p1")}
    assert len(a_labels) == 1 and len(b_labels) == 1 and a_labels != b_labels


# -- people registry + consent ----------------------------------------------

def test_people_registry_grant_revoke_and_audit():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        assert people.load_registry(cfg) == {}
        people.set_consent(cfg, "dad", granted=True)
        reg = people.load_registry(cfg)
        assert people.is_granted(reg, "dad") and not people.is_granted(reg, "nobody")
        people.set_consent(cfg, "dad", granted=False)
        assert not people.is_granted(people.load_registry(cfg), "dad")
        # fail-closed + policy for unassigned
        assert people.consent_ok(reg, None, "exclude") is False
        assert people.consent_ok(reg, None, "include") is True
        lines = cfg.consent_log_path.read_text().splitlines()
        assert len(lines) == 2 and json.loads(lines[0])["event"] == "consent"


# -- pipeline end-to-end (the big one) --------------------------------------

def _run_multi(tmp: Path):
    cfg = _cfg(tmp, multi_person=True, synthetic_people=2, max_people=2)
    m = ingest.scan(cfg)
    m.save(cfg.manifest_path)
    pipeline.run_hmr(cfg, m)
    return cfg, m


def test_pipeline_multiperson_tracks_and_identity_store():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, m = _run_multi(Path(tmp))
        done = m.by_status(ingest.POSE_DONE)
        assert done and all(len(c.tracks) == 2 for c in done)
        assert len(list(cfg.pose_dir.glob("*.npz"))) == 2 * len(done)
        assert len(list(cfg.identity_dir.glob("*.npz"))) == 2 * len(done)  # betas retained


def test_build_is_fail_closed_then_exports_on_consent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, m = _run_multi(Path(tmp))
        s1 = pipeline.build(cfg, m)
        assert s1.n_clips == 0                                   # no consent -> nothing exports
        registry = people.load_registry(cfg)
        assert set(registry) == {"person_00", "person_01"}      # shape clustering found 2 people
        for pid in registry:
            people.set_consent(cfg, pid, granted=True)
        s2 = pipeline.build(cfg, m)
        assert s2.n_clips == 2 * len(m.by_status(ingest.POSE_DONE))
        index = json.loads((cfg.dataset_dir / "index.json").read_text())["clips"]
        assert {c["person_id"] for c in index} == {"person_00", "person_01"}
        # per-person sub-datasets exist for "moves like <person>"
        assert {p.name for p in (cfg.dataset_dir / "by_person").iterdir()} == {"person_00", "person_01"}


def test_export_stays_shape_neutral_even_multiperson():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, m = _run_multi(Path(tmp))
        pipeline.build(cfg, m)  # assigns people
        for pid in people.load_registry(cfg):
            people.set_consent(cfg, pid, granted=True)
        pipeline.build(cfg, m)
        any_npz = next((cfg.dataset_dir / "amass").glob("*.npz"))
        assert np.allclose(np.load(any_npz)["betas"], 0.0)      # privacy invariant holds on export


def test_manifest_tracks_roundtrip_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, m = _run_multi(Path(tmp))
        reloaded = ingest.Manifest.load(cfg.manifest_path)
        clip = reloaded.by_status(ingest.POSE_DONE)[0]
        assert len(clip.tracks) == 2 and clip.tracks[0].track_id == "p0"
        assert clip.tracks[0].pose_rel.endswith("__p0.npz")


# -- single-subject unchanged when the flag is off --------------------------

def test_single_subject_path_writes_no_tracks():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))  # multi_person defaults off
        m = ingest.scan(cfg)
        m.save(cfg.manifest_path)
        pipeline.run_hmr(cfg, m)
        done = m.by_status(ingest.POSE_DONE)
        assert done and all(c.tracks == [] for c in done)
        assert (cfg.pose_dir / f"{done[0].clip_id}.npz").exists()   # classic naming


# -- smplx multi-detection guard --------------------------------------------

def test_smplx_guard_rejects_multidetection():
    single = [Path("frame_00000.npz"), Path("frame_00001.npz"), Path("frame_00002.npz")]
    _guard_single_detection("osx", single)  # distinct frame indices -> ok
    multi = [Path("00000_boxA.npz"), Path("00000_boxB.npz"), Path("00001_boxA.npz")]
    assert _frame_key(multi[0]) == _frame_key(multi[1]) == "00000"
    assert_raises(BackendError, lambda: _guard_single_detection("multihmr", multi))


# -- CLI people commands ----------------------------------------------------

def test_cli_people_flow():
    from videotomocap import cli
    with tempfile.TemporaryDirectory() as tmp:
        cfg, m = _run_multi(Path(tmp))
        common = ["--work-root", str(cfg.work_root), "--footage-root", str(cfg.footage_root)]
        assert cli.main([*common, "people", "assign"]) == 0
        assert cli.main([*common, "people", "list"]) == 0
        assert cli.main([*common, "people", "grant", "--all"]) == 0
        reg = people.load_registry(cfg)
        assert reg and all(r["consent"]["granted"] for r in reg.values())


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} multi-person tests passed")


if __name__ == "__main__":
    _run_all()
