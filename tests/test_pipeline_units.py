"""Unit tests for config, ingest, dataset, pipeline, CLI, and backend parsing.

Targets the structural gaps the coverage pass flagged: the CLI (previously 0%),
ingest.refresh, the pipeline failure path, config loading/validation, dataset
edges, and the mockable WHAM/TRAM/GVHMR adapter internals.

Runs under pytest OR directly: `python tests/test_pipeline_units.py`.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotomocap import ingest, pipeline
from videotomocap.backends import BackendError, get_backend
from videotomocap.backends.base import HMRBackend
from videotomocap.config import PipelineConfig, load_config
from videotomocap.ingest import Manifest


def assert_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _footage(root: Path):
    for rel in ("cam01/2024-05-01/a.mp4", "cam01/2024-05-01/b.mp4", "cam02/2024-12-24/fam.mp4"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def _noop_cfg(tmp: Path) -> PipelineConfig:
    footage = tmp / "footage"
    _footage(footage)
    return PipelineConfig(footage_root=footage, work_root=tmp / "work", backend="noop", target_fps=30.0)


# --- config -----------------------------------------------------------------

def test_load_config_happy_path_and_tuple_and_unknown_keys():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = tmp / "c.yaml"
        good.write_text("backend: noop\ntarget_fps: 24.0\nvideo_exts: ['.mp4', '.mov']\n")
        cfg = load_config(good)
        assert cfg.backend == "noop" and cfg.target_fps == 24.0
        assert cfg.video_exts == (".mp4", ".mov")  # list -> tuple

        bad = tmp / "bad.yaml"
        bad.write_text("nonsense_key: 1\n")
        assert_raises(ValueError, lambda: load_config(bad))


def test_config_rejects_bad_use_frame_and_stringifies_paths():
    assert_raises(ValueError, lambda: PipelineConfig(use_frame="bogus"))
    cfg = PipelineConfig(footage_root="/a/b", backend_repo="/opt/x")
    d = cfg.to_dict()
    assert d["footage_root"] == "/a/b" and isinstance(d["backend_repo"], str)


# --- ingest -----------------------------------------------------------------

def test_refresh_preserves_status_adds_new_drops_missing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _noop_cfg(tmp)
        m = ingest.scan(cfg)
        ingest.exclude(m, patterns=["*/2024-12-24/*"])           # exclude the family clip
        m.get([c.clip_id for c in m.clips if c.rel_path.endswith("a.mp4")][0]).status = ingest.POSE_DONE

        # add a new file, remove an old one
        (cfg.footage_root / "cam01/2024-05-01/c.mp4").write_bytes(b"")
        (cfg.footage_root / "cam01/2024-05-01/b.mp4").unlink()

        m2 = ingest.refresh(cfg, m)
        rels = {c.rel_path: c for c in m2.clips}
        assert "cam01/2024-05-01/b.mp4" not in rels                # dropped
        assert rels["cam01/2024-05-01/c.mp4"].status == ingest.PENDING   # new
        assert rels["cam02/2024-12-24/fam.mp4"].status == ingest.EXCLUDED  # preserved
        assert rels["cam01/2024-05-01/a.mp4"].status == ingest.POSE_DONE   # preserved


def test_exclude_by_id_and_include_and_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        m = ingest.scan(cfg)
        cid = m.clips[0].clip_id
        assert ingest.exclude(m, clip_ids=[cid]) == 1
        assert ingest.exclude(m, clip_ids=[cid]) == 0              # already excluded -> no-op
        assert ingest.include(m, clip_ids=[cid]) == 1
        assert m.get(cid).status == ingest.PENDING


def test_manifest_get_missing_raises_and_scan_missing_root():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        m = ingest.scan(cfg)
        assert_raises(KeyError, lambda: m.get("does-not-exist"))
    cfg2 = PipelineConfig(footage_root="/no/such/dir", backend="noop")
    assert_raises(FileNotFoundError, lambda: ingest.scan(cfg2))


def test_clip_is_processable():
    mk = lambda s: ingest.Clip(clip_id="x", camera="c", rel_path="r", status=s)
    assert mk(ingest.PENDING).is_processable() and mk(ingest.FAILED).is_processable()
    assert not mk(ingest.EXCLUDED).is_processable()
    assert not mk(ingest.POSE_DONE).is_processable()


def test_manifest_roundtrip_preserves_fields():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        m = ingest.scan(cfg)
        m.clips[0].error = "boom"
        m.clips[0].n_frames = 42
        m.clips[0].note = "hi"
        m.save(cfg.manifest_path)
        back = Manifest.load(cfg.manifest_path)
        c = back.clips[0]
        assert c.error == "boom" and c.n_frames == 42 and c.note == "hi"
        assert not cfg.manifest_path.with_suffix(".json.tmp").exists()  # atomic tmp cleaned up


# --- pipeline ---------------------------------------------------------------

class _FlakyBackend(HMRBackend):
    """Fails on one clip id, succeeds on the rest -- to exercise resilience."""
    name = "flaky"

    def run(self, video_path, out_dir, *, static=False):
        from videotomocap.backends.noop import NoopBackend
        if "b.mp4" in str(video_path):
            raise RuntimeError("simulated backend failure")
        return NoopBackend(self.cfg).run(video_path, out_dir, static=static)


def test_run_hmr_records_failure_and_continues(monkeypatch=None):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        m = ingest.scan(cfg)
        ingest.exclude(m, patterns=["*/2024-12-24/*"])
        import videotomocap.backends as be
        orig = be.get_backend
        be_pipeline = pipeline.get_backend
        pipeline.get_backend = lambda c: _FlakyBackend(c)  # inject flaky backend
        try:
            pipeline.run_hmr(cfg, m)
        finally:
            pipeline.get_backend = be_pipeline
        statuses = {c.rel_path: c.status for c in m.clips}
        assert statuses["cam01/2024-05-01/b.mp4"] == ingest.FAILED
        assert statuses["cam01/2024-05-01/a.mp4"] == ingest.POSE_DONE
        assert m.get([c.clip_id for c in m.clips if c.rel_path.endswith("b.mp4")][0]).error


def test_run_hmr_respects_limit():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        m = ingest.scan(cfg)
        pipeline.run_hmr(cfg, m, limit=1)
        assert len(m.by_status(ingest.POSE_DONE)) == 1


def test_run_all_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        m = ingest.scan(cfg)
        m.save(cfg.manifest_path)
        stats = pipeline.run_all(cfg)
        assert stats.n_clips == 3 and stats.total_seconds > 0


# --- dataset ----------------------------------------------------------------

def test_build_dataset_empty_and_min_frames_and_determinism():
    from videotomocap.dataset import build_dataset
    from videotomocap.pose import SmplMotion

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # empty input
        stats = build_dataset([], tmp / "empty", min_frames=30)
        assert stats.n_clips == 0 and stats.fps == 0.0
        assert (tmp / "empty" / "index.json").exists()

        # min_frames filtering: one short, one long
        posedir = tmp / "pose"
        posedir.mkdir()
        for name, n in (("short", 10), ("long", 40)):
            SmplMotion(poses=np.zeros((n, 72), np.float32), trans=np.zeros((n, 3), np.float32),
                       fps=30.0).save_npz(posedir / f"{name}.npz")
        s2 = build_dataset(sorted(posedir.glob("*.npz")), tmp / "ds", min_frames=30)
        assert s2.n_clips == 1  # short dropped

        # split determinism: many clips, same seed -> same val set
        many = tmp / "many"
        many.mkdir()
        for i in range(20):
            SmplMotion(poses=np.zeros((40, 72), np.float32), trans=np.zeros((40, 3), np.float32),
                       fps=30.0).save_npz(many / f"c{i}.npz")
        paths = sorted(many.glob("*.npz"))
        build_dataset(paths, tmp / "d1", min_frames=30, val_fraction=0.2, seed=7)
        build_dataset(paths, tmp / "d2", min_frames=30, val_fraction=0.2, seed=7)
        import json
        v1 = {c["clip_id"] for c in json.loads((tmp / "d1" / "index.json").read_text())["clips"] if c["split"] == "val"}
        v2 = {c["clip_id"] for c in json.loads((tmp / "d2" / "index.json").read_text())["clips"] if c["split"] == "val"}
        assert v1 == v2 and len(v1) >= 1


# --- backends: registry + error paths + mockable parsers --------------------

def test_get_backend_unknown_raises():
    assert_raises(BackendError, lambda: get_backend(PipelineConfig(backend="bogus")))


def test_require_repo_and_run_cmd_errors():
    from videotomocap.backends.gvhmr import GVHMRBackend
    be = GVHMRBackend(PipelineConfig(backend="gvhmr", backend_repo=None))
    assert_raises(BackendError, be._require_repo)
    be2 = GVHMRBackend(PipelineConfig(backend="gvhmr", backend_repo="/no/such/repo"))
    assert_raises(BackendError, be2._require_repo)
    # missing interpreter and nonzero exit both surface as BackendError
    assert_raises(BackendError, lambda: be._run_cmd(["/no/such/interpreter"]))
    assert_raises(BackendError, lambda: be._run_cmd([sys.executable, "-c", "import sys; sys.exit(3)"]))


def test_wham_parse_track_selects_world_and_raises():
    from videotomocap.backends.wham import WHAMBackend
    be = WHAMBackend(PipelineConfig(backend="wham", use_frame="global", target_fps=30.0))
    aa = np.zeros((5, 72), np.float32)
    world = np.ones((5, 72), np.float32) * 0.1
    tr = {"pose": aa, "trans": np.zeros((5, 3), np.float32),
          "pose_world": world, "trans_world": np.ones((5, 3), np.float32)}
    m = be._parse_track(tr, Path("clip.mp4"), Path("x.pkl"))
    assert m.poses.shape == (5, 72) and np.allclose(m.trans, 1.0)  # picked world frame
    assert_raises(BackendError, lambda: be._parse_track({"betas": np.zeros(10)}, Path("c"), Path("p")))


def test_wham_longest_track():
    from videotomocap.backends.wham import WHAMBackend
    data = {"a": {"pose": np.zeros((3, 72))}, "b": {"pose": np.zeros((9, 72))}}
    assert len(WHAMBackend._longest_track(data)["pose"]) == 9
    assert_raises(BackendError, lambda: WHAMBackend._longest_track({}))


def test_tram_parse_selects_world_and_raises():
    from videotomocap.backends.base import axis_angle_to_matrix
    from videotomocap.backends.tram import TRAMBackend
    be = TRAMBackend(PipelineConfig(backend="tram", use_frame="global", target_fps=30.0))
    with tempfile.TemporaryDirectory() as tmp:
        rotmat = axis_angle_to_matrix(np.zeros((6, 24, 3)))  # (6,24,3,3) identity
        d = {"pred_rotmat": rotmat, "pred_trans": np.zeros((6, 3), np.float32),
             "pred_trans_world": np.ones((6, 3), np.float32), "pred_shape": np.zeros(10, np.float32)}
        f = Path(tmp) / "hps_track_1.npy"
        np.save(f, d, allow_pickle=True)
        m = be._parse(f, Path("clip.mp4"))
        assert m.poses.shape == (6, 72) and np.allclose(m.trans, 1.0)  # world translation
        # missing rotmat -> BackendError
        bad = Path(tmp) / "bad.npy"
        np.save(bad, {"pred_trans": np.zeros((2, 3))}, allow_pickle=True)
        assert_raises(BackendError, lambda: be._parse(bad, Path("c")))


def test_gvhmr_find_results_layouts_and_missing():
    from videotomocap.backends.gvhmr import GVHMRBackend
    be = GVHMRBackend(PipelineConfig(backend="gvhmr"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "clip").mkdir()
        (out / "clip" / "hmr4d_results.pt").write_bytes(b"x")
        assert be._find_results(out, "clip").name == "hmr4d_results.pt"
    with tempfile.TemporaryDirectory() as tmp:
        assert_raises(BackendError, lambda: be._find_results(Path(tmp), "clip"))


# --- CLI --------------------------------------------------------------------

def test_cli_full_flow():
    from videotomocap import cli
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _noop_cfg(tmp)
        cfgfile = tmp / "cfg.yaml"
        cfgfile.write_text(
            f"footage_root: {cfg.footage_root}\nwork_root: {cfg.work_root}\nbackend: noop\n"
        )
        base = ["--config", str(cfgfile)]
        assert cli.main(base + ["scan"]) == 0
        assert cli.main(base + ["exclude", "--pattern", "*/2024-12-24/*"]) == 0
        assert cli.main(base + ["status"]) == 0
        assert cli.main(base + ["list", "--status", "excluded"]) == 0
        assert cli.main(base + ["run"]) == 0
        # dataset was produced through the CLI path
        assert (cfg.work_root / "dataset" / "stats.json").exists()
        # scan again -> refresh branch
        assert cli.main(base + ["scan"]) == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} pipeline/unit tests passed")


if __name__ == "__main__":
    _run_all()
