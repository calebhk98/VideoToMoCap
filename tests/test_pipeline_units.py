"""Unit tests for config, ingest, dataset, pipeline, CLI, and backend parsing.

Targets the structural gaps the coverage pass flagged: the CLI (previously 0%),
ingest.refresh, the pipeline failure path, config loading/validation, dataset
edges, and the mockable WHAM/TRAM/GVHMR adapter internals.

Runs under pytest OR directly: `python tests/test_pipeline_units.py`.
"""

from __future__ import annotations

import os
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
    """Tiny pytest.raises stand-in so tests run under plain `python` too."""
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


def test_config_mode_fields_tolerate_yaml_off_boolean():
    # YAML parses `auto_mirror: off` as the boolean False -> must become "off".
    assert PipelineConfig(auto_mirror=False).auto_mirror == "off"
    assert PipelineConfig(quality_filter=True).quality_filter == "flag"
    assert PipelineConfig(auto_mirror="correct").auto_mirror == "correct"
    assert_raises(ValueError, lambda: PipelineConfig(quality_filter="bogus"))


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


def test_exclude_is_case_sensitive_cross_platform():
    # fnmatchcase -> deterministic on Windows and Linux alike (plain fnmatch would
    # match case-insensitively on Windows, diverging between OSes).
    m = ingest.Manifest(footage_root="x", clips=[
        ingest.Clip(clip_id="a", camera="c", rel_path="cam1/Family/clip.mp4"),
    ])
    assert ingest.exclude(m, patterns=["*/family/*"]) == 0   # lowercase pattern misses
    assert ingest.exclude(m, patterns=["*/Family/*"]) == 1   # exact case hits


def test_scan_applies_config_exclusions():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.exclude_patterns = ["*/2024-12-24/*"]  # the family visit
        m = ingest.scan(cfg)
        excluded = m.by_status(ingest.EXCLUDED)
        assert len(excluded) == 1 and "2024-12-24" in excluded[0].rel_path
        # a manual include survives a later refresh (config re-exclude doesn't override it)
        ingest.include(m, patterns=["*/2024-12-24/*"])
        m2 = ingest.refresh(cfg, m)
        assert not m2.by_status(ingest.EXCLUDED)


def test_discover_config_precedence():
    from videotomocap import cli
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "videotomocap.yaml").write_text("backend: noop\n")
        assert cli._discover_config("explicit.yaml") == "explicit.yaml"      # --config wins
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            os.environ.pop("VIDEOTOMOCAP_CONFIG", None)
            assert cli._discover_config(None) == "videotomocap.yaml"         # discovered in cwd
            os.environ["VIDEOTOMOCAP_CONFIG"] = "from_env.yaml"
            assert cli._discover_config(None) == "from_env.yaml"             # env beats cwd file
        finally:
            os.environ.pop("VIDEOTOMOCAP_CONFIG", None)
            os.chdir(cwd)


def test_limit_falls_back_to_config():
    from videotomocap import cli
    ns = lambda **kw: type("A", (), kw)()
    assert cli._limit(ns(limit=5), PipelineConfig(limit=2)) == 5     # flag overrides
    assert cli._limit(ns(limit=None), PipelineConfig(limit=2)) == 2  # config used
    assert cli._limit(ns(limit=None), PipelineConfig(limit=None)) is None


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


def test_run_hmr_parallel_matches_sequential():
    # noop is deterministic per clip name, so parallel and sequential must agree.
    frames_by_mode = {}
    for workers in (1, 3):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _noop_cfg(Path(tmp))
            m = ingest.scan(cfg)
            pipeline.run_hmr(cfg, m, workers=workers, gpus=["0", "1"])
            assert len(m.by_status(ingest.POSE_DONE)) == 3
            assert len(m.by_status(ingest.FAILED)) == 0
            frames_by_mode[workers] = {c.clip_id: c.n_frames for c in m.clips}
    assert frames_by_mode[1] == frames_by_mode[3]  # identical results


def test_parallel_records_failures_and_continues():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        m = ingest.scan(cfg)
        be_pipeline = pipeline.get_backend
        pipeline.get_backend = lambda c: _FlakyBackend(c)
        try:
            pipeline.run_hmr(cfg, m, workers=2, gpus=["0", "1"])
        finally:
            pipeline.get_backend = be_pipeline
        statuses = {c.rel_path: c.status for c in m.clips}
        assert statuses["cam01/2024-05-01/b.mp4"] == ingest.FAILED
        assert statuses["cam01/2024-05-01/a.mp4"] == ingest.POSE_DONE


def test_gpu_resolution():
    from videotomocap import gpu
    assert gpu.resolve_devices(["0", "1"]) == ["0", "1"]
    assert gpu.resolve_devices([]) == [None]          # empty -> one ambient slot
    assert gpu.resolve_devices("auto")                # never empty
    # explicit worker count wins; else devices * per-gpu
    assert gpu.resolve_workers(5, ["0"], 2) == 5
    assert gpu.resolve_workers(None, ["0", "1"], 2) == 4
    assert gpu.resolve_workers(None, [None], 3) == 3   # CPU-only fallback


def test_vram_planning_pure_functions():
    from videotomocap import gpu
    # (free - headroom) // est, clamped [1, cap]
    plan = gpu.plan_workers_per_gpu({"0": 20000, "1": 8000}, est_mb=6000, headroom_mb=2000, cap=8)
    assert plan == {"0": 3, "1": 1}
    assert gpu.plan_workers_per_gpu({"0": 100000}, 6000, 2000, 4)["0"] == 4   # cap
    assert gpu.distribute_workers(5, ["0", "1"]) == {"0": 3, "1": 2}
    assert gpu.expand_devices(["0", "1"], {"0": 2, "1": 1}) == ["0", "0", "1"]
    assert gpu.expand_devices([None], {None: 0}) == [None]                    # never empty


def test_measure_peak_vram_captures_dip_and_handles_cpu():
    import time
    from videotomocap import gpu
    state = {"free": 20000}
    probe = lambda: {"0": state["free"]}

    def run():
        state["free"] = 12000        # "model loads" -> free drops
        time.sleep(0.02)             # let the sampler observe the dip

    assert gpu.measure_peak_vram(run, "0", free_mem_fn=probe, poll=0.001) == 8000
    # CPU / no device -> None (runs the fn, no measurement)
    assert gpu.measure_peak_vram(lambda: None, None, free_mem_fn=probe) is None


def test_run_hmr_auto_sizes_from_free_vram():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.workers_per_gpu = "auto"
        cfg.vram_per_worker_mb = 6000   # set -> no calibration
        m = ingest.scan(cfg)
        pipeline.run_hmr(cfg, m, gpus=["0"], free_mem_fn=lambda: {"0": 20000})
        assert len(m.by_status(ingest.POSE_DONE)) == 3


def test_run_hmr_auto_calibrates_first_clip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.workers_per_gpu = "auto"
        cfg.vram_per_worker_mb = None   # -> calibrate on clip 0 (noop uses no VRAM -> default est)
        m = ingest.scan(cfg)
        pipeline.run_hmr(cfg, m, gpus=["0"], free_mem_fn=lambda: {"0": 30000})
        assert len(m.by_status(ingest.POSE_DONE)) == 3   # calib clip + the rest all done


def test_cli_parses_workers_per_gpu_auto_and_vram():
    from videotomocap import cli
    a = cli.build_parser().parse_args(["hmr", "--workers-per-gpu", "auto"])
    cfg = PipelineConfig(backend="noop")
    cli._apply_parallel_overrides(cfg, a)
    assert cfg.workers_per_gpu == "auto"
    b = cli.build_parser().parse_args(["hmr", "--workers-per-gpu", "4", "--vram-per-worker-mb", "5000"])
    cfg2 = PipelineConfig(backend="noop")
    cli._apply_parallel_overrides(cfg2, b)
    assert cfg2.workers_per_gpu == 4 and cfg2.vram_per_worker_mb == 5000


def test_workers_per_gpu_packs_a_single_card():
    # 1 GPU, workers_per_gpu=3 -> 3 clips share the card (intra-GPU parallelism)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.workers_per_gpu = 3
        m = ingest.scan(cfg)
        pipeline.run_hmr(cfg, m, gpus=["0"])
        assert len(m.by_status(ingest.POSE_DONE)) == 3


def test_gpu_pinning_sets_cuda_visible_devices():
    from videotomocap.backends.gvhmr import GVHMRBackend
    cfg = pipeline._clone_for_device(PipelineConfig(backend="gvhmr"), "1")
    assert cfg.cuda_device == "1"
    env = GVHMRBackend(cfg)._subprocess_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    # no device assigned -> inherit parent env unchanged
    assert GVHMRBackend(PipelineConfig(backend="gvhmr"))._subprocess_env() is None


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

def test_regions_and_occlusion_tagging():
    from videotomocap.regions import joints_for_regions
    assert joints_for_regions(["head"]) == [12, 15]
    assert joints_for_regions(["legs", "left_arm"])  # union, no error
    assert_raises(ValueError, lambda: joints_for_regions(["nonsense"]))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.camera_occlusions = {"cam01": ["legs"], "cam02": ["head", "right_arm"]}
        m = ingest.scan(cfg)
        by_cam = {c.camera: c for c in m.clips}
        assert by_cam["cam01"].partial_body and by_cam["cam01"].unreliable_joints == [1, 2, 4, 5, 7, 8, 10, 11]
        assert set(by_cam["cam02"].unreliable_joints) == {12, 15, 14, 17, 19, 21, 23}


def test_partial_body_cameras_shorthand_is_legs():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.partial_body_cameras = ["cam01"]
        m = ingest.scan(cfg)
        c = next(c for c in m.clips if c.camera == "cam01")
        assert c.partial_body and c.unreliable_joints == [1, 2, 4, 5, 7, 8, 10, 11]


def test_occlusion_flows_to_pose_and_dataset():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.camera_occlusions = {"cam01": ["legs"]}
        m = ingest.scan(cfg)
        pipeline.run_hmr(cfg, m)
        # the joint-valid mask reached the saved pose npz
        cam01 = next(c for c in m.clips if c.camera == "cam01")
        from videotomocap.pose import SmplMotion
        mot = SmplMotion.load_npz(cfg.pose_dir / f"{cam01.clip_id}.npz")
        assert mot.joint_valid is not None and not mot.joint_valid[4]  # left_knee unreliable
        assert mot.joint_valid[15]                                     # head still reliable
        # and into the dataset index + AMASS npz
        pipeline.build(cfg, m)
        idx = json.loads((cfg.dataset_dir / "index.json").read_text())
        entry = next(e for e in idx["clips"] if e["clip_id"] == cam01.clip_id)
        assert entry["unreliable_joints"] == [1, 2, 4, 5, 7, 8, 10, 11]
        amass = np.load(cfg.dataset_dir / "amass" / f"{cam01.clip_id}.npz")
        assert "joint_valid_smpl24" in amass


def test_refine_flag_runs_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _noop_cfg(Path(tmp))
        cfg.refine = True
        m = ingest.scan(cfg)
        pipeline.run_hmr(cfg, m)
        assert len(m.by_status(ingest.POSE_DONE)) == 3


# (backend adapter internals -> tests/test_backends.py)


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
