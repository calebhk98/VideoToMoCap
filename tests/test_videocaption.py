"""Unit tests for Pipeline 3 (videocaption): config, manifest, segmentation,
frames, backends, store, index, windows, fine-tune bridge, pipeline, CLI.

GPU-free and decoder-free by construction -- the `noop` captioner/aggregator
exercise the whole flow with no weights/vLLM/ffmpeg, and every real backend's
command construction is asserted by capturing `_run_cmd` (so the upstream driver
+ flags are actually pinned). Runs under pytest OR directly:
`python tests/test_videocaption.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videocaption import cli, manifest as mf, pipeline, store
from videocaption.backends import (
    available_aggregators, available_captioners, get_aggregator, get_captioner,
)
from videocaption.backends.base import CaptionError, FrameCaption, SegmentLabel
from videocaption.config import CaptionConfig, load_config
from videocaption.finetune import FinetuneError, LoRAFinetune, build_dataset
from videocaption.frames import frame_times
from videocaption.index import build_index, search
from videocaption.segment import Segment, plan_segments, segment_video, subchunk
from videocaption.store import CaptionRow
from videocaption.window import build_windows, group_windows


def assert_raises(exc, fn):
    """Tiny pytest.raises stand-in so tests run under plain `python` too."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _val(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _fake_archive(root: Path, files=("a/v1.mp4", "a/v2.mp4", "b/v3.mov")) -> Path:
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


def _cfg(tmp: Path, **kw) -> CaptionConfig:
    kw.setdefault("captioner", "noop")
    kw.setdefault("aggregator", "noop")
    kw.setdefault("scene_detect", False)
    return CaptionConfig(footage_root=_fake_archive(tmp / "footage"), work_root=tmp / "work", **kw)


# -- config -----------------------------------------------------------------

def test_config_validation_rejects_bad_enums_and_window():
    assert_raises(ValueError, lambda: CaptionConfig(frame_extractor="gstreamer"))
    assert_raises(ValueError, lambda: CaptionConfig(index_format="parquet"))
    # window must be >= segment cap (a window groups whole segments)
    assert_raises(ValueError, lambda: CaptionConfig(window_seconds=60, max_segment_seconds=120))


def test_config_roundtrip_yaml_and_unknown_keys():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("captioner: noop\nmax_segment_seconds: 90\nscene_detect: off\n")
        cfg = load_config(p)
        assert cfg.captioner == "noop" and cfg.max_segment_seconds == 90 and cfg.scene_detect is False
        p.write_text("captioner: noop\nnot_a_key: 1\n")
        assert_raises(ValueError, lambda: load_config(p))


def test_config_paths_and_to_dict():
    cfg = CaptionConfig(work_root=Path("work/cap"))
    assert cfg.manifest_path == Path("work/cap/manifest.json")
    assert cfg.segments_dir == Path("work/cap/segments")
    d = cfg.to_dict()
    assert d["captioner"] == "joycaption" and isinstance(d["work_root"], str)


# -- manifest ---------------------------------------------------------------

def test_scan_exclude_include_refresh_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        m = mf.scan(cfg)
        assert len(m.videos) == 3
        assert mf.exclude(m, patterns=["b/*"]) == 1
        assert len(m.by_status(mf.EXCLUDED)) == 1
        assert mf.include(m, patterns=["b/*"]) == 1
        m.save(cfg.manifest_path)
        reloaded = mf.Manifest.load(cfg.manifest_path)
        assert reloaded.counts().get(mf.PENDING) == 3
        # refresh preserves state
        m.videos[0].status = mf.DONE
        refreshed = mf.refresh(cfg, m)
        assert refreshed.get(m.videos[0].video_id).status == mf.DONE


def test_scan_missing_root_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = CaptionConfig(footage_root=Path(tmp) / "nope", work_root=Path(tmp) / "w")
        assert_raises(FileNotFoundError, lambda: mf.scan(cfg))


# -- segmentation (pure) ----------------------------------------------------

def test_subchunk_splits_long_scenes_under_cap():
    spans = subchunk([(0, 300)], max_seconds=120, min_seconds=3)
    assert len(spans) == 3 and all(e - s <= 120 + 1e-6 for s, e in spans)
    assert spans[0][0] == 0 and spans[-1][1] == 300


def test_subchunk_merges_short_scenes():
    # a 2s stub merges into its neighbor, not its own segment
    assert subchunk([(0, 2), (2, 50)], 120, 3) == [(0, 50)]
    # short leading scene folds forward
    assert subchunk([(0, 1), (1, 40)], 120, 3) == [(0, 40)]


def test_plan_segments_empty_scenes_uses_whole_video():
    segs = plan_segments([], duration=250, max_seconds=120, min_seconds=3)
    assert [s.index for s in segs] == [0, 1, 2]
    assert segs[0].start == 0.0 and segs[-1].end == 250


def test_segment_video_with_injected_signals_reads_no_file():
    segs, dur = segment_video(Path("nope.mp4"), scene_detect=True, scene_threshold=27,
                              max_seconds=120, min_seconds=3, duration=90.0, scenes=[(0, 40), (40, 90)])
    assert dur == 90.0 and len(segs) == 2


# -- frames (pure) ----------------------------------------------------------

def test_frame_times_half_step_and_cap():
    ts = frame_times(0, 10, 1.0, 150)
    assert ts == [round(0.5 + i, 3) for i in range(10)]      # centered in each 1s bin
    assert len(frame_times(0, 1000, 1.0, 150)) == 150         # thinned to the cap
    assert frame_times(5, 5, 1.0, 150) == [5]                 # zero-duration -> one frame


# -- backends: noop + command construction ----------------------------------

def test_registries_and_unknown_fail_loud():
    assert "noop" in available_captioners() and "joycaption" in available_captioners()
    assert "noop" in available_aggregators() and "dolphin" in available_aggregators()
    assert_raises(CaptionError, lambda: get_captioner(CaptionConfig(captioner="nope")))
    assert_raises(CaptionError, lambda: get_aggregator(CaptionConfig(aggregator="nope")))


def test_noop_captioner_is_deterministic_and_per_frame():
    with tempfile.TemporaryDirectory() as tmp:
        cap = get_captioner(_cfg(Path(tmp)))
        a = cap.caption_frames(Path("a/v1.mp4"), [0.5, 1.5, 2.5], Path(tmp) / "o")
        b = cap.caption_frames(Path("a/v1.mp4"), [0.5, 1.5, 2.5], Path(tmp) / "o2")
        assert len(a) == 3 and [c.text for c in a] == [c.text for c in b]  # stable
        assert all(c.text.endswith(".") for c in a)


def test_noop_aggregator_makes_description_and_tags():
    agg = get_aggregator(CaptionConfig(aggregator="noop"))
    caps = [FrameCaption(0.5, "a dog walking outdoors."), FrameCaption(1.5, "a dog sitting outdoors.")]
    label = agg.aggregate(caps, num_tags=5)
    assert "dog" in label.tags and label.description and len(label.tags) <= 5


def _stub_driver(backend, writer):
    """Replace _run_cmd: record the cmd and fabricate the driver's --out file."""
    calls = []

    def fake(cmd, cwd=None):
        calls.append(([str(c) for c in cmd], cwd))
        writer(cmd)

    backend._run_cmd = fake
    return calls


def test_joycaption_builds_command_and_parses_output(monkeypatch=None):
    import videocaption.backends.joycaption as jc
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, captioner="joycaption")
        cap = get_captioner(cfg)
        # avoid real ffmpeg: fake the frame files
        orig = jc.frames.extract_frames
        jc.frames.extract_frames = lambda v, times, out, extractor=None: [
            (out.mkdir(parents=True, exist_ok=True) or out / f"frame_{i:04d}.jpg") for i in range(len(times))
        ]
        try:
            def writer(cmd):
                out = Path(_val([str(c) for c in cmd], "--out"))
                out.write_text(json.dumps({"captions": [{"file": "f", "text": "hi"}, {"file": "g", "text": "yo"}]}))
            calls = _stub_driver(cap, writer)
            res = cap.caption_frames(Path("a/v1.mp4"), [0.5, 1.5], tmp / "seg")
        finally:
            jc.frames.extract_frames = orig
        cmd = calls[0][0]
        assert cmd[1].endswith("joycaption_infer.py")
        assert _val(cmd, "--model") == cfg.captioner_model
        assert [c.text for c in res] == ["hi", "yo"] and [c.time for c in res] == [0.5, 1.5]


def test_joycaption_mismatched_count_fails_loud():
    import videocaption.backends.joycaption as jc
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cap = get_captioner(_cfg(tmp, captioner="joycaption"))
        jc.frames.extract_frames = lambda v, times, out, extractor=None: [Path("x") for _ in times]
        cap._run_cmd = lambda cmd, cwd=None: Path(_val([str(c) for c in cmd], "--out")).write_text(
            json.dumps({"captions": [{"file": "f", "text": "only one"}]}))
        assert_raises(CaptionError, lambda: cap.caption_frames(Path("a/v1.mp4"), [0.5, 1.5], tmp / "s"))


def test_dolphin_builds_command_and_parses_label():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, aggregator="dolphin")
        agg = get_aggregator(cfg)

        def writer(cmd):
            out = Path(_val([str(c) for c in cmd], "--out"))
            out.write_text(json.dumps({"description": "a scene", "tags": ["x", "y"]}))
        calls = _stub_driver(agg, writer)
        label = agg.aggregate([FrameCaption(0.5, "a")], num_tags=5)
        cmd = calls[0][0]
        assert cmd[1].endswith("dolphin_aggregate.py")
        assert _val(cmd, "--num-tags") == "5"
        assert label.description == "a scene" and label.tags == ["x", "y"]


def test_dolphin_missing_description_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        agg = get_aggregator(_cfg(Path(tmp), aggregator="dolphin"))
        agg._run_cmd = lambda cmd, cwd=None: Path(_val([str(c) for c in cmd], "--out")).write_text(
            json.dumps({"tags": ["x"]}))
        assert_raises(CaptionError, lambda: agg.aggregate([FrameCaption(0.5, "a")], num_tags=3))


def test_subprocess_env_pins_cuda_device():
    cfg = CaptionConfig(captioner="noop")
    cfg.cuda_device = "1"
    cap = get_captioner(cfg)
    assert cap._subprocess_env()["CUDA_VISIBLE_DEVICES"] == "1"
    cfg.cuda_device = None
    assert cap._subprocess_env() is None


# -- store ------------------------------------------------------------------

def test_store_segments_and_incremental_labels():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        segs = [Segment(0, 0, 30), Segment(1, 30, 60)]
        store.save_segments(cfg, "vid", segs, duration=60)
        assert [s.index for s in store.load_segments(cfg, "vid")] == [0, 1]
        assert store.load_labels(cfg, "vid") == {}
        store.save_label(cfg, "vid", 0, SegmentLabel("d0", ["a"]))
        store.save_label(cfg, "vid", 1, SegmentLabel("d1", ["b"]))
        labels = store.load_labels(cfg, "vid")
        assert labels[0].description == "d0" and labels[1].tags == ["b"]


def test_video_rows_only_labelled_segments():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        store.save_segments(cfg, "vid", [Segment(0, 0, 30), Segment(1, 30, 60)], 60)
        store.save_label(cfg, "vid", 0, SegmentLabel("only first", ["t"]))
        video = mf.Video(video_id="vid", rel_path="a/v.mp4")
        rows = store.video_rows(cfg, video)
        assert len(rows) == 1 and rows[0].seg_index == 0


# -- index ------------------------------------------------------------------

def test_index_build_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), index_format="both")
        rows = [CaptionRow("v", "a/v.mp4", 0, 0, 30, "a cat on a sofa", ["cat", "sofa"]),
                CaptionRow("v", "a/v.mp4", 1, 30, 60, "a car on a road", ["car", "road"])]
        summ = build_index(cfg, rows)
        assert summ["n_rows"] == 2 and Path(summ["json"]).exists()
        hits = search(cfg.index_dir / "captions.db", "cat")
        assert len(hits) == 1 and hits[0]["start"] == 0


# -- windows (pure) ---------------------------------------------------------

def test_group_windows_packs_consecutive_segments():
    rows = [CaptionRow("v", "p", i, i * 120, (i + 1) * 120, f"d{i}", []) for i in range(6)]
    wins = group_windows(rows, window_seconds=300)  # 5 min -> 2 segments (240s) per window before overflow
    assert len(wins) == 3
    assert all(w.end - w.start <= 300 for w in wins)
    assert sum(len(w.entries) for w in wins) == 6


def test_build_windows_over_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), max_segment_seconds=60, window_seconds=120)
        store.save_segments(cfg, "vid", [Segment(i, i * 60, (i + 1) * 60) for i in range(4)], 240)
        for i in range(4):
            store.save_label(cfg, "vid", i, SegmentLabel(f"d{i}", []))
        m = mf.Manifest(footage_root="x", videos=[mf.Video("vid", "a/v.mp4", status=mf.DONE)])
        wins = build_windows(cfg, m)
        assert len(wins) == 2 and all(len(w.entries) == 2 for w in wins)


# -- finetune ---------------------------------------------------------------

def test_build_dataset_target_is_window_relative():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        rows = [CaptionRow("v", "a/v.mp4", 0, 600, 660, "d0", ["t0"]),
                CaptionRow("v", "a/v.mp4", 1, 660, 720, "d1", ["t1"])]
        wins = group_windows(rows, 3600)
        path = build_dataset(cfg, wins)
        line = json.loads(path.read_text().splitlines()[0])
        assert line["video"] == "a/v.mp4" and line["window"] == [600, 720]
        labels = json.loads(line["messages"][1]["content"])
        assert labels[0]["start"] == 0.0 and labels[1]["start"] == 60.0  # window-relative


def test_finetune_commands_and_require_repo():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, lora_rank=256, lora_alpha=512)
        ft = LoRAFinetune(cfg)
        train = ft.train_command()
        assert _val(train, "--lora_rank") == "256" and _val(train, "--lora_target") == "all"
        assert _val(train, "--model_name_or_path") == cfg.finetune_base
        assert _val(ft.merge_command(), "--export_dir").endswith("merged")
        assert_raises(FinetuneError, ft.run)          # no repo set
        cfg.finetune_repo = tmp / "repo"
        (tmp / "repo").mkdir()
        assert_raises(FinetuneError, ft.run)          # repo but no dataset.jsonl


# -- pipeline (device planning + end-to-end noop) ---------------------------

def test_resolve_and_expand_devices():
    assert pipeline.resolve_devices([]) == [None]
    assert pipeline.resolve_devices(["0", "1"]) == ["0", "1"]
    assert pipeline._expand([None], 3) == [None, None, None]
    assert pipeline._expand(["0", "1"], 2) == ["0", "0", "1", "1"]


def _seed_segmented(cfg, m):
    """Pretend Step 1 already ran (decoder-free): write each video's segments."""
    for v in m.videos:
        store.save_segments(cfg, v.video_id, [Segment(0, 0, 30), Segment(1, 30, 60)], 60)
        v.status = mf.SEGMENTED
    m.save(cfg.manifest_path)


def test_pipeline_end_to_end_noop_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), max_segment_seconds=30, window_seconds=120)
        m = mf.scan(cfg)
        _seed_segmented(cfg, m)
        pipeline.caption(cfg, m)
        assert m.counts().get(mf.DONE) == 3
        assert pipeline.index(cfg, m)["n_rows"] == 6          # 3 videos x 2 segments
        assert pipeline.windows(cfg, m)["n_windows"] == 3     # 1 window/video (60s < 120s)
        assert pipeline.finetune_data(cfg, m)["n_examples"] == 3
        # resume: re-running caption is a no-op (all done), labels untouched
        pipeline.caption(cfg, m)
        assert m.counts().get(mf.DONE) == 3


def test_pipeline_isolates_failures():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))  # scene_detect False, but no segments seeded -> probe_duration reads file -> fails
        m = mf.scan(cfg)
        pipeline.caption(cfg, m)   # every video errors in segmentation, none fatal
        assert m.counts().get(mf.FAILED) == 3
        assert all(v.error for v in m.videos)


# -- CLI --------------------------------------------------------------------

def test_cli_full_flow_with_noop():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, max_segment_seconds=30, window_seconds=120)
        # a real config file makes the CLI select the noop backends (decoder-free)
        conf = tmp / "vc.yaml"
        conf.write_text(
            f"footage_root: {cfg.footage_root}\nwork_root: {cfg.work_root}\n"
            "captioner: noop\naggregator: noop\nscene_detect: off\n"
            "max_segment_seconds: 30\nwindow_seconds: 120\n"
        )
        m = mf.scan(cfg)
        _seed_segmented(cfg, m)
        common = ["--config", str(conf)]
        for sub in (["models"], ["scan"], ["status"], ["info"],
                    ["caption"], ["index"], ["windows"], ["finetune-data"], ["search", "person"]):
            assert cli.main([*common, *sub]) == 0
        # the CLI actually captioned via noop (not the failure path)
        assert mf.Manifest.load(cfg.manifest_path).counts().get(mf.DONE) == 3


def test_cli_discover_config_prefers_explicit():
    assert cli._discover_config("explicit.yaml") == "explicit.yaml"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} videocaption tests passed")


if __name__ == "__main__":
    _run_all()
