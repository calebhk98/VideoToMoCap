"""Unit tests for Pipeline 2's overfitting guard (motion_model.overfit).

GPU-free: the pre-flight risk model is pure arithmetic over the clip index, and
the post-train curve analysis reads a metrics file the `noop` trainer fabricates.
Multi-donor by construction, since the whole point of the guard is to scale from
one person to a hundred. Runs under pytest OR directly:
`python tests/test_overfit.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model import cli, overfit
from motion_model.config import MotionModelConfig
from motion_model.trainers import get_trainer


def assert_raises(exc, fn):
    """Tiny pytest.raises stand-in so tests run under plain `python` too."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _index(clips: list) -> dict:
    return {"clips": clips}


def _person_clips(n_persons: int, clips_each: int, frames: int, fps: float = 20.0) -> list:
    """A multi-donor index: n_persons donors, each with clips_each clips of `frames`."""
    out = []
    for p in range(n_persons):
        for c in range(clips_each):
            out.append({"clip_id": f"p{p:02d}c{c:02d}", "person_id": f"person_{p:02d}",
                        "n_frames": frames, "fps": fps, "split": "train"})
    return out


def _noop_cfg(tmp: Path) -> MotionModelConfig:
    """A noop config over a fabricated single-clip dataset (for the CLI/curve paths)."""
    amass = tmp / "dataset" / "amass"
    amass.mkdir(parents=True, exist_ok=True)
    (amass / "clip00.npz").write_bytes(b"")   # presence is enough; noop doesn't read it
    (tmp / "dataset" / "index.json").write_text(json.dumps(
        {"clips": [{"clip_id": "clip00", "n_frames": 600, "fps": 20.0, "split": "train"}]}))
    return MotionModelConfig(dataset_dir=tmp / "dataset", work_root=tmp / "w", method="noop")


# -- corpus stats -----------------------------------------------------------

def test_corpus_stats_totals_and_per_person():
    s = overfit.corpus_stats(_index(_person_clips(3, 2, 1200)))   # 3 donors x 2 clips x 60s
    assert s.n_clips == 6 and s.n_persons == 3
    assert abs(s.minutes - 6.0) < 1e-6                            # 6 clips * 1200/20/60 = 1 min
    assert all(abs(p.minutes - 2.0) < 1e-6 for p in s.persons)


def test_corpus_stats_single_person_unlabeled():
    s = overfit.corpus_stats(_index([{"clip_id": "a", "n_frames": 600, "fps": 20.0}]))
    assert s.n_persons == 1 and s.persons[0].person_id == "(unlabeled)"


def test_corpus_stats_excludes_val_from_train_totals():
    clips = [{"clip_id": "t", "n_frames": 1200, "fps": 20.0, "split": "train"},
             {"clip_id": "v", "n_frames": 1200, "fps": 20.0, "split": "val"}]
    s = overfit.corpus_stats(_index(clips))
    assert s.n_clips == 2 and abs(s.train_minutes - 1.0) < 1e-6   # only the train clip counts


# -- pre-flight risk (per-donor aware) --------------------------------------

def test_risk_high_for_scarce_single_person_from_scratch():
    cfg = MotionModelConfig(method="momask", num_steps=80000, batch_size=64)  # no resume
    r = overfit.assess_risk(cfg, _index(_person_clips(1, 10, 1200)))          # ~10 min, one donor
    assert r.level == "high" and r.exposures > 150
    assert any("warm-start" in rec for rec in r.recommendations)


def test_risk_low_for_broad_many_donor_corpus():
    cfg = MotionModelConfig(method="momask", conditioning="person",
                            resume_checkpoint=Path("prior.pt"))               # warm start
    r = overfit.assess_risk(cfg, _index(_person_clips(30, 5, 2400)))         # 30 donors, lots of data
    assert r.level == "low" and r.stats.n_persons == 30


def test_risk_flags_underrepresented_donor():
    cfg = MotionModelConfig(method="mdm", resume_checkpoint=Path("p.pt"))
    clips = _person_clips(2, 5, 2400) + [{"clip_id": "thin", "person_id": "person_99",
                                          "n_frames": 200, "fps": 20.0, "split": "train"}]
    r = overfit.assess_risk(cfg, _index(clips))
    assert [p.person_id for p in r.underrepresented] == ["person_99"]


def test_risk_recommends_person_conditioning_when_unset():
    cfg = MotionModelConfig(method="momask", conditioning="none", resume_checkpoint=Path("p.pt"))
    r = overfit.assess_risk(cfg, _index(_person_clips(4, 2, 2400)))
    assert any("conditioning: person" in rec for rec in r.recommendations)


def test_risk_not_applicable_for_physics_controller():
    r = overfit.assess_risk(MotionModelConfig(method="protomotions"), _index(_person_clips(2, 2, 600)))
    assert not r.applicable and r.level == "n/a"


def test_format_risk_is_readable():
    cfg = MotionModelConfig(method="momask")
    text = overfit.format_risk(overfit.assess_risk(cfg, _index(_person_clips(1, 5, 1200))))
    assert "Overfitting risk" in text and "donor" in text


# -- post-train curve analysis ----------------------------------------------

def test_analyze_curves_detects_overfitting_onset():
    steps = [0, 100, 200, 300, 400, 500, 600]
    val = [1.0, 0.7, 0.5, 0.45, 0.5, 0.6, 0.72]      # bottoms at step 300
    v = overfit.analyze_curves(steps, [0] * 7, val, patience=3)
    assert v.overfitting and v.best_step == 300 and v.onset_step == 300


def test_analyze_curves_no_upturn_when_still_improving():
    v = overfit.analyze_curves([0, 100, 200], [0, 0, 0], [1.0, 0.8, 0.6], patience=3)
    assert not v.overfitting and v.best_step == 200 and "train longer" in v.note


def test_analyze_curves_too_few_points():
    assert not overfit.analyze_curves([0], [1.0], [1.0]).overfitting


def test_read_metrics_jsonl_and_csv_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        j = Path(tmp) / "m.jsonl"
        j.write_text('{"step": 0, "train_loss": 1.0, "val_loss": 0.9}\n{"step": 100, "val_loss": 0.5}\n')
        steps, _, val = overfit.read_metrics(j)
        assert steps == [0, 100] and val == [0.9, 0.5]
        c = Path(tmp) / "m.csv"
        c.write_text("step,train_loss,val_loss\n0,1.0,0.9\n100,0.8,0.5\n")
        steps_c, _, val_c = overfit.read_metrics(c)
        assert steps_c == [0, 100] and val_c == [0.9, 0.5]


def test_read_metrics_empty_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "empty.jsonl"; p.write_text("")
        assert_raises(ValueError, lambda: overfit.read_metrics(p))


# -- integration with the trainers/CLI --------------------------------------

def test_noop_train_writes_readable_metrics_curve():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = get_trainer(_noop_cfg(Path(tmp)))
        trainer.prepare()
        save = trainer.train()
        steps, _, val = overfit.read_metrics(save / "metrics.jsonl")
        assert len(steps) == len(val) >= 5 and min(val) < val[-1]   # a real dip-then-rise curve


def test_mdm_eval_flags_wired_when_configured():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "prior.pt"; ckpt.write_bytes(b"x")
        repo = tmp / "repo"; repo.mkdir()
        cfg = _noop_cfg(tmp)
        cfg.method, cfg.repo, cfg.resume_checkpoint = "mdm", repo, ckpt
        cfg.save_every, cfg.eval_every = 1000, 1000
        trainer = get_trainer(cfg); trainer.prepare()
        calls = []
        trainer._run_cmd = lambda cmd, cwd=None: calls.append([str(c) for c in cmd])
        trainer.train()
        cmd = calls[0]
        assert cmd[cmd.index("--save_interval") + 1] == "1000"
        assert "--eval_during_training" in cmd and cmd[cmd.index("--eval_split") + 1] == "test"


def test_cli_overfit_check_and_report_with_noop():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _noop_cfg(tmp)   # writes tmp/dataset with an index
        common = ["--dataset-dir", str(tmp / "dataset"), "--work-root", str(tmp / "w"), "--method", "noop"]
        assert cli.main([*common, "overfit-check"]) == 0
        assert cli.main([*common, "train"]) == 0              # noop writes metrics.jsonl
        assert cli.main([*common, "overfit-report"]) == 0     # reads it back


def test_cli_overfit_report_missing_metrics_returns_1():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _noop_cfg(tmp)
        common = ["--dataset-dir", str(tmp / "dataset"), "--work-root", str(tmp / "w"), "--method", "noop"]
        assert cli.main([*common, "overfit-report"]) == 1     # no run yet -> no curve


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} overfit tests passed")


if __name__ == "__main__":
    _run_all()
