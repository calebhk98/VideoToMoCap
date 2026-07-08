"""Tests for Pipeline 2 adaptivity: data-driven regime (autoscale) + real early stop.

Together these are the "one config, any scale" behaviour: autoscale sizes the run to
the corpus, early-stop terminates the trainer at the overfitting onset. Both GPU-free
-- early-stop is exercised against a *fake* trainer subprocess (a stub proc + an
injected clock that writes the val curve), so no torch/CUDA. Runs under pytest OR
directly: `python tests/test_autoscale_earlystop.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model import autoscale, earlystop
from motion_model.config import MotionModelConfig


def _index(clips: list) -> dict:
    return {"clips": clips}


def _person_clips(n_persons: int, clips_each: int, frames: int, fps: float = 20.0) -> list:
    out = []
    for p in range(n_persons):
        for c in range(clips_each):
            out.append({"clip_id": f"p{p:02d}c{c:02d}", "person_id": f"person_{p:02d}",
                        "n_frames": frames, "fps": fps, "split": "train"})
    return out


# -- autoscale: data -> regime ----------------------------------------------

def test_tiny_single_person_gets_personalize_regime():
    cfg = MotionModelConfig(method="mdm", batch_size=64)
    plan = autoscale.plan_training(cfg, _index(_person_clips(1, 3, 1200)))   # ~3 min
    assert plan.regime == "personalize-tiny"
    assert plan.personalization == "lora" and plan.warm_start


def test_large_diverse_corpus_allows_from_scratch():
    cfg = MotionModelConfig(method="momask", batch_size=64)
    # 30 donors x 40 clips x 30 min each -> well past the finetune band, and diverse.
    plan = autoscale.plan_training(cfg, _index(_person_clips(30, 40, 36000)))
    assert plan.regime == "large-corpus" and plan.warm_start is False


def test_num_steps_scales_with_data_and_clamps():
    cfg = MotionModelConfig(method="momask", batch_size=64)
    tiny = autoscale.plan_training(cfg, _index(_person_clips(1, 1, 200)))
    huge = autoscale.plan_training(cfg, _index(_person_clips(50, 50, 36000)))
    assert tiny.num_steps == autoscale._MIN_STEPS        # floored
    assert huge.num_steps == autoscale._MAX_STEPS        # capped -> early-stop governs
    assert autoscale._MIN_STEPS < tiny.num_steps + 1     # sanity: within band


def test_apply_plan_sets_num_steps_and_mdm_lora():
    cfg = MotionModelConfig(method="mdm", batch_size=64, num_steps=80000, personalization="full")
    plan = autoscale.plan_training(cfg, _index(_person_clips(1, 2, 1200)))
    applied = autoscale.apply_plan(cfg, plan)
    assert cfg.num_steps == plan.num_steps
    assert cfg.personalization == "lora"                 # applied because method is mdm
    assert any("num_steps" in a for a in applied)


def test_apply_plan_leaves_personalization_when_method_not_mdm():
    cfg = MotionModelConfig(method="momask", batch_size=64, personalization="full")
    plan = autoscale.plan_training(cfg, _index(_person_clips(1, 2, 600)))
    autoscale.apply_plan(cfg, plan)
    assert cfg.personalization == "full"                 # momask has no LoRA path -> untouched


def test_multi_donor_recommends_person_conditioning():
    cfg = MotionModelConfig(method="momask", conditioning="none")
    plan = autoscale.plan_training(cfg, _index(_person_clips(5, 4, 2400)))
    assert plan.conditioning == "person"
    assert any("conditioning: person" in r for r in plan.rationale)


# -- early stop: monitor a fake trainer and kill it at the onset -------------

class _FakeProc:
    """A stand-in Popen: stays alive for `alive` polls, records termination."""

    def __init__(self, alive: int):
        self._alive = alive
        self.returncode = None
        self.terminated = self.killed = False

    def poll(self):
        if self._alive > 0:
            self._alive -= 1
            return None
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = 0
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def _curve_writer(path: Path, series: list):
    """An injected sleep_fn that appends one metrics row per call (simulates training
    writing its val curve over time)."""
    state = {"i": 0}

    def sleep_fn(_seconds):
        i = state["i"]
        if i >= len(series):
            return
        with open(path, "a") as fh:
            fh.write(json.dumps({"step": i * 100, "train_loss": 1.0 - 0.05 * i, "val_loss": series[i]}) + "\n")
        state["i"] += 1

    return sleep_fn


def test_poll_metrics_missing_and_partial_are_tolerated():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.jsonl"
        assert earlystop.poll_metrics(p, patience=3) is None      # missing
        p.write_text('{"step": 0, "val_loss": 0.9}\n{"step": 100, "val_los')  # partial last line
        assert earlystop.poll_metrics(p, patience=3) is None      # tolerated, not a crash


def test_run_with_monitor_stops_at_overfitting_onset():
    with tempfile.TemporaryDirectory() as tmp:
        metrics = Path(tmp) / "metrics.jsonl"
        series = [1.0, 0.7, 0.5, 0.45, 0.5, 0.6, 0.72]           # bottoms at step 300, rises 3
        proc = _FakeProc(alive=50)
        result = earlystop.run_with_monitor(
            ["fake-train"], metrics_path=metrics, patience=3, poll_interval=0.0,
            sleep_fn=_curve_writer(metrics, series), popen_fn=lambda *a, **k: proc)
        assert result.stopped and proc.terminated
        assert result.best_step == 300                            # keep the pre-overfit checkpoint


def test_run_with_monitor_lets_a_clean_run_finish():
    with tempfile.TemporaryDirectory() as tmp:
        metrics = Path(tmp) / "metrics.jsonl"
        series = [1.0, 0.8, 0.6, 0.5]                             # monotonically improving
        proc = _FakeProc(alive=4)                                 # dies on its own after 4 polls
        result = earlystop.run_with_monitor(
            ["fake-train"], metrics_path=metrics, patience=3, poll_interval=0.0,
            sleep_fn=_curve_writer(metrics, series), popen_fn=lambda *a, **k: proc)
        assert not result.stopped and not proc.terminated
        assert result.verdict is not None and not result.verdict.overfitting


class _StubbornProc:
    """Ignores terminate(); wait() times out until killed -> forces the kill path."""

    def __init__(self):
        self.returncode = None
        self.killed = False

    def terminate(self):
        pass

    def wait(self, timeout=None):
        if timeout is not None and not self.killed:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return -9

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_terminate_escalates_to_kill():
    proc = _StubbornProc()
    earlystop._terminate(proc, grace=0.001)
    assert proc.killed


# -- integration: the trainer seam routes through early-stop when enabled ----

def test_trainer_run_train_routes_to_monitor_when_early_stop():
    from motion_model.trainers import get_trainer

    with tempfile.TemporaryDirectory() as tmp:
        cfg = MotionModelConfig(dataset_dir=Path(tmp), work_root=Path(tmp) / "w",
                                method="noop", early_stop=True)
        trainer = get_trainer(cfg)
        seen = {}
        # Patch the monitor so we assert routing without launching a real process.
        import motion_model.earlystop as es
        orig = es.run_with_monitor

        def fake_monitor(cmd, **kw):
            seen["kw"] = kw
            return es.EarlyStopResult(stopped=False, returncode=0, verdict=None)

        es.run_with_monitor = fake_monitor
        try:
            trainer._run_train(["some", "cmd"], Path(tmp), cfg.metrics_path)
        finally:
            es.run_with_monitor = orig
        assert seen["kw"]["metrics_path"] == cfg.metrics_path
        assert seen["kw"]["patience"] == cfg.early_stop_patience


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} autoscale/early-stop tests passed")


if __name__ == "__main__":
    _run_all()
