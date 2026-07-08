"""Tests for the per-trainer metrics adapter (motion_model.metrics).

Proves early-stop / overfit-report can find a val curve regardless of which trainer
wrote it: noop's normalized JSONL and the MDM-family progress.csv both normalize to
the same (steps, train, val) triple, and an unknown trainer degrades gracefully to
"no curve" rather than a wrong guess. GPU-free -- parsers run on representative log
text, no trainer is launched. Runs under pytest OR: `python tests/test_metrics.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model import metrics, overfit
from motion_model.config import MotionModelConfig

# A baselines-style progress.csv with a clear overfitting shape in eval_loss
# (bottoms at step 300, then rises), plus a row whose eval is blank (train-only log).
_PROGRESS_CSV = """step,loss,grad_norm,eval_loss
0,1.00,0.5,1.00
100,0.80,0.4,0.70
200,0.70,0.4,0.50
250,0.68,0.4,
300,0.65,0.3,0.45
400,0.60,0.3,0.50
500,0.55,0.3,0.60
600,0.50,0.2,0.72
"""


def _mdm_cfg(tmp: Path, method: str = "mdm") -> MotionModelConfig:
    cfg = MotionModelConfig(work_root=tmp / "w", method=method)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_progress(cfg: MotionModelConfig, text: str = _PROGRESS_CSV) -> Path:
    p = cfg.checkpoint_dir / "progress.csv"
    p.write_text(text)
    return p


def test_parse_progress_csv_normalizes_and_skips_blank_val():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "progress.csv"
        p.write_text(_PROGRESS_CSV)
        steps, train, val = metrics.parse_progress_csv(p)
        assert steps == [0, 100, 200, 300, 400, 500, 600]   # the blank-eval row (250) dropped
        assert val[3] == 0.45 and train[0] == 1.0


def test_parse_progress_csv_without_val_column_raises():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "progress.csv"
        p.write_text("step,loss\n0,1.0\n100,0.8\n")
        try:
            metrics.parse_progress_csv(p)
            raise AssertionError("expected ValueError for missing val column")
        except ValueError as exc:
            assert "held-out column" in str(exc)


def test_resolve_source_per_method():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        assert metrics.resolve_source(_mdm_cfg(t, "mdm")).path.name == "progress.csv"
        assert metrics.resolve_source(_mdm_cfg(t, "closd")).path.name == "progress.csv"
        assert metrics.resolve_source(_mdm_cfg(t, "noop")).path.name == "metrics.jsonl"
        assert metrics.resolve_source(_mdm_cfg(t, "momask")) is None   # no guessed parser


def test_read_curve_missing_and_present():
    with tempfile.TemporaryDirectory() as t:
        cfg = _mdm_cfg(Path(t))
        assert metrics.read_curve(cfg) is None       # nothing logged yet
        _write_progress(cfg)
        curve = metrics.read_curve(cfg)
        assert curve is not None and curve[0][0] == 0


def test_curve_reader_detects_overfitting_live():
    with tempfile.TemporaryDirectory() as t:
        cfg = _mdm_cfg(Path(t))
        _write_progress(cfg)
        reader = metrics.curve_reader(cfg, patience=3)
        verdict = reader()
        assert verdict is not None and verdict.overfitting and verdict.best_step == 300


def test_curve_reader_none_when_no_adapter():
    with tempfile.TemporaryDirectory() as t:
        reader = metrics.curve_reader(_mdm_cfg(Path(t), "momask"), patience=3)
        assert reader() is None                       # unknown trainer -> never fires


def test_describe_source_present_vs_absent():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        assert "progress.csv" in metrics.describe_source(_mdm_cfg(t, "mdm"))
        assert "can't trigger" in metrics.describe_source(_mdm_cfg(t, "momask"))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} metrics tests passed")


if __name__ == "__main__":
    _run_all()
