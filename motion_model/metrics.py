"""Per-trainer metrics normalization: give early-stop / overfit-report a val curve
no matter which upstream trainer produced it.

Both the monitor and the report need one thing -- ``(step, val_loss)`` over time --
but each trainer emits it differently:

  * ``noop``      -> our own ``metrics.jsonl`` (already normalized)
  * ``mdm``/``closd`` -> an OpenAI-baselines-style ``progress.csv`` in ``save_dir``
                    (MDM's logger + its forks write one row per log interval)
  * ``momask``    -> its run log (format drifts across revisions; best-effort)

This module maps each native format to the normalized ``(steps, train, val)`` triple.
The mappings -- which file, and which column is the held-out (lower-is-better) signal
-- are ISOLATED here and documented as *verify against your checkout revision*,
exactly like the backend demo commands: upstream logging drifts, and this is the one
place to fix it when it does. If a trainer's val metric isn't found, the readers
return ``None`` and early-stop simply doesn't fire (with a note) rather than guessing.

GPU-free: pure stdlib parsing of text the trainer writes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import overfit
from .earlystop import VerdictReader

Curve = Tuple[List[int], List[float], List[float]]   # (steps, train_loss, val_loss)

# MDM-family progress.csv columns. Preference order; first present wins. Val columns
# are all LOWER-IS-BETTER so the "minimum then rises" overfitting test is valid --
# do NOT add R-precision/accuracy here (higher-is-better would invert the signal).
_MDM_STEP_COLS = ("step", "Step", "samples")
_MDM_TRAIN_COLS = ("loss", "train_loss", "loss_q0")
_MDM_VAL_COLS = ("eval_loss", "val_loss", "Eval/loss", "eval_fid", "fid", "eval_matching_score")


@dataclass
class MetricsSource:
    """Where a trainer writes its curve and how to normalize it."""

    path: Path
    parse: Callable[[Path], Curve]
    note: str = ""


def resolve_source(cfg) -> Optional[MetricsSource]:
    """The metrics source for the configured method, or None if we don't know one."""
    method = cfg.method.lower()
    if method == "noop":
        return MetricsSource(cfg.metrics_path, overfit.read_metrics, "noop metrics.jsonl")
    if method in ("mdm", "closd"):
        return MetricsSource(cfg.checkpoint_dir / "progress.csv", parse_progress_csv,
                             "MDM-family logger progress.csv (verify column names vs your checkout)")
    # MoMask's log path/format varies by revision; don't guess a wrong parser.
    return None


def parse_progress_csv(path: Path) -> Curve:
    """Parse an OpenAI-baselines-style progress.csv into the normalized triple.

    Only rows that carry the held-out column contribute (eval runs less often than
    the per-step train log, so most rows have no val value). Raises if the file has
    no recognizable val column -- the caller treats that as "no curve yet".
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows yet")
    step_col = _first_present(rows[0], _MDM_STEP_COLS)
    val_col = _first_present(rows[0], _MDM_VAL_COLS)
    train_col = _first_present(rows[0], _MDM_TRAIN_COLS)
    if val_col is None or step_col is None:
        raise ValueError(
            f"{path} has no held-out column (looked for {_MDM_VAL_COLS}); "
            f"enable eval logging or set the column in metrics.py")
    return _collect_rows(rows, step_col, train_col, val_col)


def _collect_rows(rows: List[dict], step_col: str, train_col: Optional[str], val_col: str) -> Curve:
    steps: List[int] = []
    train: List[float] = []
    val: List[float] = []
    for row in rows:
        raw = row.get(val_col)
        if raw in (None, ""):
            continue
        steps.append(int(float(row[step_col])))
        val.append(float(raw))
        train.append(float(row.get(train_col) or "nan") if train_col else float("nan"))
    if not steps:
        raise ValueError("no rows with a held-out value yet")
    return steps, train, val


def _first_present(row: dict, candidates) -> Optional[str]:
    """The first candidate column actually present in the CSV header."""
    return next((c for c in candidates if c in row), None)


def read_curve(cfg) -> Optional[Curve]:
    """One-shot normalized curve for `overfit-report`, or None if unavailable."""
    source = resolve_source(cfg)
    if source is None or not Path(source.path).exists():
        return None
    try:
        return source.parse(source.path)
    except (ValueError, OSError):
        return None


def curve_reader(cfg, patience: int) -> VerdictReader:
    """A VerdictReader for early-stop: resolve this method's source, judge it live."""
    source = resolve_source(cfg)
    if source is None:
        return lambda: None
    from .earlystop import poll_metrics
    return lambda: poll_metrics(source.path, patience, parse=source.parse)


def describe_source(cfg) -> str:
    """Human note for the train preflight: will a val curve actually be available?"""
    source = resolve_source(cfg)
    if source is None:
        return (f"early_stop: no metrics adapter for method {cfg.method!r} -- it can't trigger. "
                f"Add one in metrics.py, or run overfit-report with --metrics pointing at its log.")
    return f"early_stop: watching {source.path} ({source.note})"
