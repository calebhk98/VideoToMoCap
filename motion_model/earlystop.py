"""Actually stop training when it starts overfitting.

The upstream trainers run their full ``num_steps`` -- none of them self-stop on a
rising val loss. So real early stopping from an orchestration layer means: run the
trainer as a subprocess with frequent eval/checkpoints (``eval_every`` /
``save_every``), watch the val curve it writes, and **terminate the process** once
the curve has turned up for ``patience`` evals -- then keep the best checkpoint.

`run_with_monitor` is that loop. It's decoupled from any log format: the caller
passes a ``read_verdict`` closure that returns the current :class:`CurveVerdict` (or
None if there isn't enough curve yet). Each trainer's log is normalized into that
closure by :mod:`motion_model.metrics`, so MoMask's progress file and MDM's
progress.csv and noop's metrics.jsonl all drive the same loop. The clock and the
process launcher are injectable so the whole thing is testable GPU-free with a fake
trainer. The upstream training loop is untouched -- we bracket it, we don't patch it.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from . import overfit

# A zero-arg reader that returns the current curve verdict, or None when there isn't
# enough of a val curve yet. motion_model.metrics builds these per trainer.
VerdictReader = Callable[[], Optional[overfit.CurveVerdict]]


@dataclass
class EarlyStopResult:
    """Outcome of a monitored run."""

    stopped: bool                        # True = we killed it at overfitting onset
    returncode: Optional[int]
    verdict: Optional[overfit.CurveVerdict]

    @property
    def best_step(self) -> Optional[int]:
        return self.verdict.best_step if self.verdict else None


def poll_metrics(metrics_path: Path, patience: int, parse=None) -> Optional[overfit.CurveVerdict]:
    """Read a normalized metrics file and judge it, tolerating a mid-write file.

    While the trainer is appending, the last line can be partial -- that surfaces as
    a parse error, which just means "not enough yet", so we return None and retry.
    ``parse`` defaults to :func:`overfit.read_metrics` (JSONL/CSV with step/train_loss/
    val_loss); a trainer with a different log passes its own via metrics.py.
    """
    parse = parse or overfit.read_metrics
    p = Path(metrics_path)
    if not p.exists():
        return None
    try:
        steps, train, val = parse(p)
    except (ValueError, OSError):   # empty, partial last line, or a transient read race
        return None
    return overfit.analyze_curves(steps, train, val, patience)


def file_curve_reader(metrics_path: Path, patience: int, parse=None) -> VerdictReader:
    """A :data:`VerdictReader` that re-reads a metrics file each call."""
    return lambda: poll_metrics(metrics_path, patience, parse=parse)


def run_with_monitor(
    cmd: List[str],
    *,
    read_verdict: VerdictReader,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    poll_interval: float = 30.0,
    stop_grace: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    popen_fn: Callable[..., "subprocess.Popen"] = subprocess.Popen,
) -> EarlyStopResult:
    """Run ``cmd``; terminate it when ``read_verdict()`` confirms overfitting.

    Returns when the process exits on its own OR we stop it. ``read_verdict`` /
    ``sleep_fn`` / ``popen_fn`` are injected; in production the reader comes from
    metrics.py and the others are the stdlib defaults.
    """
    proc = popen_fn([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)
    while proc.poll() is None:
        sleep_fn(poll_interval)
        verdict = read_verdict()
        if verdict and verdict.overfitting:
            _terminate(proc, stop_grace)
            return EarlyStopResult(stopped=True, returncode=proc.poll(), verdict=verdict)
    # Finished its full budget without overfitting -- report the final curve verdict.
    return EarlyStopResult(stopped=False, returncode=proc.returncode, verdict=read_verdict())


def _terminate(proc, grace: float) -> None:
    """Ask the trainer to exit, then force it if it ignores the request."""
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
