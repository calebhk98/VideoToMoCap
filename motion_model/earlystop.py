"""Actually stop training when it starts overfitting.

The upstream trainers run their full ``num_steps`` -- none of them self-stop on a
rising val loss. So real early stopping from an orchestration layer means: run the
trainer as a subprocess with frequent eval/checkpoints (``eval_every`` /
``save_every``), watch the val curve it writes, and **terminate the process** once
the curve has turned up for ``patience`` evals -- then keep the best checkpoint.

`run_with_monitor` is that loop, deliberately trainer-agnostic: it works for any
command that writes a metrics file `overfit.read_metrics` can parse (JSONL or CSV).
The clock and the process launcher are injectable so the whole thing is testable
GPU-free with a fake trainer. The upstream training loop is untouched -- we bracket
it, we don't patch it, matching the rest of Pipeline 2.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from . import overfit


@dataclass
class EarlyStopResult:
    """Outcome of a monitored run."""

    stopped: bool                        # True = we killed it at overfitting onset
    returncode: Optional[int]
    verdict: Optional[overfit.CurveVerdict]

    @property
    def best_step(self) -> Optional[int]:
        return self.verdict.best_step if self.verdict else None


def poll_metrics(metrics_path: Path, patience: int) -> Optional[overfit.CurveVerdict]:
    """Read the current curve and judge it, tolerating a mid-write file.

    While the trainer is appending, the last line can be partial -- that surfaces as
    a parse error, which just means "not enough yet", so we return None and retry on
    the next poll rather than crash the run.
    """
    p = Path(metrics_path)
    if not p.exists():
        return None
    try:
        steps, train, val = overfit.read_metrics(p)
    except (ValueError, OSError):   # empty, partial last line, or a transient read race
        return None
    return overfit.analyze_curves(steps, train, val, patience)


def run_with_monitor(
    cmd: List[str],
    *,
    metrics_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    patience: int = 5,
    poll_interval: float = 30.0,
    stop_grace: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    popen_fn: Callable[..., "subprocess.Popen"] = subprocess.Popen,
) -> EarlyStopResult:
    """Run ``cmd``; terminate it when its val curve confirms overfitting.

    Returns when the process exits on its own OR we stop it. ``sleep_fn`` / ``popen_fn``
    are injected for tests; in production they're the stdlib defaults.
    """
    proc = popen_fn([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)
    while proc.poll() is None:
        sleep_fn(poll_interval)
        verdict = poll_metrics(metrics_path, patience)
        if verdict and verdict.overfitting:
            _terminate(proc, stop_grace)
            return EarlyStopResult(stopped=True, returncode=proc.poll(), verdict=verdict)
    # Finished its full budget without overfitting -- report the final curve verdict.
    return EarlyStopResult(stopped=False, returncode=proc.returncode,
                           verdict=poll_metrics(metrics_path, patience))


def _terminate(proc, grace: float) -> None:
    """Ask the trainer to exit, then force it if it ignores the request."""
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
