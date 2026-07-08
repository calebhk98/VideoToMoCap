"""Overfitting guard for Pipeline 2 -- GPU-free, numpy+stdlib only.

Training runs inside the upstream trainer's own loop (a subprocess), so this
package can't watch the loss live. What it CAN do, entirely offline, is bracket
that loop on both sides:

  * :func:`assess_risk` -- BEFORE you spend a GPU. From how much motion you have
    vs how hard you're about to train on it, estimate how likely the run is to
    overfit and say what to change (warm-start, LoRA, fewer steps, more data).
    **Multi-person aware:** it breaks the corpus down per donor, because a
    corpus of 100 people overfits nothing like hours of one person, and a single
    under-represented donor washes out (or is memorized) differently from the
    whole. This is the piece that scales as the dataset grows.

  * :func:`analyze_curves` / :func:`read_metrics` -- AFTER a run. Read the
    train/val curve the trainer logged and report the best (pre-overfit)
    checkpoint and whether validation turned back up.

The knobs that make a run *analyzable* (evaluate on the held-out split + save
checkpoints often enough to fall back to a good one) are surfaced as
``config.save_every`` / ``config.eval_every`` and wired into the trainers.

Everything here is a documented rule of thumb, not a guarantee -- overfitting is
ultimately confirmed by the val curve, which is why both halves exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Approx parameter count (millions) of each generator. Overfitting is a
# generator-fine-tune concern; physics controllers (protomotions) track motion
# rather than model a data distribution, and noop is synthetic -- both N/A.
GENERATOR_SIZES_M = {"momask": 44, "mdm": 35, "closd": 35}

# Rules of thumb (frames are at target_fps; "exposures" = times each train frame
# is fed to the optimizer = num_steps * batch_size / train_frames).
_HIGH_EXPOSURES = 150
_MED_EXPOSURES = 60
_LOW_MINUTES = 15          # below this, one style from scratch is fragile
_MED_MINUTES = 60
_UNDERREPRESENTED_MINUTES = 3.0   # a donor with less than this barely moves the model
_DIVERSE_PERSONS = 20      # a corpus this broad makes overfitting the *corpus* unlikely
_SOME_PERSONS = 5


@dataclass
class PersonCoverage:
    """How much motion one donor contributes -- the unit that scales with donors."""

    person_id: str
    n_clips: int
    frames: int
    minutes: float


@dataclass
class CorpusStats:
    """Size of the training corpus, in total and per donor (all GPU-free to compute)."""

    n_clips: int
    frames: int
    minutes: float
    train_frames: int
    train_minutes: float
    persons: List[PersonCoverage]  # sorted by minutes ascending (thinnest donor first)

    @property
    def n_persons(self) -> int:
        return len(self.persons)


@dataclass
class RiskReport:
    """Pre-flight overfitting assessment for a configured run."""

    applicable: bool
    method: str
    level: str = "n/a"                 # 'low' | 'medium' | 'high' | 'n/a'
    exposures: float = 0.0
    stats: Optional[CorpusStats] = None
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    underrepresented: List[PersonCoverage] = field(default_factory=list)


@dataclass
class CurveVerdict:
    """Post-train read of the train/val curve."""

    best_step: Optional[int]
    best_val: Optional[float]
    overfitting: bool
    onset_step: Optional[int]
    note: str


def _clip_frames_fps(clip: dict, default_fps: float) -> Tuple[int, float]:
    """Frames and fps for one clip, tolerating a missing fps (use the target)."""
    frames = int(clip.get("n_frames") or 0)
    fps = float(clip.get("fps") or default_fps) or default_fps
    return frames, fps


def _is_train(clip: dict) -> bool:
    """A clip counts toward training unless it's explicitly held out for val/test."""
    return clip.get("split") not in ("val", "test")


def corpus_stats(index: dict, default_fps: float = 20.0) -> CorpusStats:
    """Summarize Pipeline 1's clip index by total and per-donor motion volume.

    ``person_id`` is Pipeline 1's multi-person label (absent for a single-person
    corpus -> everything is attributed to one unlabeled donor). Minutes are frames
    over fps, which is all the overfitting heuristics need.
    """
    clips = index.get("clips", [])
    per_person: Dict[str, List[int]] = {}
    frames = train_frames = 0
    minutes = train_minutes = 0.0
    for clip in clips:
        f, fps = _clip_frames_fps(clip, default_fps)
        m = f / fps / 60.0 if fps else 0.0
        frames += f
        minutes += m
        if _is_train(clip):
            train_frames += f
            train_minutes += m
        pid = str(clip.get("person_id") or "(unlabeled)")
        bucket = per_person.setdefault(pid, [0, 0])
        bucket[0] += 1
        bucket[1] += f

    persons = [
        PersonCoverage(pid, n, fr, fr / default_fps / 60.0)
        for pid, (n, fr) in per_person.items()
    ]
    persons.sort(key=lambda p: p.minutes)
    return CorpusStats(len(clips), frames, minutes, train_frames, train_minutes, persons)


def _score_reasons(cfg, stats: CorpusStats, exposures: float) -> Tuple[int, List[str]]:
    """Turn the signals into a small integer score + the human reasons behind it."""
    reasons: List[str] = []
    score = 0

    if exposures > _HIGH_EXPOSURES:
        score += 2
        reasons.append(f"each training frame is seen ~{exposures:.0f}x (num_steps*batch/frames) -- high memorization pressure")
    elif exposures > _MED_EXPOSURES:
        score += 1
        reasons.append(f"each training frame is seen ~{exposures:.0f}x")

    if stats.train_minutes < _LOW_MINUTES:
        score += 2
        reasons.append(f"only {stats.train_minutes:.1f} min of training motion")
    elif stats.train_minutes < _MED_MINUTES:
        score += 1
        reasons.append(f"{stats.train_minutes:.1f} min of training motion is on the thin side")

    if cfg.resume_checkpoint is None:
        score += 1
        reasons.append("no resume_checkpoint -- training from scratch, not warm-starting a pretrained prior")

    if cfg.personalization == "lora":
        score -= 1
        reasons.append("LoRA personalization shifts only a low-rank slice, which resists overfitting")

    # A broad, many-donor corpus is the opposite regime: diversity, not scarcity.
    if stats.n_persons >= _DIVERSE_PERSONS:
        score -= 2
        reasons.append(f"{stats.n_persons} donors -- a diverse corpus makes overfitting the *corpus* unlikely")
    elif stats.n_persons >= _SOME_PERSONS:
        score -= 1
        reasons.append(f"{stats.n_persons} donors adds cross-person variety")

    return score, reasons


def _recommendations(cfg, stats: CorpusStats) -> List[str]:
    """Concrete, config-level changes that lower the risk for this exact run."""
    recs: List[str] = []
    big = GENERATOR_SIZES_M.get(cfg.method.lower(), 0) >= 35

    if cfg.resume_checkpoint is None and stats.n_persons < _SOME_PERSONS and big:
        recs.append("set resume_checkpoint to warm-start from a pretrained prior (don't train a ~35M+ model from scratch on this little data)")
    if cfg.personalization != "lora" and stats.train_minutes < _LOW_MINUTES and cfg.method.lower() == "mdm":
        recs.append("try personalization: lora -- it adapts style without washing out the prior on scarce data")
    if stats.train_minutes < _MED_MINUTES:
        recs.append(f"consider fewer num_steps (current {cfg.num_steps}) or more footage; watch the val curve with save_every/eval_every")
    if stats.n_persons > 1 and cfg.conditioning != "person":
        recs.append(f"{stats.n_persons} donors but conditioning={cfg.conditioning!r}: set conditioning: person (or train per-person sub-datasets) so styles don't average into one")
    return recs


def assess_risk(cfg, index: dict) -> RiskReport:
    """Estimate overfitting risk for the configured run, before any GPU time."""
    method = cfg.method.lower()
    stats = corpus_stats(index, getattr(cfg, "target_fps", 20))
    if method not in GENERATOR_SIZES_M:
        return RiskReport(applicable=False, method=method, stats=stats,
                          reasons=[f"method {method!r} is not a generator fine-tune -- overfitting risk is not modeled here"])

    exposures = cfg.num_steps * cfg.batch_size / max(1, stats.train_frames)
    score, reasons = _score_reasons(cfg, stats, exposures)
    level = "high" if score >= 3 else "medium" if score >= 1 else "low"
    under = [p for p in stats.persons if stats.n_persons > 1 and p.minutes < _UNDERREPRESENTED_MINUTES]
    return RiskReport(
        applicable=True, method=method, level=level, exposures=exposures, stats=stats,
        reasons=reasons, recommendations=_recommendations(cfg, stats), underrepresented=under,
    )


def format_risk(report: RiskReport) -> str:
    """Render a RiskReport as a readable block for the CLI."""
    s = report.stats
    lines = [f"Overfitting risk ({report.method}): {report.level.upper()}"]
    if s is not None:
        lines.append(f"  corpus: {s.minutes:.1f} min / {s.n_clips} clips / {s.n_persons} donor(s); train {s.train_minutes:.1f} min")
    if report.applicable:
        lines.append(f"  exposures: each training frame seen ~{report.exposures:.0f}x")
    for r in report.reasons:
        lines.append(f"  - {r}")
    if report.underrepresented:
        thin = ", ".join(f"{p.person_id} ({p.minutes:.1f} min)" for p in report.underrepresented)
        lines.append(f"  under-represented donors (< {_UNDERREPRESENTED_MINUTES:.0f} min): {thin}")
    for rec in report.recommendations:
        lines.append(f"  -> {rec}")
    return "\n".join(lines)


def read_metrics(path: Path) -> Tuple[List[int], List[float], List[float]]:
    """Read a training curve as (steps, train_loss, val_loss) from JSONL or CSV.

    JSONL: one object per line with ``step``/``train_loss``/``val_loss`` (missing
    val is skipped). CSV: a header row naming those same columns. Best-effort and
    tolerant of extra keys; raises with context if the file has neither shape.
    """
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"metrics file {path} is empty")
    rows = _parse_jsonl(text) if text[0] in "{[" else _parse_csv(text)
    steps, train, val = [], [], []
    for row in rows:
        if "val_loss" not in row or row.get("val_loss") in (None, ""):
            continue
        steps.append(int(float(row["step"])))
        train.append(float(row.get("train_loss", "nan")))
        val.append(float(row["val_loss"]))
    if not steps:
        raise ValueError(f"metrics file {path} has no rows with a val_loss column")
    return steps, train, val


def _parse_jsonl(text: str) -> List[dict]:
    """One JSON object per non-blank line."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parse_csv(text: str) -> List[dict]:
    """Header row of column names, then comma-separated values."""
    import csv
    import io

    return list(csv.DictReader(io.StringIO(text)))


def analyze_curves(steps: Sequence[int], train: Sequence[float], val: Sequence[float],
                   patience: int = 5) -> CurveVerdict:
    """Find the best-val checkpoint and whether val turned up (overfitting onset).

    ``patience`` is how many successive worse-than-best evals after the minimum
    count as a real upturn rather than noise. With too few evals we say so rather
    than guess.
    """
    if len(val) < 2:
        return CurveVerdict(None, None, False, None,
                            "too few evals to judge -- lower eval_every / train longer")

    best_idx = min(range(len(val)), key=lambda i: val[i])
    best_step, best_val = steps[best_idx], val[best_idx]
    worse_after = sum(1 for v in val[best_idx + 1:] if v > best_val)

    if best_idx == len(val) - 1:
        return CurveVerdict(best_step, best_val, False, None,
                            "val still improving at the last eval -- you could train longer")
    if worse_after >= patience:
        return CurveVerdict(best_step, best_val, True, best_step,
                            f"val bottomed at step {best_step} then rose for {worse_after} evals -- overfitting; keep the step-{best_step} checkpoint")
    return CurveVerdict(best_step, best_val, False, None,
                        f"best val at step {best_step}; no sustained upturn yet (patience {patience})")


def format_curves(v: CurveVerdict) -> str:
    """Render a CurveVerdict for the CLI."""
    head = "OVERFITTING" if v.overfitting else "ok"
    best = f"best val {v.best_val:.4f} @ step {v.best_step}" if v.best_step is not None else "no usable minimum"
    return f"Train/val curve: {head}\n  {best}\n  {v.note}"
