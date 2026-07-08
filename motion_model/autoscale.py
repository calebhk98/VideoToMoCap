"""Data-driven training regime: one config that adapts from 100 hours to 10M.

The whole point is that the *same* config works whether you point it at your own
footage or a server's corpus -- you don't hand-tune per dataset. `plan_training`
measures the prepared corpus (via `overfit.corpus_stats`, which streams the index)
and picks the regime the data can support:

  * how MUCH to train -- ``num_steps`` scaled to data volume (target a fixed number
    of times each frame is seen), clamped to a sane band. With early-stop on this is
    an upper bound; the val curve decides the actual finish.
  * HOW to train -- personalize (LoRA) on scarce single-person data, warm-start
    fine-tune in the middle, and only at large + diverse scale is training a full
    model from scratch reasonable.

Honest limit: the generator architectures (MoMask ~44M, MDM ~35M) are fixed-size --
you can't grow a transformer's width without discarding the pretrained prior. So
"scale the model" is regime selection (LoRA/full, warm-start/scratch, step budget,
recommended method), not a continuously resized net. The planner APPLIES the safe
continuous knobs (steps, and personalization where the method supports it) and
PRINTS the rest as a recommendation, because switching method or warm-starting needs
external assets (a repo, a checkpoint) this can't conjure. GPU-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from . import overfit

# Regime thresholds on training minutes (rules of thumb; tunable, documented).
_TINY_MAX_MIN = 30           # < 30 min: personalize, don't full-train a big net
_FINETUNE_MAX_MIN = 50 * 60  # up to ~50 h: warm-start fine-tune; beyond -> large-corpus
_DIVERSE_DONORS = 20         # this many donors makes from-scratch reasonable

# num_steps is chosen so each frame is seen ~this many times, then clamped. Early
# stop makes it a ceiling, so the band matters more than the exact target.
_EXPOSURES_TARGET = 40
_MIN_STEPS = 20_000
_MAX_STEPS = 500_000


@dataclass
class TrainingPlan:
    """The regime the corpus supports. `applied` are the knobs written onto the
    config; `recommended` need external assets, so they're printed not forced."""

    regime: str                       # 'personalize-tiny' | 'finetune' | 'large-corpus'
    num_steps: int
    personalization: str              # 'lora' | 'full'
    warm_start: bool                  # should resume_checkpoint be set?
    method: str                       # recommended generator for this scale
    conditioning: str                 # recommended conditioning (person when multi-donor)
    rationale: List[str] = field(default_factory=list)


def plan_training(cfg, index: dict) -> TrainingPlan:
    """Derive the training regime from the measured corpus. Pure/GPU-free."""
    stats = overfit.corpus_stats(index, getattr(cfg, "target_fps", 20))
    minutes, donors = stats.train_minutes, stats.n_persons
    frames = max(1, stats.train_frames)

    num_steps = _steps_for(frames, cfg.batch_size)
    regime = _regime(minutes, donors)
    method, personalization, warm_start = _regime_recipe(regime, donors)
    conditioning = "person" if donors > 1 else cfg.conditioning

    rationale = [
        f"corpus: {minutes:.1f} min train / {stats.n_clips} clips / {donors} donor(s) -> regime '{regime}'",
        f"num_steps={num_steps} (~{_EXPOSURES_TARGET} passes over {frames} frames, clamped to [{_MIN_STEPS}, {_MAX_STEPS}])",
    ]
    if warm_start:
        rationale.append("warm-start recommended: set resume_checkpoint to a pretrained prior")
    else:
        rationale.append(f"{donors} donors is diverse enough that training from scratch is reasonable (no warm-start required)")
    if donors > 1 and cfg.conditioning != "person":
        rationale.append(f"multi-donor: recommend conditioning: person (got {cfg.conditioning!r}) so styles don't average")
    return TrainingPlan(regime, num_steps, personalization, warm_start, method, conditioning, rationale)


def _steps_for(frames: int, batch_size: int) -> int:
    """Step budget that scales with data, clamped to a sane band."""
    raw = _EXPOSURES_TARGET * frames / max(1, batch_size)
    return int(min(_MAX_STEPS, max(_MIN_STEPS, raw)))


def _regime(minutes: float, donors: int) -> str:
    if minutes < _TINY_MAX_MIN and donors <= 2:
        return "personalize-tiny"
    if minutes < _FINETUNE_MAX_MIN:
        return "finetune"
    return "large-corpus"


def _regime_recipe(regime: str, donors: int):
    """(method, personalization, warm_start) for a regime."""
    if regime == "personalize-tiny":
        return "mdm", "lora", True          # keep the prior's vocabulary; adapt only style
    if regime == "finetune":
        return "momask", "full", True       # the default high-fidelity fine-tune
    # large-corpus: from scratch only makes sense once the corpus is genuinely diverse
    return "momask", "full", donors < _DIVERSE_DONORS


def apply_plan(cfg, plan: TrainingPlan) -> List[str]:
    """Write the SAFE knobs onto the config; return what was applied vs left to you.

    Applies num_steps always, and personalization only when the current method
    supports it (LoRA is mdm-only). Method switch + warm-start are recommendations,
    not silent changes, because they need a repo/checkpoint we can't verify here.
    """
    applied = [f"num_steps -> {plan.num_steps}"]
    cfg.num_steps = plan.num_steps
    if cfg.method.lower() == "mdm" and plan.personalization != cfg.personalization:
        cfg.personalization = plan.personalization
        applied.append(f"personalization -> {plan.personalization}")
    return applied


def format_plan(plan: TrainingPlan, applied: List[str]) -> str:
    """Render the plan + what auto-scale changed for the CLI."""
    lines = [f"Auto-scaled training plan: regime '{plan.regime}'"]
    for r in plan.rationale:
        lines.append(f"  - {r}")
    lines.append(f"  recommended: method={plan.method}, personalization={plan.personalization}, "
                 f"conditioning={plan.conditioning}, warm_start={plan.warm_start}")
    lines.append(f"  applied: {', '.join(applied)}")
    return "\n".join(lines)
