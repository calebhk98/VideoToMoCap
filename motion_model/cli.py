"""Command-line interface: ``python -m motion_model <command>``.

Config-first, exactly like Pipeline 1: put every knob (method, repo paths, steps,
simulator, ...) in a YAML and the commands read it; flags are optional overrides.
The config is auto-discovered (``--config`` > ``$MOTIONMODEL_CONFIG`` >
./motion_model.yaml > ./configs/motion_model.yaml), so the common case is:

    python -m motion_model train        # prepare + train the configured method

Individual steps, each reading the same config:
    python -m motion_model methods      # list selectable methods
    python -m motion_model info         # what the configured method will do
    python -m motion_model prepare      # AMASS dataset -> this method's training data
    python -m motion_model train        # prepare (if needed) then launch training
    python -m motion_model autoscale       # data-driven regime + num_steps for the corpus
    python -m motion_model overfit-check   # pre-flight overfitting risk (per-donor, no GPU)
    python -m motion_model overfit-report  # post-train: best pre-overfit checkpoint

One config, any scale: set ``auto_scale: true`` to adapt num_steps/regime to the
corpus and ``early_stop: true`` (with ``eval_every``) to stop at the overfitting
onset -- the same YAML then works for 100 hours or a server's corpus.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from . import data
from .config import MotionModelConfig, load_config
from .trainers import available_methods, get_trainer

_DEFAULT_CONFIG_NAMES = ("motion_model.yaml", "motion_model.yml", "configs/motion_model.yaml")


def _discover_config(explicit) -> Optional[str]:
    """Resolve which config to load: --config, else $MOTIONMODEL_CONFIG, else a
    conventional file in the working dir. Returns None -> built-in defaults."""
    if explicit:
        return explicit
    env = os.environ.get("MOTIONMODEL_CONFIG")
    if env:
        return env
    for name in _DEFAULT_CONFIG_NAMES:
        if Path(name).exists():
            return name
    return None


def _cfg(args) -> MotionModelConfig:
    path = _discover_config(args.config)
    if path:
        print(f"Using config: {path}")
    cfg = load_config(path) if path else MotionModelConfig()
    if getattr(args, "method", None):
        cfg.method = args.method
    if getattr(args, "dataset_dir", None):
        cfg.dataset_dir = Path(args.dataset_dir)
    if getattr(args, "work_root", None):
        cfg.work_root = Path(args.work_root)
    return cfg


def cmd_config_template(args) -> int:
    """Print a fully-commented YAML with every config option (redirect to a file)."""
    from .config import config_template

    print(config_template(), end="")
    return 0


def cmd_methods(args) -> int:
    """List the selectable methods and their roles."""
    for name in available_methods():
        trainer = get_trainer(MotionModelConfig(method=name))
        print(f"  {name:14s} {trainer.role}")
    return 0


def cmd_info(args) -> int:
    """Show the resolved config and what the configured method will do."""
    cfg = _cfg(args)
    trainer = get_trainer(cfg)
    print(f"method:      {cfg.method}  ({trainer.describe()})")
    print(f"dataset:     {cfg.dataset_dir}")
    print(f"prepared:    {cfg.prepared_dir}")
    print(f"checkpoints: {cfg.checkpoint_dir}")
    print(f"repo:        {cfg.repo}")
    if trainer.feature_format == "humanml3d_263":
        warn = data.fps_warning(cfg.target_fps)
        if warn:
            print(f"WARNING: {warn}")
    return 0


def cmd_prepare(args) -> int:
    """Convert the AMASS dataset into the configured method's training data."""
    cfg = _cfg(args)
    trainer = get_trainer(cfg)
    print(f"Preparing data for method={cfg.method} ...")
    out = trainer.prepare()
    print(f"Prepared: {out}")
    return 0


def cmd_train(args) -> int:
    """Prepare (unless --skip-prepare) then launch the configured trainer."""
    cfg = _cfg(args)
    trainer = get_trainer(cfg)
    if not args.skip_prepare:
        print(f"Preparing data for method={cfg.method} ...")
        trainer.prepare()
    if cfg.auto_scale:
        _apply_auto_scale(cfg)     # adapt num_steps/regime to the corpus (same config, any scale)
    _print_overfit_risk(cfg)       # surface the risk BEFORE spending GPU time
    if cfg.early_stop and cfg.eval_every <= 0:
        print("  NOTE: early_stop is on but eval_every=0 -- no val curve will be logged, so it "
              "can't trigger. Set eval_every>0.")
    print(f"Training method={cfg.method} ...")
    save = trainer.train()
    print(f"Checkpoints: {save}")
    return 0


def _apply_auto_scale(cfg) -> None:
    """Apply the data-driven training plan to the config (best-effort; needs an index)."""
    from . import autoscale

    try:
        index = data.load_index(cfg.dataset_dir)
    except FileNotFoundError:
        print("  auto_scale: no dataset index yet -- skipping (run prepare/build first)")
        return
    plan = autoscale.plan_training(cfg, index)
    applied = autoscale.apply_plan(cfg, plan)
    print(autoscale.format_plan(plan, applied))


def cmd_autoscale(args) -> int:
    """Show the data-driven training plan for the corpus (no training)."""
    from . import autoscale

    cfg = _cfg(args)
    index = data.load_index(cfg.dataset_dir)
    plan = autoscale.plan_training(cfg, index)
    print(autoscale.format_plan(plan, ["(preview -- run `train` with auto_scale to apply)"]))
    return 0


def _print_overfit_risk(cfg) -> None:
    """Best-effort pre-flight risk block; never blocks training if the index is odd."""
    from . import overfit

    try:
        index = data.load_index(cfg.dataset_dir)
    except FileNotFoundError:
        return
    print(overfit.format_risk(overfit.assess_risk(cfg, index)))


def cmd_overfit_check(args) -> int:
    """Pre-flight: estimate overfitting risk for the configured run (no GPU/training)."""
    from . import overfit

    cfg = _cfg(args)
    index = data.load_index(cfg.dataset_dir)
    print(overfit.format_risk(overfit.assess_risk(cfg, index)))
    return 0


def cmd_overfit_report(args) -> int:
    """Post-train: read the train/val curve and report the best pre-overfit checkpoint."""
    from . import overfit

    cfg = _cfg(args)
    metrics = Path(args.metrics) if args.metrics else cfg.checkpoint_dir / "metrics.jsonl"
    if not metrics.exists():
        print(f"No metrics file at {metrics}. Set eval_every so the trainer logs a val "
              f"curve, or pass --metrics <path> to point at the trainer's own log.")
        return 1
    steps, train, val = overfit.read_metrics(metrics)
    verdict = overfit.analyze_curves(steps, train, val, cfg.early_stop_patience)
    print(overfit.format_curves(verdict))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``methods``/``info``/``prepare``/``train`` subcommands."""
    p = argparse.ArgumentParser(prog="motion_model", description=__doc__)
    p.add_argument("--config", help="path to motion-model YAML config")
    p.add_argument("--method", help="override the configured method")
    p.add_argument("--dataset-dir", dest="dataset_dir", help="override Pipeline 1 dataset dir")
    p.add_argument("--work-root", dest="work_root", help="override work/output root")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("methods", help="list selectable methods").set_defaults(func=cmd_methods)
    sub.add_parser("info", help="show what the configured method will do").set_defaults(func=cmd_info)
    sub.add_parser("config-template",
                   help="print a fully-commented YAML of every config option").set_defaults(func=cmd_config_template)
    sub.add_parser("prepare", help="build the method's training data").set_defaults(func=cmd_prepare)

    tr = sub.add_parser("train", help="prepare + launch training")
    tr.add_argument("--skip-prepare", action="store_true", help="assume prepare already ran")
    tr.set_defaults(func=cmd_train)

    sub.add_parser("autoscale",
                   help="preview the data-driven training plan (regime + num_steps) for the corpus"
                   ).set_defaults(func=cmd_autoscale)
    sub.add_parser("overfit-check",
                   help="pre-flight: estimate overfitting risk (per-donor aware, no GPU)"
                   ).set_defaults(func=cmd_overfit_check)
    orp = sub.add_parser("overfit-report", help="post-train: best pre-overfit checkpoint from the val curve")
    orp.add_argument("--metrics", help="path to the trainer's metrics log (default: checkpoints/metrics.jsonl)")
    orp.set_defaults(func=cmd_overfit_report)
    return p


def main(argv=None) -> int:
    """Parse argv and dispatch to the selected subcommand's handler."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
