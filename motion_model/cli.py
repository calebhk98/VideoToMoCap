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
    print(f"Training method={cfg.method} ...")
    save = trainer.train()
    print(f"Checkpoints: {save}")
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
    sub.add_parser("prepare", help="build the method's training data").set_defaults(func=cmd_prepare)

    tr = sub.add_parser("train", help="prepare + launch training")
    tr.add_argument("--skip-prepare", action="store_true", help="assume prepare already ran")
    tr.set_defaults(func=cmd_train)
    return p


def main(argv=None) -> int:
    """Parse argv and dispatch to the selected subcommand's handler."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
