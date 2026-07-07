"""Command-line interface: ``python -m videotomocap <command>``.

Typical session
---------------
    # 1. discover all footage
    python -m videotomocap scan --config configs/pipeline.yaml

    # 2. pull out the two family visits before anything is processed
    python -m videotomocap exclude --pattern "*/2024-12-24/*" --pattern "*/2025-06-*"

    # 3. sanity-check what will be processed
    python -m videotomocap status

    # 4. run human-mesh recovery (GPU box; resumable, --limit to smoke-test)
    python -m videotomocap hmr --limit 5

    # 5. aggregate the anonymized clips into an AMASS-format dataset
    python -m videotomocap build
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ingest, pipeline
from .config import PipelineConfig, load_config
from .ingest import Manifest


def _cfg(args) -> PipelineConfig:
    cfg = load_config(args.config) if args.config else PipelineConfig()
    if getattr(args, "footage_root", None):
        cfg.footage_root = Path(args.footage_root)
    if getattr(args, "work_root", None):
        cfg.work_root = Path(args.work_root)
    return cfg


def _load_or_scan(cfg: PipelineConfig) -> Manifest:
    if cfg.manifest_path.exists():
        return Manifest.load(cfg.manifest_path)
    manifest = ingest.scan(cfg)
    manifest.save(cfg.manifest_path)
    return manifest


def cmd_scan(args) -> int:
    """Discover footage and (re)build the manifest, preserving existing state."""
    cfg = _cfg(args)
    if cfg.manifest_path.exists():
        manifest = ingest.refresh(cfg, Manifest.load(cfg.manifest_path))
        print("Refreshed existing manifest (preserving exclusions/progress).")
    else:
        manifest = ingest.scan(cfg)
        print("Created new manifest.")
    manifest.save(cfg.manifest_path)
    print(f"{len(manifest.clips)} clips across {len(set(c.camera for c in manifest.clips))} cameras")
    print(f"Manifest: {cfg.manifest_path}")
    return 0


def cmd_exclude(args) -> int:
    """Exclude (or, with --undo, re-include) clips by id or glob pattern."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    fn = ingest.include if args.undo else ingest.exclude
    n = fn(manifest, clip_ids=args.id or [], patterns=args.pattern or [])
    manifest.save(cfg.manifest_path)
    verb = "re-included" if args.undo else "excluded"
    print(f"{verb} {n} clips")
    return 0


def cmd_status(args) -> int:
    """Print clip-state counts and total recovered motion duration."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    counts = manifest.counts()
    print("Clip states:")
    for k in (ingest.PENDING, ingest.EXCLUDED, ingest.HMR_DONE, ingest.POSE_DONE, ingest.FAILED):
        if k in counts:
            print(f"  {k:10s} {counts[k]}")
    frames = sum(c.n_frames or 0 for c in manifest.clips)
    if frames:
        print(f"  frames of motion recovered: {frames} (~{frames / cfg.target_fps / 3600:.2f} h)")
    return 0


def cmd_list(args) -> int:
    """List clips (optionally filtered by status), one per line."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    for c in manifest.clips:
        if args.status and c.status != args.status:
            continue
        line = f"{c.status:10s} {c.camera:14s} {c.clip_id}  {c.rel_path}"
        if c.error:
            line += f"  !! {c.error}"
        print(line)
    return 0


def cmd_hmr(args) -> int:
    """Run human-mesh recovery on pending/failed clips, then report status."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    pipeline.run_hmr(cfg, manifest, limit=args.limit)
    cmd_status(args)
    return 0


def cmd_build(args) -> int:
    """Aggregate anonymized clips into the AMASS-format dataset."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    stats = pipeline.build(cfg, manifest)
    print("Dataset written to", cfg.dataset_dir)
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def cmd_run(args) -> int:
    """Run HMR and build the dataset in one go."""
    cfg = _cfg(args)
    _load_or_scan(cfg)
    stats = pipeline.run_all(cfg, limit=args.limit)
    print("Dataset written to", cfg.dataset_dir)
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``scan``/``exclude``/``status``/``list``/``hmr``/``build``/``run`` subcommands."""
    p = argparse.ArgumentParser(prog="videotomocap", description=__doc__)
    p.add_argument("--config", help="path to pipeline YAML config")
    p.add_argument("--footage-root", dest="footage_root", help="override footage root")
    p.add_argument("--work-root", dest="work_root", help="override work/output root")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="discover footage and (re)build the manifest").set_defaults(func=cmd_scan)

    ex = sub.add_parser("exclude", help="exclude clips by id or glob pattern (e.g. family visits)")
    ex.add_argument("--pattern", action="append", help="glob against rel path; repeatable")
    ex.add_argument("--id", action="append", help="explicit clip_id; repeatable")
    ex.add_argument("--undo", action="store_true", help="re-include instead of exclude")
    ex.set_defaults(func=cmd_exclude)

    sub.add_parser("status", help="show clip-state counts").set_defaults(func=cmd_status)

    ls = sub.add_parser("list", help="list clips")
    ls.add_argument("--status", help="filter by status")
    ls.set_defaults(func=cmd_list)

    hr = sub.add_parser("hmr", help="run human-mesh recovery on pending clips")
    hr.add_argument("--limit", type=int, help="process at most N clips (smoke test)")
    hr.set_defaults(func=cmd_hmr)

    sub.add_parser("build", help="aggregate anonymized clips into an AMASS dataset").set_defaults(func=cmd_build)

    rn = sub.add_parser("run", help="hmr + build in one go")
    rn.add_argument("--limit", type=int, help="process at most N clips")
    rn.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    """Parse argv and dispatch to the selected subcommand's handler."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
