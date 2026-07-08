"""Command-line interface: ``python -m videotomocap <command>``.

Config-first: put every knob (footage root, backend, exclude_patterns, gpus, ...)
in a YAML and the commands read it -- flags are optional overrides. The config is
auto-discovered (``--config`` > ``$VIDEOTOMOCAP_CONFIG`` > ./videotomocap.yaml >
./configs/pipeline.yaml), so the common case needs no flags at all:

    python -m videotomocap run        # scan -> exclude -> hmr -> build, from config

Individual steps, each reading the same config:
    python -m videotomocap scan       # (re)build manifest; applies exclude_patterns
    python -m videotomocap status     # what will be processed
    python -m videotomocap hmr        # human-mesh recovery (resumable)
    python -m videotomocap build      # aggregate into the AMASS dataset

Overrides when you want them: --limit, --gpus, --workers-per-gpu, --config, etc.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from . import ingest, pipeline
from .config import PipelineConfig, load_config
from .ingest import Manifest


# Where we look for a config when --config isn't passed, in order. This is what
# lets you keep every knob in one YAML and just run `videotomocap run`.
_DEFAULT_CONFIG_NAMES = ("videotomocap.yaml", "videotomocap.yml", "configs/pipeline.yaml")


def _discover_config(explicit) -> Optional[str]:
    """Resolve which config to load: --config, else $VIDEOTOMOCAP_CONFIG, else a
    conventional file in the working dir. Returns None -> built-in defaults."""
    if explicit:
        return explicit
    env = os.environ.get("VIDEOTOMOCAP_CONFIG")
    if env:
        return env
    for name in _DEFAULT_CONFIG_NAMES:
        if Path(name).exists():
            return name
    return None


def _cfg(args) -> PipelineConfig:
    path = _discover_config(args.config)
    if path:
        print(f"Using config: {path}")
    cfg = load_config(path) if path else PipelineConfig()
    if getattr(args, "footage_root", None):
        cfg.footage_root = Path(args.footage_root)
    if getattr(args, "work_root", None):
        cfg.work_root = Path(args.work_root)
    if getattr(args, "multi_person", False):
        cfg.multi_person = True
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
    suspected = sum(1 for c in manifest.clips if c.suspected_mirrored)
    corrected = sum(1 for c in manifest.clips if c.mirrored)
    if suspected or corrected:
        print(f"  mirrored: {suspected} suspected, {corrected} corrected")
    low_q = sum(1 for c in manifest.clips if c.low_quality)
    flagged_q = sum(1 for c in manifest.clips if c.quality_issues)
    if flagged_q:
        print(f"  quality: {flagged_q} flagged, {low_q} hard-failing")
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


def _gpus(args):
    """Parse --gpus: 'auto' -> ['auto']; '0,1' -> ['0','1']; unset -> None (config)."""
    val = getattr(args, "gpus", None)
    return [g.strip() for g in val.split(",") if g.strip()] if val else None


def _apply_parallel_overrides(cfg, args) -> None:
    wpg = getattr(args, "workers_per_gpu", None)
    if wpg is not None:
        cfg.workers_per_gpu = wpg if wpg.lower() == "auto" else int(wpg)
    if getattr(args, "vram_per_worker_mb", None) is not None:
        cfg.vram_per_worker_mb = args.vram_per_worker_mb


def _limit(args, cfg):
    """--limit overrides config; otherwise use cfg.limit (may be None = all)."""
    return args.limit if args.limit is not None else cfg.limit


def cmd_hmr(args) -> int:
    """Run human-mesh recovery on pending/failed clips, then report status."""
    cfg = _cfg(args)
    _apply_parallel_overrides(cfg, args)
    manifest = _load_or_scan(cfg)
    pipeline.run_hmr(cfg, manifest, limit=_limit(args, cfg), workers=args.workers, gpus=_gpus(args))
    cmd_status(args)
    return 0


def cmd_mirror(args) -> int:
    """Detect (and, per config, correct) left/right-mirrored clips corpus-wide."""
    cfg = _cfg(args)
    if getattr(args, "correct", False):
        cfg.auto_mirror = "correct"
    elif cfg.auto_mirror == "off":
        cfg.auto_mirror = "flag"
    manifest = _load_or_scan(cfg)
    result = pipeline.detect_mirroring(cfg, manifest)
    print(json.dumps(result, indent=2))
    return 0


def cmd_people(args) -> int:
    """Manage the people registry + consent (multi_person). Actions:
    list | assign (cluster tracks -> person_id) | grant | revoke."""
    from . import identity, people

    cfg = _cfg(args)
    if args.action == "assign":
        manifest = Manifest.load(cfg.manifest_path)
        counts = identity.assign_people(cfg, manifest)
        print("assigned tracks per person:")
        print(json.dumps(counts, indent=2))
        return 0
    if args.action == "list":
        registry = people.load_registry(cfg)
        if not registry:
            print("no people yet -- run `people assign` after `hmr` with multi_person on.")
        for pid, rec in sorted(registry.items()):
            granted = rec.get("consent", {}).get("granted")
            print(f"  {pid:14s} {rec.get('display_name', ''):18s} consent={'granted' if granted else 'DENIED'}")
        return 0
    # grant / revoke
    registry = people.load_registry(cfg)
    ids = sorted(registry) if args.all else (args.id or [])
    if not ids:
        print("no person ids -- pass --id <id> (repeatable) or --all")
        return 1
    for pid in ids:
        people.set_consent(cfg, pid, granted=(args.action == "grant"))
    print(f"{args.action}ed consent for {len(ids)} people; audit -> {cfg.consent_log_path}")
    return 0


def cmd_build(args) -> int:
    """Aggregate anonymized clips into the AMASS-format dataset."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    stats = pipeline.build(cfg, manifest)  # runs mirror + quality passes, then aggregates
    print("Dataset written to", cfg.dataset_dir)
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def cmd_caption_dataset(args) -> int:
    """Pair Pipeline 3 captions with this pipeline's motion -> text-to-motion dataset.

    Slices each clip's recovered motion at the caption-segment boundaries and writes
    an AMASS + index.json dataset (one snippet+caption per segment) that Pipeline 2
    trains on with ``conditioning: text``.
    """
    from .captioned_dataset import build_captioned_dataset  # lazy: pulls in videocaption

    cfg = _cfg(args)
    out = Path(args.out) if args.out else cfg.work_root / "dataset_captioned"
    stats = build_captioned_dataset(
        cfg.pose_dir, args.caption_work, out,
        min_frames=cfg.min_clip_frames, val_fraction=cfg.val_fraction,
    )
    print(f"Captioned dataset written to {out}")
    print(json.dumps(stats, indent=2))
    return 0


def cmd_run(args) -> int:
    """Run HMR and build the dataset in one go."""
    cfg = _cfg(args)
    _apply_parallel_overrides(cfg, args)
    _load_or_scan(cfg)
    stats = pipeline.run_all(cfg, limit=_limit(args, cfg), workers=args.workers, gpus=_gpus(args))
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

    for name, fn in (("hmr", cmd_hmr), ("run", cmd_run)):
        sp = sub.add_parser(name, help=("run human-mesh recovery on pending clips"
                                        if name == "hmr" else "hmr + build in one go"))
        sp.add_argument("--limit", type=int, help="process at most N clips (smoke test)")
        sp.add_argument("--workers", type=int, help="clips to process at once (default: auto)")
        sp.add_argument("--gpus", help="'auto' to detect, or comma-separated ids e.g. 0,1")
        sp.add_argument("--workers-per-gpu", dest="workers_per_gpu",
                        help="clips per GPU: an int, or 'auto' to size from free VRAM")
        sp.add_argument("--vram-per-worker-mb", dest="vram_per_worker_mb", type=int,
                        help="VRAM/clip estimate (MiB) for auto sizing; skips calibration")
        sp.add_argument("--multi-person", dest="multi_person", action="store_true",
                        help="recover every person per clip (needs a multi-person backend)")
        sp.set_defaults(func=fn)

    mr = sub.add_parser("mirror", help="detect (--correct to fix) left/right-mirrored clips")
    mr.add_argument("--correct", action="store_true", help="flip flagged clips (else just flag)")
    mr.set_defaults(func=cmd_mirror)

    sub.add_parser("build", help="aggregate anonymized clips into an AMASS dataset").set_defaults(func=cmd_build)

    pp = sub.add_parser("people", help="manage people + consent (multi_person)")
    pp.add_argument("action", choices=["list", "assign", "grant", "revoke"],
                    help="list; assign (cluster tracks->person_id); grant/revoke consent")
    pp.add_argument("--id", action="append", help="person id (repeatable)")
    pp.add_argument("--all", action="store_true", help="apply to all registered people")
    pp.set_defaults(func=cmd_people)

    cd = sub.add_parser("caption-dataset",
                        help="pair Pipeline 3 captions with motion -> text-to-motion dataset")
    cd.add_argument("--caption-work", dest="caption_work", default="work/caption",
                    help="Pipeline 3 work_root holding the caption manifest + labels")
    cd.add_argument("--out", help="output dataset dir (default: <work_root>/dataset_captioned)")
    cd.set_defaults(func=cmd_caption_dataset)
    return p


def main(argv=None) -> int:
    """Parse argv and dispatch to the selected subcommand's handler."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
