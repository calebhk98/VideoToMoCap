"""Command-line interface: ``python -m videocaption <command>``.

Config-first, exactly like the other pipelines: put every knob (footage root,
captioner, aggregator, window size, gpus, ...) in a YAML and the commands read it
-- flags are optional overrides. The config is auto-discovered (``--config`` >
``$VIDEOCAPTION_CONFIG`` > ./videocaption.yaml > ./configs/videocaption.yaml):

    python -m videocaption run        # scan -> caption -> index -> windows -> ft-data

Individual steps, each reading the same config:
    python -m videocaption scan          # (re)build manifest; applies exclude_patterns
    python -m videocaption status        # what will be processed
    python -m videocaption caption       # segment + caption (resumable, parallel)
    python -m videocaption index         # build the search index
    python -m videocaption windows       # group segments into training windows
    python -m videocaption finetune-data # build the Step 6 bootstrap dataset
    python -m videocaption search "..."  # query the index
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from . import manifest as mf
from . import pipeline
from .backends import available_aggregators, available_captioners, get_aggregator, get_captioner
from .config import CaptionConfig, load_config
from .index import search as index_search
from .manifest import Manifest

_DEFAULT_CONFIG_NAMES = ("videocaption.yaml", "videocaption.yml", "configs/videocaption.yaml")


def _discover_config(explicit) -> Optional[str]:
    """Resolve which config to load: --config, else $VIDEOCAPTION_CONFIG, else a
    conventional file in the working dir. Returns None -> built-in defaults."""
    if explicit:
        return explicit
    env = os.environ.get("VIDEOCAPTION_CONFIG")
    if env:
        return env
    for name in _DEFAULT_CONFIG_NAMES:
        if Path(name).exists():
            return name
    return None


def _cfg(args) -> CaptionConfig:
    path = _discover_config(args.config)
    if path:
        print(f"Using config: {path}")
    cfg = load_config(path) if path else CaptionConfig()
    if getattr(args, "footage_root", None):
        cfg.footage_root = Path(args.footage_root)
    if getattr(args, "work_root", None):
        cfg.work_root = Path(args.work_root)
    return cfg


def _load_or_scan(cfg: CaptionConfig) -> Manifest:
    if cfg.manifest_path.exists():
        return Manifest.load(cfg.manifest_path)
    manifest = mf.scan(cfg)
    manifest.save(cfg.manifest_path)
    return manifest


def _gpus(args):
    val = getattr(args, "gpus", None)
    return [g.strip() for g in val.split(",") if g.strip()] if val else None


def _limit(args, cfg):
    return args.limit if args.limit is not None else cfg.limit


def cmd_scan(args) -> int:
    """Discover footage and (re)build the manifest, preserving existing state."""
    cfg = _cfg(args)
    if cfg.manifest_path.exists():
        manifest = mf.refresh(cfg, Manifest.load(cfg.manifest_path))
        print("Refreshed existing manifest (preserving exclusions/progress).")
    else:
        manifest = mf.scan(cfg)
        print("Created new manifest.")
    manifest.save(cfg.manifest_path)
    print(f"{len(manifest.videos)} videos")
    print(f"Manifest: {cfg.manifest_path}")
    return 0


def cmd_exclude(args) -> int:
    """Exclude (or, with --undo, re-include) videos by id or glob pattern."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    fn = mf.include if args.undo else mf.exclude
    n = fn(manifest, video_ids=args.id or [], patterns=args.pattern or [])
    manifest.save(cfg.manifest_path)
    print(f"{'re-included' if args.undo else 'excluded'} {n} videos")
    return 0


def cmd_status(args) -> int:
    """Print video-state counts and total captioned duration."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    counts = manifest.counts()
    print("Video states:")
    for k in (mf.PENDING, mf.EXCLUDED, mf.SEGMENTED, mf.DONE, mf.FAILED):
        if k in counts:
            print(f"  {k:10s} {counts[k]}")
    hours = sum(v.duration or 0 for v in manifest.videos) / 3600.0
    segs = sum(v.n_segments or 0 for v in manifest.videos)
    if hours:
        print(f"  footage: ~{hours:.2f} h across {segs} caption segments")
    return 0


def cmd_list(args) -> int:
    """List videos (optionally filtered by status), one per line."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    for v in manifest.videos:
        if args.status and v.status != args.status:
            continue
        line = f"{v.status:10s} {v.video_id}  {v.rel_path}"
        if v.error:
            line += f"  !! {v.error}"
        print(line)
    return 0


def cmd_info(args) -> int:
    """Show the resolved models and what will run."""
    cfg = _cfg(args)
    print(f"captioner:  {get_captioner(cfg).describe()}")
    print(f"aggregator: {get_aggregator(cfg).describe()}")
    print(f"segment cap: {cfg.max_segment_seconds}s   window: {cfg.window_seconds}s   "
          f"sample: {cfg.frame_sample_fps} fps")
    print(f"finetune base: {cfg.finetune_base}  (LoRA r={cfg.lora_rank}, a={cfg.lora_alpha})")
    print(f"work_root: {cfg.work_root}")
    return 0


def cmd_models(args) -> int:
    """List the selectable captioner/aggregator backends."""
    print("captioners:  " + ", ".join(available_captioners()))
    print("aggregators: " + ", ".join(available_aggregators()))
    return 0


def cmd_caption(args) -> int:
    """Segment + caption pending videos (resumable, parallel), then report status."""
    cfg = _cfg(args)
    if getattr(args, "workers_per_gpu", None) is not None:
        cfg.workers_per_gpu = args.workers_per_gpu
    manifest = _load_or_scan(cfg)
    pipeline.caption(cfg, manifest, limit=_limit(args, cfg), workers=args.workers, gpus=_gpus(args))
    return cmd_status(args)


def cmd_index(args) -> int:
    """Build the search index from captioned rows."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    print(json.dumps(pipeline.index(cfg, manifest), indent=2))
    return 0


def cmd_windows(args) -> int:
    """Group captioned segments into training windows."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    print(json.dumps(pipeline.windows(cfg, manifest), indent=2))
    return 0


def cmd_finetune_data(args) -> int:
    """Build the Step 6 bootstrap fine-tune dataset from the windows."""
    cfg = _cfg(args)
    manifest = _load_or_scan(cfg)
    print(json.dumps(pipeline.finetune_data(cfg, manifest), indent=2))
    return 0


def cmd_search(args) -> int:
    """Query the SQLite index and print matching (video, time range) rows."""
    cfg = _cfg(args)
    db = cfg.index_dir / "captions.db"
    if not db.exists():
        print(f"No index at {db} -- run `index` first.")
        return 1
    for row in index_search(db, args.query, limit=args.limit):
        print(f"{row['rel_path']}  [{row['start']:.1f}-{row['end']:.1f}s]  {row['description'][:80]}")
    return 0


def cmd_run(args) -> int:
    """Caption + index + windows + fine-tune dataset in one go."""
    cfg = _cfg(args)
    if getattr(args, "workers_per_gpu", None) is not None:
        cfg.workers_per_gpu = args.workers_per_gpu
    _load_or_scan(cfg)
    result = pipeline.run_all(cfg, limit=_limit(args, cfg), workers=args.workers, gpus=_gpus(args))
    print(json.dumps(result, indent=2))
    return 0


def _add_parallel_flags(sp) -> None:
    sp.add_argument("--limit", type=int, help="process at most N videos (smoke test)")
    sp.add_argument("--workers", type=int, help="videos to process at once (default: auto)")
    sp.add_argument("--gpus", help="'auto' to detect, or comma-separated ids e.g. 0,1")
    sp.add_argument("--workers-per-gpu", dest="workers_per_gpu", type=int,
                    help="videos per GPU (default from config)")


def build_parser() -> argparse.ArgumentParser:
    """Build the videocaption subcommands."""
    p = argparse.ArgumentParser(prog="videocaption", description=__doc__)
    p.add_argument("--config", help="path to caption YAML config")
    p.add_argument("--footage-root", dest="footage_root", help="override footage root")
    p.add_argument("--work-root", dest="work_root", help="override work/output root")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="discover footage and (re)build the manifest").set_defaults(func=cmd_scan)
    sub.add_parser("status", help="show video-state counts").set_defaults(func=cmd_status)
    sub.add_parser("info", help="show resolved models/knobs").set_defaults(func=cmd_info)
    sub.add_parser("models", help="list selectable captioner/aggregator backends").set_defaults(func=cmd_models)

    ex = sub.add_parser("exclude", help="exclude videos by id or glob pattern")
    ex.add_argument("--pattern", action="append", help="glob against rel path; repeatable")
    ex.add_argument("--id", action="append", help="explicit video_id; repeatable")
    ex.add_argument("--undo", action="store_true", help="re-include instead of exclude")
    ex.set_defaults(func=cmd_exclude)

    ls = sub.add_parser("list", help="list videos")
    ls.add_argument("--status", help="filter by status")
    ls.set_defaults(func=cmd_list)

    cap = sub.add_parser("caption", help="segment + caption pending videos")
    _add_parallel_flags(cap)
    cap.set_defaults(func=cmd_caption)

    sub.add_parser("index", help="build the search index").set_defaults(func=cmd_index)
    sub.add_parser("windows", help="group segments into training windows").set_defaults(func=cmd_windows)
    sub.add_parser("finetune-data", help="build the Step 6 bootstrap dataset").set_defaults(func=cmd_finetune_data)

    se = sub.add_parser("search", help="query the search index")
    se.add_argument("query", help="free-text / keyword query")
    se.add_argument("--limit", type=int, default=20, help="max hits")
    se.set_defaults(func=cmd_search)

    rn = sub.add_parser("run", help="caption + index + windows + ft-data in one go")
    _add_parallel_flags(rn)
    rn.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    """Parse argv and dispatch to the selected subcommand's handler."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
