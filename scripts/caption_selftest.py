#!/usr/bin/env python
"""End-to-end self-test of Pipeline 3 with zero GPU / weights / video decoders.

Fabricates a small archive, then drives the real pipeline with the synthetic
'noop' captioner + aggregator:  scan -> exclude -> segment -> caption -> aggregate
-> search index -> windows -> Step 6 bootstrap dataset.  Asserts the label format,
index rows, window grouping, and the window-relative training target hold.  Run:
    python scripts/caption_selftest.py

Segmentation reads a video's duration (OpenCV), which we don't have here, so Step 1
is exercised two ways: its pure logic directly (``segment_video`` with injected
duration/scenes), and the rest of the flow on those segments seeded to disk --
exactly as a resumed run sees them. This mirrors how Pipeline 1's self-test uses
the 'noop' backend to stay decoder-free.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videocaption import manifest as mf, pipeline, store
from videocaption.config import CaptionConfig
from videocaption.manifest import Manifest
from videocaption.segment import Segment, segment_video


def make_fake_archive(root: Path) -> None:
    """Create empty video placeholders in a realistic tree (noop reads no pixels)."""
    layout = {
        "phone/2024-05-01": ["morning.mp4", "walk.mp4"],
        "gopro/2024-06-10": ["ride.mp4"],
        "phone/2024-12-25": ["family.mp4"],  # to be excluded
    }
    for rel, files in layout.items():
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_bytes(b"")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        footage = tmp / "footage"
        make_fake_archive(footage)

        cfg = CaptionConfig(
            footage_root=footage, work_root=tmp / "work",
            captioner="noop", aggregator="noop", scene_detect=False,
            max_segment_seconds=120.0, window_seconds=300.0, num_tags=8,
        )

        print("1) scan")
        manifest = mf.scan(cfg)
        manifest.save(cfg.manifest_path)
        check(len(manifest.videos) == 4, f"discovered 4 videos (got {len(manifest.videos)})")

        print("2) exclude the family visit")
        n = mf.exclude(manifest, patterns=["*/2024-12-25/*"])
        manifest.save(cfg.manifest_path)
        check(n == 1, f"excluded exactly 1 video (got {n})")

        print("3) segment (Step 1 logic, decoder-free via injected signals)")
        segs, dur = segment_video(footage / "phone/2024-05-01/walk.mp4", scene_detect=True,
                                  scene_threshold=27, max_seconds=120, min_seconds=3,
                                  duration=600.0, scenes=[(0, 130), (130, 600)])
        check(all(s.duration <= 120 + 1e-6 for s in segs), "no segment exceeds the 2-min cap")
        # seed each processable video with 4 x 120s segments (a 10-min video)
        for v in manifest.by_status(mf.PENDING):
            store.save_segments(cfg, v.video_id, [Segment(i, i * 120, (i + 1) * 120) for i in range(4)], 480.0)
            v.status = mf.SEGMENTED
        manifest.save(cfg.manifest_path)

        print("4) caption + aggregate (synthetic)")
        pipeline.caption(cfg, manifest)
        done = manifest.by_status(mf.DONE)
        check(len(done) == 3, f"3 videos captioned, family visit skipped (got {len(done)})")
        check(len(manifest.by_status(mf.FAILED)) == 0, "no failures")

        print("5) label format: (description, tags) per segment")
        rows = list(store.iter_rows(cfg, manifest))
        check(len(rows) == 12, f"3 videos x 4 segments = 12 rows (got {len(rows)})")
        check(bool(rows[0].description) and isinstance(rows[0].tags, list) and rows[0].tags,
              "each row carries a non-empty description and tag list")

        print("6) search index (Step 5)")
        summary = pipeline.index(cfg, manifest)
        check(summary["n_rows"] == 12, f"index has 12 rows (got {summary['n_rows']})")
        from videocaption.index import search
        hits = search(cfg.index_dir / "captions.db", rows[0].tags[0])
        check(len(hits) >= 1, "a tag query returns at least one hit with a time range")
        check("start" in hits[0] and "end" in hits[0], "hits carry the jump-to time range")

        print("7) windows (Step 5.5) -- larger than the caption segments")
        win = pipeline.windows(cfg, manifest)
        # 4 x 120s = 480s of segments, 300s window -> 2 windows/video (2 segs + 2 segs)
        check(win["n_windows"] == 6, f"6 windows across 3 videos (got {win['n_windows']})")
        check(win["n_labels"] == 12, "every caption segment lands in exactly one window")

        print("8) Step 6 bootstrap dataset -- window-relative timestamped targets")
        ft = pipeline.finetune_data(cfg, manifest)
        check(ft["n_examples"] == 6, f"one training example per window (got {ft['n_examples']})")
        first = json.loads((cfg.finetune_dir / "dataset.jsonl").read_text().splitlines()[0])
        labels = json.loads(first["messages"][1]["content"])
        check(labels[0]["start"] == 0.0, "target timestamps are relative to the window start")
        check({"start", "end", "description", "tags"} <= set(labels[0]), "target is the (start,end,description,tags) format")

        print("9) reload manifest from disk (resumability)")
        reloaded = Manifest.load(cfg.manifest_path)
        check(reloaded.counts().get(mf.DONE) == 3, "state survives round-trip to disk")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
