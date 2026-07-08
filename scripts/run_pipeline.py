#!/usr/bin/env python
"""One command, video -> motion model.

Runs Pipeline 1 (scan -> HMR -> anonymize -> AMASS dataset) then Pipeline 2
(prepare -> launch training) back to back, each from its own config. This is the
whole chain; the two stages keep separate configs because they run in different
environments (the HMR backend and the trainer usually live in different conda
envs), but you invoke them together:

    python scripts/run_pipeline.py \
        --video-config configs/dropzone.yaml \
        --model-config configs/motion_model.yaml

Use --dataset-only to stop after the dataset (e.g. to inspect it before training),
or --model-only to (re)train from an already-built dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model.cli import main as model_main
from videotomocap.cli import main as video_main


def main(argv=None) -> int:
    """Drive Pipeline 1 then Pipeline 2 from their configs; honour the stage flags."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-config", default="configs/dropzone.yaml", help="Pipeline 1 (video->dataset) config")
    ap.add_argument("--model-config", default="configs/motion_model.yaml", help="Pipeline 2 (dataset->model) config")
    ap.add_argument("--dataset-only", action="store_true", help="stop after building the dataset")
    ap.add_argument("--model-only", action="store_true", help="skip Pipeline 1; train from an existing dataset")
    args = ap.parse_args(argv)

    if not args.model_only:
        print("=== Pipeline 1: video -> AMASS dataset ===")
        rc = video_main(["--config", args.video_config, "run"])
        if rc != 0 or args.dataset_only:
            return rc

    print("\n=== Pipeline 2: dataset -> motion model ===")
    return model_main(["--config", args.model_config, "train"])


if __name__ == "__main__":
    raise SystemExit(main())
