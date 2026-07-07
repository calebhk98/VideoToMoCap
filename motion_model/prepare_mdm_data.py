#!/usr/bin/env python
"""Thin CLI shim: build the MDM/HumanML3D data skeleton from Pipeline 1's dataset.

Kept for backwards compatibility and quick one-off use. The real, config-driven
entry point is now ``python -m motion_model prepare`` (pick the method in a YAML,
exactly like Pipeline 1). This script just wires the shared helpers in
``motion_model.data`` to argparse and prints the same asset-gated hand-off.

    python motion_model/prepare_mdm_data.py --dataset work/dataset --out work/mdm_data \
        [--humanml3d /opt/HumanML3D --smpl-model /opt/body_models/smpl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

# Allow running as a bare script (`python motion_model/prepare_mdm_data.py`): put
# the repo root on sys.path so the shared package imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motion_model import data  # noqa: E402


def main() -> int:
    """Build the HumanML3D skeleton and print the feature-extraction hand-off."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="work/dataset from Pipeline 1")
    ap.add_argument("--out", required=True, type=Path, help="output MDM data dir")
    ap.add_argument("--humanml3d", type=Path, help="path to cloned HumanML3D repo (enables feature step)")
    ap.add_argument("--smpl-model", type=Path, help="path to SMPL body model dir")
    ap.add_argument("--conditioning", default="none", choices=["none", "text", "action"],
                    help="caption scaffolding mode (default: none)")
    args = ap.parse_args()

    index = data.load_index(args.dataset)
    clips = index["clips"]
    print(f"{len(clips)} clips in dataset")

    dirs = data.make_humanml3d_skeleton(args.out)
    data.copy_amass(args.dataset, dirs["amass"], clips)
    data.write_splits(args.out, clips)
    data.write_captions(dirs["texts"], clips, args.conditioning)
    print(f"Skeleton written to {args.out}")

    # Reuse the package's hand-off text, driven by a tiny config-shaped object.
    cfg = SimpleNamespace(humanml3d_repo=args.humanml3d, smpl_model=args.smpl_model)
    print(data.humanml3d_handoff(cfg, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
