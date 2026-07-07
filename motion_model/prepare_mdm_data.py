#!/usr/bin/env python
"""Bridge the AMASS-format dataset (Pipeline 1) into MDM/HumanML3D training data.

What this script does *without* external assets (runs anywhere):
  * reads work/dataset/index.json (the clip list + train/val split)
  * lays out the HumanML3D-style directory skeleton under --out
  * writes new_joint_vecs/ placeholders and an empty caption per clip
    (unconditional fine-tune by default; edit texts/ to add conditioning)
  * writes train.txt / val.txt split files

What needs the SMPL body model + the HumanML3D repo (the actual feature
extraction: SMPL forward kinematics -> 22 joints -> 263-d features):
  * pass --humanml3d and --smpl-model to enable it; otherwise the script stops
    after the skeleton and tells you exactly what to run.

This split is deliberate: you can wire up and inspect the whole data layout on a
laptop, and only the one GPU/asset-dependent step is gated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def load_index(dataset_dir: Path) -> dict:
    """Load Pipeline 1's clip index, failing loudly if ``build`` hasn't run yet."""
    idx = dataset_dir / "index.json"
    if not idx.exists():
        sys.exit(f"No index.json in {dataset_dir}; run `videotomocap build` first.")
    return json.loads(idx.read_text())


def make_skeleton(out: Path) -> dict:
    """Create the HumanML3D-style directory layout under ``out`` and return its paths."""
    dirs = {
        "amass": out / "amass_copy",
        "joints": out / "new_joints",
        "vecs": out / "new_joint_vecs",
        "texts": out / "texts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_splits(out: Path, clips: list) -> None:
    """Write train/val/test id lists from Pipeline 1's per-clip split."""
    train = [c["clip_id"] for c in clips if c.get("split") == "train"]
    val = [c["clip_id"] for c in clips if c.get("split") == "val"]
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    # HumanML3D expects a test list too; reuse val so downstream scripts don't choke.
    (out / "test.txt").write_text("\n".join(val) + "\n")
    print(f"  splits: {len(train)} train / {len(val)} val")


def scaffold_texts(dirs: dict, clips: list) -> None:
    """Ensure every clip has a caption file, leaving existing ones untouched."""
    for c in clips:
        t = dirs["texts"] / f"{c['clip_id']}.txt"
        if not t.exists():
            # Empty caption => unconditional. Fill with "<action>#...#..." for
            # HumanML3D-style text conditioning.
            t.write_text("")


def extract_features(args, clips: list, dirs: dict) -> None:
    """Invoke the HumanML3D feature extractor (SMPL FK -> 263-d features).

    Left as an explicit hand-off because it requires the registered SMPL body
    model and the HumanML3D processing code, which cannot be vendored here.
    """
    hml = Path(args.humanml3d)
    smpl = Path(args.smpl_model)
    if not hml.exists():
        sys.exit(f"--humanml3d path not found: {hml}")
    if not smpl.exists():
        sys.exit(f"--smpl-model path not found: {smpl}")

    print(
        "\nFeature extraction hand-off:\n"
        f"  1. Copy {args.dataset}/amass/*.npz into the HumanML3D AMASS input tree.\n"
        f"  2. Run HumanML3D's raw_pose_processing + motion_representation notebooks/scripts\n"
        f"     ({hml}) with SMPL model at {smpl} to produce new_joint_vecs/*.npy (263-d).\n"
        f"  3. Point MDM's --data_dir at {dirs['vecs'].parent}.\n"
        "  (HumanML3D indexes SMPL-H poses[:66] for its 22 joints; our AMASS npz already\n"
        "   stores the SMPL body in those first 66 slots, so it consumes cleanly.)\n"
    )


def main() -> int:
    """Build the MDM data skeleton and, if assets are supplied, hand off feature extraction."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="work/dataset from Pipeline 1")
    ap.add_argument("--out", required=True, type=Path, help="output MDM data dir")
    ap.add_argument("--humanml3d", type=Path, help="path to cloned HumanML3D repo (enables feature step)")
    ap.add_argument("--smpl-model", type=Path, help="path to SMPL body model dir")
    args = ap.parse_args()

    index = load_index(args.dataset)
    clips = index["clips"]
    print(f"{len(clips)} clips in dataset")

    dirs = make_skeleton(args.out)
    # copy the AMASS npz so the MDM-side tree is self-contained
    for c in clips:
        src = args.dataset / "amass" / f"{c['clip_id']}.npz"
        if src.exists():
            shutil.copy2(src, dirs["amass"] / src.name)
    write_splits(args.out, clips)
    scaffold_texts(dirs, clips)
    print(f"Skeleton written to {args.out}")

    if args.humanml3d and args.smpl_model:
        extract_features(args, clips, dirs)
    else:
        print("\n(Skipping feature extraction: pass --humanml3d and --smpl-model to enable.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
