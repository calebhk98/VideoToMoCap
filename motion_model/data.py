"""GPU-free data helpers: lay out training data from Pipeline 1's AMASS dataset.

Reads ``work/dataset`` (``amass/*.npz`` + ``index.json``) and builds the
directory skeleton each upstream trainer expects. Only numpy + stdlib here -- the
heavy step (SMPL forward-kinematics -> HumanML3D 263-d features) is shelled out to
the upstream repos by the generator trainers, never done in-process. This split
lets you wire up and inspect the whole layout on a laptop with no GPU/assets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


# MDM's HumanML3D loader parses each caption as "text#tok/POS ...#start#end" and
# CRASHES on an empty file -- so the placeholder must be a valid line, not "".
PLACEHOLDER_CAPTION = "a person moves#a/DET person/NOUN moves/VERB#0.0#0.0\n"


def load_index(dataset_dir: Path) -> dict:
    """Load Pipeline 1's clip index, failing loudly if ``build`` hasn't run yet."""
    idx = Path(dataset_dir) / "index.json"
    if not idx.exists():
        raise FileNotFoundError(
            f"No index.json in {dataset_dir}; run `python -m videotomocap build` first."
        )
    return json.loads(idx.read_text())


def make_humanml3d_skeleton(out: Path) -> Dict[str, Path]:
    """Create the HumanML3D-style directory layout under ``out`` and return it."""
    dirs = {
        "amass": out / "amass_copy",
        "joints": out / "new_joints",
        "vecs": out / "new_joint_vecs",
        "texts": out / "texts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_amass(dataset_dir: Path, dst: Path, clips: List[dict]) -> int:
    """Copy each clip's AMASS npz into ``dst`` so the training tree is self-contained."""
    n = 0
    for c in clips:
        src = Path(dataset_dir) / "amass" / f"{c['clip_id']}.npz"
        if src.exists():
            shutil.copy2(src, dst / src.name)
            n += 1
    return n


def write_splits(out: Path, clips: List[dict]) -> None:
    """Write train/val/test id lists from Pipeline 1's per-clip split."""
    train = [c["clip_id"] for c in clips if c.get("split") == "train"]
    val = [c["clip_id"] for c in clips if c.get("split") == "val"]
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    # HumanML3D expects a test list too; reuse val so downstream scripts don't choke.
    (out / "test.txt").write_text("\n".join(val) + "\n")


def _caption_for(clip: dict, conditioning: str) -> str:
    """The caption line to write for a clip under the chosen conditioning mode.

    'action' turns Pipeline 1's unsupervised ``action_cluster`` id into a coarse
    label so you get promptable control without hand-captioning; 'none'/'text'
    fall back to the loader-valid placeholder (fill texts/ yourself for real text).
    """
    if conditioning == "action" and clip.get("action_cluster") is not None:
        label = f"action {clip['action_cluster']}"
        return f"{label}#{label.replace(' ', '/NOUN ')}/NUM#0.0#0.0\n"
    return PLACEHOLDER_CAPTION


def write_captions(texts_dir: Path, clips: List[dict], conditioning: str) -> None:
    """Ensure every clip has a loader-valid caption file, leaving existing ones be."""
    for c in clips:
        t = texts_dir / f"{c['clip_id']}.txt"
        if not t.exists():
            t.write_text(_caption_for(c, conditioning))


def fps_warning(target_fps: int) -> Optional[str]:
    """Return a warning if ``target_fps`` would misfeed HumanML3D's int(fps/20) stride."""
    if target_fps % 20 != 0:
        return (
            f"target_fps={target_fps} is not a multiple of 20: HumanML3D decimates "
            f"with int(fps/20), so clips would train at the wrong speed. Re-export "
            f"Pipeline 1 at 20 fps (PipelineConfig.target_fps)."
        )
    return None


def humanml3d_handoff(cfg, out: Path) -> str:
    """The exact, asset-gated feature-extraction hand-off text for generator methods."""
    hml = cfg.humanml3d_repo
    smpl = cfg.smpl_model
    if hml is None or smpl is None:
        return (
            "\nFeature extraction is gated on external assets. To finish it:\n"
            "  set humanml3d_repo (EricGuo5513/HumanML3D checkout) and smpl_model\n"
            "  (registration-gated SMPL+H / DMPL body models) in the config, then\n"
            "  re-run `prepare`. The 263-d features cannot be produced without them.\n"
        )
    return (
        "\nFeature extraction hand-off (the one asset-gated step; see motion_model/README.md):\n"
        f"  1. In {hml}, run raw_pose_processing on {out/'amass_copy'}/*.npz with the\n"
        f"     SMPL+H model at {smpl} (needs human_body_prior) -> (T,22,3) joints.\n"
        "  2. Run motion_representation's process_file -> new_joint_vecs/*.npy (263-d);\n"
        "     its tgt_offsets is the one-time KIT-000021 reference baked into checkpoints.\n"
        "  3. Fine-tuning? Use the checkpoint's SHIPPED Mean.npy/Std.npy (only recompute\n"
        "     via cal_mean_variance if training from scratch).\n"
        f"  4. Point the trainer's data dir at {out}.\n"
        "  (Or use method: protomotions, which needs none of this -- it consumes the\n"
        "  AMASS npz directly.)\n"
    )


def prepare_humanml3d(cfg) -> Path:
    """Build the HumanML3D training skeleton from the AMASS dataset (generator methods).

    Does everything that needs no GPU/assets -- skeleton, amass copy, splits,
    captions -- then returns the prepared dir. The 263-d feature extraction itself
    is the asset-gated hand-off in :func:`humanml3d_handoff`; the caller prints it.
    """
    index = load_index(cfg.dataset_dir)
    clips = index["clips"]
    out = cfg.prepared_dir
    dirs = make_humanml3d_skeleton(out)
    copy_amass(cfg.dataset_dir, dirs["amass"], clips)
    write_splits(out, clips)
    write_captions(dirs["texts"], clips, cfg.conditioning)
    return out
