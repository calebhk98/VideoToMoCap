"""Headless Blender: generate + rig a parametric human with MPFB2, export FBX.

Run by character/generative.py (Mpfb2Backend):
    blender --background --python character/scripts/mpfb2_gen.py -- \
        --rig mixamo --out path/to/character.fbx --prompt "a dwarf"

Runs inside Blender's Python (bpy) with MPFB2 installed -- not unit-tested by the
repo's GPU-free suite. MPFB2 is slider/parameter driven (no text conditioning), so
``--prompt`` is advisory metadata only; art-direction happens via the modeling params
you set below. Verify against your Blender 4.2+ / MPFB2 version.

MPFB2's rig is MakeHuman/Mixamo topology, NOT SMPL -- drive it by retargeting the
SMPL motion with the map from `videotomocap export-retarget-config --target mixamo`.
"""

import argparse
import sys

import bpy


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rig", default="mixamo",
                   help="built-in MPFB2 rig: default|game_engine|mixamo|openpose|cmu_mb")
    p.add_argument("--prompt", default="", help="advisory only (MPFB2 is parametric, not text-driven)")
    return p.parse_args(argv)


def _enable_mpfb():
    """MPFB2 must be enabled in the loaded Blender profile; enable it defensively."""
    try:
        bpy.ops.preferences.addon_enable(module="mpfb")
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the caller
        raise SystemExit(f"mpfb2_gen: could not enable MPFB2 addon (install it in Blender): {exc}")


def _dynamic_import(module_name, class_name):
    """MPFB2 extensions land at an unpredictable module path -- resolve dynamically
    (the boilerplate MPFB2's own script_samples use)."""
    import importlib

    for prefix in ("", "mpfb.", "bl_ext.user_default.mpfb.", "bl_ext.blender_org.mpfb."):
        try:
            mod = importlib.import_module(prefix + module_name if prefix else module_name)
            return getattr(mod, class_name)
        except (ImportError, AttributeError):
            continue
    raise SystemExit(f"mpfb2_gen: could not import {class_name} from {module_name} (MPFB2 layout changed).")


def main():
    args = _args()
    _enable_mpfb()
    HumanService = _dynamic_import("mpfb.services.humanservice", "HumanService")
    ExportService = _dynamic_import("mpfb.services.exportservice", "ExportService")

    basemesh = HumanService.create_human()
    # Art-direction hook: set modeling params here (age/build/proportions) to realize the
    # described character. MPFB2 is slider-driven -- map desired traits to HumanService
    # modeling targets. Left neutral by default.
    HumanService.add_builtin_rig(basemesh, args.rig)

    export_basemesh = ExportService.create_character_copy(basemesh, name_suffix="_export")
    ExportService.bake_modifiers_remove_helpers(
        export_basemesh, bake_masks=True, bake_subdiv=True, remove_helpers=True, also_proxy=True)
    bpy.ops.object.select_all(action="DESELECT")
    export_basemesh.select_set(True)
    bpy.ops.export_scene.fbx(filepath=args.out, use_selection=True, add_leaf_bones=False, bake_anim=False)


if __name__ == "__main__":
    main()
