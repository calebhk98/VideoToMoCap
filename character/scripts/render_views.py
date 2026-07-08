"""Headless Blender: render a character asset to N orbit views for the mesh critic.

Run by character/render.py:
    blender --background --python character/scripts/render_views.py -- \
        --asset path/to/character.glb --out path/to/renders --views 4

Runs inside Blender's Python (bpy) -- not unit-tested by the repo's GPU-free suite.
Verify against your Blender version (tested conceptually against Blender 3.6/4.x).
"""

import argparse
import math
import sys

import bpy


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--asset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--res", type=int, default=512)
    return p.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import(path):
    lower = path.lower()
    if lower.endswith(".glb") or lower.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif lower.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    elif lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif lower.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise SystemExit(f"render_views: unsupported asset format: {path}")


def _scene_bounds():
    """Center + radius of all mesh objects, to frame the orbit camera."""
    import mathutils

    coords = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            coords.append(obj.matrix_world @ mathutils.Vector(corner))
    if not coords:
        return mathutils.Vector((0, 0, 1)), 1.0
    lo = mathutils.Vector((min(c[i] for c in coords) for i in range(3)))
    hi = mathutils.Vector((max(c[i] for c in coords) for i in range(3)))
    center = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 0.5)
    return center, radius


def _setup_light_and_camera(center, radius):
    import mathutils

    bpy.ops.object.light_add(type="SUN")
    bpy.context.object.data.energy = 3.0
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam, mathutils.Vector(center), radius


def _render_orbit(cam, center, radius, n_views, out, res):
    import mathutils

    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = res
    dist = radius * 3.0
    for i in range(max(1, n_views)):
        ang = 2.0 * math.pi * i / max(1, n_views)
        cam.location = center + mathutils.Vector((math.sin(ang) * dist, -math.cos(ang) * dist, radius * 0.3))
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = f"{out}/view_{i:02d}.png"
        bpy.ops.render.render(write_still=True)


def main():
    args = _args()
    _clear_scene()
    _import(args.asset)
    center, radius = _scene_bounds()
    cam, center, radius = _setup_light_and_camera(center, radius)
    _render_orbit(cam, center, radius, args.views, args.out, args.res)


if __name__ == "__main__":
    main()
