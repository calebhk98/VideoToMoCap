"""Multi-view render seam for the mesh critic.

The mesh critic judges the actual 3D output, not the pre-lift concept image, so it
needs the character rendered from several angles. Rendering is the one place that
touches a heavy tool (Blender), so it's isolated behind ``render_views`` and only
invoked for the ``vlm``/``combined`` critic. The ``noop`` renderer writes placeholder
image files so the whole loop is testable GPU-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BackendError, Character
from .config import CharacterConfig

# Path to the Blender script we'd drive headless. Kept as data (not shipped yet) so
# the command is centralized + verifiable against your Blender/checkout, like the
# HMR backends' demo commands.
_BLENDER_SCRIPT = "character/scripts/render_views.py"


def render_views(cfg: CharacterConfig, character: Character) -> List[Path]:
    """Render the character to ``cfg.n_views`` images; dispatch on ``cfg.renderer``."""
    out = cfg.render_dir
    out.mkdir(parents=True, exist_ok=True)
    if cfg.renderer == "noop":
        return _noop_views(out, cfg.n_views)
    if cfg.renderer == "blender":
        return _blender_views(cfg, character, out)
    raise BackendError(f"unknown renderer {cfg.renderer!r} (have: 'noop', 'blender')")


def _noop_views(out: Path, n_views: int) -> List[Path]:
    """Placeholder views so the critic path runs without a GPU/Blender."""
    paths = []
    for i in range(max(1, n_views)):
        p = out / f"view_{i:02d}.png"
        p.write_bytes(b"PNG-NOOP")
        paths.append(p)
    return paths


def _blender_views(cfg: CharacterConfig, character: Character, out: Path) -> List[Path]:
    """Drive Blender headless to render N orbit views of the asset. The subprocess is
    built here; the render script itself is verified against your Blender version."""
    import subprocess

    blender = cfg.blender or "blender"
    cmd = [str(blender), "--background", "--python", _BLENDER_SCRIPT, "--",
           "--asset", str(character.asset_path), "--out", str(out), "--views", str(cfg.n_views)]
    try:
        subprocess.run([str(c) for c in cmd], check=True)
    except FileNotFoundError as exc:
        raise BackendError(f"Blender not found (set cfg.blender): {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise BackendError(f"Blender render failed (status {exc.returncode}); verify {_BLENDER_SCRIPT}") from exc
    views = sorted(out.glob("view_*.png"))
    if not views:
        raise BackendError(f"Blender produced no views in {out}; check {_BLENDER_SCRIPT}")
    return views
