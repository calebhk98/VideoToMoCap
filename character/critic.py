"""Mesh critics -- judge the GENERATED 3D character (not the concept image).

Two flavours, because "does this mesh match the prompt" splits into two questions:

  * :class:`GeometricCritic` -- pure-NumPy geometry sanity that needs NO reference and
    NO GPU: is the mesh finite, are its proportions human-plausible, is it left-right
    symmetric? This literally analyses the vertices, catching the classic AI-mesh
    failures (melted/lopsided/degenerate) before you spend anything on a VLM.
  * :class:`VLMCritic` -- renders the mesh to multi-view (see render.py) and scores the
    renders against the prompt with a LOCAL vision model. This is the "compare the mesh
    to what I asked for" semantic check, and because it judges the actual 3D output it
    sees the back-of-head/occluded regions a concept-image critic never would.

``combined`` averages the two. ``noop`` is the GPU-free stand-in for the loop tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import BackendError, Character
from .config import CharacterConfig


@dataclass
class CritiqueResult:
    """A critic's verdict on one character."""

    score: float                       # in [0, 1]; higher = better match/quality
    passed: bool                       # score >= accept threshold
    feedback: str = ""                 # human-readable why
    details: Dict = field(default_factory=dict)


class MeshCritic(ABC):
    """Score a generated character in [0,1]. See module docstring for the flavours."""

    name = "base"

    def __init__(self, cfg: CharacterConfig):
        self.cfg = cfg

    @abstractmethod
    def score(self, character: Character, prompt: str) -> CritiqueResult:
        """Judge ``character`` against ``prompt``; return a CritiqueResult."""

    def _passed(self, score: float) -> bool:
        return score >= self.cfg.accept_score


class NoopCritic(MeshCritic):
    """Deterministic stand-in: returns a fixed/scripted score so the refine loop is
    testable GPU-free. With no ``scores`` it accepts immediately."""

    name = "noop"

    def __init__(self, cfg: CharacterConfig, scores: Optional[List[float]] = None):
        super().__init__(cfg)
        self._scores = list(scores) if scores else None
        self._i = 0

    def score(self, character: Character, prompt: str) -> CritiqueResult:
        if self._scores is None:
            s = self.cfg.accept_score
        else:
            s = self._scores[min(self._i, len(self._scores) - 1)]
            self._i += 1
        return CritiqueResult(s, self._passed(s), "noop critic", {"synthetic": True})


class GeometricCritic(MeshCritic):
    """Pure-NumPy geometry sanity -- finiteness, human-plausible proportions, left-right
    symmetry. Coarse by design: it rejects broken meshes, it doesn't judge beauty."""

    name = "geometric"

    def score(self, character: Character, prompt: str) -> CritiqueResult:
        verts = _load_vertices(character.asset_path)
        if verts is None or not np.isfinite(verts).all() or len(verts) < 4:
            return CritiqueResult(0.0, False, "mesh missing/degenerate/non-finite", {})
        prop, prop_note = _proportion_score(verts)
        sym, sym_note = _symmetry_score(verts)
        s = float(0.5 * prop + 0.5 * sym)
        return CritiqueResult(s, self._passed(s), f"{prop_note}; {sym_note}",
                              {"proportion": prop, "symmetry": sym, "n_verts": int(len(verts))})


class VLMCritic(MeshCritic):
    """Render the mesh multi-view, then score the renders against the prompt with a
    local VLM (subprocess). The command is centralized here; verify it against your
    VLM env. GPU-bound -- exercised only when configured, never in the tests."""

    name = "vlm"

    def score(self, character: Character, prompt: str) -> CritiqueResult:
        from . import render

        views = character.renders or render.render_views(self.cfg, character)
        character.renders = views
        self._require(self.cfg.vlm_python, "vlm_python (env for the local VLM)")
        self._require(self.cfg.vlm_model, "vlm_model (local VLM weights)")
        s, fb = self._run_vlm(views, prompt)
        return CritiqueResult(s, self._passed(s), fb, {"n_views": len(views)})

    def _require(self, path, what):
        if path is None or not Path(path).exists():
            raise BackendError(f"vlm critic needs {what} (got {path!r}).")

    def _run_vlm(self, views: List[Path], prompt: str) -> tuple:
        """Shell out to a local VLM to score views vs the prompt -> (score, feedback).
        Left as the seam to your VLM runner; wire it to Qwen2-VL/InternVL/MiniCPM-V."""
        raise BackendError(
            "vlm critic is not wired to a local VLM yet. Point vlm_python/vlm_model at a "
            "Qwen2-VL/InternVL/MiniCPM-V runner that scores the rendered views against the "
            "prompt (0-1). Until then use critic: geometric (no VLM) or noop.")


class CombinedCritic(MeshCritic):
    """Average of the geometric and VLM scores -- geometry sanity AND semantic match."""

    name = "combined"

    def score(self, character: Character, prompt: str) -> CritiqueResult:
        g = GeometricCritic(self.cfg).score(character, prompt)
        v = VLMCritic(self.cfg).score(character, prompt)
        s = float(0.5 * g.score + 0.5 * v.score)
        return CritiqueResult(s, self._passed(s), f"geom={g.score:.2f}, vlm={v.score:.2f}",
                              {"geometric": g.details, "vlm": v.details})


def get_critic(cfg: CharacterConfig) -> MeshCritic:
    """Instantiate the critic named by ``cfg.critic``."""
    table = {"noop": NoopCritic, "geometric": GeometricCritic, "vlm": VLMCritic, "combined": CombinedCritic}
    return table[cfg.critic](cfg)


# -- pure-NumPy geometry helpers -------------------------------------------

def _load_vertices(asset_path: Path):
    """Vertices (N,3) for the geometric critic. Reads an npz with 'vertices' directly
    (test/synthetic path); otherwise lazy-imports trimesh for a real mesh file."""
    p = Path(asset_path)
    if p.suffix == ".npz":
        data = np.load(p)
        return np.asarray(data["vertices"], dtype=np.float64) if "vertices" in data else None
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - trivial
        raise BackendError("geometric critic needs trimesh to read a mesh file (`pip install trimesh`).") from exc
    mesh = trimesh.load(str(p), force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64)


def _axes_by_extent(verts: np.ndarray):
    """Order the 3 axes by extent: (up=largest, width=middle, depth=smallest)."""
    ranges = verts.max(0) - verts.min(0)
    order = np.argsort(ranges)[::-1]
    return int(order[0]), int(order[1]), int(order[2]), ranges


def _proportion_score(verts: np.ndarray):
    """Human standing figures are tall: height >> width ~ depth. Score how human the
    bbox aspect ratio is (peaks around 3-4x, tolerant across [1.5, 7])."""
    up, width, _depth, ranges = _axes_by_extent(verts)
    tall = ranges[up] / max(ranges[width], 1e-9)
    score = float(np.clip(1.0 - abs(tall - 3.5) / 3.5, 0.0, 1.0))
    return score, f"height/width={tall:.1f} (human ~3.5)"


def _symmetry_score(verts: np.ndarray, sample: int = 800):
    """Left-right symmetry: mirror across the width-axis median and measure how far
    each (sampled) vertex is from its nearest mirrored neighbour, normalized by size."""
    _up, width, _depth, ranges = _axes_by_extent(verts)
    diag = float(np.linalg.norm(ranges)) or 1.0
    idx = np.linspace(0, len(verts) - 1, min(sample, len(verts))).astype(int)
    pts = verts[idx]
    mirrored = pts.copy()
    center = float(np.median(verts[:, width]))
    mirrored[:, width] = 2.0 * center - mirrored[:, width]
    # nearest mirrored neighbour distance, in chunks to bound memory
    dmin = _nearest_dist(pts, mirrored)
    score = float(np.clip(1.0 - (dmin.mean() / (0.1 * diag)), 0.0, 1.0))
    return score, f"symmetry residual={dmin.mean() / diag:.3f} of size"


def _nearest_dist(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Min distance from each row of ``a`` to any row of ``b`` (chunked)."""
    out = np.empty(len(a))
    for s in range(0, len(a), chunk):
        block = a[s:s + chunk]
        d = np.linalg.norm(block[:, None, :] - b[None, :, :], axis=2)
        out[s:s + chunk] = d.min(axis=1)
    return out
