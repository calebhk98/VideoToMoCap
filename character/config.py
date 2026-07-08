"""Configuration for Pipeline 3 -- generating an invented character to drive.

Mirrors ``motion_model.config``: a plain dataclass you build in code or load from
YAML, with one ``method`` knob that selects the generation backend (exactly like
Pipeline 1's ``backend`` and Pipeline 2's ``method``). Swapping the tool -- IDOL,
LHM, MakeHuman, SO-SMPL, ... -- is one config line, so you can A/B the whole open
zoo by editing the YAML.

Only the fields relevant to the *selected* method matter; the rest are ignored. A
feed-forward lifter (idol/lhm/pshuman) uses ``input_image`` + ``repo``/``weights``;
a parametric generator (makehuman/mpfb2) uses its own params; the critic-loop knobs
apply to any method.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _one_of(value, allowed: set, field_name: str) -> str:
    v = str(value).lower()
    if v not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {value!r}")
    return v


@dataclass
class CharacterConfig:
    """All knobs for Pipeline 3, plain enough to build in code or load from YAML."""

    # --- Method selection ------------------------------------------------
    method: str = "noop"
    """Which generation backend to run. Feed-forward SMPL-X lifters (image->avatar):
    'idol', 'lhm', 'pshuman', 'human3diffusion'. Clothed/implicit: 'sifu', 'econ',
    'icon', 'anigs'. Generative/parametric: 'en3d', 'so_smpl', 'makehuman', 'mpfb2',
    'mblab'. Testing: 'noop'. Swapping the tool is this one line."""

    # --- What to make ----------------------------------------------------
    prompt: str = ""
    """Text description of the character (e.g. 'a high elf ranger, freckles')."""

    input_image: Optional[Path] = None
    """Concept image for image-native lifters (idol/lhm/pshuman/...). Generate one
    from ``prompt`` with an image backend first, or point at an existing file."""

    # --- Working dirs + upstream assets ---------------------------------
    work_root: Path = Path("work/character")
    """Where each method's generated assets + critic renders are written."""

    backend_python: str = "python"
    """Interpreter used to launch the upstream tool (usually a dedicated env)."""

    repo: Optional[Path] = None
    """Cloned upstream repo for the method (IDOL/LHM/PSHuman/...)."""

    weights: Optional[Path] = None
    """Model checkpoint dir/file for the method, when it needs one separately."""

    smpl_model: Optional[Path] = None
    """SMPL-X body model, for methods that fit/animate against it (registration-gated;
    reuse the one your HMR backend already required)."""

    cuda_device: Optional[str] = None
    """Exported as CUDA_VISIBLE_DEVICES to the subprocess. None = inherit."""

    # --- Critic loop (any method) ---------------------------------------
    critic: str = "noop"
    """Judge the GENERATED MESH against the prompt: 'noop', 'geometric' (pure-NumPy
    geometry sanity: manifold/symmetry/proportion/fragments -- no reference needed),
    'vlm' (render the mesh to multi-view, score with a local VLM), or 'combined'."""

    renderer: str = "noop"
    """How the vlm/combined critic renders the mesh to views: 'noop' or 'blender'."""

    blender: Optional[str] = None
    """Blender executable for the 'blender' renderer (headless `--background`)."""

    vlm_python: Optional[Path] = None
    """Interpreter/env for the local VLM critic (Qwen2-VL/InternVL/MiniCPM-V)."""

    vlm_model: Optional[Path] = None
    """Local VLM weights dir for the 'vlm' critic."""

    n_views: int = 4
    """How many views to render for the vlm critic (front/side/back/…)."""

    max_attempts: int = 3
    """Refine loop: regenerate up to this many times to beat ``accept_score``."""

    n_candidates: int = 1
    """Best-of-N generations per attempt; the critic keeps the best."""

    accept_score: float = 0.75
    """Critic score in [0,1] at/above which the character is accepted and the loop stops."""

    seed: int = 0
    """Base RNG seed; the loop offsets it per attempt so refinement isn't identical."""

    # --- Passthrough -----------------------------------------------------
    extra_args: List[str] = field(default_factory=list)
    """Extra argv tokens appended verbatim to the upstream generation command."""

    def __post_init__(self) -> None:
        self.work_root = Path(self.work_root)
        for name in ("input_image", "repo", "weights", "smpl_model", "blender",
                     "vlm_python", "vlm_model"):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, Path(val))
        self.critic = _one_of(self.critic, {"noop", "geometric", "vlm", "combined"}, "critic")
        self.renderer = _one_of(self.renderer, {"noop", "blender"}, "renderer")

    # Convenience paths -------------------------------------------------
    @property
    def method_dir(self) -> Path:
        """Per-method output root: assets, renders, and the run log live here."""
        return self.work_root / self.method

    @property
    def asset_dir(self) -> Path:
        """Where the generated 3D character asset(s) are written."""
        return self.method_dir / "asset"

    @property
    def render_dir(self) -> Path:
        """Where the critic's multi-view renders are written."""
        return self.method_dir / "renders"

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def load_config(path: str | Path) -> CharacterConfig:
    """Load a :class:`CharacterConfig` from YAML, rejecting unknown keys loudly."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - trivial
        raise RuntimeError("PyYAML is required to load a config file (`pip install pyyaml`).") from exc

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    fields = {f.name for f in dataclasses.fields(CharacterConfig)}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return CharacterConfig(**raw)
