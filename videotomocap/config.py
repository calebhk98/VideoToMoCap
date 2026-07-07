"""Configuration for the video-to-mocap pipeline.

Config is a plain dataclass so it can be built in code or loaded from a YAML
file.  YAML is optional: if PyYAML is not installed you can still construct a
``PipelineConfig`` directly (the self-test does exactly that).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Video containers we treat as candidate footage during ingestion.
DEFAULT_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts")


@dataclass
class PipelineConfig:
    """All tunable knobs for the pipeline, plain enough to build in code or load from YAML."""

    # --- Ingestion -------------------------------------------------------
    footage_root: Path = Path("footage")
    """Root directory holding one sub-directory per camera (or arbitrary tree)."""

    video_exts: tuple = DEFAULT_VIDEO_EXTS
    """File extensions treated as candidate footage during ``scan`` (case-insensitive)."""

    camera_dir_depth: int = 1
    """How many path components below ``footage_root`` name the camera. With a
    layout of ``footage/cam03/2024-05-01/clip.mp4`` a depth of 1 -> camera 'cam03'."""

    # --- Working directories --------------------------------------------
    work_root: Path = Path("work")
    """Where the manifest and all intermediate artifacts are written."""

    # --- HMR backend -----------------------------------------------------
    backend: str = "gvhmr"
    """A registered backend. Body-only: 'gvhmr','wham','tram'. Whole-body SMPL-X
    (with hands): 'smplestx','whac','osx','hand4whole','multihmr'. Combined:
    'fusion' (body + hand net). Testing: 'noop'."""

    backend_repo: Optional[Path] = None
    """Path to the cloned upstream repo (e.g. the GVHMR checkout)."""

    backend_python: str = "python"
    """Interpreter used to run the backend (often a dedicated conda env)."""

    static_cameras: List[str] = field(default_factory=list)
    """Camera ids known to be fixed/static -> skip visual odometry (GVHMR ``-s``)."""

    partial_body_cameras: List[str] = field(default_factory=list)
    """Shorthand for the common case: cameras that only see the upper body (legs
    out of frame). Equivalent to ``camera_occlusions: {cam: ['legs']}``. Clips are
    tagged ``partial_body`` with the leg joints marked unreliable."""

    camera_occlusions: Dict[str, List[str]] = field(default_factory=dict)
    """Per-camera list of body regions that are out of frame / never reliably
    seen, e.g. ``{cam_desk: ['legs'], cam_high: ['head'], cam_left: ['right_arm']}``.
    Region names are from ``videotomocap.regions.BODY_REGIONS`` (legs, left_arm,
    head, hands, ...). The union of their SMPL joints is recorded per clip as
    ``unreliable_joints`` so you can filter or mask those joints downstream. HMR
    still outputs a full body -- this just labels which joints are guessed."""

    backend_extra_args: List[str] = field(default_factory=list)
    """Extra argv tokens appended verbatim to the backend's demo command."""

    # --- Fusion backend (body + dedicated hand estimator) ---------------
    body_backend: str = "gvhmr"
    """When backend='fusion': which body/whole-body backend supplies the body."""

    hand_backend: str = "wilor"
    """When backend='fusion': which hand specialist supplies fingers ('wilor'|'hamer')."""

    hand_repo: Optional[Path] = None
    """Cloned checkout of the hand tool (WiLoR/HaMeR)."""

    hand_python: Optional[str] = None
    """Interpreter for the hand tool; falls back to backend_python."""

    graft_wrist: bool = False
    """Compose the hand-net wrist into the body chain (advanced; see fusion.py).
    Off by default -- keeps the body's wrist and only grafts finger articulation."""

    # --- Anonymization ---------------------------------------------------
    drop_shape: bool = True
    """Zero out SMPL betas so body identity does not leave the pipeline."""

    keep_translation: bool = True
    """Keep root translation (global trajectory). Rotations alone are gait-agnostic
    in scale but you usually want trajectory for a world-grounded motion model."""

    use_frame: str = "global"
    """'global' (world-grounded) or 'incam' (camera-relative) SMPL params."""

    refine: bool = False
    """Apply post-processing (temporal de-jitter + stationary anti-drift) after
    HMR, before anonymization. Pure-NumPy, no weights; see ``videotomocap.refine``.
    Off by default -- turn on if your backend's output is jittery."""

    # --- Dataset ---------------------------------------------------------
    target_fps: float = 30.0
    """Frame rate every clip is resampled to before anonymization/export, so the
    exported dataset has a uniform rate regardless of source camera fps."""

    min_clip_frames: int = 30
    """Drop motion snippets shorter than this after HMR (too short to be useful)."""

    val_fraction: float = 0.05
    """Fraction of *clips* (not frames) held out for validation -- split by clip id
    to avoid frame leakage between train/val."""

    gender: str = "neutral"
    """SMPL body gender label written into every exported AMASS npz."""

    # --- Parallelism -----------------------------------------------------
    workers: Optional[int] = None
    """How many clips to process concurrently. Clips are independent, so this
    scales near-linearly until GPUs saturate. ``None`` -> auto = detected GPUs x
    ``workers_per_gpu`` (or ``workers_per_gpu`` on a CPU-only box)."""

    gpus: object = field(default_factory=list)
    """GPU ids to spread work across. 'auto' (or ['auto']) detects them; an
    explicit list ['0','1'] pins those; empty inherits the ambient device.
    Workers are assigned round-robin, so >1 worker per GPU packs a single card."""

    workers_per_gpu: object = 1
    """Clips to run concurrently ON EACH GPU. An int pins that many; ``'auto'``
    sizes it from free VRAM (see below) -- fill each card as deep as its memory
    allows. 1 = one clip per card. Higher overlaps one clip's GPU phase with
    another's decode/IO and saturates a fast card, at the cost of more VRAM."""

    vram_per_worker_mb: Optional[int] = None
    """For ``workers_per_gpu: auto`` -- estimated VRAM one clip needs (MiB). If
    None, the runner CALIBRATES by measuring the first clip's peak usage, then
    fills each GPU accordingly. Set it explicitly to skip calibration."""

    vram_headroom_mb: int = 2000
    """Safety margin (MiB) always left free per GPU when auto-sizing workers."""

    max_workers_per_gpu: int = 8
    """Hard cap on auto-sized workers per GPU (guards against over-subscription)."""

    cuda_device: Optional[str] = None
    """Internal: the GPU id assigned to THIS run, exported as CUDA_VISIBLE_DEVICES
    to the backend subprocess. Set per-worker by the parallel runner; not YAML."""

    def __post_init__(self) -> None:
        self.footage_root = Path(self.footage_root)
        self.work_root = Path(self.work_root)
        if self.backend_repo is not None:
            self.backend_repo = Path(self.backend_repo)
        if self.hand_repo is not None:
            self.hand_repo = Path(self.hand_repo)
        if self.use_frame not in ("global", "incam"):
            raise ValueError(f"use_frame must be 'global' or 'incam', got {self.use_frame!r}")

    # Convenience paths -------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        """Path to the manifest (the pipeline's single source of truth)."""
        return self.work_root / "manifest.json"

    @property
    def hmr_dir(self) -> Path:
        """Scratch root for per-clip backend artifacts."""
        return self.work_root / "hmr"

    @property
    def pose_dir(self) -> Path:
        """Where anonymized per-clip ``SmplMotion`` npz files are written."""
        return self.work_root / "pose"

    @property
    def dataset_dir(self) -> Path:
        """Where the aggregated AMASS-format dataset is written."""
        return self.work_root / "dataset"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain (JSON/YAML-friendly) dict, stringifying Paths."""
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def load_config(path: str | Path) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - trivial
        raise RuntimeError(
            "PyYAML is required to load a config file. `pip install pyyaml` "
            "or construct PipelineConfig directly in code."
        ) from exc

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if "video_exts" in raw and raw["video_exts"] is not None:
        raw["video_exts"] = tuple(raw["video_exts"])
    return PipelineConfig(**raw)
