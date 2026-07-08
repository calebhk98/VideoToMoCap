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


def _coerce_mode(value, allowed: set, field_name: str) -> str:
    """Normalize a string-enum config field, tolerating YAML's off/on booleans.

    `off` -> the "off" mode; `on`/True -> "flag" (on-but-non-destructive). Any
    other value must already be one of ``allowed`` or it's a loud error.
    """
    if isinstance(value, bool):
        return "flag" if value else "off"
    v = str(value).lower()
    if v not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {value!r}")
    return v


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

    exclude_patterns: List[str] = field(default_factory=list)
    """Glob patterns (against each clip's relative path) auto-excluded on ``scan``
    -- the set-once-and-forget home for the family-visit ranges, so you never
    retype ``exclude --pattern``. Manual ``include`` still wins and survives
    re-scans."""

    exclude_ids: List[str] = field(default_factory=list)
    """Specific clip ids to auto-exclude on ``scan`` (companion to patterns)."""

    # --- Working directories --------------------------------------------
    work_root: Path = Path("work")
    """Where the manifest and all intermediate artifacts are written."""

    limit: Optional[int] = None
    """Optional cap on clips processed per ``hmr``/``run`` (mostly for smoke
    tests). ``--limit`` overrides it. None = process everything."""

    # --- HMR backend -----------------------------------------------------
    backend: str = "gvhmr"
    """A registered backend. Body-only: 'gvhmr','wham','tram'. Whole-body SMPL-X
    (with hands): 'smplestx','camenduru_smplerx','whac','osx','hand4whole',
    'multihmr'. Combined: 'fusion' (body + hand net). Testing: 'noop'."""

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

    refine_method: str = "savgol"
    """De-jitter method when ``refine`` is on: 'savgol' (fast local fit) or
    'variational' (global acceleration-penalized smoother -- the HTD-Refine
    objective solved directly; stronger, slightly slower)."""

    auto_mirror: str = "off"
    """Automatic left/right-mirror handling for flipped (e.g. selfie) footage,
    detected corpus-relative from handedness (no per-video tags). One of:
    'off' (default), 'flag' (annotate suspected clips in the manifest, no data
    change), 'correct' (also flip flagged clips' pose so handedness is fixed)."""

    mirror_margin: float = 0.15
    """Confidence margin for ``auto_mirror`` -- how far a clip's handedness must
    oppose the corpus consensus before it's flagged. Higher = fewer, surer flags."""

    # --- Automatic quality filtering (from the recovered motion) ---------
    quality_filter: str = "flag"
    """Auto-detect degenerate/implausible clips from the motion itself. One of:
    'off', 'flag' (annotate ``quality_issues`` in the manifest; default), 'exclude'
    (also drop hard-failing clips from the dataset). Pure NumPy, no extra deps."""

    quality_max_speed_ms: float = 12.0
    """Root speed above this (m/s) is a tracking teleport, not human locomotion."""

    quality_max_joint_step: float = 1.5
    """Per-frame joint rotation jump above this (radians) is a tracking glitch."""

    quality_min_motion: float = 1.0e-3
    """Mean per-frame pose change below this = a near-static clip (low training
    value); flagged as info, not auto-excluded."""

    # --- Automatic occlusion tagging (from motion) -----------------------
    auto_occlusion: str = "off"
    """Automatically mark joints that never move across a clip as unreliable (a
    proxy for out-of-frame joints, which HMR freezes) -> they get masked out via
    joint_valid. 'off' or 'flag'. Complements manual ``camera_occlusions``."""

    # --- Unsupervised action pseudo-labels -------------------------------
    cluster_actions: int = 0
    """If > 0, k-means the clips into this many motion clusters and tag each with
    an ``action_cluster`` id (for conditioning the motion model). 0 = off."""

    # --- Raw-video pre-analysis (needs OpenCV; lazy) ---------------------
    skip_empty: bool = False
    """Sample each clip and skip HMR on ones with no activity (empty security
    footage) -- excluded with note 'empty'. Needs opencv-python."""

    empty_activity_threshold: float = 0.01
    """Mean inter-frame difference (0-1) below this = an empty/static clip."""

    auto_camera_motion: str = "off"
    """Auto-detect static vs moving cameras from raw video (global pixel shift) to
    skip visual odometry on locked-off cams. 'off' or 'flag'. Needs opencv-python."""

    camera_motion_threshold: float = 2.0
    """Median global pixel shift below this = a static camera (skip VO)."""

    # --- Dataset ---------------------------------------------------------
    target_fps: float = 20.0
    """Frame rate every clip is resampled to before anonymization/export, so the
    exported dataset has a uniform rate regardless of source camera fps.

    Defaults to 20 to match the downstream motion-model stack: HumanML3D (and the
    MDM prior trained on it) operate at 20 fps and decimate AMASS with a naive
    ``int(source_fps / 20)`` stride. Any value that is NOT a multiple of 20 makes
    that stride truncate wrong -- e.g. 30 fps gives ``int(30/20) == 1`` (no
    downsampling), so clips would silently train 1.5x too fast. Keep this at 20
    (or another multiple of 20) unless a different downstream consumer needs it."""

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
        # YAML parses bare off/on/yes/no as booleans, so `auto_mirror: off` arrives
        # as False -- coerce these mode fields back to their string values.
        self.auto_mirror = _coerce_mode(self.auto_mirror, {"off", "flag", "correct"}, "auto_mirror")
        self.quality_filter = _coerce_mode(self.quality_filter, {"off", "flag", "exclude"}, "quality_filter")
        self.auto_occlusion = _coerce_mode(self.auto_occlusion, {"off", "flag"}, "auto_occlusion")
        self.auto_camera_motion = _coerce_mode(self.auto_camera_motion, {"off", "flag"}, "auto_camera_motion")

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
