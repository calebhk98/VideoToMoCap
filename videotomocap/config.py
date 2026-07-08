"""Configuration for the video-to-mocap pipeline.

Config is a plain dataclass so it can be built in code or loaded from a YAML
file.  YAML is optional: if PyYAML is not installed you can still construct a
``PipelineConfig`` directly (the self-test does exactly that).
"""

from __future__ import annotations

import dataclasses
import json
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


def _one_of_ci(value, allowed: set, field_name: str) -> str:
    """Validate a plain string-enum config field (case-insensitive), failing loud."""
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
    """A registered backend. Body-only: 'gvhmr','wham','tram','trace','hmr2'.
    Whole-body SMPL-X (with hands): 'smplestx','camenduru_smplerx','hybrik','whac',
    'osx','hand4whole','multihmr','sam3dbody'. Combined: 'fusion' (body + hand
    net). Testing: 'noop'."""

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

    # --- Multi-person / identity (opt-in) --------------------------------
    multi_person: bool = False
    """Recover EVERY person in each clip (not just the dominant track) and carry
    them through as separate per-person tracks. Off = the original single-subject
    behavior (one motion per clip), byte-for-byte. Requires a backend that returns
    multiple tracks (``run_tracks``); most already track everyone and just discard
    all but one -- see the backend notes. Turning this on enables the identity +
    consent machinery below."""

    person_assignment: str = "shape"
    """How per-clip tracks are assigned to people across the corpus (multi_person
    only): 'shape' (cluster on SMPL betas -- needs biometric consent, see below),
    'manual' (operator labels tracks via the CLI; no biometrics computed), or
    'single' (every clip is assumed to be the same one person)."""

    max_people: int = 0
    """Expected number of distinct people for 'shape' assignment. 0 = auto (pick
    the cluster count from the data). Set it when you know the family size -- it is
    more reliable than auto and the recommended setting for a known household."""

    shape_gap_ratio: float = 3.0
    """Auto (``max_people: 0``) sensitivity: two people are split only when the jump
    in body-shape distance between them is at least this many times the within-person
    spread. Higher = more conservative (fewer, surer identities -- a single person's
    natural jitter never splits); lower = splits more eagerly. Ignored when
    ``max_people`` is set."""

    consent_required: bool = True
    """Gate the dataset on per-person consent: a track only exports if its assigned
    person's consent is granted (see ``people.yaml``). Non-consenting / unassigned
    tracks are handled by ``unassigned_policy``. Keep this on for real footage."""

    unassigned_policy: str = "exclude"
    """What to do at ``build`` with a track that has no consenting person assigned:
    'exclude' (fail-closed -- drop it; the safe default) or 'include' (keep it, for
    a single-consenting-user project where assignment is moot)."""

    synthetic_people: int = 1
    """Testing only: how many synthetic people the ``noop`` backend fabricates per
    clip (each with a distinct fake shape). Lets the whole multi-person flow run
    GPU-free. Ignored by real backends."""

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
    """De-jitter method when ``refine`` is on: 'savgol' (fast uniform local fit),
    'variational' (global acceleration-penalized smoother -- the HTD-Refine
    objective solved directly; stronger, slightly slower), or 'confidence'
    (per-joint adaptive: smooths each joint in proportion to its own local jitter,
    so inferred/occluded joints get denoised hard while clean ones stay sharp)."""

    confidence_kappa: float = 0.02
    """For ``refine_method: confidence`` -- the per-joint acceleration (rad/frame^2)
    at which a joint receives half its maximum smoothing. Lower = smooth more
    aggressively (more joints treated as noisy); higher = only the jerkiest joints
    are touched. Ignored by the other refine methods."""

    # --- Learned refinement (refine_method: dposer) ----------------------
    # Heavy, opt-in: the DPoser-X pose prior runs in its OWN env as a subprocess
    # (see videotomocap/refine_learned.py). Unused unless refine_method='dposer'.
    dposer_repo: Optional[Path] = None
    """Cloned DPoser-X checkout (moonbow721/DPoser-X). Required for the 'dposer'
    refine method; the bridging driver runs with this as its working directory."""

    dposer_python: Optional[str] = None
    """Interpreter for the DPoser-X env (its deps pin torch 1.12.1 / CUDA 11.3).
    Falls back to 'python' on PATH."""

    dposer_config: str = "configs/body/subvp/timefc.py"
    """DPoser-X model config (relative to ``dposer_repo``) selecting the prior."""

    dposer_strength: float = 1.0
    """Blend of the denoised pose vs the original for 'dposer' (0 = off/no-op,
    1 = fully replace with the prior's output)."""

    scorehmr_repo: Optional[Path] = None
    """Cloned ScoreHMR checkout (statho/ScoreHMR). Required for the 'scorehmr'
    refine method. Unlike 'dposer', ScoreHMR is image-guided -- it re-reads the
    source video -- so it only runs during the ``hmr`` stage."""

    scorehmr_python: Optional[str] = None
    """Interpreter for the ScoreHMR env. Falls back to 'python' on PATH."""

    scorehmr_strength: float = 1.0
    """Blend of ScoreHMR's refined pose vs the original (0 = off/no-op, 1 = full)."""

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
        if self.dposer_repo is not None:
            self.dposer_repo = Path(self.dposer_repo)
        if self.scorehmr_repo is not None:
            self.scorehmr_repo = Path(self.scorehmr_repo)
        if self.use_frame not in ("global", "incam"):
            raise ValueError(f"use_frame must be 'global' or 'incam', got {self.use_frame!r}")
        # YAML parses bare off/on/yes/no as booleans, so `auto_mirror: off` arrives
        # as False -- coerce these mode fields back to their string values.
        self.auto_mirror = _coerce_mode(self.auto_mirror, {"off", "flag", "correct"}, "auto_mirror")
        self.quality_filter = _coerce_mode(self.quality_filter, {"off", "flag", "exclude"}, "quality_filter")
        self.auto_occlusion = _coerce_mode(self.auto_occlusion, {"off", "flag"}, "auto_occlusion")
        self.auto_camera_motion = _coerce_mode(self.auto_camera_motion, {"off", "flag"}, "auto_camera_motion")
        if isinstance(self.multi_person, str):
            self.multi_person = self.multi_person.strip().lower() in ("1", "true", "yes", "on")
        self.person_assignment = _one_of_ci(self.person_assignment, {"shape", "manual", "single"}, "person_assignment")
        self.unassigned_policy = _one_of_ci(self.unassigned_policy, {"exclude", "include"}, "unassigned_policy")

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

    @property
    def identity_dir(self) -> Path:
        """Per-track SMPL betas kept for cross-clip identity (multi_person only).

        This is the ONE place body shape is retained -- consent-gated, and never
        copied into the exported dataset (the AMASS export stays shape-neutral, as
        the self-test still asserts). It lives here so identity assignment is
        re-runnable without re-running HMR."""
        return self.work_root / "identity"

    @property
    def people_path(self) -> Path:
        """The people registry + consent ledger (``people.json``; stdlib, hand-editable)."""
        return self.work_root / "people.json"

    @property
    def consent_log_path(self) -> Path:
        """Append-only audit trail of consent/assignment changes."""
        return self.work_root / "consent_log.jsonl"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain (JSON/YAML-friendly) dict, stringifying Paths."""
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def _yaml_scalar(v) -> str:
    """Render a Python default as a valid YAML (JSON-subset) scalar."""
    if isinstance(v, Path):
        v = str(v)
    if isinstance(v, tuple):
        v = list(v)
    try:
        return json.dumps(v)
    except TypeError:
        return json.dumps(str(v))


def _yaml_template(cls, skip=()) -> str:
    """Render a config dataclass as commented YAML: every field + its docstring.

    Reads the attribute docstrings from the source so the template is always
    complete and self-documenting -- it can't drift from the dataclass.
    """
    import ast
    import inspect
    import textwrap

    body = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0].body
    docs = {}
    for i, node in enumerate(body):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        nxt = body[i + 1] if i + 1 < len(body) else None
        if isinstance(nxt, ast.Expr) and isinstance(getattr(nxt, "value", None), ast.Constant) \
                and isinstance(nxt.value.value, str):
            docs[node.target.id] = nxt.value.value

    out = []
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        if f.default is not dataclasses.MISSING:
            val = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            val = f.default_factory()
        else:
            val = None
        for line in textwrap.wrap(" ".join(docs.get(f.name, "").split()), 76):
            out.append(f"# {line}")
        out.append(f"{f.name}: {_yaml_scalar(val)}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def config_template() -> str:
    """A fully-commented YAML template with EVERY PipelineConfig option + its docs.

    ``cuda_device`` is omitted -- it is set per-run by the parallel runner, not YAML.
    Use via ``python -m videotomocap config-template > my.yaml``.
    """
    return _yaml_template(PipelineConfig, skip=("cuda_device",))


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
