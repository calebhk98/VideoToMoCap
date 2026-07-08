"""Configuration for Pipeline 3 -- caption a personal video archive + index it.

Mirrors ``videotomocap.config`` and ``motion_model.config``: a plain dataclass
you can build in code or load from YAML, with role-selecting knobs (``captioner``,
``aggregator``) exactly like Pipeline 1's ``backend``. Swapping a model is one
config line. Only the fields relevant to the selected models matter.

The heavy models (JoyCaption VLM, Dolphin LLM, the Qwen2.5-VL fine-tune) run in
their own environments behind subprocess bridges; this module stays importable
with just the standard library (+ optional PyYAML for ``load_config``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Video containers treated as candidate footage during ingestion (matches
# Pipeline 1's default set so both pipelines see the same archive).
DEFAULT_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts")


def _coerce_bool(value) -> bool:
    """YAML parses bare off/on/yes/no as booleans already; be tolerant of strings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _one_of(value, allowed: set, field_name: str) -> str:
    """Validate a string-enum config field, failing loud with the allowed set."""
    v = str(value).lower()
    if v not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {value!r}")
    return v


@dataclass
class CaptionConfig:
    """All knobs for Pipeline 3, plain enough to build in code or load from YAML."""

    # --- Ingestion -------------------------------------------------------
    footage_root: Path = Path("footage")
    """Root directory holding the personal video archive (arbitrary tree)."""

    video_exts: tuple = DEFAULT_VIDEO_EXTS
    """File extensions treated as candidate footage during ``scan`` (case-insensitive)."""

    work_root: Path = Path("work/caption")
    """Where the manifest, per-video segment/label artifacts, and index are written."""

    limit: Optional[int] = None
    """Optional cap on videos processed per ``caption``/``run`` (mostly smoke tests)."""

    exclude_patterns: List[str] = field(default_factory=list)
    """Glob patterns (against each video's relative path) auto-excluded on ``scan``."""

    exclude_ids: List[str] = field(default_factory=list)
    """Specific video ids to auto-exclude on ``scan`` (companion to patterns)."""

    # --- Step 1: Segmentation -------------------------------------------
    scene_detect: bool = True
    """Cut on natural scene boundaries with PySceneDetect (lazy import). When off
    -- or when PySceneDetect isn't installed -- fall back to fixed-interval cuts of
    ``max_segment_seconds`` over the whole video."""

    scene_threshold: float = 27.0
    """PySceneDetect ContentDetector threshold. Lower = more (shorter) scenes."""

    max_segment_seconds: float = 120.0
    """Hard cap on caption-segment length (the 2-min default). Any detected scene
    longer than this is sub-chunked at fixed intervals, so no segment ever exceeds
    what's practical for frame extraction / captioning / a fine-tune example."""

    min_segment_seconds: float = 3.0
    """Detected scenes shorter than this are merged into an adjacent scene before
    sub-chunking, so rapid-cut sequences don't yield a flood of 1-frame segments."""

    # --- Step 2: Frame extraction ---------------------------------------
    frame_sample_fps: float = 1.0
    """Frames sampled per second within a segment (1 fps -> ~60-120 frames for a
    1-2 min segment). Tune against the throughput benchmark for your archive."""

    frame_extractor: str = "ffmpeg"
    """How frames are decoded: 'ffmpeg' (subprocess) or 'decord' (lazy import)."""

    max_frames_per_segment: int = 150
    """Ceiling on frames captioned per segment, so a long segment at a high sample
    rate can't blow up VRAM/time. Sampling is thinned evenly to fit."""

    # --- Step 3: Captioning (per-frame VLM) -----------------------------
    captioner: str = "joycaption"
    """Registered per-frame captioner. 'joycaption' (JoyCaption via vLLM),
    'noop' (synthetic, tests). Selects the Step 3 model in one line."""

    captioner_model: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    """HF model id the captioner serves (default: the JoyCaption beta)."""

    captioner_python: str = "python"
    """Interpreter that runs the captioner (usually a dedicated vLLM env)."""

    captioner_repo: Optional[Path] = None
    """Optional cloned helper repo / server dir for the captioner, if it needs one."""

    caption_prompt: str = "Write a detailed description of this image."
    """Instruction sent to the VLM for every frame."""

    # --- Step 4: Aggregation (caption -> structured label) --------------
    aggregator: str = "dolphin"
    """Registered text aggregator that folds the frame captions into one
    (description, tags) label. 'dolphin' (Dolphin3.0 LLM), 'noop' (synthetic)."""

    aggregator_model: str = "cognitivecomputations/Dolphin3.0-Llama3.1-8B"
    """HF model id the aggregator serves (default: the Dolphin3.0 instruct tune)."""

    aggregator_python: str = "python"
    """Interpreter that runs the aggregator (its own vLLM/transformers env)."""

    aggregator_repo: Optional[Path] = None
    """Optional cloned helper repo / server dir for the aggregator, if it needs one."""

    num_tags: int = 12
    """Target number of keyword/entity tags per segment label."""

    caption_focus: str = "scene"
    """What the Step 4 label should describe: 'scene' (appearance/setting -- who and
    where, the default) or 'motion' (body movement and actions -- 'walks forward,
    reaches up, turns left'). 'motion' steers the aggregator toward the verbs a
    text-to-motion model wants, so the caption<->motion pairing bridge produces
    sharper training targets. Frame captions (Step 3) are unchanged either way --
    only how they're synthesized differs."""

    # --- Step 5: Search index -------------------------------------------
    index_format: str = "both"
    """How the (video_id, start, end, description, tags) rows are stored: 'sqlite'
    (queryable, FTS if available), 'json' (portable), or 'both' (default)."""

    # --- Step 5.5: Training/inference windows ---------------------------
    window_seconds: float = 600.0
    """Larger grouping (the ~10-min default) that several caption segments are
    packed into -- the unit the fine-tuned video model actually consumes/produces
    a full timestamped label list for. Distinct from ``max_segment_seconds`` (the
    caption unit). Pin against memory testing on your hardware."""

    # --- Step 6: Video-native fine-tune (bridge; runs upstream) ---------
    finetune_base: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    """Video-native base model LoRA-fine-tuned on the bootstrap dataset (Apache-2.0)."""

    finetune_repo: Optional[Path] = None
    """Cloned upstream trainer that does the Qwen2.5-VL LoRA run (e.g. LLaMA-Factory)."""

    finetune_python: str = "python"
    """Interpreter for the fine-tune env (peft/transformers/accelerate)."""

    lora_rank: int = 128
    """LoRA rank. Start 128-256 targeting attention+MLP broadly -- a capability
    fine-tune (teach descriptive vocabulary), not a light style touch-up."""

    lora_alpha: int = 256
    """LoRA alpha (scaling). Convention: ~2x rank."""

    finetune_epochs: int = 3
    """Passes over the bootstrap dataset."""

    merge_adapter: bool = True
    """After training, merge the LoRA adapter into the base weights
    (peft ``merge_and_unload``) so the distributed model is one standalone repo."""

    # --- GPU / parallelism ----------------------------------------------
    gpus: object = field(default_factory=list)
    """GPU ids to spread work across. 'auto' (or ['auto']) detects them; an
    explicit list ['0','1'] pins those; empty inherits the ambient device. With
    2x3090 the captioner runs as one independent instance per card."""

    workers_per_gpu: int = 1
    """Videos processed concurrently per GPU. Each holds a captioner/aggregator
    instance, so raise only if a single card has spare VRAM."""

    cuda_device: Optional[str] = None
    """Internal: the GPU id assigned to THIS run, exported as CUDA_VISIBLE_DEVICES
    to the model subprocess. Set per-worker by the parallel runner; not YAML."""

    def __post_init__(self) -> None:
        self.footage_root = Path(self.footage_root)
        self.work_root = Path(self.work_root)
        for name in ("captioner_repo", "aggregator_repo", "finetune_repo"):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, Path(val))
        self.video_exts = tuple(self.video_exts)
        self.scene_detect = _coerce_bool(self.scene_detect)
        self.merge_adapter = _coerce_bool(self.merge_adapter)
        self.frame_extractor = _one_of(self.frame_extractor, {"ffmpeg", "decord"}, "frame_extractor")
        self.index_format = _one_of(self.index_format, {"sqlite", "json", "both"}, "index_format")
        self.caption_focus = _one_of(self.caption_focus, {"scene", "motion"}, "caption_focus")
        if self.max_segment_seconds <= 0:
            raise ValueError(f"max_segment_seconds must be > 0, got {self.max_segment_seconds}")
        if self.window_seconds < self.max_segment_seconds:
            raise ValueError(
                "window_seconds (the model's input unit) must be >= max_segment_seconds "
                f"(the caption unit); got {self.window_seconds} < {self.max_segment_seconds}"
            )

    # --- Convenience paths ----------------------------------------------
    @property
    def manifest_path(self) -> Path:
        """Path to the manifest (this pipeline's single source of truth)."""
        return self.work_root / "manifest.json"

    @property
    def frames_dir(self) -> Path:
        """Scratch root for per-segment extracted frames."""
        return self.work_root / "frames"

    @property
    def segments_dir(self) -> Path:
        """Where each video's segmentation (one json per video) is written."""
        return self.work_root / "segments"

    @property
    def labels_dir(self) -> Path:
        """Where each video's per-segment (description, tags) labels are written."""
        return self.work_root / "labels"

    @property
    def index_dir(self) -> Path:
        """Where the aggregated search index (sqlite/json) is written."""
        return self.work_root / "index"

    @property
    def finetune_dir(self) -> Path:
        """Where the bootstrap fine-tune dataset + adapter/checkpoints are written."""
        return self.work_root / "finetune"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain (JSON/YAML-friendly) dict, stringifying Paths."""
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def load_config(path: str | Path) -> CaptionConfig:
    """Load a :class:`CaptionConfig` from a YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - trivial
        raise RuntimeError(
            "PyYAML is required to load a config file. `pip install pyyaml` "
            "or construct CaptionConfig directly in code."
        ) from exc

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    fields = {f.name for f in dataclasses.fields(CaptionConfig)}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if "video_exts" in raw and raw["video_exts"] is not None:
        raw["video_exts"] = tuple(raw["video_exts"])
    return CaptionConfig(**raw)
