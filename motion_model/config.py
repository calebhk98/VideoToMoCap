"""Configuration for Pipeline 2 -- training a motion model on the dataset.

Mirrors ``videotomocap.config``: a plain dataclass you can build in code or load
from YAML, with one ``method`` knob that selects the trainer (exactly like
Pipeline 1's ``backend``). Swapping method is one config line.

Only the fields relevant to the *selected* method matter; the rest are ignored.
A generator method (momask/mdm) uses the HumanML3D + training knobs; a physics
controller (protomotions/closd) uses the simulator + algorithm knobs.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ProtoMotions ships one experiment file per algorithm; masked_mimic's is a
# transformer, the others an MLP. Single source of truth: the config validates
# `algorithm` against these keys and the ProtoMotions trainer maps them to
# --experiment-path, so the two never drift.
PROTOMOTIONS_EXPERIMENTS = {
    "mimic": "examples/experiments/mimic/mlp.py",
    "amp": "examples/experiments/amp/mlp.py",
    "ase": "examples/experiments/ase/mlp.py",
    "masked_mimic": "examples/experiments/masked_mimic/transformer.py",
}


def _one_of(value, allowed: set, field_name: str) -> str:
    """Validate a string-enum config field, failing loud with the allowed set."""
    v = str(value).lower()
    if v not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {value!r}")
    return v


@dataclass
class MotionModelConfig:
    """All knobs for Pipeline 2, plain enough to build in code or load from YAML."""

    # --- Method selection ------------------------------------------------
    method: str = "noop"
    """Which trainer to run. Generators (HumanML3D features): 'momask' (default
    pick), 'mdm'. Physics controllers (consume AMASS npz directly): 'protomotions',
    'closd'. Testing: 'noop'. Swapping method is this one line."""

    # --- Working directories --------------------------------------------
    dataset_dir: Path = Path("work/dataset")
    """Pipeline 1's output dir -- must contain ``amass/*.npz`` and ``index.json``."""

    work_root: Path = Path("work/motion_model")
    """Where prepared training data and checkpoints for THIS method are written."""

    trainer_python: str = "python"
    """Interpreter used to launch the upstream trainer (usually a dedicated env).
    Named like Pipeline 1's ``backend_python`` for parity."""

    cuda_device: Optional[str] = None
    """Exported as CUDA_VISIBLE_DEVICES to the training subprocess. None = inherit."""

    # --- Upstream repos + assets (only the selected method's are needed) -
    repo: Optional[Path] = None
    """Cloned upstream training repo for the method (MoMask/MDM/ProtoMotions/CLoSD)."""

    tmr_repo: Optional[Path] = None
    """Mathux/TMR checkout -- its ``joints_to_guofeats`` does the offline, byte-exact
    263-d conversion (and ships the reference skeleton, so no gated AMASS clip). Set
    this + ``smpl_model`` and generator ``prepare`` extracts features automatically."""

    humanml3d_repo: Optional[Path] = None
    """Deprecated alias kept for configs that set it; TMR (``tmr_repo``) is the
    offline feature path now. Only used in the printed hand-off when TMR is unset."""

    smpl_model: Optional[Path] = None
    """Neutral SMPL-family body model for the feature-step FK. SMPL-H *or* SMPL-X
    works (only the 22 body joints are used), so point this at the model your HMR
    backend already required -- no separate registration. Registration-gated (one
    free MPI academic sign-up); a *commercial* SMPL licence is required for paid
    output -- the pipeline cannot grant it, so that's your responsibility."""

    resume_checkpoint: Optional[Path] = None
    """Pretrained prior to WARM-START from, then full fine-tune. Do not train a
    >35M model from scratch on hours of one person -- it overfits. See README.md."""

    # --- Generator training (momask / mdm) ------------------------------
    conditioning: str = "none"
    """'none' (unconditional style), 'text' (needs captions in texts/), 'action'
    (uses each clip's ``action_cluster`` label), or 'person' (uses each clip's
    ``person_id`` from Pipeline 1's multi-person export -> promptable 'moves like
    <person>'). For a single person's model, point ``dataset_dir`` at that person's
    ``dataset/by_person/<id>`` sub-dataset instead."""

    personalization: str = "full"
    """'full' fine-tune (what you asked for) or 'lora' (LoRA-MDM adapters, mdm only)."""

    num_steps: int = 80000
    """Training steps. 80k/batch-64/lr-1e-4 mirrors the priorMDM fine-tune recipe."""

    batch_size: int = 64
    """Samples per optimization step. Lower it if the trainer runs out of VRAM."""

    lr: float = 1.0e-4
    """Learning rate. 1e-4 is deliberately low for fine-tuning -- it adapts the
    pretrained prior toward your style without washing it out."""

    target_fps: int = 20
    """HumanML3D + MDM operate at 20 fps. Pipeline 1 should already export at 20;
    values that are not a multiple of 20 break HumanML3D's int(fps/20) decimation."""

    guidance_param: float = 2.5
    """Classifier-free-guidance scale used at sampling time (generator methods)."""

    # --- Overfitting guard (generator methods) --------------------------
    save_every: int = 0
    """Checkpoint interval in steps (0 = trainer default). Frequent checkpoints let
    the `overfit-report` fall back to the best PRE-overfit one instead of the last.
    MDM/CLoSD map it to --save_interval. Fine-tuning a big model on one person is the
    case where the last checkpoint is often not the best -- see motion_model/overfit.py."""

    eval_every: int = 0
    """Evaluate on the held-out val split every N steps (0 = off). Produces the val
    curve `overfit-report` reads to detect the overfitting upturn. MDM/CLoSD map it to
    --eval_during_training --eval_split test (needs the t2m evaluator bundle present)."""

    early_stop_patience: int = 5
    """How many successive val-worsening evals after the minimum count as a real
    overfitting onset (vs curve noise). Used by `overfit-report` AND, when
    ``early_stop`` is on, to actually terminate training at the onset."""

    auto_scale: bool = False
    """Adapt the training regime to the measured corpus (num_steps scaled to data
    volume, personalization/warm-start/method recommended) so ONE config works from
    100 hours to a server's corpus. Applies the safe knobs, prints the rest. See
    motion_model/autoscale.py."""

    early_stop: bool = False
    """Actually stop training at the overfitting onset (not just report it): the run
    is monitored and the trainer subprocess is terminated once the val curve turns up
    for ``early_stop_patience`` evals. Needs ``eval_every`` on so a val curve exists.
    Generator methods only. See motion_model/earlystop.py."""

    early_stop_poll_s: float = 30.0
    """How often (seconds) the early-stop monitor re-reads the val curve while training."""

    # --- Physics controller (protomotions / closd) ----------------------
    simulator: str = "isaaclab"
    """'isaaclab' (maintained) / 'isaacgym' (deprecated) / 'mujoco' (CPU) / 'newton'."""

    algorithm: str = "masked_mimic"
    """ProtoMotions experiment: 'mimic', 'amp', 'ase', or 'masked_mimic'."""

    num_envs: int = 4096
    """Parallel sim environments. Cut this if VRAM is tight (throughput ~linear)."""

    ngpu: int = 1
    """GPUs for one training run (DDP). 2x3090 works without NVLink; Linux only."""

    robot: str = "smpl"
    """Humanoid the controller tracks. 'smpl' matches Pipeline 1's SMPL-H export."""

    # --- Passthrough -----------------------------------------------------
    extra_args: List[str] = field(default_factory=list)
    """Extra argv tokens appended verbatim to the upstream training command."""

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.work_root = Path(self.work_root)
        for name in ("repo", "tmr_repo", "humanml3d_repo", "smpl_model", "resume_checkpoint"):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, Path(val))
        self.conditioning = _one_of(self.conditioning, {"none", "text", "action", "person"}, "conditioning")
        self.personalization = _one_of(self.personalization, {"full", "lora"}, "personalization")
        self.simulator = _one_of(self.simulator, {"isaaclab", "isaacgym", "mujoco", "newton"}, "simulator")
        self.algorithm = _one_of(self.algorithm, set(PROTOMOTIONS_EXPERIMENTS), "algorithm")

    # Convenience paths -------------------------------------------------
    @property
    def amass_dir(self) -> Path:
        """Where Pipeline 1 wrote one AMASS-SMPL-H npz per clip."""
        return self.dataset_dir / "amass"

    @property
    def index_path(self) -> Path:
        """Pipeline 1's clip index (list + train/val split)."""
        return self.dataset_dir / "index.json"

    @property
    def prepared_dir(self) -> Path:
        """Method-specific training data produced by ``prepare`` (features/splits)."""
        return self.work_root / self.method / "prepared"

    @property
    def checkpoint_dir(self) -> Path:
        """Where the upstream trainer writes checkpoints for this run."""
        return self.work_root / self.method / "checkpoints"

    @property
    def metrics_path(self) -> Path:
        """Val curve the early-stop monitor / overfit-report read. The trainer should
        write it here (JSONL/CSV: step, train_loss, val_loss); noop does."""
        return self.checkpoint_dir / "metrics.jsonl"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict, stringifying Paths (JSON/YAML-friendly)."""
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
    """Render a config dataclass as commented YAML: every field + its docstring
    (read from the source, so the template never drifts from the dataclass)."""
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
    """Fully-commented YAML with EVERY MotionModelConfig option + its docs.
    Use via ``python -m motion_model config-template > my.yaml``."""
    return _yaml_template(MotionModelConfig, skip=("cuda_device",))


def load_config(path: str | Path) -> MotionModelConfig:
    """Load a :class:`MotionModelConfig` from a YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - trivial
        raise RuntimeError(
            "PyYAML is required to load a config file. `pip install pyyaml` "
            "or construct MotionModelConfig directly in code."
        ) from exc

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    fields = {f.name for f in dataclasses.fields(MotionModelConfig)}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return MotionModelConfig(**raw)
