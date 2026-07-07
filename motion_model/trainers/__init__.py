"""Motion-model trainers (Pipeline 2).

Each trainer wraps one upstream training tool behind a common interface so the
orchestration layer never has to know which one is in use. Pick with
``MotionModelConfig.method`` -- swapping is one config line, exactly like
Pipeline 1's ``backend``.

Availability of the underlying code was verified 2026-07 (see ARCHITECTURE.md):

Generators (train on HumanML3D 263-d features):
    momask -> https://github.com/EricGuo5513/momask-codes   (CVPR 2024, MIT)  [default pick]
    mdm    -> https://github.com/GuyTevet/motion-diffusion-model
              (+ priorMDM / LoRA-MDM for the fine-tune/personalization mechanics)

Physics controllers (consume the AMASS npz directly -- no HumanML3D step):
    protomotions -> https://github.com/NVlabs/ProtoMotions  (Apache-2.0, maintained)
    closd        -> https://github.com/GuyTevet/CLoSD       (ICLR 2025, MIT; closed-loop A+B)

Testing:
    noop   -> synthetic trainer (no GPU, no weights, no repo)

Licence note: the framework code above is permissive, but the SMPL/SMPL-H body
models every method depends on are registration-gated and non-commercial by
default -- a paid product needs a commercial SMPL licence from Meshcapade.
"""

from __future__ import annotations

from .base import MotionTrainer, TrainerError
from .momask import MoMaskTrainer
from .mdm import MDMTrainer
from .protomotions import ProtoMotionsTrainer
from .closd import CLoSDTrainer
from .noop import NoopTrainer

_REGISTRY = {
    # generators (HumanML3D features)
    "momask": MoMaskTrainer,
    "mdm": MDMTrainer,
    # physics controllers (AMASS npz direct)
    "protomotions": ProtoMotionsTrainer,
    "closd": CLoSDTrainer,
    # testing
    "noop": NoopTrainer,
}


def get_trainer(cfg) -> MotionTrainer:
    """Instantiate the trainer named by ``cfg.method`` (case-insensitive)."""
    name = cfg.method.lower()
    if name not in _REGISTRY:
        raise TrainerError(f"Unknown method {cfg.method!r}. Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[name](cfg)


def available_methods():
    """List registered method names, sorted."""
    return sorted(_REGISTRY)


__all__ = [
    "MotionTrainer",
    "TrainerError",
    "MoMaskTrainer",
    "MDMTrainer",
    "ProtoMotionsTrainer",
    "CLoSDTrainer",
    "NoopTrainer",
    "get_trainer",
    "available_methods",
]
