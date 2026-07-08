"""Pipeline 3 -- generate an invented character to drive with the motion model.

Each backend wraps one open character-generation tool behind a common interface so
the orchestration/critic layer never has to know which is in use. Pick with
``CharacterConfig.method`` -- swapping the tool is one config line, exactly like
Pipeline 1's ``backend`` and Pipeline 2's ``method``.

The seam that matters: backends whose ``native_smplx`` is True (IDOL/LHM/PSHuman/…)
produce SMPL-X-native avatars that Pipeline 2's motion drives with NO retargeting.
Parametric tools (MakeHuman/MPFB2) emit their own rig and need a retarget step.

Availability verified 2026-07 (see character/README.md). Tools with no public code
yet (AniGS, HumanOrbit) are registered as blocked stubs that fail loud.
"""

from __future__ import annotations

from .base import BackendError, Character, CharacterBackend
from .config import CharacterConfig, load_config
from .noop import NoopBackend

# Real backends are added to the registry as their adapters land. Import lazily
# inside the registry build so a missing optional adapter never breaks `import`.
_REGISTRY = {"noop": NoopBackend}


def _register_optional() -> None:
    """Add each real-tool group if its module is present (groups land incrementally)."""
    import importlib

    for modname in ("lifters", "reconstruct", "generative", "blocked"):
        try:
            mod = importlib.import_module(f".{modname}", __package__)
        except ImportError:
            continue
        _REGISTRY.update(getattr(mod, "BACKENDS", {}))


def get_backend(cfg: CharacterConfig) -> CharacterBackend:
    """Instantiate the backend named by ``cfg.method`` (case-insensitive)."""
    _ensure_registry()
    name = cfg.method.lower()
    if name not in _REGISTRY:
        raise BackendError(f"Unknown method {cfg.method!r}. Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[name](cfg)


def available_methods():
    """List registered method names, sorted."""
    _ensure_registry()
    return sorted(_REGISTRY)


_REGISTRY_READY = False


def _ensure_registry() -> None:
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return
    try:
        _register_optional()
    except ImportError:
        pass  # adapters not present yet -> noop still works
    _REGISTRY_READY = True


__all__ = [
    "BackendError", "Character", "CharacterBackend", "CharacterConfig",
    "load_config", "get_backend", "available_methods", "NoopBackend",
]
