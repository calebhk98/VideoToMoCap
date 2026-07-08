"""Pipeline 2 -- train a motion model on Pipeline 1's anonymized dataset.

Config-driven and method-selectable, mirroring Pipeline 1: choose the trainer
with ``MotionModelConfig.method`` and everything else follows. Stays importable
with only numpy + stdlib -- heavy training runs upstream, behind subprocess
adapters. See README.md for methods, the data path, licensing, and hardware.
"""

from __future__ import annotations

from .config import MotionModelConfig, load_config
from .trainers import available_methods, get_trainer

__all__ = ["MotionModelConfig", "load_config", "get_trainer", "available_methods"]
