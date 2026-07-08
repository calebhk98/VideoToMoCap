"""Backend interface + shared helpers for the caption/aggregate models.

Two roles, each selected by config exactly like Pipeline 1's ``backend``:
  * :class:`Captioner`  -- Step 3, a per-frame vision-language model.
  * :class:`Aggregator` -- Step 4, a text model folding the frame captions into
    one structured ``(description, tags)`` label.

Both shell out to their heavy model (vLLM / transformers) in a dedicated env via
the shared :class:`_Bridge` helpers, so this package stays importable with just
the standard library. The heavy code lives in the driver scripts under
``scripts/`` (``joycaption_infer.py`` / ``dolphin_aggregate.py``) and is only ever
run as a subprocess -- never imported here.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..config import CaptionConfig


class CaptionError(RuntimeError):
    """Raised for backend setup/execution problems (missing repo/driver, subprocess
    failure, unparseable output) -- always with actionable context on how to fix it."""


@dataclass
class FrameCaption:
    """One frame's caption at a timestamp within its segment."""

    time: float
    text: str


@dataclass
class SegmentLabel:
    """The structured Step 4 label -- the exact format the Step 6 model learns."""

    description: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"description": self.description, "tags": list(self.tags)}


class _Bridge:
    """Subprocess plumbing shared by both backend roles (GPU pinning, driver paths)."""

    name = "base"

    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg

    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
        env = self._subprocess_env()
        try:
            subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True, env=env)
        except FileNotFoundError as exc:
            raise CaptionError(
                f"Could not launch {self.name}: {exc}. Is the interpreter on PATH and the "
                f"driver/repo set up? See videocaption/README.md."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise CaptionError(f"{self.name} exited with status {exc.returncode} on: {printable}") from exc

    def _subprocess_env(self) -> Optional[dict]:
        """Env for the model subprocess -- pins its GPU when the runner assigned one."""
        device = getattr(self.cfg, "cuda_device", None)
        if device is None:
            return None  # inherit the parent env unchanged
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        return env

    def _require(self, path: Optional[Path], what: str) -> Path:
        if path is None or not Path(path).exists():
            raise CaptionError(f"{self.name} needs {what} (got {path!r}). Set it in the config.")
        return Path(path)

    @staticmethod
    def _driver(script_name: str) -> Path:
        """Absolute path to a driver under the repo's ``scripts/`` dir."""
        return Path(__file__).resolve().parents[2] / "scripts" / script_name


class Captioner(_Bridge, ABC):
    """Step 3: recover a caption for each sampled frame of a segment."""

    role = "per-frame VLM"

    @abstractmethod
    def caption_frames(self, video_path: Path, times: List[float], out_dir: Path) -> List[FrameCaption]:
        """Caption the frames at ``times`` in ``video_path``; return one per frame.

        ``out_dir`` is a per-segment scratch dir the backend may fill with the
        extracted frames and its own artifacts.
        """

    def describe(self) -> str:
        return f"{self.name}: {self.role} ({self.cfg.captioner_model})"


class Aggregator(_Bridge, ABC):
    """Step 4: fold the per-frame captions into one structured segment label."""

    role = "caption -> (description, tags)"

    @abstractmethod
    def aggregate(self, captions: List[FrameCaption], *, num_tags: int) -> SegmentLabel:
        """Synthesize ``captions`` into a single :class:`SegmentLabel`."""

    def describe(self) -> str:
        return f"{self.name}: {self.role} ({self.cfg.aggregator_model})"
