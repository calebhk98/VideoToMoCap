"""Dolphin aggregator (Step 4) -- fold frame captions into a structured label.

This is text-to-text (the frame captions already carry the vision), so it needs
no VLM. ``cognitivecomputations/Dolphin3.0-Llama3.1-8B`` is a community instruct
tune with a strong IFEval score and little refusal overhead, so it reliably obeys
"return JSON with exactly these two fields". The heavy model runs in
``scripts/dolphin_aggregate.py``; nothing here imports it.

The chosen ``(description, tags)`` format IS the Step 6 training-label format --
whatever this emits is what the fine-tuned video model learns to produce.

Dolphin3.0 is Llama-3.1-derived: the same Llama licence note as the captioner
applies if a downstream model is distributed publicly (see README).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .base import Aggregator, CaptionError, FrameCaption, SegmentLabel


class DolphinBackend(Aggregator):
    """Structure frame captions into (description, tags) via Dolphin3.0."""

    name = "dolphin"

    def aggregate(self, captions: List[FrameCaption], *, num_tags: int) -> SegmentLabel:
        work = self.cfg.work_root / "agg_scratch"
        work.mkdir(parents=True, exist_ok=True)
        request = work / "agg_request.json"
        result = work / "agg_result.json"
        request.write_text(json.dumps({
            "model": self.cfg.aggregator_model,
            "num_tags": num_tags,
            "captions": [{"time": c.time, "text": c.text} for c in captions],
        }))
        cmd = [
            self.cfg.aggregator_python, str(self._driver("dolphin_aggregate.py")),
            "--model", self.cfg.aggregator_model,
            "--request", str(request),
            "--num-tags", str(num_tags),
            "--out", str(result),
        ]
        self._run_cmd(cmd, cwd=self.cfg.aggregator_repo)
        return _parse_label(result)


def _parse_label(result: Path) -> SegmentLabel:
    """Read the driver's ``{"description": ..., "tags": [...]}`` output, failing loud."""
    if not result.exists():
        raise CaptionError(f"dolphin driver wrote no output at {result}")
    payload = json.loads(result.read_text())
    description = payload.get("description")
    if not isinstance(description, str) or not description:
        raise CaptionError(f"dolphin output missing a 'description' string: {payload!r}")
    tags = [str(t) for t in payload.get("tags", [])]
    return SegmentLabel(description=description, tags=tags)
