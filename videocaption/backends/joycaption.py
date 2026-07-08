"""JoyCaption captioner (Step 3) -- per-frame VLM served by vLLM.

Extracts the segment's frames, then shells out to ``scripts/joycaption_infer.py``
(which loads ``fancyfeast/llama-joycaption-beta-one-hf-llava`` under vLLM with
continuous batching) and reads back one caption per frame. The heavy model code
lives entirely in that driver's dedicated env -- nothing here imports torch/vLLM.

JoyCaption is built on a Llama 3.1 backbone: fine for private/local use, but if a
model trained on its outputs is distributed publicly the Llama 3.1 Community
Licence naming/notice terms apply. See videocaption/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .. import frames
from .base import Captioner, CaptionError, FrameCaption


class JoyCaptionBackend(Captioner):
    """Per-frame descriptions via JoyCaption on vLLM; see module docstring."""

    name = "joycaption"

    def caption_frames(self, video_path: Path, times: List[float], out_dir: Path) -> List[FrameCaption]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = frames.extract_frames(video_path, times, out_dir, extractor=self.cfg.frame_extractor)
        request = out_dir / "caption_request.json"
        result = out_dir / "captions.json"
        request.write_text(json.dumps({
            "model": self.cfg.captioner_model,
            "prompt": self.cfg.caption_prompt,
            "frames": [p.name for p in frame_paths],
        }))
        cmd = [
            self.cfg.captioner_python, str(self._driver("joycaption_infer.py")),
            "--model", self.cfg.captioner_model,
            "--frames-dir", str(out_dir),
            "--request", str(request),
            "--out", str(result),
        ]
        self._run_cmd(cmd, cwd=self.cfg.captioner_repo)
        return _parse_captions(result, times)


def _parse_captions(result: Path, times: List[float]) -> List[FrameCaption]:
    """Read the driver's ``{"captions": [{"file","text"}, ...]}`` output.

    Pairs each caption with its frame timestamp by position -- the driver emits
    them in the same order the frames were requested.
    """
    if not result.exists():
        raise CaptionError(f"joycaption driver wrote no output at {result}")
    payload = json.loads(result.read_text())
    texts = [c.get("text", "") for c in payload.get("captions", [])]
    if len(texts) != len(times):
        raise CaptionError(
            f"joycaption returned {len(texts)} captions for {len(times)} frames "
            f"(expected one per frame)."
        )
    return [FrameCaption(time=t, text=text) for t, text in zip(times, texts)]
