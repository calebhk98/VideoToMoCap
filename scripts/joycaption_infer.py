#!/usr/bin/env python
"""Driver: per-frame captioning with JoyCaption served by vLLM.

Runs INSIDE the captioner's own env (vLLM + the JoyCaption weights) -- NOT
importable by the videocaption core. ``videocaption/backends/joycaption.py``
shells out to it, one call per segment.

Contract
--------
  input : ``--frames-dir`` holding the segment's frames + ``--request`` json
          ``{"model", "prompt", "frames": [filename, ...]}`` (frame order == the
          order the pipeline sampled them).
  output: ``--out`` json ``{"captions": [{"file", "text"}, ...]}`` in that same
          order -- one entry per frame.

vLLM's continuous batching turns the whole segment's frames into one batched
``chat`` call, which is where the throughput win over single-request decoding
comes from. Images are passed inline as base64 data URLs so this stays robust to
vLLM API drift; if you pin a version with a nicer multimodal path, swap
``run_joycaption`` -- everything else is plain json plumbing on our contract.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import List


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JoyCaption per-frame captioning (vLLM)")
    p.add_argument("--model", required=True, help="HF model id to serve")
    p.add_argument("--frames-dir", required=True, type=Path)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args(argv)


def _data_url(path: Path) -> str:
    """Inline a frame as a base64 data URL for the chat request."""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def run_joycaption(model: str, prompt: str, frame_paths: List[Path], max_tokens: int) -> List[str]:
    """Caption each frame; return texts in the same order. Adjust for your vLLM pin."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.3, max_tokens=max_tokens)
    conversations = [
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(fp)}},
            {"type": "text", "text": prompt},
        ]}]
        for fp in frame_paths
    ]
    outputs = llm.chat(conversations, sampling)  # batched: continuous batching does the work
    return [o.outputs[0].text.strip() for o in outputs]


def main(argv=None):
    args = parse_args(argv)
    request = json.loads(args.request.read_text())
    prompt = request.get("prompt", "Describe this image.")
    frame_paths = [args.frames_dir / name for name in request["frames"]]

    texts = run_joycaption(args.model, prompt, frame_paths, args.max_tokens)
    if len(texts) != len(frame_paths):
        raise SystemExit(f"JoyCaption produced {len(texts)} captions for {len(frame_paths)} frames")

    captions = [{"file": fp.name, "text": t} for fp, t in zip(frame_paths, texts)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"captions": captions}, indent=2))
    print(f"wrote {len(captions)} captions to {args.out}")


if __name__ == "__main__":
    main()
