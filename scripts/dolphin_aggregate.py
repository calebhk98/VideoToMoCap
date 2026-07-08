#!/usr/bin/env python
"""Driver: fold a segment's frame captions into one (description, tags) label.

Runs INSIDE the aggregator's own env (vLLM + the Dolphin3.0 weights) -- NOT
importable by the videocaption core. ``videocaption/backends/dolphin.py`` shells
out to it, one call per segment.

Text-to-text only (the frame captions already carry the vision), so no VLM.
Dolphin3.0 reliably obeys a strict "return JSON with exactly these fields"
instruction, which is why it's the aggregator: the JSON it emits IS the Step 6
training-label format.

Contract
--------
  input : ``--request`` json ``{"model", "num_tags", "captions": [{"time","text"}, ...]}``
  output: ``--out`` json ``{"description": str, "tags": [str, ...]}``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

# The two label focuses. 'scene' is the search-index default (who/where/appearance);
# 'motion' steers toward the verbs a text-to-motion model wants, so the caption<->
# motion pairing bridge produces sharper training targets.
_SYSTEM_SCENE = (
    "You summarize a short video segment from its per-frame captions. Respond with "
    "a single JSON object and nothing else, with exactly two keys: 'description' (a "
    "rich natural-language description of the whole segment -- who is present, the "
    "setting, and what is happening) and 'tags' (concise keyword/entity strings "
    "suited to search)."
)
_SYSTEM_MOTION = (
    "You summarize the BODY MOVEMENT in a short video segment from its per-frame "
    "captions. Respond with a single JSON object and nothing else, with exactly two "
    "keys: 'description' (describe the physical actions and how bodies move across "
    "the segment -- e.g. 'walks forward, reaches up, then turns left' -- not the "
    "static scenery or appearance) and 'tags' (concise action/movement verbs). If a "
    "person's motion can't be read from the captions, describe what movement is "
    "implied and keep it about the body, not the background."
)


def _system_for(focus: str) -> str:
    return _SYSTEM_MOTION if focus == "motion" else _SYSTEM_SCENE


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate frame captions -> (description, tags)")
    p.add_argument("--model", required=True, help="HF model id to serve")
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--num-tags", type=int, default=12)
    p.add_argument("--focus", choices=["scene", "motion"], default="scene",
                   help="what the label describes: scene appearance or body motion")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args(argv)


def _build_prompt(captions: List[dict], num_tags: int) -> str:
    lines = [f"- t={c['time']:.1f}s: {c['text']}" for c in captions]
    return (
        f"Here are the frame captions for one video segment:\n" + "\n".join(lines) +
        f"\n\nWrite about {num_tags} tags. Return only the JSON object."
    )


def run_dolphin(model: str, prompt: str, system: str, max_tokens: int) -> str:
    """Return the model's raw text for the aggregation prompt. Adjust for your pin."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.2, max_tokens=max_tokens)
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    outputs = llm.chat([conversation], sampling)
    return outputs[0].outputs[0].text.strip()


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of the model's reply, tolerating stray prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise SystemExit(f"aggregator returned no JSON object:\n{text}")
    return json.loads(match.group(0))


def main(argv=None):
    args = parse_args(argv)
    request = json.loads(args.request.read_text())
    # focus travels in the request too; the --focus flag is the override the backend passes.
    focus = args.focus or request.get("focus", "scene")
    prompt = _build_prompt(request["captions"], args.num_tags)

    raw = run_dolphin(args.model, prompt, _system_for(focus), args.max_tokens)
    obj = _extract_json(raw)
    label = {
        "description": str(obj.get("description", "")).strip(),
        "tags": [str(t).strip() for t in obj.get("tags", []) if str(t).strip()],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(label, indent=2))
    print(f"wrote label ({len(label['tags'])} tags) to {args.out}")


if __name__ == "__main__":
    main()
