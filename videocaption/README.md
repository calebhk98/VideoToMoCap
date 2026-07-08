# Pipeline 3 — archive captioning & search

Turn a large personal video archive (~100–1,000 h) into a **searchable, captioned
index**, and use that as bootstrap data to fine-tune a **video-native model that
captions a whole clip in one pass** — the primary deliverable, shared so others
don't have to repeat the data-generation step.

Config-selectable from a YAML exactly like the other two pipelines. The heavy
models run in their own environments behind subprocess bridges; everything here
is plain **standard library** (no numpy even), so the whole flow runs and tests
GPU-free with the `noop` backends.

```bash
python -m videocaption --config configs/videocaption.yaml scan     # discover footage
python -m videocaption --config configs/videocaption.yaml caption  # segment + caption (resumable)
python -m videocaption --config configs/videocaption.yaml index    # build the search index
python -m videocaption --config configs/videocaption.yaml search "cooking in the kitchen"
python -m videocaption --config configs/videocaption.yaml run      # caption → index → windows → ft-data
python scripts/caption_selftest.py                                 # GPU-free end-to-end, must stay green
```

## The pipeline

```
raw video (up to ~2 h)
   │  [1] segment: PySceneDetect boundaries, capped at max_segment_seconds (~2 min)
   ▼
caption segments  ──[2] sample frames (~1 fps)──▶  [3] JoyCaption (per-frame VLM, vLLM)
   │                                                     │
   │                        [4] Dolphin3.0 folds the frame captions into ▼
   │                            one structured (description, tags) label
   ▼
[5] search index rows: (video_id, start, end, description, tags)   ← sqlite + json
   │
   │  [5.5] group consecutive segments into ~10-min windows
   ▼
[6] LoRA-fine-tune Qwen2.5-VL-7B: window in → full timestamped label list out
     (the deliverable; merged to a standalone checkpoint for distribution)
```

Two granularities are deliberately different: **caption segments** (~2 min, the
unit captions are generated at) vs **windows** (~10 min, the unit the fine-tuned
model actually consumes). Steps 1–5 only build the *labels*; the trained model
runs on windows.

## Model choices (`captioner:` / `aggregator:` in the config)

| role | default | why | swap to |
|---|---|---|---|
| captioner (Step 3) | `joycaption` (`fancyfeast/llama-joycaption-beta-one-hf-llava`) | 8B-class LLaVA, ~16–17 GB bf16 on one 3090; run via **vLLM** continuous batching for throughput | `noop` (tests) |
| aggregator (Step 4) | `dolphin` (`cognitivecomputations/Dolphin3.0-Llama3.1-8B`) | text-to-text (no vision needed); strong IFEval (~76), low refusal → reliably obeys "return JSON with these two fields" | `noop` (tests) |
| fine-tune base (Step 6) | `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0, native multi-frame video, base training already does temporal grounding (timestamped events) | — |

Each backend shells out to its driver (`scripts/joycaption_infer.py`,
`scripts/dolphin_aggregate.py`) in a dedicated env and fails loud with an
actionable message when a repo/interpreter/output is missing. The `(description,
tags)` format Step 4 emits **is** the Step 6 training-label format — choose it
here and the fine-tuned model learns to produce both in one call.

## Step 6 — the fine-tune (bridge)

`finetune-data` turns the windows into a bootstrap dataset (`finetune/dataset.jsonl`,
one example per window; the assistant target is the dense `(start, end, description,
tags)` list with **window-relative** timestamps). `videocaption/finetune.py`'s
`LoRAFinetune` then builds the LoRA training + adapter-merge commands for an
upstream trainer (LLaMA-Factory by default — set `finetune_repo`).

- **LoRA rank 128–256, `lora_target: all`** (attention + MLP broadly): this is a
  capability fine-tune that teaches descriptive vocabulary the base video model
  doesn't have, not a light style touch-up. Start with LoRA; if held-out output
  stays generic/vague despite training, that's your evidence a full fine-tune is
  justified — cheaper than committing to it upfront.
- **Merge to distribute:** `merge_adapter: true` runs `merge_and_unload`-equivalent
  export → one self-contained checkpoint, so downstream users just download and run
  (no separate adapter load).
- Validate on held-out windows before running across the rest of the archive.

## Hardware (2× RTX 3090, 48 GB, NVLink, local-only)

- **Captioning throughput:** benchmark first. Rough (unvalidated) single-GPU
  batched range is ~1.5–3 images/sec; both cards as **independent instances**
  (`gpus: auto`, one per card) roughly doubles it. Total time ≈ (hours × sample
  fps × 3600) ÷ images-per-sec — measure on a few hundred–thousand frames before
  sizing the full run.
- **Fine-tune:** LoRA/QLoRA of a 7B fits comfortably on 2×24 GB; NVLink only
  matters if you tensor/data-parallel the run (not required for 7B LoRA).
- Scale note: a ~10-min window over 1,000 h is ~6,000 windows for the trained
  model vs ~1.8M frame-level calls under the raw pipeline — the fine-tune is the
  payoff.

## Licensing — read before sharing a fine-tuned model

`Qwen2.5-VL-7B-Instruct` is **Apache-2.0** (only Qwen's 3B/72B carry a custom
licence). JoyCaption and Dolphin3.0 are both **Llama-3.1-derived**, so both fall
under the **Llama 3.1 Community License** — one lineage, not two obligations. That
licence permits using their outputs to train other models, but if you *distribute*
the resulting model publicly it requires: the new model's name to **start with
"Llama"**, a copy of the licence included, and a **"Built with Llama"** notice.
Purely private/local use has no such requirement. This is a naming/notice
obligation on the *shared artifact*, not a restriction on building it.

## Open items (tuning knobs, not architecture)

Set as documented defaults; pin against your own hardware/footage:

- caption-segment cap (`max_segment_seconds`, default 120)
- window size (`window_seconds`, default 600 — 5–15 min pending memory testing)
- frame sample rate (`frame_sample_fps`, default 1.0)
- **real throughput benchmark** to size total processing time
- how many bootstrap examples generalize well; per-segment vs per-frame targets —
  both are producible from the artifacts this pipeline already writes.

## Layout

```
videocaption/
  config.py     CaptionConfig (+ YAML loader) — every knob
  manifest.py   scan archive → manifest; video state machine; exclude/include
  segment.py    Step 1: pure scene sub-chunking + lazy scene-detect/duration
  frames.py     Step 2: pure frame-time sampling + lazy ffmpeg/decord extraction
  store.py      per-video segments + incremental labels on disk; row joins
  index.py      Step 5: sqlite (FTS when available) + json search index
  window.py     Step 5.5: pure grouping of segments into training windows
  finetune.py   Step 6: bootstrap dataset builder + LoRA/merge command bridge
  pipeline.py   orchestration (scan→segment→caption→index→windows), parallel+resumable
  cli.py        `python -m videocaption ...`
  backends/
    base.py       Captioner + Aggregator ABCs + shared subprocess bridge
    joycaption.py per-frame VLM (driver: scripts/joycaption_infer.py)
    dolphin.py    caption → (description, tags) (driver: scripts/dolphin_aggregate.py)
    noop.py       synthetic captioner + aggregator (no GPU/weights/decoder) for tests
```
