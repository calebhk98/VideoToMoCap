# VideoToMoCap

**Goal:** take ~2 years of personal security-camera footage and turn it into a
model that moves like you.

That goal is really **two pipelines**, and this repo builds the first one
end-to-end and scaffolds the second:

1. **Data pipeline — video → anonymized motion data** (`videotomocap/`).
   Recover human motion from your footage, strip your body identity, and emit a
   standard motion dataset. This is fully implemented here.
2. **Motion model — data → an agent that moves like you** (`motion_model/`).
   Fine-tune a small motion-diffusion model on that dataset. Bridged and
   documented here; the heavy training runs in the upstream MDM repo.

Everything except the two neural stages (HMR inference, motion-model training)
is plain NumPy and runs on a laptop. Those two stages shell out to the upstream
research tools, behind clean adapters. A GPU-free self-test exercises the whole
data pipeline: `python scripts/selftest.py`.

---

## What the research actually says (verified 2026-07)

You asked me to look these up and verify code availability before committing.
Here's the ground truth, with the design decisions that follow from it.

### Step 1 — human mesh/motion recovery: which method

Every candidate you listed has **public code**. Speed numbers below are
throughput on a large backlog, which is the thing that matters for ~2 years of
footage.

| Method | Code | Moving camera | Bulk throughput | Verdict |
|---|---|---|---|---|
| **GVHMR** | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) (SIGGRAPH Asia 2024) | yes (VO/SLAM) | ~real-time class on a 4090 | **default backend** |
| **WHAM** | [yohanshin/WHAM](https://github.com/yohanshin/WHAM) (CVPR 2024) | yes | full pipeline much slower (~1 h / 10-min clip reported) | supported alt |
| **TRAM** | [yufu-wang/tram](https://github.com/yufu-wang/tram) (ECCV 2024) | yes (DROID-SLAM) | multi-stage; not benchmarked here | supported alt |
| **SLAHMR** | [vye16/slahmr](https://github.com/vye16/slahmr) | yes (optimization) | ~78 h / 10-min clip → **infeasible at this volume** | not wrapped |
| **DanceHMR** | [project page](https://shenwenhao01.github.io/dancehmr/) (ByteDance, Jun 2026) | — | — | hands add-on (see below) |
| **SAM-Body4D** | [gaomingqi/sam-body4d](https://github.com/gaomingqi/sam-body4d) (training-free) | yes | — | future backend |

**Decision:** GVHMR is the default. It's the throughput leader among the
world-grounded methods and its output format (SMPL-X params in
`hmr4d_results.pt`) is clean to parse. WHAM and TRAM are implemented as
drop-in alternatives; SLAHMR is intentionally *not* wrapped — at ~78 hours per
10-minute clip it would take longer than the footage spans to process it.

The backend is one config line, so you can benchmark two on a day of footage
and pick the winner without touching code.

### High-detail hands — swappable, and now a config line

Body-only methods (GVHMR/WHAM/TRAM) under-recover fingers — "pick up an apple
and eat it" comes out with a neutral hand. Because your work needs hands, the
pipeline now ships two families that recover articulated hands, all selectable
via `backend:`:

| `backend:` | Kind | Hands | Notes |
|---|---|---|---|
| `gvhmr` `wham` `tram` | body-only | ✗ (neutral) | fastest bulk; best when hands don't matter |
| `smplestx` (a.k.a. `smplerx`) | whole-body SMPL-X | ✓ | recommended general whole-body model |
| `whac` | whole-body SMPL-X | ✓ | **moving-camera + world-grounded** with hands |
| `hand4whole` | whole-body SMPL-X | ✓✓ | CVPR 2026, MIT, best single-model hands |
| `osx` `multihmr` | whole-body SMPL-X | ✓ | MIT / fast alternatives |
| `fusion` | body + hand net | ✓✓ | GVHMR/WHAM/… body **+ WiLoR/HaMeR** fingers |

The **`fusion`** backend is the most flexible: it runs any body backend for the
body + camera, runs WiLoR or HaMeR for the fingers, and grafts the MANO finger
pose into SMPL-X's `left_hand_pose`/`right_hand_pose` (with One-Euro smoothing
and last-good-frame gap filling; optional FK-based wrist relocalization). Config:

```yaml
backend: fusion
body_backend: whac       # any body/whole-body backend (moving-camera here)
hand_backend: wilor      # 'wilor' (fast, default) or 'hamer'
backend_repo: /opt/WHAC          # the BODY tool
hand_repo:    /opt/WiLoR         # the HAND tool
graft_wrist: false       # advanced: compose the hand-net wrist into the chain
```

Every hand-capable backend carries hands all the way through anonymization and
into the AMASS-SMPL-H hand slots, so the motion model can learn them. DanceHMR
(the ideal single video-native model) was **withdrawn with no public code** as
of 2026-07, so it is documented but not wrapped.

**Hardware note (dual RTX 3090 / 48 GB):** none of these need more than ~10 GB
for batch-1 inference, so a single 3090 runs any of them. The second card is
headroom — run `fusion`'s body and hand tools on separate GPUs, or process two
clips at once, to chew through the backlog faster.

### Step 2 — separate shape from pose (the privacy mechanism)

SMPL factors a human into **shape** (`betas` — build/limb-lengths, strongly
identifying) and **pose** (per-joint rotations over time — the movement).
`videotomocap/pose.py::anonymize()` keeps pose and **discards `betas`**,
re-expressing your motion on a neutral canonical body. The self-test asserts
this: after processing, `betas` are all zero and the AMASS export's hand/face
slots are neutral.

Honest caveat, stated in code: motion *style* (gait, posture) is itself weakly
identifying — but that's the entire point of the project. The strongest
biometric here, metric body shape, is dropped.

### Step 3 — camera handling

Corner mounts and rotating cameras mean you cannot assume a static camera. All
three wrapped backends are the SLAM/VO-based world-grounded family, so this is
handled by construction. Two knobs:
- `static_cameras: [...]` — list truly-fixed cameras to skip visual odometry
  (GVHMR's `-s`), saving time.
- Discrete pan positions — if a camera snaps between fixed presets rather than
  rotating continuously, split each preset into its own "clip" (folder), and the
  manifest treats them independently. Left as a manual step; not confirmed
  necessary.

### Step 2 (pipeline 2) — the motion model

- Motion-diffusion models are **small** (MDM: tens of millions of params, not
  billions) because joint-rotation sequences are far lower-dimensional than
  pixels/text. Confirm the exact count from the checkpoint you download.
- **No motion-diffusion scaling law exists** publicly. Don't import Chinchilla's
  ~20 tokens/param (that's autoregressive text). Use the data:param ratio of the
  checkpoint you fine-tune as your empirical reference.
- **Fine-tune, don't train from scratch** — you have hours of one person: plenty
  to specialize a pretrained motion prior, far too little to learn general human
  motion. ([MDM](https://github.com/GuyTevet/motion-diffusion-model),
  [priorMDM](https://github.com/priorMDM/priorMDM).)
- **Two problems, not one.** A motion *generator* (A) is not a *behavior* layer
  (B) that decides what to do. This repo produces (A); (B) is a separate
  control/RL project. Details in `motion_model/README.md`.

### ⚠️ Licensing — read before any commercial use

The **SMPL / SMPL-X body models are non-commercial** (Max Planck license): no
commercial products/services, and explicitly **no training methods for
commercial use**. Since GVHMR/WHAM/TRAM and HumanML3D/MDM all depend on SMPL,
a model trained through this pipeline inherits that restriction. For commercial
rights, license SMPL via Meshcapade. This tooling is provided for personal /
research use; the licensing is your call to clear.

---

## Install

Orchestration layer (the GPU-free parts — this is all you need to run the
self-test and manage the dataset):

```bash
pip install -r requirements.txt      # numpy, pyyaml
python scripts/selftest.py           # end-to-end, no GPU/weights
python tests/test_conversions.py     # rotation-math unit tests
```

### Installing a backend (GVHMR)

Backends have heavy, mutually-incompatible environments (torch/CUDA, SMPL-X
body models, SLAM deps). Install each in its **own** env from upstream — don't
mix them into this one.

```bash
git clone https://github.com/zju3dv/GVHMR /opt/GVHMR
# follow GVHMR/docs/INSTALL.md (conda env + SMPL-X weights via registration)
```

Then point the config at it (`configs/pipeline.yaml`):

```yaml
backend: gvhmr
backend_repo: /opt/GVHMR
backend_python: /opt/miniconda3/envs/gvhmr/bin/python
```

## Usage

```bash
# 1. discover all footage into a manifest (nothing is copied/transcoded)
python -m videotomocap --config configs/pipeline.yaml scan

# 2. exclude the two family visits BEFORE processing (glob on path)
python -m videotomocap --config configs/pipeline.yaml \
    exclude --pattern "*/2024-12-24/*" --pattern "*/2025-06-1*"

# 3. see what will be processed
python -m videotomocap --config configs/pipeline.yaml status

# 4. run HMR + anonymize (GPU box; resumable, checkpoints every clip)
python -m videotomocap --config configs/pipeline.yaml hmr --limit 5   # smoke test
python -m videotomocap --config configs/pipeline.yaml hmr             # the rest

# 5. aggregate into an AMASS-format dataset
python -m videotomocap --config configs/pipeline.yaml build

# then hand off to pipeline 2
python motion_model/prepare_mdm_data.py --dataset work/dataset --out work/mdm_data
```

The manifest is the single source of truth; every stage is idempotent and
resumable, so a crash 400 GB into the backlog just means re-running the command.
One clip failing (bad file, no person detected) is recorded and skipped, never
fatal to the batch.

## Layout

```
videotomocap/
  config.py            PipelineConfig (+ YAML loader)
  ingest.py            scan footage → manifest; exclude/include clips
  pose.py              SmplMotion container; anonymize() = drop shape, keep pose
  dataset.py           aggregate → AMASS-SMPL npz + train/val split + stats
  pipeline.py          orchestration (scan→hmr→anonymize→build), resumable
  cli.py               `python -m videotomocap ...`
  backends/
    base.py            HMRBackend ABC + rotation/SMPL-family conversion helpers
    gvhmr.py           default body-only: wraps GVHMR, parses hmr4d_results.pt
    wham.py  tram.py   alternative body-only world-grounded backends
    smplx_frames.py    whole-body SMPL-X (smplestx/whac/osx/hand4whole/multihmr)
    fusion.py          body + hand net (WiLoR/HaMeR) → SMPL-X with real hands
    noop.py            synthetic backend (no GPU) for tests/dry-runs
motion_model/          pipeline 2: MDM fine-tuning bridge, config, and docs
scripts/selftest.py    GPU-free end-to-end test of pipeline 1
tests/                 unit tests (rotation math, hands/fusion, config/ingest/CLI)
configs/               example + dropzone configs
dropzone/              drop your videos here
```

## Honest scope

- **Done and tested (83% line coverage; core modules 95–100%):** ingestion,
  manual exclusion, backend adapter layer, SMPL-72 + MANO-hand normalization and
  rotation conversions, shape/pose anonymization, fps resampling, the
  hand-graft/FK/smoothing glue, AMASS-SMPL-H dataset aggregation with splits,
  resumable orchestration, CLI, MDM data-prep bridge. GPU-free throughout via
  the `noop` backend.
- **Wired but needs your GPU + registered SMPL-X/MANO weights to actually run:**
  the neural inference itself (GVHMR/WHAM/TRAM, the SMPL-X whole-body backends,
  and WiLoR/HaMeR for fusion). Each backend's demo command and output-file
  layout are centralized and commented — verify them against your checkout
  revision (upstream demos drift).
- **Deliberately left as phase 2:** HumanML3D 263-d feature extraction (needs
  their repo + SMPL model), a learned wrist-correction regressor (the geometric
  `graft_wrist` is a first cut), and the entire behavior/decision layer (B).
