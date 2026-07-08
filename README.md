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
| `camenduru_smplerx` (a.k.a. `camenduru`) | whole-body SMPL-X | ✓ | SMPLer-X via [camenduru's runnable repackaging](https://github.com/camenduru/SMPLer-X) |
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

### Partial-body / occluded footage (legs, an arm, the head — any part)

Cameras often frame a person waist-up, or cut off an arm, or (mounted high) miss
the head. **HMR methods still output a full SMPL body either way** — they
*infer* the out-of-frame joints rather than observing them. So the pipeline never
crashes on truncation, but those inferred joints are plausible-but-fake motion
you don't want polluting a "move like me" dataset.

How this pipeline helps — **tag which body regions each camera can't see**, and
the unseen joints get labeled so you can filter or mask them:

```yaml
# whichever regions a camera never reliably shows
camera_occlusions:
  cam_desk:  [legs]              # waist-up desk view
  cam_high:  [head]             # ceiling mount, head cropped
  cam_left:  [right_arm]        # subject's right arm out of frame
partial_body_cameras: [cam_desk] # shorthand: same as camera_occlusions {cam: [legs]}
```

Region names (`videotomocap/regions.py`): `legs`, `left_leg`, `right_leg`,
`feet`, `arms`, `left_arm`, `right_arm`, `hands`, `head`. For each clip the union
of their SMPL joints is recorded as `unreliable_joints` in the manifest (visible
in `list`/`status`) **and** carried all the way through: the anonymized pose npz
and the exported AMASS npz get a `joint_valid` mask (24 bools, `False` =
inferred), and the dataset `index.json` lists the unreliable joints per clip. So
downstream you can drop those clips, or **mask the loss on the guessed joints**
during training and still use the good ones (e.g. keep the arms/torso from a
waist-up camera, ignore its legs).

Also **pick a truncation-robust backend** — `smplestx`, `multihmr` (its CUFFS
close-up data), and the fusion path degrade more gracefully under partial framing
than the body-only trio.

The honest state of the art: there is **no** published method dedicated to
"partial framing → full-body motion." The closest breaking work (FactorizedHMR's
torso-anchor + generative limb completion; DanceHMR's close-up-aware
augmentation) has no usable code yet — tracked in
[`RESEARCH_WATCHLIST.md`](RESEARCH_WATCHLIST.md). Until then, tag-and-mask is the
pragmatic answer.

### Step 2 (pipeline 2) — the motion model

- Motion-diffusion models are **small** (MDM: tens of millions of params, not
  billions) because joint-rotation sequences are far lower-dimensional than
  pixels/text. Confirm the exact count from the checkpoint you download.
- **Motion-diffusion scaling laws now exist** — don't import Chinchilla's ~20
  tokens/param (that's autoregressive text). Use the motion-specific work:
  [ScaMo](https://github.com/shunlinlu/ScaMo_code) (CVPR'25, log test-loss vs
  compute), Being-M0/MotionLib (ICML'25, data×model), and NVIDIA's Kimodo (2026,
  700 hrs). All three have public code — real reference points for your
  data:param budget.
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

**Everything lives in the config — you don't need to remember flags.** Set your
options once (footage root, backend, exclusions, GPUs, ...) in a YAML, and the
commands read them. The config is auto-discovered, so you don't even pass
`--config`: put a `videotomocap.yaml` in your working directory (or point
`$VIDEOTOMOCAP_CONFIG` at one), and:

```bash
python -m videotomocap run      # scan → exclude → hmr → build, all from the config
```

That's the whole pipeline. Under the hood `run` = `hmr` + `build`, and `scan`
runs automatically the first time. The family visits are excluded because they're
listed once in the config (`exclude_patterns`), not retyped each run:

```yaml
# videotomocap.yaml
footage_root: dropzone
backend: gvhmr
backend_repo: /opt/GVHMR
backend_python: /opt/miniconda3/envs/gvhmr/bin/python
exclude_patterns: ["*/2024-12-24/*", "*/2025-06-1*"]   # the family visits, set once
gpus: auto
workers_per_gpu: auto
```

Individual steps (each still reads the config; flags are optional overrides):

```bash
python -m videotomocap scan       # (re)build the manifest; applies exclude_patterns
python -m videotomocap status     # what will be processed
python -m videotomocap hmr        # HMR + anonymize (resumable, checkpoints each clip)
python -m videotomocap build      # aggregate into the AMASS dataset
python -m motion_model train      # dataset -> motion model (Pipeline 2; see motion_model/)
```

Or the whole chain in one command:

```bash
python scripts/run_pipeline.py \
    --video-config configs/dropzone.yaml --model-config configs/motion_model.yaml
```

Prefer an isolated, offline container (no host pollution)? Three short commands
(see `docker/README.md`):

```bash
make build                      # build the image
make setup                      # one-time online: repos, envs, weights, SMPL-H
make run                        # videos in ./dropzone -> trained model in ./work
```

Override the backend/method with `make setup BACKEND=wham METHOD=mdm`.

Need a one-off override? Any config field that matters at the CLI has a flag
(`--limit 5` for a smoke test, `--gpus 0`, `--workers-per-gpu 2`, `--config other.yaml`,
`--footage-root ...`) — but you never *have* to use them. Extra ad-hoc exclusions
still work too: `videotomocap exclude --pattern "*/badcam/*"`.

The manifest is the single source of truth; every stage is idempotent and
resumable, so a crash 400 GB into the backlog just means re-running the command.
One clip failing (bad file, no person detected) is recorded and skipped, never
fatal to the batch.

## Running at scale (parallelism)

Clips are fully independent, so HMR is embarrassingly parallel — the right lever
for hundreds of hours of footage.

**It finds your GPUs automatically.** `gpus: auto` (the default in the example
configs) runs `nvidia-smi` to detect every card and respects an existing
`CUDA_VISIBLE_DEVICES`. Worker count then defaults to *(#GPUs × `workers_per_gpu`)*
— so on your dual 3090s, just:

```bash
python -m videotomocap --config configs/pipeline.yaml hmr        # auto: 2 GPUs, 1 clip each
```

**Saturate a card / one-GPU parallelism.** You also asked for "do multiple at
once even on one GPU." `workers_per_gpu` packs N clips onto each card: while one
clip's model is on the GPU the others are decoding video / doing IO, so the card
stays busy instead of idling between clips.

```bash
python -m videotomocap ... hmr --gpus 0 --workers-per-gpu 3   # 3 clips share GPU 0
python -m videotomocap ... hmr --gpus auto --workers-per-gpu 2 # both cards, 2 deep
```

**Or let it pick N automatically** — `workers_per_gpu: auto` sizes the depth from
each card's *free VRAM*: `(free − headroom) ÷ per-clip footprint`, clamped to
`max_workers_per_gpu`. It learns the per-clip footprint by **calibrating on the
first clip** (measuring how far its free VRAM dips while it runs), unless you set
`vram_per_worker_mb` to skip that. So on a bigger future card it just fills up:

```bash
python -m videotomocap ... hmr --gpus auto --workers-per-gpu auto
# -> "Calibrated ~5400 MiB/clip on gpu0"
# -> "Auto workers/GPU from ~5400 MiB/clip, 2000 MiB headroom: {'0': 3, '1': 3}"
```

Tune the safety margin and ceiling with `vram_headroom_mb` (default 2000) and
`max_workers_per_gpu` (default 8). Because it reads *free* (not total) VRAM, it
also backs off if another process is already using a card.

Mechanics and honest limits:
- Each worker runs the backend in its own subprocess with `CUDA_VISIBLE_DEVICES`
  pinned to its GPU (heavy work is in that subprocess, GIL released → real
  speedup from the thread pool).
- **Across-clip overlap, not within-clip.** The upstream HMR tools are opaque
  subprocesses, so we can't pipeline *inside* one clip (decode→detect→regress
  stages of a single clip aren't ours to interleave). Instead we overlap *whole
  clips*: `workers_per_gpu > 1` is exactly the "shift the next batch in while the
  previous one finishes" idea, at clip granularity. More workers than a card's
  VRAM allows will OOM — 2–3 per 24 GB is the usual sweet spot.
- Manifest checkpointed under a lock after every clip → a parallel run is as
  crash-resumable as a sequential one.
- **Beyond one machine:** the manifest is the coordination point. Point several
  machines at the same footage share with disjoint `--limit`/exclusions, or split
  the footage tree per machine — each writes its own pose npz, then run `build`
  once over the merged `work/pose`.

Roughly: GVHMR at ~real-time on a 4090 → on a 3090 pair, ~2× real-time
throughput, so hundreds of hours become days, not weeks — and it resumes if
interrupted.

### Refinement (optional post-processing)

Set `refine: true` to run a pure-NumPy cleanup pass after HMR (before
anonymization): a temporal de-jitter and a stationary anti-drift fix. No weights,
no GPU. Two de-jitter methods (`refine_method`), both realizing **HTD-Refine**'s
"penalize high-order temporal dynamics" objective without its learned network:
`savgol` (fast local polynomial fit) and `variational` (a global smoother that
directly minimizes `‖x−y‖² + λ‖accel(x)‖²`). See `videotomocap/refine.py` and the
feasibility notes in [`RESEARCH_WATCHLIST.md`](RESEARCH_WATCHLIST.md).

## Code health (pre-commit)

A commit gate keeps files small and shallow (large/deeply-nested files trip up
smaller LLMs and humans alike). Enable it once per clone:

```bash
bash scripts/install_hooks.sh     # sets git core.hooksPath -> .githooks
```

On every commit it runs `scripts/check_code_health.py`, which flags Python files
over 500 lines, any file over 1 MB, and code nested deeper than 6 blocks (true
block nesting via `tokenize`, so docstring tables and wrapped call arguments
don't false-positive). Run it anytime: `python scripts/check_code_health.py --all`.
Thresholds are env-overridable (`CH_MAX_LINES`, `CH_MAX_BYTES`, `CH_MAX_INDENT`);
bypass a single commit with `git commit --no-verify`.

## Platforms (Windows + Linux)

**This orchestration package runs natively on both, no changes needed.** It's
pure Python: `pathlib` everywhere, atomic manifest writes via `os.replace`
(atomic on Windows too), GPU detection via `nvidia-smi` (present on both), and
glob exclusions use `fnmatchcase` so `exclude_patterns` match *identically* on
Windows and Linux (plain `fnmatch` is case-insensitive on Windows only — we
avoid it). The GPU-free path — ingest, exclude, anonymize, dataset build, the
`noop` backend, all tests — works the same on either OS. Config paths can be
Windows-style; patterns match the clip's relative path in `/`-form.

**The one caveat is the heavy neural backends, not this code.** GVHMR / WHAM /
TRAM / SMPLest-X pull in Linux-oriented CUDA extensions (DROID-SLAM, Detectron2,
PyTorch3D) that are painful or unsupported on *native* Windows. On Windows the
reliable path for those is **WSL2** (Ubuntu + CUDA), where they run exactly as on
Linux — this package sits on top unchanged. Dev-only note: the pre-commit hook is
a shell script, so on Windows enable it from **Git Bash** (bundled with Git for
Windows); it's not needed to *run* the pipeline.

## Input types (phone videos, selfies, ...)

Any monocular video works — it doesn't have to be security footage.

- **Phone / handheld recordings: yes, well.** Moving-camera is exactly what the
  world-grounded backends (GVHMR/WHAM/TRAM, and **WHAC** for whole-body+hands)
  are built for. Just *don't* list that source under `static_cameras`, so visual
  odometry/SLAM handles the motion. Variable frame rate is fine — every clip is
  resampled to `target_fps`. Portrait orientation is fine (rotation metadata is
  honored by the decoder).
- **Selfie *videos* (front camera, arm's length): yes.** They're usually
  upper-body / close-up, i.e. the partial-body case — tag the source with
  `camera_occlusions: {selfie: [legs]}` so the inferred legs are flagged, and
  prefer a close-up-robust backend (`smplestx`, `multihmr`, or `fusion` for good
  hands). Front cameras sometimes save a *mirrored* file (left/right swapped) —
  handled automatically, see below; no per-video tagging.
- **A single selfie *photo*: no.** One still is one frame — below
  `min_clip_frames`, so it's dropped. This pipeline is about *motion over time*;
  it needs video (or a burst), not a snapshot.

So: your phone clips are great input; selfie videos work if you tag them as
upper-body; single photos don't (nothing moves to capture).

### Mirrored / flipped footage — detected automatically (no tagging)

Some clips (front-camera selfies especially) are horizontally flipped, which
swaps left and right in the recovered motion and corrupts handedness. There's no
metadata flag for this and a mirrored person still looks valid, so per-video
pixel detection is unreliable — **but across your corpus it's you, and your
handedness is consistent.** So the pipeline scores each clip's handedness from
the recovered motion, takes the corpus consensus as your true dominant side
(self-calibrating — a left-handed user isn't mass-flagged), and flags the clips
that confidently disagree. Zero per-video tagging.

Controlled by `auto_mirror` in the config:
- `flag` (default) — annotate suspected clips in the manifest (`status` shows the
  count); no data change. You can eyeball just the flagged few.
- `correct` — also flip the flagged clips' pose so the dataset handedness is
  consistent. The fix is exact (`mirror_motion`: the standard SMPL left/right
  mirror, applied in motion space — equivalent to flipping the video, no HMR
  re-run).
- `off` — skip entirely.

It runs automatically inside `run`/`build`, or on demand:

```bash
python -m videotomocap mirror            # flag suspected clips
python -m videotomocap mirror --correct  # and fix them
```

Honest limits: it's a heuristic. It can't distinguish a mirrored clip from one
where you genuinely used your non-dominant hand a lot, so it only flags
*confident* disagreements (tune `mirror_margin`) and defaults to off (set it to
`flag` or `correct`). Gross motion (walking, sitting) is nearly symmetric and
barely affected either way; handedness-specific tasks (writing, eating) matter.

### Automatic quality filtering (garbage clips)

HMR over hours of footage produces some junk: frames where tracking teleports the
root across the scene, a limb snaps 180° between frames, a NaN from a failed
solve, or a clip where nothing moves. These pollute the dataset. The pipeline
scores each clip's plausibility **from the recovered motion alone** — no video
decode, no detector, just NumPy — and flags or drops the bad ones. Controlled by
`quality_filter`:

- `flag` (default) — annotate `quality_issues` in the manifest (`status` shows
  the counts); nothing dropped.
- `exclude` — also drop *hard* failures from the dataset.
- `off` — skip.

Findings, split into hard (physically impossible → drop candidates) and soft
(valid but low-value):

| issue | tier | meaning |
|---|---|---|
| `non_finite` | hard | NaN/Inf in the pose — failed solve |
| `root_teleport` | hard | root faster than `quality_max_speed_ms` (12 m/s) — tracking jumped |
| `pose_jump` | hard | a joint rotates > `quality_max_joint_step` (1.5 rad) in one frame |
| `static_low_motion` | soft | barely moving — real, just low training value (never auto-dropped) |

It runs automatically inside `run`/`build`. This is the cheap, safe win among the
auto-tags — no heavy deps, and it directly cleans the training data.

### Auto-occlusion (`auto_occlusion`) and action clusters (`cluster_actions`)

Two more motion-derived tags, no per-video work:

- **`auto_occlusion: flag`** — marks joints that never move across a clip as
  unreliable (masked via `joint_valid`, same as manual `camera_occlusions`).
  Since HMR freezes out-of-frame joints at a default, a frozen joint is a decent
  proxy for occluded — with the honest caveat that it also flags genuinely-still
  limbs, which is fine for the purpose (masking non-informative joints out of
  training). Complements `camera_occlusions`; the two are unioned.
- **`cluster_actions: <k>`** — k-means the clips into `k` motion clusters and
  tags each with an `action_cluster` id (surfaced in the dataset `index.json`).
  Your footage has no action labels; this gives the motion model a free
  conditioning signal ("cluster 3" ≈ walking, etc.). It's clustering, not
  recognition — coarse buckets you can name later, not ground-truth actions.

### Raw-video pre-analysis (`skip_empty`, `auto_camera_motion`)

The only auto-tags that look at *pixels* (before HMR), so they need
`opencv-python` — installed lazily; the rest of the pipeline runs without it.
Both sample ~16 frames per clip:

- **`skip_empty: true`** — the big compute saver for security footage, most of
  which is nobody there. If a clip's mean inter-frame difference is below
  `empty_activity_threshold`, it's excluded ("empty (no activity)") *before* it
  reaches the GPU, so you don't waste HMR on empty scenes.
- **`auto_camera_motion: flag`** — estimates each clip's global pixel shift
  (phase correlation); a locked-off camera (shift below `camera_motion_threshold`)
  is tagged `static`, and HMR then skips visual odometry for it (faster) —
  automating the manual `static_cameras` list. *Caveat: a moving subject filling
  the frame can inflate the shift, so treat it as a hint; it's conservative
  (only near-zero shift → static).*

Both run automatically at the start of `hmr`/`run`. If OpenCV isn't installed
they print a warning and no-op — nothing breaks.

## Layout

```
videotomocap/
  config.py            PipelineConfig (+ YAML loader)
  ingest.py            scan footage → manifest; exclude/include clips
  pose.py              SmplMotion container; anonymize() = drop shape, keep pose
  dataset.py           aggregate → AMASS-SMPL npz + train/val split + stats
  pipeline.py          orchestration (scan→hmr→refine→anonymize→build), parallel+resumable
  gpu.py               GPU auto-detection + worker/device resolution
  regions.py           body regions → SMPL joints (occlusion tagging)
  refine.py            optional post-proc: temporal de-jitter + anti-drift
  mirror.py            auto left/right-mirror detection + correction
  quality.py           auto clip-quality assessment (teleports/jumps/NaN/static)
  cluster.py           unsupervised motion clusters (action pseudo-labels)
  video.py             optional raw-video pre-analysis (skip-empty, camera motion)
  cli.py               `python -m videotomocap ...`
  backends/
    base.py            HMRBackend ABC + rotation/SMPL-family conversion helpers
    gvhmr.py           default body-only: wraps GVHMR, parses hmr4d_results.pt
    wham.py  tram.py   alternative body-only world-grounded backends
    smplx_frames.py    whole-body SMPL-X (smplestx/camenduru_smplerx/whac/osx/hand4whole/multihmr)
    fusion.py          body + hand net (WiLoR/HaMeR) → SMPL-X with real hands
    noop.py            synthetic backend (no GPU) for tests/dry-runs
motion_model/          pipeline 2: MDM fine-tuning bridge, config, and docs
scripts/
  selftest.py          GPU-free end-to-end test of pipeline 1
  check_code_health.py commit-time size/indentation gate
  install_hooks.sh     enable the pre-commit hook
tests/                 unit tests (rotation math, hands/fusion, config/ingest/CLI/parallel)
configs/               example + dropzone + fusion configs
dropzone/              drop your videos here
RESEARCH_WATCHLIST.md  breaking papers with no usable code yet (what to watch)
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
