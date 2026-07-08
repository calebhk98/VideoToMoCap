# VideoToMoCap

Turn a large backlog of personal camera footage into an anonymized, AMASS-format
motion dataset — and from there into a motion model that moves like the person in
the video.

The project is **three pipelines**; this repo builds the first end-to-end,
scaffolds the second, and adds an independent captioning/search pipeline over the
same footage:

1. **Data pipeline — video → anonymized motion data** (`videotomocap/`).
   Recover human motion from footage, strip body identity (drop SMPL `betas`),
   and export a standard motion dataset. Fully implemented here.
2. **Motion model — data → an agent that moves like you** (`motion_model/`).
   Fine-tune a small motion-diffusion model on that dataset. Bridged and
   documented; the heavy training runs in the upstream MDM repo.
3. **Archive captioning & search — video → searchable captions → a video-native
   captioner** (`videocaption/`). Segment, caption per-frame (JoyCaption), and
   aggregate into a `(description, tags)` search index; then LoRA-fine-tune
   Qwen2.5-VL on those labels so it captions a whole clip in one pass. Its light
   layer is fully built (stdlib-only, GPU-free tests); the models are bridged.
   Its captions can also **pair with Pipeline 1's motion** (`videotomocap
   caption-dataset`) to train a *text-conditioned* movement model. See
   [`videocaption/README.md`](videocaption/README.md).

Everything except the neural stages (HMR inference, motion-model training, the
caption VLMs) is plain NumPy/stdlib and runs on a laptop. Those stages shell out
to upstream research tools behind clean adapters, so the core imports and tests
with no GPU.

## Quickstart

Install the orchestration layer (the GPU-free parts — enough to run the self-test
and manage a dataset):

```bash
pip install -r requirements.txt      # numpy, pyyaml
python scripts/selftest.py           # end-to-end, no GPU/weights — must stay green
python tests/test_conversions.py     # rotation-math unit tests
```

Then drop videos in `dropzone/`, point a config at an installed backend, and run
the whole pipeline with one command:

```bash
python -m videotomocap run           # scan → exclude → hmr → build, all from the config
```

Every knob lives in the YAML config. For a fully-commented file listing **every**
option with its documentation (each of the three pipelines has one):

```bash
python -m videotomocap config-template > my_config.yaml   # also: videocaption / motion_model
```

## Backends (HMR methods)

The HMR backend is **one config line**, so you can benchmark two on a day of
footage and switch without touching code. Body-only backends are fastest for bulk
work; the whole-body SMPL-X and `fusion` families recover articulated hands.

| `backend:` | Kind | Hands | Notes |
|---|---|---|---|
| `gvhmr` `wham` `tram` | body-only | ✗ (neutral) | world-grounded (VO/SLAM); fastest bulk. **`gvhmr` is the default** (throughput leader; clean `hmr4d_results.pt` output) |
| `trace` | body-only | ✗ (neutral) | world-grounded [TRACE](https://github.com/Arthur151/ROMP) (`pip install simple-romp`); **Apache-2.0** — the most permissive backend license |
| `hmr2` (a.k.a. `4dhumans`) | body-only | ✗ (neutral) | [4D-Humans](https://github.com/shubham-goel/4D-Humans) HMR2, per-frame + PHALP tracking (camera-relative, not world-grounded) |
| `smplestx` (a.k.a. `smplerx`) | whole-body SMPL-X | ✓ | recommended general whole-body model |
| `camenduru_smplerx` (a.k.a. `camenduru`) | whole-body SMPL-X | ✓ | SMPLer-X via [camenduru's runnable repackaging](https://github.com/camenduru/SMPLer-X) |
| `hybrik` (a.k.a. `hybrikx`) | whole-body SMPL-X | ✓ | [HybrIK-X](https://github.com/jeffffffli/HybrIK) analytical-neural IK; **MIT** license (vs non-commercial peers) |
| `whac` | whole-body SMPL-X | ✓ | **moving-camera + world-grounded** with hands |
| `hand4whole` | whole-body SMPL-X | ✓✓ | CVPR 2026, MIT, best single-model hands |
| `osx` `multihmr` | whole-body SMPL-X | ✓ | MIT / fast alternatives |
| `sam3dbody` (a.k.a. `sam3d`) | whole-body SMPL-X | ✓ | [Meta SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) foundation model → SMPL-X via MHR fit (heavy; SAM License) |
| `fusion` | body + hand net | ✓✓ | any body backend **+ WiLoR/HaMeR** fingers |

Body-only methods under-recover fingers ("pick up an apple and eat it" comes out
with a neutral hand). Every hand-capable backend carries hands all the way
through anonymization and into the AMASS-SMPL-H hand slots, so the motion model
can learn them.

The **`fusion`** backend is the most flexible: it runs any body backend for the
body + camera, runs WiLoR or HaMeR for the fingers, and grafts the MANO finger
pose into SMPL-X's `left_hand_pose`/`right_hand_pose` (One-Euro smoothing,
last-good-frame gap filling, optional FK-based wrist relocalization):

```yaml
backend: fusion
body_backend: whac       # any body/whole-body backend (moving-camera here)
hand_backend: wilor      # 'wilor' (fast, default) or 'hamer'
backend_repo: /opt/WHAC          # the BODY tool
hand_repo:    /opt/WiLoR         # the HAND tool
graft_wrist: false       # advanced: compose the hand-net wrist into the chain
```

DanceHMR (the ideal single video-native hands model) was withdrawn with no public
code as of 2026-07, so it is documented but not wrapped.

### Installing a backend

Backends have heavy, mutually-incompatible environments (torch/CUDA, SMPL-X body
models, SLAM deps). Install each in its **own** env from upstream — don't mix them
into this one — then point the config at it:

```bash
git clone https://github.com/zju3dv/GVHMR /opt/GVHMR
# follow GVHMR/docs/INSTALL.md (conda env + SMPL-X weights via registration)
```

```yaml
backend: gvhmr
backend_repo: /opt/GVHMR
backend_python: /opt/miniconda3/envs/gvhmr/bin/python
```

Most backends need SMPL/SMPL-X/MANO body models plus their own checkpoints,
usually behind separate registrations. [`lithiumice/models_hub`](https://huggingface.co/lithiumice/models_hub)
is a Hugging Face mirror bundling many of them (SMPL/SMPL-H/SMPL-X, MANO, FLAME,
and the HMR2/4D-Humans checkpoints the `hmr2` backend uses) in one git-LFS repo —
check each model's license before relying on it.

**Hardware:** none of these need more than ~10 GB for batch-1 inference, so a
single 24 GB card runs any of them. A second card is headroom — run `fusion`'s
body and hand tools on separate GPUs, or process two clips at once.

## Usage

**Everything lives in the config — you don't need to remember flags.** Set your
options once (footage root, backend, exclusions, GPUs, …) in a YAML and the
commands read them. The config is auto-discovered: put a `videotomocap.yaml` in
the working directory (or point `$VIDEOTOMOCAP_CONFIG` at one), no `--config`
needed.

```yaml
# videotomocap.yaml
footage_root: dropzone
backend: gvhmr
backend_repo: /opt/GVHMR
backend_python: /opt/miniconda3/envs/gvhmr/bin/python
exclude_patterns: ["*/2024-12-24/*", "*/2025-06-1*"]   # e.g. family visits, set once
gpus: auto
workers_per_gpu: auto
```

`run` = `hmr` + `build`, with `scan` run automatically the first time. Individual
steps (each still reads the config; flags are optional overrides):

```bash
python -m videotomocap scan       # (re)build the manifest; applies exclude_patterns
python -m videotomocap status     # what will be processed
python -m videotomocap hmr        # HMR + anonymize (resumable, checkpoints each clip)
python -m videotomocap build      # aggregate into the AMASS dataset
python -m motion_model train      # dataset → motion model (Pipeline 2)
```

Or the whole chain, including the motion model, in one command:

```bash
python scripts/run_pipeline.py \
    --video-config configs/dropzone.yaml --model-config configs/motion_model.yaml
```

Prefer an isolated, offline container? Three commands (see `docker/README.md`):

```bash
make build                      # build the image
make setup                      # one-time online: repos, envs, weights, SMPL-H
make run                        # videos in ./dropzone → trained model in ./work
```

Override the backend/method with `make setup BACKEND=wham METHOD=mdm`. Any config
field that matters at the CLI also has a flag (`--limit 5`, `--gpus 0`,
`--workers-per-gpu 2`, `--config other.yaml`, `--footage-root …`), and ad-hoc
exclusions still work: `videotomocap exclude --pattern "*/badcam/*"`.

The manifest is the single source of truth; every stage is idempotent and
resumable, so a crash deep into the backlog just means re-running the command. One
clip failing (bad file, no person detected) is recorded and skipped, never fatal.

## Running at scale (parallelism)

Clips are fully independent, so HMR is embarrassingly parallel.

- **GPU auto-detection.** `gpus: auto` (the default in the example configs) runs
  `nvidia-smi` and respects an existing `CUDA_VISIBLE_DEVICES`. Worker count
  defaults to *(#GPUs × `workers_per_gpu`)*.
- **Pack a card.** `workers_per_gpu: N` runs N clips per card, overlapping one
  clip's GPU phase with another's decode/IO. `auto` sizes N from *free* VRAM
  (`(free − headroom) ÷ per-clip footprint`), calibrated on the first clip unless
  you set `vram_per_worker_mb`. Tune with `vram_headroom_mb` (default 2000) and
  `max_workers_per_gpu` (default 8). 2–3 per 24 GB is the usual sweet spot.

```bash
python -m videotomocap hmr --gpus auto --workers-per-gpu 2   # both cards, 2 deep
```

Mechanics and honest limits:
- Each worker runs the backend in its own subprocess with `CUDA_VISIBLE_DEVICES`
  pinned (heavy work is in the subprocess, GIL released → real speedup).
- **Across-clip overlap, not within-clip.** Upstream HMR tools are opaque
  subprocesses, so we overlap *whole clips*, not stages inside one clip. More
  workers than a card's VRAM allows will OOM.
- Manifest is checkpointed under a lock after every clip → parallel runs are as
  crash-resumable as sequential ones.
- **Beyond one machine:** point several machines at the same footage share with
  disjoint `--limit`/exclusions (or split the tree per machine); each writes its
  own pose npz, then run `build` once over the merged `work/pose`.

## Multiple people (a whole family)

Single-subject is the default. Set **`multi_person: true`** (or `hmr --multi-person`)
to recover *every* person per clip, cluster them into identities by body shape,
gate each on **consent**, and export one motion dataset per person — so everyone
in a consenting household can get a model that moves like them.

```bash
python -m videotomocap hmr --multi-person     # recover everyone (needs a multi-person backend)
python -m videotomocap people assign          # cluster tracks -> person_00, person_01, ...
python -m videotomocap people grant --all     # consent is fail-closed: nothing exports without it
python -m videotomocap build                  # per-person datasets in dataset/by_person/<id>/
python scripts/multiperson_selftest.py        # GPU-free end-to-end
```

Identity is a project-local **label**, never a stored biometric; body shape is
retained only in a consent-gated identity store and never enters the (still
shape-neutral) exported dataset. Consenting to participate means consenting to
biometric use — that's the whole premise, and it's enforced (fail-closed, audit
log, real revocation). Full design, backends, and honest caveats:
**[`docs/MULTI_PERSON.md`](docs/MULTI_PERSON.md)**.

## Optional processing

All motion-derived tags below run automatically inside `run`/`build`, are pure
NumPy (no weights, no GPU), and default to non-destructive `flag` unless noted.

- **`refine`** (default off) — post-HMR cleanup: temporal de-jitter + stationary
  anti-drift. `refine_method` is `savgol` (fast uniform local fit), `variational`
  (global smoother minimizing `‖x−y‖² + λ‖accel(x)‖²`), or `confidence` (per-joint
  adaptive: smooths each joint in proportion to its own local jitter — denoises
  inferred/occluded joints hard while leaving cleanly-tracked ones sharp; knee set
  by `confidence_kappa`). These three are pure NumPy (`videotomocap/refine.py`).
  Two more are **learned** passes (`videotomocap/refine_learned.py`), heavy and
  opt-in — each runs in its own env as a subprocess, off unless selected, and
  swapped by one config line:
  - `dposer` — [DPoser-X](https://github.com/moonbow721/DPoser-X) diffusion pose
    prior; motion-only, so it works after any backend. Set `dposer_repo` /
    `dposer_python`.
  - `scorehmr` — [ScoreHMR](https://github.com/statho/ScoreHMR) diffusion
    *image-guided* refinement; re-reads the source video for reprojection
    guidance, so it only runs in the `hmr` stage. Set `scorehmr_repo` /
    `scorehmr_python`.

  ```yaml
  refine: true
  refine_method: dposer     # savgol | variational | confidence | dposer | scorehmr
  dposer_repo: /opt/DPoser-X
  dposer_python: /opt/miniconda3/envs/dposer/bin/python
  ```
- **`auto_mirror`** (`off`/`flag`/`correct`) — some clips (front-camera selfies)
  are horizontally flipped, corrupting handedness. There's no metadata flag and a
  mirrored person still looks valid, so the pipeline scores each clip's handedness
  from the recovered motion, takes the corpus consensus as the true dominant side
  (self-calibrating), and flags confident disagreements. `correct` applies the
  exact SMPL left/right mirror in motion space (no HMR re-run). Tune with
  `mirror_margin`. Heuristic: gross motion is near-symmetric; handedness-specific
  tasks (writing, eating) are what it protects.
- **`quality_filter`** (`off`/`flag`/`exclude`) — scores plausibility from the
  motion alone. Hard failures (`non_finite`, `root_teleport` > `quality_max_speed_ms`,
  `pose_jump` > `quality_max_joint_step`) are drop candidates under `exclude`;
  `static_low_motion` is soft (never auto-dropped).
- **`auto_occlusion: flag`** — marks joints that never move as unreliable (masked
  via `joint_valid`), a proxy for out-of-frame joints HMR freezes. Complements
  manual `camera_occlusions`; the two are unioned.
- **`cluster_actions: <k>`** — k-means the clips into `k` motion clusters and tags
  each with an `action_cluster` id (in the dataset `index.json`) as a free
  conditioning signal. Clustering, not recognition — coarse buckets to name later.
- **`skip_empty` / `auto_camera_motion`** — the only tags that look at *pixels*
  (before HMR), so they need `opencv-python` (lazy; the rest runs without it).
  `skip_empty` excludes clips below `empty_activity_threshold` before they reach
  the GPU — the big saver for mostly-empty security footage. `auto_camera_motion`
  tags locked-off cameras `static` (below `camera_motion_threshold`) so HMR skips
  visual odometry, automating the manual `static_cameras` list.

### Partial-body / occluded footage

Cameras often frame a person waist-up, cut off an arm, or (mounted high) miss the
head. HMR still outputs a full SMPL body — it *infers* the out-of-frame joints —
so the pipeline never crashes on truncation, but those inferred joints are
plausible-but-fake motion. Tag which regions a camera can't see and the unseen
joints get labeled so you can filter or mask them:

```yaml
camera_occlusions:
  cam_desk:  [legs]              # waist-up desk view
  cam_high:  [head]             # ceiling mount, head cropped
  cam_left:  [right_arm]        # subject's right arm out of frame
partial_body_cameras: [cam_desk] # shorthand: same as camera_occlusions {cam: [legs]}
```

Region names (`videotomocap/regions.py`): `legs`, `left_leg`, `right_leg`,
`feet`, `arms`, `left_arm`, `right_arm`, `hands`, `head`. For each clip the union
of their SMPL joints is recorded as `unreliable_joints` in the manifest and
carried through: the anonymized pose npz and exported AMASS npz get a `joint_valid`
mask (24 bools, `False` = inferred), and the dataset `index.json` lists them per
clip. So downstream you can drop those clips, or **mask the loss on the guessed
joints** during training and still use the good ones. Truncation-robust backends
(`smplestx`, `multihmr`, `fusion`) degrade more gracefully than the body-only trio.

## Input types (phone videos, selfies, …)

Any monocular video works — it doesn't have to be security footage.

- **Phone / handheld: yes, well.** Moving-camera is what the world-grounded
  backends (GVHMR/WHAM/TRAM, and WHAC for whole-body+hands) are built for. Don't
  list the source under `static_cameras`. Variable frame rate and portrait
  orientation are fine (every clip is resampled to `target_fps`).
- **Selfie *videos*: yes.** Usually upper-body/close-up — tag with
  `camera_occlusions: {selfie: [legs]}` and prefer a close-up-robust backend.
  Mirrored front-camera files are handled automatically (see `auto_mirror`).
- **A single selfie *photo*: no.** One still is below `min_clip_frames` and gets
  dropped; this pipeline is about motion over time, not snapshots.

## Platforms (Windows + Linux)

The orchestration package runs natively on both, no changes: pure Python with
`pathlib`, atomic manifest writes via `os.replace`, GPU detection via `nvidia-smi`,
and glob exclusions via `fnmatchcase` so `exclude_patterns` match identically on
both OSes. The GPU-free path (ingest, exclude, anonymize, dataset build, `noop`
backend, all tests) works the same either way.

The caveat is the heavy neural backends: GVHMR/WHAM/TRAM/SMPLest-X pull in
Linux-oriented CUDA extensions (DROID-SLAM, Detectron2, PyTorch3D) that are
painful on *native* Windows. There the reliable path is **WSL2** (Ubuntu + CUDA),
where they run as on Linux and this package sits on top unchanged. The pre-commit
hook is a shell script — on Windows enable it from Git Bash (not needed to *run*
the pipeline).

## Layout

```
videotomocap/
  config.py            PipelineConfig (+ YAML loader)
  ingest.py            scan footage → manifest; exclude/include clips
  pose.py              SmplMotion container; anonymize() = drop shape, keep pose
  dataset.py           aggregate → AMASS-SMPL npz + train/val split + stats
  captioned_dataset.py bridge: slice motion at Pipeline 3's caption-segment spans → text-to-motion dataset
  multiperson.py       opt-in multi_person plumbing (MotionUnit, consent gate, per-person export)
  identity.py          cross-clip identity: cluster per-track betas → person_id (consent-gated store)
  people.py            people registry + consent ledger + audit log
  pipeline.py          orchestration (scan→hmr→refine→anonymize→build), parallel+resumable
  gpu.py               GPU auto-detection + worker/device resolution
  regions.py           body regions → SMPL joints (occlusion tagging)
  refine.py            optional post-proc: temporal de-jitter + anti-drift (pure NumPy)
  refine_learned.py    optional learned refine (DPoser-X / ScoreHMR; opt-in, own env)
  mirror.py            auto left/right-mirror detection + correction
  quality.py           auto clip-quality assessment (teleports/jumps/NaN/static)
  cluster.py           unsupervised motion clusters (action pseudo-labels)
  video.py             optional raw-video pre-analysis (skip-empty, camera motion)
  cli.py               `python -m videotomocap ...`
  backends/
    base.py            HMRBackend ABC + rotation/SMPL-family conversion helpers
    gvhmr.py           default body-only: wraps GVHMR, parses hmr4d_results.pt
    wham.py  tram.py   alternative body-only world-grounded backends
    trace.py           body-only world-grounded TRACE (simple-romp; Apache-2.0)
    fourdhumans.py     body-only HMR2 / 4D-Humans (per-frame, camera-relative)
    hybrik.py          whole-body SMPL-X HybrIK-X (MIT)
    sam3dbody.py       Meta SAM 3D Body → SMPL-X adapter (driver: scripts/sam3d_to_smplx.py)
    smplx_frames.py    whole-body SMPL-X (smplestx/camenduru_smplerx/whac/osx/hand4whole/multihmr)
    fusion.py          body + hand net (WiLoR/HaMeR) → SMPL-X with real hands
    noop.py            synthetic backend (no GPU) for tests/dry-runs
motion_model/          pipeline 2: MDM fine-tuning bridge, config, and docs
videocaption/          pipeline 3: archive captioning + search index + Qwen2.5-VL LoRA bridge
scripts/
  selftest.py          GPU-free end-to-end test of pipeline 1
  multiperson_selftest.py  GPU-free end-to-end test of multi_person + consent
  caption_selftest.py  GPU-free end-to-end test of pipeline 3
  joycaption_infer.py  driver: per-frame captioning (JoyCaption via vLLM)
  dolphin_aggregate.py driver: frame captions → (description, tags) (Dolphin3.0)
  check_code_health.py commit-time size/indentation gate
  install_hooks.sh     enable the pre-commit hook
tests/                 unit tests (rotation math, hands/fusion, config/ingest/CLI/parallel, captioning)
configs/               example + dropzone + fusion configs
dropzone/              drop your videos here
RESEARCH_WATCHLIST.md  breaking papers with no usable code yet (what to watch)
```

## Design notes

Rationale behind the defaults, and the state of the art as verified 2026-07.

**Why these HMR methods.** All candidates were checked for public code before
wrapping. Speed below is bulk throughput, the thing that matters for years of
footage.

| Method | Code | Moving camera | Bulk throughput | Verdict |
|---|---|---|---|---|
| **GVHMR** | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | yes (VO/SLAM) | ~real-time class on a 4090 | **default backend** |
| **WHAM** | [yohanshin/WHAM](https://github.com/yohanshin/WHAM) | yes | ~1 h / 10-min clip reported | supported alt |
| **TRAM** | [yufu-wang/tram](https://github.com/yufu-wang/tram) | yes (DROID-SLAM) | multi-stage; not benchmarked here | supported alt |
| **SLAHMR** | [vye16/slahmr](https://github.com/vye16/slahmr) | yes (optimization) | ~78 h / 10-min clip | not wrapped (infeasible at volume) |
| **DanceHMR** | [project page](https://shenwenhao01.github.io/dancehmr/) | — | — | withdrawn, no public code |
| **SAM-Body4D** | [gaomingqi/sam-body4d](https://github.com/gaomingqi/sam-body4d) | yes | — | future backend |

**The privacy mechanism.** SMPL factors a human into **shape** (`betas` —
build/limb-lengths, strongly identifying) and **pose** (per-joint rotations over
time — the movement). `videotomocap/pose.py::anonymize()` keeps pose and
**discards `betas`**, re-expressing motion on a neutral canonical body; the
self-test asserts betas are zero after processing and the export's hand/face slots
are neutral. Honest caveat: motion *style* (gait, posture) is itself weakly
identifying — but that's the point of the project; the strongest biometric, metric
body shape, is dropped.

**Camera handling.** Corner mounts and rotating cameras mean you can't assume a
static camera, so the wrapped world-grounded backends handle it by construction.
`static_cameras: [...]` skips visual odometry on truly-fixed cameras. For a camera
that snaps between fixed presets, split each preset into its own clip/folder.

**The motion model (pipeline 2).** Motion-diffusion models are *small* (MDM: tens
of millions of params) because joint-rotation sequences are far lower-dimensional
than pixels/text. With hours of one person, **fine-tune a pretrained prior**
([MDM](https://github.com/GuyTevet/motion-diffusion-model),
[priorMDM](https://github.com/priorMDM/priorMDM)) — too little data to train from
scratch. For a data:param budget, use motion-specific scaling work
([ScaMo](https://github.com/shunlinlu/ScaMo_code), Being-M0/MotionLib, NVIDIA's
Kimodo), not Chinchilla's text ratio. Note that a motion *generator* is not a
*behavior* layer that decides what to do — this repo produces the former; the
latter is a separate control/RL project (`motion_model/README.md`).

### ⚠️ Licensing — read before any commercial use

The **SMPL / SMPL-X body models are non-commercial** (Max Planck license): no
commercial products/services, and explicitly no training methods for commercial
use. Since GVHMR/WHAM/TRAM and HumanML3D/MDM all depend on SMPL, a model trained
through this pipeline inherits that restriction. For commercial rights, license
SMPL via Meshcapade. This tooling is for personal/research use; clearing the
licensing is your responsibility.

## Honest scope

- **Done and tested (83% line coverage; core modules 95–100%):** ingestion,
  manual exclusion, backend adapter layer, SMPL-72 + MANO-hand normalization and
  rotation conversions, shape/pose anonymization, fps resampling, the
  hand-graft/FK/smoothing glue, AMASS-SMPL-H dataset aggregation with splits,
  resumable orchestration, CLI, MDM data-prep bridge. GPU-free throughout via the
  `noop` backend.
- **Wired but needs a GPU + registered SMPL-X/MANO weights to run:** the neural
  inference itself. Each backend's demo command and output-file layout are
  centralized and commented — verify them against your checkout revision (upstream
  demos drift).
- **Deliberately phase 2:** HumanML3D 263-d feature extraction, a learned
  wrist-correction regressor (the geometric `graft_wrist` is a first cut), and the
  behavior/decision layer.
```
