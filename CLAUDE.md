# CLAUDE.md — working notes for this repo

Guidance for Claude (and humans) editing this codebase. Read this before
changing code.

## What this project is

Turn hours of personal camera footage into a model that moves like the user.
Two pipelines:

1. **Data pipeline** (`videotomocap/`) — video → recover human motion → strip
   body identity → emit an AMASS-format motion dataset. **Fully built here.**
2. **Motion model** (`motion_model/`) — fine-tune a small motion-diffusion model
   (MDM) on that dataset. **Bridged + documented; training runs upstream.**

The heavy neural stages (HMR inference, motion-model training) shell out to
upstream research tools behind adapters. Everything else is plain NumPy and runs
on a laptop. See `README.md` for the method research and rationale.

## Quick-and-dirty code layout

```
videotomocap/
  config.py       PipelineConfig dataclass + YAML loader. All knobs live here.
  ingest.py       Scan footage → Manifest (JSON). Clip states + exclude/include.
  pose.py         SmplMotion container (SMPL-72 body + optional MANO hands).
                  anonymize() drops betas = THE privacy step. resample_fps().
  dataset.py      Aggregate anonymized clips → AMASS-SMPL-H npz + train/val split.
  pipeline.py     Orchestration: scan → hmr → anonymize → build. Resumable.
  cli.py          `python -m videotomocap <cmd>`. Thin wrapper over the above.
  backends/
    base.py         HMRBackend ABC + rotation/SMPL conversion helpers (pure NumPy).
    gvhmr/wham/tram Body-only backends (fast; hands zero-padded).
    smplx_frames.py Whole-body SMPL-X backends (smplestx/whac/osx/hand4whole/
                    multihmr) — one per-frame SMPL-X parser, recovers hands.
    fusion.py       Body backend + hand net (WiLoR/HaMeR) → SMPL-X with real
                    hands. FK wrist relocalization + smoothing glue lives here.
    noop.py         Synthetic backend — no GPU/weights. Powers the tests.
motion_model/     Pipeline 2: MDM data-prep bridge, finetune config, docs.
scripts/selftest.py   GPU-free end-to-end test of pipeline 1.
tests/            Unit tests (rotation math, pose helpers).
dropzone/         Where the user drops videos (git-ignores the media).
configs/          Example + dropzone YAML configs.
```

### Data flow / the one contract that matters

```
video ──backend.run()──▶ SmplMotion ──anonymize()──▶ SmplMotion(betas=None)
                          (SMPL-72)                    ──save_npz──▶ work/pose/*.npz
                                                       ──build_dataset──▶ work/dataset/amass/*.npz
```

**Every backend must return a `SmplMotion` with `poses` shape `(T, 72)`
axis-angle** (`global_orient[3] + body_pose[69]`). That is the seam that keeps
the rest of the pipeline backend-agnostic. `backends/base.py` has helpers
(`to_axis_angle`, `assemble_smpl72`, `assemble_hand`, `axis_angle_to_matrix`) to
convert from SMPL-X 63-dim body, rotation matrices, or 6D. If you add a backend,
convert to SMPL-72 there and nowhere else.

**Hands are optional and carried alongside the body:** a backend may also set
`left_hand_pose`/`right_hand_pose` `(T, 45)` MANO. Body-only backends leave them
`None` (→ neutral). `anonymize`/`resample_fps`/npz round-trip all preserve
hands; the AMASS export drops them into the SMPL-H hand slots. Never fold hand
articulation into the SMPL-72 body block.

### Invariants — do not break these

- **Privacy:** SMPL `betas` (body shape / identity) must never leave the
  pipeline. `anonymize()` zeroes them; the self-test asserts it. Any new export
  path must keep betas neutral.
- **Resumable:** state lives in the manifest on disk, checkpointed after every
  clip. One clip failing is recorded (`status=failed`) and skipped, never fatal.
- **Core stays light:** only `numpy` (and optional `pyyaml`) may be imported at
  module top-level in `videotomocap/`. Heavy deps (`torch`, etc.) are
  **lazy-imported inside backend methods** so the package imports and tests run
  without a GPU.
- **Tests stay GPU-free:** the `noop` backend must be enough to exercise the
  whole flow. Don't add a test that needs real weights/CUDA.

## Coding conventions

Match the existing style. Concretely:

- **Never deeply nest.** Prefer early returns / guard clauses over `if/else`
  pyramids. Aim for ≤ 2 levels of indentation inside a function. If you're
  writing a third nested `if`, extract a helper or invert a condition.
- **Comment the *why*, not the *what*.** Assume the reader knows Python. Explain
  intent, domain quirks (SMPL-X vs SMPL joint counts, why betas are dropped,
  atomic manifest writes), and anything a future reader would trip on. Don't
  narrate obvious lines. Module/class docstrings carry the mental model.
- **Small, single-purpose functions.** One reason to change each.
- **Fail loud with context.** Raise `BackendError`/`ValueError` with a message
  that says what was expected vs got and how to fix it (see `_require_repo`).
- **Descriptive names**, no abbreviations that aren't already domain terms
  (`betas`, `transl`, `hmr` are fine; invented shorthand is not).
- **Determinism in tests.** Seed RNGs; split by clip id, not by frame.
- **Atomic writes** for anything that's a source of truth (see `Manifest.save`
  writing a `.tmp` then `replace()`).
- **No new heavy dependency** without a strong reason — this repo's value is
  being a light orchestration layer over the heavy upstream tools.

## How to add a new HMR backend

1. New file `backends/<name>.py`, subclass `HMRBackend`, implement `run()`.
2. Shell out to the tool with `self._run_cmd(...)`, `self._require_repo()`.
3. Parse its output, convert rotations with `assemble_smpl72(...)`, return a
   `SmplMotion` (frame = `self.cfg.use_frame`, set `betas` if available).
4. Register it in `backends/__init__.py` `_REGISTRY`.
5. Backends are selected by `PipelineConfig.backend` — swapping is one config
   line. Keep that true.
6. `_run_cmd` already pins the subprocess to `cfg.cuda_device` when the parallel
   runner assigns a GPU — use it for any subprocess so parallelism works.

## Parallelism

`pipeline.run_hmr(..., workers=N, gpus=[...])` fans clips out over a thread pool,
pinning one clip per GPU (heavy work is in the backend subprocess, GIL released).
Keep it crash-resumable: mutate the manifest and `save()` under the lock, one
clip at a time. Clip work must stay independent (own scratch dir, own pose npz) —
don't introduce shared mutable state between clips.

## Code health (enforced on commit)

`scripts/check_code_health.py` runs as a pre-commit hook (enable with
`bash scripts/install_hooks.sh`). It blocks files > 500 lines, > 1 MB, or nested
> 6 blocks deep. If you're about to exceed these, that's the signal to split the
file or extract a helper — not to bump the threshold. It measures *true* block
nesting (tokenize), so it enforces the "never nest" rule directly.

## Running things

```bash
pip install -r requirements.txt
python scripts/selftest.py            # end-to-end, no GPU — must stay green
python -m pytest tests/               # unit tests — must stay green
python scripts/check_code_health.py --all   # size/indent gate
python -m videotomocap --config configs/dropzone.yaml scan   # real usage
```

Before committing non-trivial changes, run the tests + self-test + health check
above. Keep commit messages descriptive; branch off the default branch (don't
commit straight to it).
