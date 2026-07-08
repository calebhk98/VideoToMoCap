# Multi-person & multi-user support

Single-subject ("moves like *me*") is the default. Turn on `multi_person` to
recover **every person** in the footage, assign them to identities, gate every
person on consent, and export one motion dataset per person — so a whole family
(everyone consenting) can each get a model that moves like them.

GPU-free end-to-end check: `python scripts/multiperson_selftest.py`.

**Works for one person too.** `multi_person` degrades cleanly to a single subject:
one track per clip, shape-clustering resolves to a single identity (it won't
over-split one person's natural jitter — auto cluster-count uses an MST gap test),
and the per-person export is just that one person. You still grant consent once.
If you only ever have one subject, the plain single-subject default is simpler and
needs no consent step — but turning `multi_person` on does not break the 1-person
case. Every option here is in the config; `videotomocap config-template` prints a
fully-commented reference.

## Quick start

```bash
# 1. recover everyone (needs a multi-person backend; see below)
python -m videotomocap hmr --multi-person          # or multi_person: true in the config

# 2. cluster tracks into people by body shape, then review/label + consent
python -m videotomocap people assign               # -> person_00, person_01, ...
python -m videotomocap people list
python -m videotomocap people grant --all          # or --id person_00 --id person_01

# 3. build — only consenting people export; also writes dataset/by_person/<id>/
python -m videotomocap build

# 4. train "moves like dad" from his sub-dataset
python -m motion_model --dataset-dir work/dataset/by_person/person_00 train
#   or one shared, promptable model:  motion_model conditioning: person
```

## Identity model

- **`track_id`** — a per-clip label (`p0`, `p1`) from the backend's tracker. Not a person.
- **`person_id`** — a **project-local label** ("dad", `person_00`) that crosses clips.
  It is *just a string* — never an embedding, never a stored biometric vector.
- Assignment (`person_assignment`):
  - `shape` (default) — cluster tracks by SMPL **betas** (body shape is a stable
    per-person signature). Deterministic k-means over the retained betas.
  - `manual` — assignment is left to you (set each track's `person_id` in the
    manifest); no biometrics are computed. A per-track labelling CLI is a follow-up.
  - `single` — every clip is the same one person (`people assign` maps all to `person_00`).
- Body shape is a strong identity cue but ambiguous for similar builds; the
  assignment seam returns a `{unit_id: person_id}` map, so stronger cues
  (**face recognition / gait / appearance re-ID**, see Research below) can be
  dropped in later without touching the rest of the pipeline.

## Consent (fail-closed)

Participating means consenting to biometric use — identity needs body shape, so
"no to biometrics" is "no to multi-person." That is enforced, not assumed:

- `people.json` is the registry: per `person_id`, a display name + a consent record.
- **A track only exports if its person's consent is `granted`.** Unassigned tracks
  follow `unassigned_policy` (default `exclude` = fail-closed). Non-consenting people
  who merely appear in frame are dropped as whole tracks.
- Every grant/revoke/assign is appended to `consent_log.jsonl` — the "who agreed,
  when" audit trail.
- **Revocation is real:** `build` is idempotent, so revoking a person and
  re-running `build` removes them from the dataset. Honest caveat: motion already
  *exported or trained on* upstream can't be recalled — same limitation as the SMPL
  licence note.

## The privacy invariant, precisely

The core rule still holds where it matters: **the exported dataset is
shape-neutral.** `anonymize()` zeroes betas before any pose npz is written, and
`to_amass_npz` writes `betas = 0`; the self-tests assert it for both the single
and multi-person paths.

What multi-person adds: body shape is **retained in one consent-gated place** —
`work/identity/*.npz` — purely so identity clustering is re-runnable. It never
enters `work/pose/` or the dataset. So the training data a downstream user
receives carries motion + a person *label*, never a body model. Two honesty notes:

- The identity store *does* hold biometrics on the operator's own machine (needed
  for assignment). It's intermediate `work/` data, not shared output.
- Multiple people in a shared world frame create a **relational fingerprint**
  (who-is-with-whom, spacing) that shape-dropping doesn't hide. Motion style is
  already weakly identifying; multi-person adds co-occurrence. Know this before
  sharing multi-person data.

## Backends (who can recover everyone)

`multi_person` needs a backend that returns *all* people from `run_tracks()`. The
default wraps single-subject `run()` as one track, so nothing breaks — but you get
one person until you use a multi-person-capable backend:

- **Already integrated, permissive, multi-person (just promote to all tracks):**
  `trace` (Apache-2.0, world-grounded) and `hmr2`/4D-Humans (MIT, PHALP tracks
  everyone). Body-only; add hands via the `fusion` path.
- **New, strongest fit (non-commercial licences — fine for personal use):**
  **Human3R** (multi-person SMPL-X + world trajectories + camera, one pass) and
  **PromptHMR** (mature, world-grounded, whole-body, from the TRAM lineage). See
  `RESEARCH_WATCHLIST.md`.
- **Guarded:** the per-frame `smplx_frames` backends (Multi-HMR, camenduru
  SMPLer-X) emit one file *per detection* and would silently stack two people as
  consecutive frames — the pipeline now **fails loud** on that instead of
  corrupting, until per-frame track assembly is implemented.

## Captions with multiple people

Scene captions handle multiple people at the *scene* level ("two people cooking"),
and the caption↔motion bridge is track-aware: each *consenting* person's motion
snippet is paired with the segment caption and tagged with `person_id`. But the
caption is still scene-level — it isn't attributed to that specific person. Making
it person-accurate ("*dad* is chopping") needs **per-person grounded captioning**
(DAM-3B-Video / VideoRefer per track, or extending the Qwen2.5-VL fine-tune with
person boxes) — deferred; see Research.

## Multi-user (others running it)

A "project" is already just `{footage_root, work_root, config, people.json,
consent_log}` — so a second user is a second directory, no shared state. The
per-user isolation is emergent from the existing config-first design; formal
`init`/scaffolding is a small follow-up.

## What's implemented vs deferred

| Area | Status |
|---|---|
| Multi-track contract (`run_tracks`) + GPU-free noop | ✅ |
| Per-track manifest + per-track pose npz | ✅ |
| Shape-clustering identity assignment | ✅ |
| People registry + consent + audit + revoke | ✅ |
| Consent-gated per-person datasets + per-person mirror | ✅ |
| Track-aware caption bridge + `conditioning: person` | ✅ |
| smplx multi-detection corruption guard | ✅ |
| Real multi-person parsing in each GPU backend | ⏳ contract ready; wire per backend |
| Face/gait/appearance re-ID (stronger than shape alone) | ⏳ assignment seam ready |
| Per-person grounded captioning ("who did what") | ⏳ research pointers in watchlist |

## Research pointers

See `RESEARCH_WATCHLIST.md` for the surveyed models: multi-person HMR (Human3R,
PromptHMR, CoMotion), re-identification (InsightFace/ArcFace, OpenGait, SOLIDER),
and per-person captioning (DAM-3B-Video, VideoRefer, Sa2VA).
