# Pipeline 2 — the motion model

Pipeline 1 (`videotomocap`) turns footage into an anonymized **AMASS-SMPL-H
dataset**. This package turns that dataset into a model that *moves like you*,
method-selectable from a YAML exactly like Pipeline 1's backend.

```bash
python -m motion_model methods                                     # list methods
python -m motion_model --config configs/motion_model.yaml info     # what it will do
python -m motion_model --config configs/motion_model.yaml prepare   # dataset -> training data
python -m motion_model --config configs/motion_model.yaml train     # prepare + launch training
```

Whole chain in one command: `python scripts/run_pipeline.py`.

## Methods (`method:` in the config)

| method | role | features | notes |
|---|---|---|---|
| `momask` | generator, ~44M (**default**) | HumanML3D 263-d | best quality/effort; MIT |
| `mdm` | generator, ~35M | HumanML3D 263-d | `personalization: lora` → LoRA-MDM (keeps text control) |
| `protomotions` | physics controller | **AMASS npz direct** | fully automated data path; Apache-2.0, maintained |
| `closd` | closed-loop planner+controller | HumanML3D 263-d | your generator driving a physics tracker; MIT |
| `noop` | synthetic (tests) | — | no GPU/repo/weights |

Each trainer shells out to its upstream repo (set `repo:` in the config) and
fails loud with an actionable message when a repo/asset is missing. It stages your
prepared data where the upstream loader expects it and builds the correct command
(verified against each repo). Upstream-specific hyperparameters go through
`extra_args:`.

## Two problems, not one

Training a generator (A) produces movement on demand. Making the character *decide*
what to do (B) is separate. For rendered content you usually **author** the action
timeline (a behaviour tree or a small LLM planner emitting text goals) rather than
train an autonomous policy — so you often don't need (B) at all. If you do want
physics-realistic autonomy, `protomotions`/`closd` are the (B) layer; the final
render is SMPL→MetaHuman via UE5's IK Retargeter (mature, off-the-shelf).

## HumanML3D features (generator methods) — automated & offline

`momask`/`mdm`/`closd` train on the HumanML3D 263-d representation. `prepare`
extracts it **automatically** when `tmr_repo` + `smpl_model` are set: a forward
pass (`human_body_prior`) gets the 22 joints, then `Mathux/TMR`'s
`joints_to_guofeats` (byte-exact HumanML3D, ships its own reference skeleton) makes
the 263-d vector — offline, no notebooks, no gated AMASS clip. `setup` clones TMR
and installs the deps. `protomotions` skips this entirely (consumes the AMASS npz).

Body model: only the 22 body joints are used, so a neutral **SMPL-H or SMPL-X**
works — reuse the one your HMR backend already required (no second registration).
When fine-tuning, use the pretrained checkpoint's shipped `Mean.npy`/`Std.npy`
(not corpus stats). Conditioning: `none` (style), `action` (reuses Pipeline 1's
`action_cluster` labels automatically), or `text`.

For `text`, point `dataset_dir` at the **captioned dataset** the Pipeline 3 bridge
produces (`python -m videotomocap caption-dataset` — it slices each clip's motion
at Pipeline 3's caption-segment boundaries and writes one `(motion snippet,
caption)` pair per segment). `prepare` then reads each clip's `caption` field and
writes real `texts/<clip_id>.txt` automatically; without that field it falls back
to a loader-valid placeholder (so you can still hand-fill `texts/`).

## One config, any scale (`auto_scale` + `early_stop`)

The same YAML should work whether you point it at 100 hours of your own footage or a
server's corpus — you shouldn't hand-tune per dataset. Two switches make it adaptive:

```yaml
auto_scale: true      # size the run to the corpus (num_steps + regime)
early_stop: true      # actually stop at the overfitting onset (needs eval_every > 0)
eval_every: 2000
```

- **`auto_scale`** (`motion_model/autoscale.py`, preview with `python -m motion_model
  autoscale`) measures the prepared corpus and picks the regime it can support:
  `num_steps` scaled to data volume (≈40 passes over the frames, clamped), and
  `personalize-tiny → finetune → large-corpus` selecting LoRA-vs-full, warm-start-vs-
  scratch, and the recommended method. It **applies** the safe knobs (`num_steps`, and
  `personalization` where the method supports it) and **prints** the rest, because
  switching method or warm-starting needs a repo/checkpoint it can't conjure.
  *Honest limit:* the generators are fixed-size (MoMask ~44M, MDM ~35M) — you can't
  grow a transformer's width without discarding the prior — so this scales the
  *regime*, not the parameter count.
- **`early_stop`** (`motion_model/earlystop.py`) does real early stopping the only way
  an orchestration layer can: the upstream trainers run their full `num_steps` and
  never self-stop, so the run is monitored and the **trainer subprocess is terminated**
  once the val curve turns up for `early_stop_patience` evals — then you keep the best
  checkpoint. This is why `num_steps` from `auto_scale` is just a ceiling: the data
  decides the real stopping point. Needs `eval_every > 0` so a val curve exists.

## Overfitting guard

Training happens inside the upstream loop (a subprocess), so this package can't
watch the loss live — but it brackets that loop on both sides (`motion_model/overfit.py`):

```bash
python -m motion_model --config c.yaml overfit-check    # BEFORE training: risk estimate, no GPU
python -m motion_model --config c.yaml overfit-report   # AFTER training: best pre-overfit checkpoint
```

- **`overfit-check`** (also printed automatically at the top of `train`) estimates
  risk from how much motion you have vs how hard you're about to train on it —
  *exposures* (`num_steps·batch / frames`), total minutes, warm-start, LoRA — and
  prints concrete fixes. It's **per-donor aware**: it breaks the corpus down by
  `person_id`, flags under-represented donors, and *lowers* the risk verdict as the
  donor count grows (a 100-person corpus is a different regime from hours of one
  person — see below).
- **`save_every` / `eval_every`** turn on frequent checkpointing + evaluation on the
  held-out split (Pipeline 1's val clips, written to `test.txt`). MDM/CLoSD map these
  to `--save_interval` / `--eval_during_training`. Off by default.
- **`overfit-report`** reads the resulting train/val curve (`checkpoints/metrics.jsonl`,
  or `--metrics <path>` for the trainer's own log) and tells you the best-val step and
  whether val turned back up — i.e. which checkpoint to keep instead of the last one.
  `early_stop_patience` sets how many worsening evals count as a real upturn.

### Scaling to many donors

The guard is built for a corpus that grows from one person to a hundred. It reads
`person_id` straight from Pipeline 1's multi-person export, so as more people donate
footage: per-donor coverage is reported, thin donors are flagged, the overfitting
verdict relaxes with diversity, and — when there's more than one donor and
`conditioning` isn't `person` — it reminds you to set `conditioning: person` (or train
per-person sub-datasets) so distinct styles don't average into one. No config changes
are needed as the donor count scales; the same `overfit-check` reflects the new corpus.

## Model & training notes

- **Don't train >35M from scratch on hours of one person — it overfits.** Warm-start
  from a pretrained checkpoint (`resume_checkpoint:`), then full fine-tune.
  (`overfit-check` will say exactly this when it applies — and stop saying it once
  the corpus is broad enough that from-scratch becomes reasonable.)
- **Hands:** the data carries MANO hands, but HumanML3D's 263-d body features drop
  them. Real hands need a whole-body representation (Motion-X + HumanTOMATO) — real
  but immature; deferred. Nothing is lost by waiting.
- **Moving/handheld footage:** prefer `backend: gvhmr` in Pipeline 1 (gravity-view,
  least trajectory drift) and keep `refine`/`quality_filter` on, so HMR jitter
  isn't learned as your style.

## Hardware (2× RTX 3090)

Generator fine-tune: one 3090, hours. Physics (`protomotions`): fits 24 GB for
state-based tracking (DDP across both cards, no NVLink); the real cost is Isaac
Lab setup, budget 1–3 months. The compute sink is Pipeline 1 (HMR over the
backlog), both cards in parallel.

## Body-model registration & licence

There is no clean zero-account path: every FK-compatible body model
(SMPL/SMPL-H/SMPL-X/STAR/SUPR) is MPI-licensed with a no-redistribution clause,
and the HMR backends require one just to run. The honest floor is **one free MPI
academic registration** (~2 min) — the feature step reuses that same model, so you
don't register twice. Un-gated mirrors exist but violate the licence; this repo
doesn't script them. For **paid** output you additionally need a commercial SMPL
licence from Meshcapade — that's yours to obtain.
