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
`action_cluster` labels automatically), or `text` (fill `texts/`).

## Model & training notes

- **Don't train >35M from scratch on hours of one person — it overfits.** Warm-start
  from a pretrained checkpoint (`resume_checkpoint:`), then full fine-tune.
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
