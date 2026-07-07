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

## The one manual step: HumanML3D features (generator methods only)

`momask`/`mdm`/`closd` train on the HumanML3D 263-d representation. Converting the
AMASS npz to it is the single step this repo cannot run for you — it needs
registration-gated assets and reproduces a normalization the pretrained
checkpoints bake in, so an in-repo reimplementation would silently diverge.
`protomotions` has **no** such step (it consumes the AMASS npz directly) — prefer
it if you want the fully-automated path.

Precise recipe (one-time), driven from a cloned `EricGuo5513/HumanML3D`:
1. `pip install git+https://github.com/nghorbani/human_body_prior`; get the gated
   **SMPL+H** model (`mano.is.tue.mpg.de`) and **DMPL** (`smpl.is.tue.mpg.de`).
2. `raw_pose_processing`: SMPL-H forward pass on `poses[:, :66]` (+ the Y/Z
   `trans_matrix` swap) → `(T, 22, 3)` joints. `mocap_framerate=20` means no
   resampling. Segment/mirror cells are dataset-curation — skip them.
3. `motion_representation`: `process_file(joints, 0.002)` → 263-d. Its
   `tgt_offsets` is computed once from the KIT clip `000021` (baked into every
   pretrained checkpoint) — reproduce it once and cache.
4. **Use the checkpoint's shipped `Mean.npy`/`Std.npy`** when fine-tuning; only
   recompute (`cal_mean_variance`, block-uniform per the 7 feature groups) if you
   train from scratch.

`prepare` lays out everything around this (splits, captions, amass copy) and
prints the hand-off. Conditioning: `none` (style only), `action` (reuses
Pipeline 1's `action_cluster` labels automatically), or `text` (fill `texts/`).

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

## Licence gate (paid use)

Framework code above is permissive (MIT/Apache-2.0), but the **SMPL/SMPL-H body
models every method depends on are non-commercial by default** — a paid product
needs a commercial SMPL licence from Meshcapade. That's yours to obtain; the
`smpl_model:` path is your responsibility.
