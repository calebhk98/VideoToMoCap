# Pipeline 2 — the motion model (making an agent move like you)

Pipeline 1 (the `videotomocap` package) turns your footage into an
**AMASS-format motion dataset** of anonymized SMPL pose. This second pipeline
turns that dataset into a model that *generates* your movement. Two genuinely
separate problems live here — do not conflate them:

| | What it does | Status |
|---|---|---|
| **(A) Motion generator** | produces movement given a condition (text, action label, or nothing) | this folder — fine-tune MDM |
| **(B) Behavior / decision layer** | decides *what* to do at each moment in a virtual environment | out of scope of "train a model"; see the note at the bottom |

Training a motion model only solves (A). A generator that can produce "you
walking / sitting / reaching" is not yet an agent that *chooses* to do those
things — that's (B), a control/RL/planning problem that consumes (A) as a
primitive.

## Why fine-tune MDM rather than train from scratch

- **Size.** Motion-diffusion models are small — MDM's transformer is on the
  order of tens of millions of parameters, not billions, because joint
  rotations over time are far lower-dimensional than pixels or text. (Confirm
  the exact count from the checkpoint you download; published configs are ~8
  transformer layers, latent 512.)
- **Data.** You have hours of one person. That is plenty to *specialize* a
  pretrained motion prior toward your style, and far too little to learn general
  human motion from zero. There is **no published scaling law for motion
  diffusion specifically** — don't import Chinchilla's ~20 tokens/param ratio
  (that's for autoregressive text). Use the data:param ratio of the checkpoint
  you fine-tune as your empirical reference point.
- **Mirrors the video-gen playbook:** start from a general prior, adapt on your
  data.

Repos (availability verified 2026-07):
- MDM — https://github.com/GuyTevet/motion-diffusion-model
- priorMDM (fine-tuning recipes) — https://github.com/priorMDM/priorMDM

## Steps

### 1. Convert the AMASS dataset to HumanML3D features

MDM trains on the HumanML3D 263-dim feature representation, not raw SMPL. The
conversion is: SMPL params → 22 joint positions (SMPL forward kinematics) →
HumanML3D features (`motion_representation`). It needs the SMPL body model
(register at https://smpl.is.tue.mpg.de) and the HumanML3D repo.

```bash
python motion_model/prepare_mdm_data.py \
    --dataset work/dataset \
    --humanml3d /opt/HumanML3D \
    --smpl-model /opt/body_models/smpl \
    --out work/mdm_data
```

`prepare_mdm_data.py` handles the parts that don't need those external assets
(indexing, splits, text-annotation scaffolding) and invokes the HumanML3D
feature extractor for the rest. Read its header for exactly which step needs
what.

### 2. Fine-tune

```bash
python -m train.train_mdm \
    --save_dir save/mymotion \
    --dataset humanml \
    --data_dir work/mdm_data \
    --resume_checkpoint save/humanml_trans_enc_512/model000475000.pt \
    --num_steps 80000 --batch_size 64
```

See `finetune_mdm.yaml` here for the recommended starting hyperparameters
(matches the priorMDM control-task recipe: 80k steps, batch 64).

### 3. Conditioning choice

Your footage has no text labels. Options, cheapest first:

1. **Unconditional fine-tune** — drop text, learn "motion that looks like you."
   Simplest; pair with (B) to drive it.
2. **Action labels** — cluster clips (or hand-label a few) into coarse actions
   (walk, sit, cook, ...) and condition on the label.
3. **Auto-captioning** — run a video captioner over each clip to get text, then
   train the standard text-conditioned MDM. Most work, most controllable.

`prepare_mdm_data.py` writes an empty caption per clip by default (option 1);
fill `texts/<clip_id>.txt` to move toward option 2/3.

## The hands problem (unsolved, flagged not hidden)

Body-only HMR (GVHMR/WHAM/TRAM) does **not** recover fine hand articulation, so
"pick up an apple and eat it" will have a floating/neutral hand in the current
dataset (Pipeline 1 zero-pads the hand joints). Paths forward:
- Run **DanceHMR** (hand-aware whole-body, ByteDance, 2026) on the eating/manual
  clips and merge its hand params into the SMPL-X hand slots.
- Move the whole pipeline to **SMPL-X** and train a whole-body motion model.

Either is a real extension, not a config flag — treat it as phase 2.

## (B) The behavior layer — pointer

Once (A) exists, "acts like me in a virtual environment" is a control problem:
a policy that selects goals/actions and uses the motion generator (or a
physics-based tracking controller like those in the PHC / PhysHOI / Ke line of
work) to realize them. That's a separate project; this repo deliberately stops
at producing a high-quality, personal motion generator it can build on.
