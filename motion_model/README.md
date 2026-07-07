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

> **You probably don't need (B) to make content.** "No live performer" splits
> into *authored* (you script the sequence of action prompts — cheap,
> controllable, works today) vs *autonomous* (a learned policy decides actions
> live). For rendered video, author the timeline and drive a **text-conditioned**
> generator. See **`NEXT_STEPS.md`** for the full ordered plan, and the
> recommendation to personalize with **LoRA-MDM** (keeps text control, shifts
> style toward you) rather than the unconditional fine-tune below — MDM's
> unconditional path on HumanML3D is not officially supported and its loader
> crashes on empty captions.

## Why fine-tune MDM rather than train from scratch

- **Size.** Motion-diffusion models are small — MDM's transformer is on the
  order of tens of millions of parameters, not billions, because joint
  rotations over time are far lower-dimensional than pixels or text. (Confirm
  the exact count from the checkpoint you download; published configs are ~8
  transformer layers, latent 512.)
- **Data.** You have hours of one person. That is plenty to *specialize* a
  pretrained motion prior toward your style, and far too little to learn general
  human motion from zero. Don't import Chinchilla's ~20 tokens/param ratio
  (that's for autoregressive text) — use the **motion-specific** scaling work
  that now exists: [ScaMo](https://github.com/shunlinlu/ScaMo_code) (CVPR'25, log
  test-loss vs compute), Being-M0/MotionLib (ICML'25, data×model), and NVIDIA's
  Kimodo (2026). All have public code, so you have real reference points for the
  data:param budget rather than a guess.
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

## Hands

Body-only HMR (GVHMR/WHAM/TRAM) does not recover fine hand articulation. If your
clips need hands ("pick up an apple and eat it"), pick a hand-capable backend in
Pipeline 1 — it's now a config line, not phase 2:
- **Whole-body SMPL-X:** `backend: smplestx | whac | hand4whole | osx | multihmr`.
- **Fusion:** `backend: fusion` with `body_backend` + `hand_backend`
  (WiLoR/HaMeR) — grafts MANO fingers onto any body estimate.

Those backends fill `left_hand_pose`/`right_hand_pose`, and the AMASS export
writes them into the SMPL-H hand slots, so the dataset carries real fingers.

To train a motion model that *uses* the hands, you need a **whole-body (SMPL-H
or SMPL-X) motion representation**, not the 22-joint HumanML3D body features MDM
uses by default — otherwise the finger channels are discarded at feature-
extraction time. Options: extend the HumanML3D feature set to include hand
joints, or train a whole-body motion diffusion model. That extension is the
remaining phase-2 work on the model side; the *data* already has the hands.

DanceHMR (the ideal single video-native whole-body+hands model) was withdrawn
with no public code as of 2026-07 — watch for a re-release.

## (B) The behavior layer — pointer

Once (A) exists, "acts like me in a virtual environment" is a control problem:
a policy that selects goals/actions and uses the motion generator (or a
physics-based tracking controller like those in the PHC / PhysHOI / Ke line of
work) to realize them. That's a separate project; this repo deliberately stops
at producing a high-quality, personal motion generator it can build on.
