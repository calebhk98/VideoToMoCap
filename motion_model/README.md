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

## Config-driven usage (like Pipeline 1)

Pipeline 2 is now method-selectable from a YAML, exactly like Pipeline 1's
backend. Pick the method in `configs/motion_model.yaml` and run:

```bash
python -m motion_model methods                      # list selectable methods
python -m motion_model --config configs/motion_model.yaml info      # what it will do
python -m motion_model --config configs/motion_model.yaml prepare   # dataset -> training data
python -m motion_model --config configs/motion_model.yaml train     # prepare + launch training
```

Methods (`method:` in the config — swapping is one line):

| method | role | features | licence-friendly for paid use |
|---|---|---|---|
| `momask` | generator, ~44M (**default pick**) | HumanML3D 263-d | code MIT; SMPL gate applies |
| `mdm` | generator, ~35M (`personalization: lora` for LoRA-MDM) | HumanML3D 263-d | code MIT; SMPL gate |
| `protomotions` | physics controller | AMASS npz **direct** | Apache-2.0; SMPL gate |
| `closd` | closed-loop planner+controller (A+B) | HumanML3D 263-d | MIT; SMPL gate |
| `noop` | synthetic (tests) | — | — |

Each trainer shells out to its upstream repo (set `repo:` etc. in the config) and
fails loud with an actionable message when a repo/asset is missing — the heavy
training runs upstream, this is the orchestration layer. **The SMPL/SMPL-H body
models every method needs are non-commercial by default; a paid product needs a
commercial SMPL licence from Meshcapade** (`smpl_model:` is your responsibility).
See `ARCHITECTURE.md` for the full four-layer stack and `NEXT_STEPS.md` for the
ordered build.

The subsections below explain the manual HumanML3D feature-extraction hand-off
the generator methods orchestrate.

## Steps

### 1. Convert the AMASS dataset to HumanML3D features

MDM trains on the HumanML3D 263-dim feature representation, not raw SMPL. The
conversion is: SMPL params → 22 joint positions (SMPL forward kinematics) →
HumanML3D features (`motion_representation`). It needs the SMPL body model
(register at https://smpl.is.tue.mpg.de) and the HumanML3D repo.

`python -m motion_model prepare` (with a generator method + `humanml3d_repo` +
`smpl_model` set in the config) lays out the parts that need no external assets
(indexing, splits, caption scaffolding) and prints the exact HumanML3D
feature-extraction hand-off for the rest — `raw_pose_processing` →
`motion_representation` → `cal_mean_variance` (the last makes `Mean.npy`/`Std.npy`
over *your* corpus). It needs the SMPL body model (register at
https://smpl.is.tue.mpg.de) and the HumanML3D repo.

### 2. Fine-tune

`python -m motion_model train` shells out to the configured method's trainer.
The trainer stages your prepared data where the upstream loader expects it
(`<repo>/dataset/HumanML3D`) and builds the right command — MoMask's two-stage
RVQ+transformer recipe, MDM's `--resume_checkpoint` full fine-tune, or LoRA-MDM's
`--lora_finetune --starting_checkpoint`. Upstream-specific hyperparameters
(diffusion steps, noise schedule, ...) go through `extra_args` in the config.

### 3. Conditioning choice

Your footage has no text labels. Options, cheapest first (set `conditioning:` in
the config):

1. **`none`** — learn "motion that looks like you." Simplest; pair with (B) to drive it.
2. **`action`** — reuse Pipeline 1's unsupervised `action_cluster` ids as coarse
   labels (walk, sit, cook, ...); `prepare` writes them as captions automatically.
3. **`text`** — run a video captioner over each clip, fill `texts/<clip_id>.txt`,
   and train the standard text-conditioned model. Most work, most controllable.

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
