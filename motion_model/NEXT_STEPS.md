# NEXT_STEPS — from dataset to a motion model that moves like you

A concrete, ordered plan for taking Pipeline 1's output all the way to a trained,
promptable motion generator. Synthesized from a July 2026 review of the MDM /
HumanML3D / priorMDM stack and the current motion-model landscape. Read
`motion_model/README.md` first for the "why fine-tune MDM" rationale; this doc is
the "what to actually do, in order," plus the gotchas that bite silently.

## The one decision that shapes everything: what does "train a model" get you?

The motion generator (MDM) is a function **condition -> a clip of motion**. Three
kinds of condition, and the choice decides whether you also need a separate
"decision AI":

| Conditioning | You feed it | You get | Need a decision layer? |
|---|---|---|---|
| Unconditional | nothing | random plausible clip "in your style" | **yes** — something must choose/sequence |
| **Text** | `"walk forward"`, `"wave"` | that action, in your style | **no** — you author the prompts |
| Action label | a class id (walk/sit/cook) | that class, in your style | no — you pick the label |

**Recommended target: text-conditioned, personalized via LoRA.** You want
promptable control ("walk forward", "do jumping jacks"), and you're producing
*rendered content*, so you (the director) author the sequence of action prompts.
That means:

- You do **not** need the autonomous decision/behavior layer to make finished
  video. "No live performer" splits into *authored* (you script the action
  timeline — cheap, controllable, works today) vs *autonomous* (a learned
  RL/planning policy picks actions live — a separate research project, e.g. the
  PHC / PhysHOI physics-controller line). Authored is the pragmatic path for a
  content platform. Treat the autonomous layer as explicit maybe-later.
- Text conditioning also **sidesteps the unconditional-training trap**: MDM's
  `--unconstrained` mode was only ever tested on a *different* data format
  (HumanAct12), and its HumanML3D data loader crashes on missing captions. Going
  text-conditioned avoids that whole unsupported path.

## Model choice (verified available, July 2026)

- **Base: MDM** (`GuyTevet/motion-diffusion-model`) — still maintained, ~35M
  params, uses the exact HumanML3D 263-dim features this pipeline targets. Still
  the substrate the whole personalization ecosystem builds on.
- **Personalization: LoRA-MDM** (`haimsaw/LoRA-MDM`, "Dance Like a Chicken",
  2025) — adapts the prior toward your style with small adapters while
  **keeping MDM's text vocabulary**. So `"do jumping jacks"` -> jumping jacks,
  moving like you. Preferred over full fine-tuning, which risks mode collapse
  (personal footage is a few repeated activities) and catastrophic forgetting of
  text control.
- **Quality-bump option: MoMask** (`EricGuo5513/momask-codes`) — same 263-dim
  features, better published FID, drop-in if MDM output disappoints. No LoRA
  tooling though.
- **Hands ("eat an apple"): defer.** Your data already stores MANO hands, but
  HumanML3D's body features discard them. The whole-body route (Motion-X +
  HumanTOMATO) is real but immature (no confirmed pretrained weights). Watch
  NVIDIA Kimodo. Nothing is lost by adding hands later — the data keeps them.

## Ordered steps

### 0. Run Pipeline 1 on a small batch first
Manually exclude the two family-visit clips, then process ~a handful of clips so
you can eyeball `work/dataset/index.json` + `stats.json` before committing the
whole backlog. This is also where you confirm the backend's world-frame output
looks sane (feet below head, no obvious drift).

### 1. Turn on the quality gates BEFORE exporting
Set `quality_filter` and `refine` (temporal de-jitter + anti-drift) in the
config. The motion scaling work (ScaMo) found **data quality dominates
quantity** — HMR jitter learned as "your style" is the dominant failure mode.
HumanML3D has no notion of the `joint_valid` occlusion mask, so out-of-frame
limb guesses get trained as ground truth unless you filter here.

### 2. Build the dataset (Pipeline 1 -> AMASS npz)
`python -m videotomocap --config configs/dropzone.yaml build`. Confirm
`target_fps` is **20** (now the default — see the FPS gotcha below). Output:
`work/dataset/amass/*.npz` (betas=0, world-frame, 20 fps).

### 3. Stand up the MDM environment + assets (one-time)
```bash
git clone https://github.com/GuyTevet/motion-diffusion-model && cd motion-diffusion-model
conda env create -f environment.yml && conda activate mdm
bash prepare/download_smpl_files.sh          # MDM's own SMPL assets
bash prepare/download_glove.sh
# Pretrained HumanML3D prior (manual Google Drive): humanml-encoder-512,
# contains model000475000.pt -> unzip into ./save/
```
Separately, for HumanML3D *preprocessing* you need two registration-gated body
models (different accounts, easy to conflate):
- **SMPL+H** from `mano.is.tue.mpg.de` (this is what AMASS/HumanML3D FK uses)
- **DMPL** from `smpl.is.tue.mpg.de`

### 4. Convert AMASS -> HumanML3D 263-dim features
```bash
python -m motion_model --config configs/motion_model.yaml prepare
```
(with `humanml3d_repo` + `smpl_model` set in the config). Then run HumanML3D's
`raw_pose_processing` -> `motion_representation` -> **`cal_mean_variance`** (the
last one is easy to forget — it makes `Mean.npy` / `Std.npy` over YOUR corpus; do
not reuse the shipped HumanML3D stats). `prepare` prints the exact hand-off.

### 5. Fine-tune with LoRA-MDM (text-conditioned)
Fill `texts/<clip_id>.txt` with real captions first — start from `cluster.py`
action pseudo-labels (walk/sit/cook…), or auto-caption clips with a video
captioner. The bridge writes a loader-valid placeholder so nothing crashes if you
run before captioning, but placeholders give you no real control.
Follow LoRA-MDM's README to attach adapters to the `model000475000.pt` prior and
train. Smoke-test a few hundred steps, watch a few sampled clips for mode
collapse, then the full run.
(If you ever do a plain MDM full fine-tune instead, use priorMDM's
`train_mdm_motion_control.py` path — vanilla `train_mdm` has no working
`--resume_checkpoint`. See README's recipe: ~80k steps, batch 64, lr 1e-4.)

### 6. Later, in order of value
1. Better captions -> better text control (auto-captioning).
2. Hands (whole-body representation; Motion-X + HumanTOMATO) — data already has them.
3. The behavior/decision layer — only if you want autonomous (non-authored) action.

## Your hardware (2x RTX 3090, Ryzen 9, 32 GB) — where the time goes

- **The motion model is NOT the bottleneck.** MDM (~35M params), let alone LoRA,
  fine-tunes comfortably on **one 3090** in **hours**, not weeks. The 80k-step
  recipe is under a day.
- **The compute sink is Pipeline 1 — HMR over ~2 years of footage.** That's where
  the "weeks" go. `pipeline.py` pins one clip per GPU, so run **both 3090s in
  parallel** on the backlog (`gpus: ['0','1']`).
- **Watch the 32 GB system RAM** during bulk video decode — the tightest
  resource. Process in batches; don't load everything at once.
- Sane sequencing: spend the weeks on HMR (both GPUs), then the fine-tune is an
  afternoon on one GPU.

## Silent gotchas already fixed in this repo (July 2026)

- **FPS (was a real bug):** `target_fps` defaulted to 30. HumanML3D decimates
  with `int(source_fps/20)` = `int(30/20)` = 1 -> no downsampling, but it still
  treats the clip as 20 fps, so every clip trained **1.5x too fast**. Default is
  now **20**; keep any override a multiple of 20. (`videotomocap/config.py`)
- **Frame guard:** `to_amass_npz` now refuses non-`global` (camera-relative)
  clips, which HumanML3D's floor/up-axis normalization would silently corrupt.
  (`videotomocap/dataset.py`)
- **Empty captions crash MDM's loader:** the bridge now writes a loader-valid
  placeholder caption, not `""`. (`motion_model/data.py`)
- **Mean/Std + gender + quality-filter gotchas** are now printed in the bridge's
  feature-extraction hand-off.

## Open items to verify yourself before a big run
- Up-axis: confirm your chosen backend's `global` output is Z-up (AMASS
  convention) — sanity check average foot-Y below head-Y on a sample clip.
- 263-dim foot-contact block: confirm the concatenation order against
  `cal_mean_variance` when you inspect features (a dim mismatch corrupts
  normalization silently).
- Min clip length: reconcile `min_clip_frames` (30) with MDM's t2m loader's
  minimum-length / `unit_length` expectations at 20 fps.
