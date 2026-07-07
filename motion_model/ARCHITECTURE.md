# ARCHITECTURE — from motion dataset to an autonomous MetaHuman character

The full system behind "a fictional character that moves like me and acts on its
own." Pipeline 1 (`videotomocap/`) produces the input to all of this; everything
below consumes it. Synthesized from a July 2026 review of the motion-generation
and physics-based-control ecosystems — see `NEXT_STEPS.md` for the narrower
"just get a generator trained" path.

## The stack: four layers, not one "model"

```
 DECISION      picks WHAT to do (idle -> gesture -> "pick up X")   [unbuilt]
      |  text goal / action id
 (A) GENERATOR your personal motion model (moves like you)         [scaffolded, not runnable]
      |  SMPL motion
 (B) CONTROLLER physics: make a simulated body actually follow it  [unbuilt; use upstream]
      |  simulated SMPL poses
 RENDER        retarget SMPL -> MetaHuman skeleton in UE5          [mature, off-the-shelf]
```

Pipeline 1 emits anonymized SMPL-H (AMASS npz), which is the clean input to (A)
and — importantly — directly to (B). "Train the models" really means "build the
DECISION, (A), and (B) layers"; only (A) is scaffolded today.

## Layer (A) — the motion generator

**Do not train >35M params from scratch on hours of one person — it overfits and
memorizes you instead of learning a generative prior.** MDM-35M already sits at
the data ceiling of HumanML3D's ~29 hours. Motion scaling work (ScaMo, Being-M0)
is unanimous: scale data and model *together*. The recipe is therefore
**warm-start from a pretrained checkpoint, then FULL fine-tune** on your data
(full FT is fine — it's *from scratch* that's the trap).

| Want | Model | Size | Keeps HumanML3D bridge? |
|---|---|---|---|
| Best quality, least disruption | **MoMask** | ~44M | yes (native 263-d) |
| Genuinely larger, LM-style | MotionGPT | ~220M | yes (VQ over HumanML3D) |
| Demonstrated scaling (needs big pretrain corpus) | LMM | 90M–760M | needs adapter |
| Preserve MANO **hands** | Being-M0 / whole-body | large | no; immature |

**Pick: MoMask, warm-started + fully fine-tuned.** Biggest quality-per-effort
step up from MDM with the existing AMASS->HumanML3D bridge intact.

**Hands trade-off is real:** every HumanML3D-263 model silently discards the MANO
fingers the pipeline carries (263-d is body-only, 22 joints). Real hands need the
whole-body path (Motion-X features + a whole-body model), which is immature (no
confirmed weights). Defer hands; the *data* keeps them for later.

## Layer (B) — the physics controller

Makes a simulated humanoid actually follow generated motion under gravity/contact
(fixes foot-sliding, falls, real object interaction). Consumes AMASS/SMPL
directly — **no HumanML3D step** — so it eats Pipeline 1's output more directly
than (A) does.

**Pick: ProtoMotions (NVlabs).** Actively maintained (commits landing monthly
into 2026), **Apache-2.0**, unifies AMP/ASE/**MaskedMimic**/motion-tracking in one
codebase, and runs on the *maintained* simulators (Isaac Lab 2.3, NVIDIA Newton,
MuJoCo) — not only the deprecated Isaac Gym. Its `convert_amass_to_proto.py`
requires npz keys `poses` + `trans` + `mocap_framerate`, which is exactly what
`videotomocap/dataset.py` writes, so your dataset should feed it with minimal
glue (`--humanoid-type smpl`).

**CLoSD is the closed-loop A+B reference**, not the build target: it pairs an
MDM-family diffusion planner (DiP) with a PHC tracker and closes the loop
(simulated state re-conditions the next plan). Conceptually it's *exactly* the
A+B unification you want, and your generator is the right lineage for its planner
slot — but the repo is dormant (no commits since mid-2025), Isaac-Gym-only, and
its closed-loop training stage needs ~50 GB VRAM (see hardware). Read it for the
composition; build on ProtoMotions.

Architectural reference only: NVIDIA's **Kimodo** (kinematic generator) + **SONIC**
(RL tracker) stack is real and on HF, but it targets the SOMA/Unitree-G1 robot
skeleton, not SMPL/MetaHuman — instructive, not a drop-in.

### Licensing — a project-wide gate, because this is a *paid* product
- **Commercial-friendly:** ProtoMotions / MimicKit (Apache-2.0), PHC (BSD-3),
  InterMimic (MIT), CLoSD (MIT).
- **Non-commercial — avoid for paid output:** ASE, CALM, PhysHOI (NVIDIA research
  license).
- **The compounding trap:** the **SMPL / SMPL-X body models themselves** are
  registration-gated and academic/non-commercial by default, and every layer
  (including Pipeline 1's export) depends on them. **A commercial product needs a
  commercial SMPL license from Meshcapade.** Resolve this before building — it
  sits under the whole stack.

## DECISION layer — what picks the action

Lightest to heaviest: **behavior tree / FSM over clips** (mature, deterministic,
zero ML — recommended start) -> **LLM-as-planner** emitting text goals to the
generator (~200 lines of glue, most expressive, pairs with text conditioning) ->
hierarchical RL task policy (research-grade, heavy). For authored *content* you
often don't need this layer at all — you script the prompt timeline.

## RENDER — SMPL motion to MetaHuman

Generated SMPL/SMPL-H -> FBX (SMPL-X Blender add-on / `smpl2fbx`) -> UE5 **IK
Retargeter** onto the MetaHuman skeleton -> sequence with a state machine /
Sequencer. Mature, production-proven (same tool as Mixamo/Rokoko -> MetaHuman).
Patch residual foot-sliding with UE5 foot IK.

## Hardware reality (2x RTX 3090, 24 GB each)

- **(A) generator fine-tune:** easy — one 3090, hours to a day.
- **(B) physics training:** the tight spot. ProtoMotions does 40 h AMASS in ~12 h
  on 4x A100; on 2x 3090 you reduce `--num-envs` (default 8192 -> 1024) and accept
  slower wall-clock. CLoSD's closed-loop stage (~50 GB) **won't fit one 3090** at
  defaults — another reason to prefer ProtoMotions and dial env count.
- **Avoid the deprecated Isaac Gym** where possible: run ProtoMotions on Newton /
  Isaac Lab / MuJoCo instead. Isaac Gym is not just deprecated — its reference
  env repo (IsaacGymEnvs) was archived read-only in 2026.
- **VRAM is NOT the blocker for (B).** State-based humanoid tracking (no vision
  sensors) fits comfortably in 24 GB — reference AMP/ASE ran ~4096 envs on a
  single V100; ~5000+ envs are reported on a 24 GB card. Two 3090s drive one run
  via documented DDP (RL-Games/RSL-RL/skrl + torchrun), **no NVLink required**.
  If memory is tight, cut `--num-envs` (throughput scales ~linearly, so wall-clock
  rises proportionally).
- **The real cost of (B) is setup churn, not compute.** Isaac Lab/Sim installs are
  driver/CUDA/conda-fragile, and AMP/ASE/PHC configs predate Isaac Lab's config
  system (migration required). Budget **1-3 months for the physics layer**, mostly
  tooling — not the "few weeks" the compute alone would suggest.
- **The real compute sink is Pipeline 1 (HMR over the video backlog)** — both
  3090s in parallel; that's where the *video-processing* weeks go, separate from
  the model work.

## Recommended sequence

1. **Kinematic first (days, no sim, no RL):** MoMask generator -> SMPL -> UE5
   retarget -> MetaHuman, sequenced by a behavior tree or small LLM planner.
   Gets autonomous rendered content out immediately; covers idle/gestures/
   "walk over and gesture." Fake object pickup via hand socket.
2. **Generator:** MoMask, warm-start + full fine-tune on your anonymized data.
3. **Physics later (weeks), only for shots needing real contact** ("eat an
   apple"): ProtoMotions/MaskedMimic, fed your AMASS npz; retrain/track on your
   motion. Reserve CLoSD-style closed loop for when you want your *own* generator
   driving the physics character.
4. **Resolve the commercial SMPL license** before shipping anything paid.
5. Hands and the autonomous DECISION layer are last, and optional for authored
   content.

## Status of "is step 2 ready?"

No — it's a scaffold. Even with a perfect dataset, (A) still needs the manual
HumanML3D 263-d conversion (gated body models, no in-repo training), and (B) + the
decision layer are unbuilt. The good news: the *data contract* is right, and (B)
(ProtoMotions) can consume Pipeline 1's npz nearly as-is. See `NEXT_STEPS.md` for
the (A) build steps and the already-fixed bridge bugs.
