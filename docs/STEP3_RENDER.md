# Step 3 — a character that moves *and* looks real

Pipelines 1-2 give you SMPL motion (your recovered mocap, and what the trained
motion model emits). Step 3 puts that motion on a photoreal character. The render
and rig are off-the-shelf DCC work — this repo's job is the **export bridge** that
gets the motion into a format those tools ingest. That bridge is `videotomocap/export.py`
(`export-bvh`), and this doc is the workflow around it.

```
motion model / recovered clip ──▶ SMPL pose npz ──export-bvh──▶ clip.bvh
                                                                  │
                          ┌───────────────────────────────────────┤
                          ▼                                        ▼
                 Blender / UE5 import                    3D-Gaussian avatar
                 → IK Retarget → MetaHuman               (consumes SMPL pose direct)
                 → Movie Render Queue                    → real-time / offline render
```

## Export: SMPL motion → BVH

```bash
python -m videotomocap export-bvh --pose work/pose/clip0.npz \
    --model /path/to/neutral_smpl.npz --out clip0.bvh
# or --clip clip0 to resolve it under the work pose_dir
```

BVH is the universal skeletal-animation interchange: it imports into Blender and
Unreal and drives MetaHuman via UE5's IK Retargeter. `export.py` writes the 24-joint
SMPL body — SMPL's per-joint pose is already parent-relative (BVH's model), so no FK
is needed for the animation; the rest skeleton comes from the neutral SMPL model you
already registered for the HMR backends (`--model`; reused, not a second sign-up).
Default scale is metres→centimetres (what Blender/UE mocap import expects).

*Current limits (honest):* body only — MANO fingers aren't in the BVH yet (a
follow-on); and if `--model` is SMPL-X the 22 body joints map exactly while the two
hand joints are approximate. Retargeting fixes proportions regardless.

## Path A (recommended for MetaHuman): UE5 IK Retargeter

The 2025 EULA change makes this hobbyist-friendly — MetaHumans can be sold and used
outside UE. Steps:

1. **Import `clip0.bvh`** into Blender (or directly to UE via a BVH importer). In
   Blender, the SMPL Blender addon (Meshcapade) or a plain BVH import both work;
   export an **FBX** (UE's IK Retargeter wants a skeletal-mesh animation).
2. In UE5, build an **IK Rig** for the SMPL skeleton and one for `metahuman_base_skel`,
   define retarget chains (spine/arms/legs), and reconcile the A-pose↔T-pose reference
   pose. This is the one fiddly seam; the MetaHuman→MetaHuman side is mature.
3. **Retarget** the animation onto your MetaHuman and render with **Movie Render
   Queue** (path-traced for top-tier offline quality).

Softer-landing equivalent if UE's rigging friction annoys you: **Reallusion Character
Creator 4 + iClone** (friendlier photoreal-human retargeting, free RTX path-tracer).

## Path B (best identity match, skips the rig): 3D-Gaussian avatar

Build a personal avatar from a 1–3 min self-video with **ExAvatar** (SMPL-X + 3DGS)
or **GART**, then drive it **directly with SMPL pose** — no BVH/FBX/retarget bridge at
all, because these consume SMPL parameters natively. It genuinely looks like you, is
real 3D (re-shoot any angle), and trains/renders on one 3090. Best when identity
fidelity matters more than full cinematic scene control. (Convert the 24-joint SMPL
body to SMPL-X for whole-body/hands+face; quality falls off far outside the captured
pose range.)

## Path C (fastest to a clip, no 3D asset): pose→video diffusion

Drive a single reference photo with your motion via **Champ** (SMPL-guidance native:
renders depth/normal/semantic maps from the SMPL sequence) or **StableAnimator++**.
Near-zero rigging, photoreal identity, but offline and short (2–5 s before drift).
Good for social-length "me doing X"; poor for long, camera-moving, multi-character
scenes.

## The behaviour layer (what the character *decides* to do)

Separate from making it move. Pick by autonomy needed:

- **Authored timeline / behaviour tree** — you sequence motion clips. Simplest; right
  for rendered content and scripted NPCs.
- **LLM planner → motion-text prompts → your model** — language-driven autonomy
  ("walk to the door, wave"). Kinematic only (can foot-slide).
- **Physics policy (CLoSD / protomotions)** — physically grounded, environment-aware
  autonomy; pairs with any of the render paths above. This is Pipeline 2's
  `protomotions`/`closd` methods.

## Which to pick

| Want | Path |
|---|---|
| Cinematic scenes, moving cameras, relight, multi-character | A (UE5 MetaHuman) |
| "Looks like me", real 3D, minimal rigging | B (3DGS avatar) |
| A quick photoreal clip from one photo | C (video diffusion) |

## Licensing (read before monetizing)

- **SMPL/SMPL-X** is free for research; commercial use of the model (which shipped
  content embeds) needs a Meshcapade commercial licence. The pipeline strips `betas`,
  but the body topology is still SMPL.
- **MetaHuman** (post-2025 EULA) can be sold/used outside UE, but its assets **must not
  be used to train or enhance an AI model** — driving a MetaHuman with your already-
  trained model is fine; feeding MetaHuman data back into training is not.
- **AMASS** (if you mixed any in) is non-commercial research only.

Sources for the tool landscape are in the step-3 research notes; verify tool versions
and UE/MetaHuman EULA terms against their current docs before shipping.
