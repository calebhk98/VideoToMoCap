# Research watchlist — promising work with no usable public code (as of 2026-07)

Papers relevant to this pipeline whose code is **not usable today** (unreleased /
"coming soon" / stub repo). These are things to *watch* or *reimplement* — not
wired in. Anything already wired (GVHMR/WHAM/TRAM, SMPLest-X/WHAC/OSX/
Hand4Whole++/Multi-HMR, WiLoR/HaMeR) lives in `README.md`, not here.

Code-status confidence is noted; very recent arXiv posts may ship a repo shortly
after — re-check before assuming absent.

## Feasibility for this repo — what's reimplementable *without training* (2026-07)

Investigated the "reimplement" candidates against this repo's constraint (pure
NumPy post-processing, no training/weights/GPU). Status of each:

| Idea | Tractable slice (done here) | Hard part (deferred) |
|---|---|---|
| **HTD-Refine** (de-jitter/de-drift) | ✅ **Done.** `refine.py` implements the objective two ways: Savitzky–Golay (`temporal_dejitter`) and now a **variational second-difference smoother** (`variational_smooth`) that directly minimizes `‖x−y‖²+λ‖accel(x)‖²` — the paper's "penalize high-order temporal dynamics" objective, *solved* instead of learned. Set `refine_method: variational`. | The learned PVA-Net that predicts per-joint velocity/accel *constraints* from images — needs their network + training data. |
| **FactorizedHMR** (torso-anchor + generative limb completion for out-of-frame limbs) | ◑ **Partial.** We detect & mask unreliable/out-of-frame limbs (`auto_occlusion`, `camera_occlusions` → `joint_valid`), which is the "don't trust hallucinated limbs" half. | The **generative completion** (flow-matching model that *fills in* plausible occluded-limb motion conditioned on the torso) needs training. A training-free stand-in — retrieve the person's own reliable limb motion from a similar corpus clip — is possible but speculative; left for later. |
| **PersonaAnimator** (personal-style motion model) | — | This *is* Pipeline 2 (fine-tune a motion model on your data). Scaffolded in `motion_model/`; needs GPU training. |

Bottom line: HTD-Refine's core is now fully realized here; FactorizedHMR and
PersonaAnimator's remaining value is genuinely training-bound, so they stay on
the watchlist rather than in the pipeline.

## Most actionable for this pipeline

| Paper | Why it matters here | Code | Move |
|---|---|---|---|
| **FactorizedHMR** (arXiv:2605.14854, UCF) | **Best fit for out-of-frame legs:** solves a solid torso/root anchor, then *generatively completes uncertain distal limbs* (arms/legs) via flow-matching. Exactly the prior you want for waist-up footage. | dead placeholder link | **Watch → reimplement** if truncation is central |
| **HTD-Refine** (arXiv:2605.26879, ZJU3DV, CVPR'26 Oral) | Drop-in de-jitter/de-drift refinement for *any* estimator (TRAM/GVHMR): e.g. EMDB-2 jitter 17.2→7.2. From the GVHMR authors. | "Code Coming Soon" (confirmed) | **Watch → reimplement** (it's a post-proc module, tractable) |
| **DanceHMR** (arXiv:2605.18102) | Whole-body SMPL-X with genuinely detailed *video* hands; **close-up-aware augmentation** targets upper-body framing. Your core hand need + partial-body. | none linked | **Watch closely** |
| **PersonaAnimator** (arXiv:2508.19895) | Learns one person's motion *style* from unconstrained (non-mocap) video — essentially your Pipeline-2 goal, published. | none mentioned | **Watch → possible reimplement** |

## By topic

### World-grounded video HMR (GVHMR/WHAM/TRAM lineage)
- **HTD-Refine** — see above. Temporal de-drift, reimplementable.
- **WAT/World-aware Allied Trajectory** (arXiv:2509.04600) — joint camera+human trajectory; "code will be available" but none found. *Watch.*
- **RAM — Recover Any 3D Human Motion** (arXiv:2603.19929, CVPR'26) — occlusion-robust multi-person tracking + memory-augmented temporal HMR. No code. *Watch.*

### High-detail hands / hand-object
- **DanceHMR** — see above. Top hands match.
- **WHOLE** (arXiv:2602.22209, Stanford, CVPR'26) — hand-object prior that reasons through **out-of-frame** moments (egocentric). "Code Coming Soon." *Watch.*
- **CHOIR** (arXiv:2605.20992) — monocular video → reusable 4D hand-object primitives with explicit contact (grasping/eating). No code. *Note/watch.*
- **DyTact** (arXiv:2506.03103) — dynamic contact capture under occlusion; repo is a README-only stub. *Watch.*

### Occlusion / truncation / partial-body  ← your "only torso + arms visible" case
- **FactorizedHMR** — see above. The single best conceptual fit for inferring out-of-frame limbs.
- **Pulp Motion** (arXiv:2510.05097) — framing-aware; explicitly calls out "close-up shots capturing only the upper body" and refines out-of-frame regions. But it's *generation*, not recovery from real video — an idea source/prior. *Note/watch.*
- **Discriminative-Generative Synergy** (arXiv:2604.21712) — diffusion completes occluded body (single-image; occlusion, not framing-truncation). *Note.*
- Honest bottom line: **no 2025-26 paper is dedicated purely to waist-up security-camera framing → full-body motion.** It's a real gap; FactorizedHMR's design is the closest lever.

### Long-video / temporal / identity-agnostic
- **TopoCap** (arXiv:2606.12153, SIGGRAPH'26) — topology-agnostic motion prior, retargets to any skeleton; **dataset public, no code**. *Watch (dataset useful now).*
- **EgoForce** (arXiv:2605.13041) — streaming online reconstruction via diffusion forcing; anti-drift. Code unconfirmed. *Watch.*
- **HMRMamba** (arXiv:2601.21376) — first Mamba/SSM video HMR for long-range temporal modeling. No code. *Note.*

### Motion generation / personal-style + scaling
- **PersonaAnimator** — see above. Closest to Pipeline 2.
- **Repetitive-motion personal model** (arXiv:2503.15225) — per-person LSTM "motor signature"; narrow but a clean proof-of-concept. *Note.*
- **ClusterStyle** (arXiv:2512.02453), **AnyMo** (arXiv:2605.29488, ships OmniHuMo 5000+ hrs) — stylized / any-modality motion generation. Code unconfirmed/none. *Note.*
- **Scaling laws DO exist** (and have working code — use directly, not "watch"): **ScaMo** (CVPR'25, log test-loss vs compute), **Being-M0/MotionLib** (ICML'25, data×model), **Kimodo** (NVIDIA, 2026, 700 hrs, dataset/model scaling + control). This settles the "what data:param ratio?" question with real references — see `motion_model/README.md`.

## Suggested strategy if truncated framing is central to your footage
Watch **FactorizedHMR**'s repo and be ready to reimplement its torso-anchor +
generative-limb-completion design; pair with a working occlusion-robust baseline
today (**SAM-Body4D**) and, once released, a temporal de-drift pass
(**HTD-Refine**). For hands under close-up framing, watch **DanceHMR**.

## Multi-person & identity (2026-07 survey)

For `multi_person` mode (`docs/MULTI_PERSON.md`). The `run_tracks()` contract and
the shape-clustering assignment are wired; these are the upgrade paths.

### Multi-person HMR backends (recover everyone, with tracking)
| Model | What it adds | Code / license | Move |
|---|---|---|---|
| **Human3R** (arXiv 2510.06219, 2025) | Multi-person **SMPL-X** + world trajectories + camera in one online pass (~15 FPS). Best single fit for our SMPL-X-with-hands contract. | released, weights on HF; **CC-BY-NC-SA** | **Add as flagship multi-person backend** |
| **PromptHMR** (CVPR 2025) | Mature, world-grounded, whole-body, promptable; TRAM lineage. | released; **Meshcapade non-commercial** | Add as the higher-accuracy alternative |
| **CoMotion** (Apple, ICLR 2025) | Best-in-class concurrent tracking / ID stability through occlusion; **SMPL-only, camera-space**. | released; **Apple research-only weights** | Use if track-ID stability is the bottleneck |
| `trace`, `hmr2`/4D-Humans (already wired) | Already multi-person + tracked, **Apache/MIT** — just promote from dominant-track to all tracks. | in repo | Cheapest permissive win: override `run_tracks` |

### Cross-clip re-identification (assign tracks to a known family)
A family is a small **closed, enrolled** gallery — 1:N verification, not open-world.
- **InsightFace / ArcFace** — primary matcher when faces are visible; mature, offline, MIT code (NC weights).
- **OpenGait** (SMPLGait/DeepGaitV2) — clothing-invariant gait ID; supports SMPL input directly.
- **SOLIDER-REID / OSNet** — body-appearance ReID for faceless, same-outfit tracks.
- **SMPL betas clustering** — the near-free cue we already use (`identity.py`); good confirmation, weak alone. Extend the assignment seam to fuse face+gait+shape.

### Per-person grounded captioning ("who did what")
Makes the caption↔motion pairing person-accurate (currently scene-level).
- **DAM-3B-Video** (NVIDIA, ICCV 2025) — region-conditioned localized video captioning; give it a per-person mask, get that person's actions. Most on-target; Apache code, NC weights.
- **VideoRefer / PixelRefer** (Alibaba, 2025) — object/region-level video LLM, timestamped regions; lighter 2B option.
- **Sa2VA** (ByteDance, 2025) — SAM-2 + LLaVA, grounded segmentation + captioning; **Apache-2.0**.
- Or extend the existing **Qwen2.5-VL** fine-tune (Apache-2.0) to ingest person boxes/IDs and emit ID-tagged timestamped captions. Split detect/track (identity) from describe (VLM) — don't ask one model to do both.
