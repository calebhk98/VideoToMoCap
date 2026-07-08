# Pipeline 3 — invented characters to drive

Generate a stylized-but-humanoid character (elf, dwarf, "green-man", realistic human)
and drive it with Pipeline 2's motion model. **Every tool here is open-source and runs
locally** — no cloud, no API, nothing that can be shut down. Pick one with `method:` in
a YAML; swapping is one line, so you can A/B the whole zoo.

```bash
python -m character methods                         # list tools + status
python -m character --config c.yaml info            # what the configured tool will do
python -m character --config c.yaml generate        # one character
python -m character --config c.yaml refine          # generate -> critique MESH -> refine
```

## The tools (`method:`)

**"Drivable" = SMPL-X-native, so Pipeline 2's motion drives it with NO retargeting.**
The others need a retarget step (their own rig) or a fit (no rig).

| method | in | drivable? | VRAM (3090) | license | status |
|---|---|---|---|---|---|
| **lhm** | image | ✅ SMPL-X | 16–24 GB | Apache-2.0 | **recommended** — clean CLI, `.ply` Gaussian |
| **idol** | image | ✅ SMPL-X | ~24 GB | MIT* | demo hardcodes inputs + exports video; patch `run_demo.py` |
| **en3d** | text/seed/img | ✅ SMPL-24† | >24 GB | Apache-2.0 | generative; textured mesh needs the full synthesis chain |
| **so_smpl** | text | ✅ SMPL-X | 24 GB+ | non-commercial | native SMPL-X, **disentangles body+clothes** (skin under clothes!); slow SDS (~hrs) |
| **econ** | image | SMPL-X (via avatarizer) | >12 GB | MPI non-commercial | clothed; ships `avatarizer`+`animation` (pairs with our HybrIK-X) |
| **icon** / **sifu** | image | SMPL-X (needs binding) | >12/16 GB | non-comm / MIT | clothed reconstruction |
| **pshuman** | image | ✗ (fit needed) | **>40 GB — won't fit** | MIT | high-detail mesh, no rig |
| **human3diffusion** | image | ✗ (fit needed) | unverified | MIT | Gaussian + TSDF mesh, no rig |
| **mpfb2** | params | ✗ (retarget) | CPU/GPU | GPLv3/CC0 | headless parametric (MakeHuman model), own rig |
| **makehuman** | — | — | — | AGPL/CC0 | **GUI-only** → use `mpfb2` |
| **mblab** | — | — | — | GPLv3/AGPL | **end-of-life** → use `mpfb2` |
| **anigs** / **humanorbit** | — | — | — | — | **code not released** → use `lhm`/`idol` |
| **noop** | — | ✅ | none | — | synthetic, for tests |

\* IDOL's repo has no LICENSE file despite an MIT badge. † En3D's SMPL-24 rig is inferred
from its code — verify driving on your data.

Each backend shells out to the upstream repo (set `repo:`) and **fails loud** with what's
missing. The exact commands are centralized in each adapter and flagged *verify against
your checkout* — upstream demos drift. Set `repo`/`weights`/`backend_python` per tool.

## The mesh critic + refine loop

`refine` runs generate → **critique the mesh** → regenerate-if-poor, keeping the best:

- `critic: geometric` — **pure-NumPy** geometry sanity (finite, human proportions,
  left-right symmetry). No GPU, no reference; rejects melted/lopsided meshes.
- `critic: vlm` — renders the mesh multi-view (`renderer: blender`) and scores the
  renders against the prompt with a **local VLM** (wire `vlm_python`/`vlm_model` to
  Qwen2-VL/InternVL/MiniCPM-V). Judges the actual 3D output, not the concept image.
- `critic: combined` — average of both. `critic: noop` — for tests.

Knobs: `accept_score`, `max_attempts`, `n_candidates` (best-of-N). Because the lifters
are feed-forward, "refine" = regenerate with a fresh seed (swap the concept image between
rounds for image-native tools), not iterate one asset.

## Recommended local path

**FLUX/SDXL concept image → `lhm` → SMPL-X avatar → Pipeline 2 motion drives it directly.**
For a dressable body with skin underneath, `so_smpl` (disentangled body+clothes). For a
free parametric base, `mpfb2` (then retarget). The generation step is heavy (GPU + the
upstream repo); this package is the light orchestration + config seam over them, GPU-free
tested via `noop`.
