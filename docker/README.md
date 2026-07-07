# Containerized, offline-capable pipeline

Runs the whole thing (video → dataset → motion model) isolated in a container, with
no host pollution. After a one-time online setup, every stage runs offline from
mounted volumes.

## Quick start (three commands)

```bash
make build                                # slim image: only the core env
make setup                                # one-time, ONLINE (BACKEND=gvhmr METHOD=momask)
make run                                  # video → trained model, offline
```

Override the stages: `make setup BACKEND=wham METHOD=mdm`. Other targets:
`make dataset`, `make train`, `make shell`, `make selftest`.

Drop videos in `./dropzone`; outputs land in `./work`. `setup` clones the repos you
chose into `./repos`, builds a conda env per stage (from each repo's own env file),
fetches the scriptable weights, and (with your MPI creds) the SMPL-H model.

Without `make`, the raw form is `docker compose run --rm [--network none] pipeline
<stage>` where stage ∈ {setup, dataset, train, all, selftest, shell}.

## Why one image, many envs

The stages need conflicting torch/CUDA (GVHMR py3.10/torch2.3+cu121, MoMask+MDM
py3.7/torch1.7+cu110, ProtoMotions py3.8/torch2.2+cu121). They're batch steps
chained by files, not services, so one CUDA base image (`nvidia/cuda:12.1.0-devel-
ubuntu20.04`, matching ProtoMotions/GVHMR) with a conda env per stage is simplest.
Only the light `core` env is baked in — and verified at build time (selftest +
pytest). Everything heavy lives in `./repos` and `./models` volumes, so the image
stays slim and reruns are offline.

## Offline

`setup` is the only step that needs the network. Weights are cached into `./models`
(and `./models/.cache` for CLIP/torch-hub); ProtoMotions weights come via git-lfs.
For a hard guarantee, run stages with no network:

```bash
docker compose run --rm --network none pipeline all
```

If a stage errors under `--network none`, something wasn't pre-cached during setup
— re-run `setup` for that component.

## Gated: the SMPL body model (one academic registration — unavoidable)

The SMPL family is the only license-gated piece, and there is **no clean
zero-account path**: every FK-compatible body model (SMPL/SMPL-H/SMPL-X/STAR/SUPR)
is MPI-licensed with a no-redistribution clause, and the HMR backends require it
just to run — not only the feature step. Un-gated Hugging Face mirrors exist
(e.g. `camenduru/SMPLer-X`, `lithiumice/models_hub`) and work technically, but
they violate that license, so this repo does not script them.

The honest floor is **one free MPI academic registration** (~2 min, no purchase):

- The feature step reuses the **same** neutral model your HMR backend already
  needed (SMPL-H *or* SMPL-X — only the 22 body joints are used), so you don't
  register twice. Point `smpl_model` at it.
- If you go the SMPL-H route explicitly, `setup` can download it for you: set
  `MANO_USERNAME` / `MANO_PASSWORD` (env or compose) and it POSTs to MPI's backend
  (the mechanism ICON/PIXIE/WHAM/ARCTIC use) and extracts
  `./models/smplh/neutral/model.npz`. No creds → it prints the manual step.
- **protomotions** as the *model* needs no feature-step SMPL, but still needs a
  body model for the sim humanoid.

## CPU-only smoke test (no GPU)

`selftest` runs the whole flow GPU-free via the synthetic `noop` backend/method.
Of the real stages, MoMask/MDM and ProtoMotions' MuJoCo backend can run CPU-only
(slow) to validate the offline plumbing; GVHMR needs a GPU (use `noop` to stub it).
