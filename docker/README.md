# Containerized, offline-capable pipeline

Runs the whole thing (video → dataset → motion model) isolated in a container, with
no host pollution. After a one-time online setup, every stage runs offline from
mounted volumes.

## Quick start

```bash
docker compose build                                          # slim: only the core env
docker compose run --rm pipeline setup --backend gvhmr --method momask   # one-time, ONLINE
# → place your ONE gated download under ./models (see "Gated" below)
docker compose run --rm pipeline all                          # video → model
```

Drop videos in `./dropzone`; outputs land in `./work`. `setup` clones the repos you
chose into `./repos`, builds a conda env per stage (from each repo's own env file),
fetches the scriptable weights, and checks for the gated model.

Stages: `setup`, `dataset`, `train`, `all`, `selftest`, `shell`.

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

## Gated: the SMPL body models (one academic registration)

The SMPL family is the only license-gated piece. Register once at
mano.is.tue.mpg.de, then **`setup` downloads it for you** using your own
credentials — set `MANO_USERNAME` / `MANO_PASSWORD` (env or compose) and it POSTs
to MPI's download backend (the same mechanism ICON/PIXIE/WHAM/ARCTIC use) and
extracts `./models/smplh/neutral/model.npz`. No creds set → it prints the manual
step instead. What each stage needs:

- **Generators** (momask/mdm/closd): the **neutral SMPL-H** — auto-downloaded as above.
- **HMR backend / ProtoMotions**: their own SMPL / SMPL-X (same registration).
- **protomotions** as the *model*: needs no feature-step SMPL at all.

## CPU-only smoke test (no GPU)

`selftest` runs the whole flow GPU-free via the synthetic `noop` backend/method.
Of the real stages, MoMask/MDM and ProtoMotions' MuJoCo backend can run CPU-only
(slow) to validate the offline plumbing; GVHMR needs a GPU (use `noop` to stub it).
