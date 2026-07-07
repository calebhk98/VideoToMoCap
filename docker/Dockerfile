# Isolated, offline-capable image for the whole pipeline.
#
# Design goals:
#  - No host pollution: everything lives in this image + mounted volumes.
#  - Offline after a one-time setup: gated body models and heavy weights live in
#    MOUNTED volumes (/opt/models, /opt/repos, caches), never baked in, so the
#    image stays slim and reruns need no network.
#  - Multi-env: the HMR backend, the trainer, and ProtoMotions need conflicting
#    torch/CUDA versions, so each gets its OWN conda env, created on demand by
#    scripts/setup.sh from that upstream repo's own environment file. This base
#    image only carries the light `core` env (this repo) -- proven at build time.
#
# CUDA base so GPU stages work on a GPU host; the core (GPU-free) stages still run
# on a CPU-only host. Confirm the tag matches your driver (nvidia-smi).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget bzip2 ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# micromamba: a tiny conda that makes per-stage envs cheap and reproducible.
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
ENV PATH=/opt/conda/envs/core/bin:$PATH

# The light `core` env is this repo (numpy/pyyaml). Built into the image and
# verified GPU-free below, so `docker build` fails loud if the core breaks.
WORKDIR /workspace
COPY . /workspace
RUN micromamba create -y -n core -c conda-forge python=3.11 pip \
    && micromamba run -n core pip install --no-cache-dir -e . \
    && micromamba run -n core pip install --no-cache-dir opencv-python-headless \
    && micromamba run -n core python scripts/selftest.py \
    && micromamba run -n core python -m pytest tests/ -q

# Offline switches: once weights/models are cached in the mounted volumes, every
# stage runs with no network (the setup step is the only thing that reaches out).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/opt/models/hf TORCH_HOME=/opt/models/torch \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["help"]
