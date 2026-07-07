#!/usr/bin/env bash
# Stage dispatcher for the container. Each stage runs in its own conda env.
#
#   docker compose run --rm pipeline <stage> [args...]
#
# Stages:
#   setup   [--backend X --method Y]  one-time ONLINE: clone repos, make envs, fetch weights
#   dataset                           run Pipeline 1 (video -> AMASS dataset) [core env]
#   train                             run Pipeline 2 (dataset -> model)       [core env]
#   all                               dataset + train in one go
#   selftest                          GPU-free end-to-end smoke test          [core env]
#   shell                             drop into a shell in the core env
set -euo pipefail
cd /workspace

stage="${1:-help}"; shift || true

run_core() { micromamba run -n core "$@"; }

case "$stage" in
  setup)     bash scripts/setup.sh "$@" ;;
  dataset)   run_core python -m videotomocap --config "${VIDEO_CONFIG:-configs/dropzone.yaml}" run "$@" ;;
  train)     run_core python -m motion_model --config "${MODEL_CONFIG:-configs/motion_model.yaml}" train "$@" ;;
  all)       run_core python scripts/run_pipeline.py \
               --video-config "${VIDEO_CONFIG:-configs/dropzone.yaml}" \
               --model-config "${MODEL_CONFIG:-configs/motion_model.yaml}" "$@" ;;
  selftest)  run_core python scripts/selftest.py ;;
  shell)     exec micromamba run -n core bash ;;
  help|*)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
