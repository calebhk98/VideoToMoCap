#!/usr/bin/env bash
# One-time ONLINE provisioning: clone the upstream repos you chose, build a conda
# env for each (from the repo's OWN environment file, so we never hand-maintain
# their deps), and run their OWN weight-download scripts. After this, every stage
# runs offline from the mounted volumes.
#
#   bash scripts/setup.sh --backend gvhmr --method momask
#
# Gated body models (SMPL/SMPL-H) cannot be auto-downloaded -- they need a login.
# Register once, drop the files under ./models (mounted at /opt/models), and this
# script verifies they're present. Nothing gated is ever baked into the image.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

BACKEND=""; METHOD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --method)  METHOD="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPOS="${REPOS_DIR:-/opt/repos}"
MODELS="${MODELS_DIR:-/opt/models}"
mkdir -p "$REPOS" "$MODELS"

have() { command -v "$1" >/dev/null 2>&1; }
MAMBA="micromamba"; have "$MAMBA" || MAMBA="conda"

clone() {  # clone <dir> <url>   (idempotent)
  local dir="$REPOS/$1"
  [ -d "$dir/.git" ] && { echo "  have $1"; return; }
  echo "  cloning $1 ..."; git clone --depth 1 "$2" "$dir"
}

make_env() {  # make_env <env-name> <repo-dir>   (from the repo's own env file, else pip)
  local name="$1" dir="$REPOS/$2"
  if "$MAMBA" env list | grep -q "\b$name\b"; then echo "  env $name exists"; return; fi
  local yml
  yml=$(ls "$dir"/environment.y*ml 2>/dev/null | head -1 || true)
  if [ -n "$yml" ]; then
    echo "  creating env $name from $(basename "$yml") ..."
    "$MAMBA" env create -n "$name" -f "$yml"
  else
    echo "  creating env $name (python 3.10 + repo requirements) ..."
    "$MAMBA" create -y -n "$name" -c conda-forge python=3.10 pip
    [ -f "$dir/requirements.txt" ] && "$MAMBA" run -n "$name" pip install -r "$dir/requirements.txt"
  fi
}

fetch_weights() {  # run the repo's OWN download scripts (non-gated weights)
  local dir="$REPOS/$1"
  for s in "$dir"/prepare/download_*.sh "$dir"/scripts/download*.sh "$dir"/download_*.sh; do
    [ -f "$s" ] || continue
    echo "  running $(basename "$s") ..."; ( cd "$dir" && bash "$s" ) || echo "  (skipped $(basename "$s"); run it by hand if needed)"
  done
}

check_gated() {  # warn (do not fail) if a gated model file is absent
  local rel="$1" what="$2"
  [ -e "$MODELS/$rel" ] && { echo "  ok: $what"; return; }
  echo "  MISSING (gated): $what -> place at $MODELS/$rel   ($3)"
}

echo "== core env =="
"$MAMBA" run -n core python scripts/selftest.py >/dev/null && echo "  ok: core pipeline runs"

# --- HMR backend (video -> SMPL) ------------------------------------------
case "$BACKEND" in
  ""|none) : ;;
  gvhmr) clone gvhmr https://github.com/zju3dv/GVHMR; make_env gvhmr gvhmr; fetch_weights gvhmr ;;
  wham)  clone wham  https://github.com/yohanshin/WHAM;  make_env wham  wham;  fetch_weights wham ;;
  tram)  clone tram  https://github.com/yufu-wang/tram;  make_env tram  tram;  fetch_weights tram ;;
  *) echo "unknown backend: $BACKEND" >&2; exit 2 ;;
esac

# --- motion model (dataset -> model) --------------------------------------
case "$METHOD" in
  ""|noop) : ;;
  momask) clone momask https://github.com/EricGuo5513/momask-codes; make_env momask momask; fetch_weights momask
          clone humanml3d https://github.com/EricGuo5513/HumanML3D ;;  # feature extraction
  mdm)    clone mdm https://github.com/GuyTevet/motion-diffusion-model; make_env mdm mdm; fetch_weights mdm
          "$MAMBA" run -n mdm pip install "git+https://github.com/nghorbani/human_body_prior" || true
          clone humanml3d https://github.com/EricGuo5513/HumanML3D ;;
  protomotions) clone protomotions https://github.com/NVlabs/ProtoMotions; make_env protomotions protomotions; fetch_weights protomotions ;;
  closd)  clone closd https://github.com/GuyTevet/CLoSD; make_env closd closd; fetch_weights closd ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

# --- gated body models (register once; auto-download impossible) -----------
echo "== gated body models (one-time registration; then fully offline) =="
check_gated "smpl/SMPL_NEUTRAL.pkl"      "SMPL neutral"  "smpl.is.tue.mpg.de"
check_gated "smplh/neutral/model.npz"    "SMPL-H (AMASS)" "mano.is.tue.mpg.de"

echo
echo "Setup done. Edit configs/*.yaml so repo/smpl_model paths point under $REPOS and $MODELS,"
echo "then:  docker compose run --rm pipeline all"
