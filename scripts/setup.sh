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

# Downloaders for scriptable (non-gated) weights live in the light core env.
"$MAMBA" run -n core pip install -q gdown "huggingface_hub[cli]" 2>/dev/null || true

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

fetch_gvhmr_weights() {  # ~5.6 GB, non-gated: HF community mirror, else the official Drive folder
  local dst="$REPOS/gvhmr/inputs/checkpoints"
  [ -d "$dst" ] && [ -n "$(ls -A "$dst" 2>/dev/null)" ] && { echo "  have GVHMR weights"; return; }
  mkdir -p "$dst"
  echo "  fetching GVHMR weights (~5.6 GB, best-effort) ..."
  if "$MAMBA" run -n core huggingface-cli download camenduru/GVHMR --local-dir "$dst" >/dev/null 2>&1; then
    echo "  ok: GVHMR weights from HF mirror -> $dst"
  elif "$MAMBA" run -n core gdown --folder \
        https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD -O "$dst" >/dev/null 2>&1; then
    echo "  ok: GVHMR weights from Google Drive -> $dst"
  else
    echo "  could not auto-fetch GVHMR weights; follow $REPOS/gvhmr/docs/INSTALL.md"
  fi
  echo "  (confirm layout is inputs/checkpoints/{gvhmr,hmr2,vitpose,yolo,dpvo}/ per INSTALL.md)"
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
  gvhmr) clone gvhmr https://github.com/zju3dv/GVHMR; make_env gvhmr gvhmr; fetch_weights gvhmr; fetch_gvhmr_weights ;;
  wham)  clone wham  https://github.com/yohanshin/WHAM;  make_env wham  wham;  fetch_weights wham ;;
  tram)  clone tram  https://github.com/yufu-wang/tram;  make_env tram  tram;  fetch_weights tram ;;
  *) echo "unknown backend: $BACKEND" >&2; exit 2 ;;
esac

# The offline 263-d feature path for the generators: TMR's joints_to_guofeats
# (byte-exact HumanML3D, ships its own reference skeleton) + human_body_prior FK.
setup_features() {  # setup_features <env-name>
  clone tmr https://github.com/Mathux/TMR
  "$MAMBA" run -n "$1" pip install "git+https://github.com/nghorbani/human_body_prior" einops || \
    echo "  (install human_body_prior + einops into env $1 by hand if this failed)"
}

# --- motion model (dataset -> model) --------------------------------------
case "$METHOD" in
  ""|noop) : ;;
  momask) clone momask https://github.com/EricGuo5513/momask-codes; make_env momask momask; fetch_weights momask; setup_features momask ;;
  mdm)    clone mdm https://github.com/GuyTevet/motion-diffusion-model; make_env mdm mdm; fetch_weights mdm; setup_features mdm ;;
  protomotions) clone protomotions https://github.com/NVlabs/ProtoMotions; make_env protomotions protomotions; fetch_weights protomotions
                echo "  pulling ProtoMotions pretrained trackers (git-lfs, in-repo) ..."
                ( cd "$REPOS/protomotions" && git lfs pull ) || echo "  (run 'git lfs pull' in $REPOS/protomotions by hand)" ;;
  closd)  clone closd https://github.com/GuyTevet/CLoSD; make_env closd closd; fetch_weights closd; setup_features closd ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

# --- gated body model (ONE registration; then fully offline) ---------------
# Only the generators need it (for FK). protomotions needs none of this.
if [ "$METHOD" != "protomotions" ] && [ -n "$METHOD" ] && [ "$METHOD" != "noop" ]; then
  echo "== gated body model (one-time registration; then fully offline) =="
  check_gated "smplh/neutral/model.npz" "SMPL-H neutral (no DMPL, no gender split)" "mano.is.tue.mpg.de"
fi

echo
echo "Setup done. Edit configs/*.yaml so repo/smpl_model paths point under $REPOS and $MODELS,"
echo "then:  docker compose run --rm pipeline all"
