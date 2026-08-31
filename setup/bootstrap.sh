#!/usr/bin/env bash
# One-command setup for an AEGIS reproduce box. Safe to re-run: every phase
# leaves a marker under $AEGIS_DATA_ROOT/.bootstrap and is skipped next time,
# so a killed download resumes by running this again.
#
#   bash setup/bootstrap.sh                     # everything
#   bash setup/bootstrap.sh --only assets       # one phase
#   bash setup/bootstrap.sh --skip envs         # e.g. envs already exist
#   bash setup/bootstrap.sh --status            # what is done, what is missing
#
# Phases: preflight secrets envs assets verify
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

PHASES=(preflight secrets envs assets verify)
ONLY=(); SKIP=(); STATUS_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --only)   ONLY+=("$2"); shift 2 ;;
    --skip)   SKIP+=("$2"); shift 2 ;;
    --status) STATUS_ONLY=1; shift ;;
    --force)  rm -rf "${AEGIS_DATA_ROOT:-/workspace/aegis-data}/.bootstrap"; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- config
[ -f setup/.env ] && set -a && . setup/.env && set +a
: "${AEGIS_DATA_ROOT:=/workspace/aegis-data}"
MARK="$AEGIS_DATA_ROOT/.bootstrap"
mkdir -p "$MARK"

# One shared HF cache on the data volume. Without this every entry point
# re-downloads SD v1.4's diffusers weights into a different directory.
export HF_HOME="$AEGIS_DATA_ROOT/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
skip() { printf '   %s\n' "$*"; }
done_() { touch "$MARK/$1.done"; }
is_done() { [ -f "$MARK/$1.done" ]; }

wants() {
  local p=$1
  [ ${#ONLY[@]} -gt 0 ] && { printf '%s\n' "${ONLY[@]}" | grep -qx "$p"; return; }
  printf '%s\n' "${SKIP[@]:-}" | grep -qx "$p" && return 1
  return 0
}

if [ "$STATUS_ONLY" = 1 ]; then
  echo "data root: $AEGIS_DATA_ROOT"
  for p in "${PHASES[@]}"; do
    is_done "$p" && echo "  done    $p" || echo "  pending $p"
  done
  exit 0
fi

# ---------------------------------------------------------------- preflight
if wants preflight && ! is_done preflight; then
  log "preflight"
  mkdir -p "$AEGIS_DATA_ROOT"
  FREE=$(df -BG --output=avail "$AEGIS_DATA_ROOT" | tail -1 | tr -dc '0-9')
  echo "   disk free at $AEGIS_DATA_ROOT: ${FREE} GB (need ~100)"
  [ "$FREE" -lt 60 ] && { echo "   too small -- set AEGIS_DATA_ROOT to a bigger volume" >&2; exit 1; }
  [ "$FREE" -lt 100 ] && echo "   WARN: under 100 GB; drop a concept or delete generated images as you go"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/   gpu /'
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
    if   [ "$VRAM" -ge 40000 ]; then echo "   ${VRAM} MiB: can train and attack"
    elif [ "$VRAM" -ge 16000 ]; then echo "   ${VRAM} MiB: attacks and generation only -- training needs >=40 GB"
    else echo "   WARN: ${VRAM} MiB is below the 16 GB the attacks need"; fi
  else
    echo "   no nvidia-smi: setup will complete, but nothing can be run here"
  fi
  for c in conda git curl unzip python; do
    command -v $c >/dev/null || { echo "   missing: $c" >&2; exit 1; }
  done

  # Everything heavy lives on the data volume and the repo only holds symlinks.
  # On a host where the volume outlives the container (RunPod /workspace, a
  # mounted disk), a rebuilt box re-clones the repo and finds the 25 GB already
  # there. external/ is in the list because the cloned harness carries the
  # 900 MB style classifier and the diffusers .cache.
  for d in models data/imgs external; do
    mkdir -p "$AEGIS_DATA_ROOT/$d"
    if [ ! -L "$d" ]; then
      [ -d "$d" ] && { cp -rn "$d"/. "$AEGIS_DATA_ROOT/$d"/ 2>/dev/null || true; rm -rf "$d"; }
      ln -s "$AEGIS_DATA_ROOT/$d" "$d"
    fi
    echo "   $d -> $AEGIS_DATA_ROOT/$d"
  done
  done_ preflight
else wants preflight && skip "preflight already done"; fi

# ---------------------------------------------------------------- secrets
if wants secrets && ! is_done secrets; then
  log "secrets"
  [ -f setup/.env ] || echo "   no setup/.env (cp setup/env.example setup/.env) -- using the environment"
  if [ -n "${HF_TOKEN:-}" ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer $HF_TOKEN" \
      https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt)
    case $code in
      200|302) echo "   HF_TOKEN ok, ckpt reachable" ;;
      401|403)  echo "   HF_TOKEN rejected (HTTP $code): accept the CompVis licence on the model page" >&2; exit 1 ;;
      *)        echo "   WARN: unexpected HTTP $code probing the ckpt" ;;
    esac
  else
    echo "   HF_TOKEN unset -- the SD v1.4 checkpoint download will be skipped" >&2
  fi
  [ -n "${WANDB_API_KEY:-}" ] && echo "   W&B enabled" || echo "   WANDB_API_KEY unset -- training logs locally only"
  [ -n "${COCO_DRIVE_ID:-}" ] && echo "   COCO_DRIVE_ID set" || echo "   COCO_DRIVE_ID unset -- FID reference set will be skipped"
  done_ secrets
else wants secrets && skip "secrets already checked"; fi

# ---------------------------------------------------------------- envs
if wants envs && ! is_done envs; then
  log "envs (~20 min)"
  eval "$(conda shell.bash hook)"
  # conda's own package cache also belongs on the data volume
  conda config --add pkgs_dirs "$AEGIS_DATA_ROOT/conda-pkgs" 2>/dev/null || true

  if conda env list | grep -qE '^AEGIS\s'; then skip "conda env AEGIS exists"
  else conda env create -f environment.yaml; fi
  conda activate AEGIS
  pip install -e . -q
  pip install -r requirements-eval.txt -q
  conda deactivate

  # The attack harness pins different versions; keep it in its own env.
  if conda env list | grep -qE '^dma\s'; then skip "conda env dma exists"
  else
    [ -d external/Diffusion-MU-Attack ] || \
      git clone --depth 1 https://github.com/OPTML-Group/Diffusion-MU-Attack.git external/Diffusion-MU-Attack
    conda create -y -n dma python=3.9 -q
    conda activate dma
    pip install -q -r external/Diffusion-MU-Attack/requirements.txt 2>/dev/null \
      || pip install -q torch torchvision diffusers==0.21.4 transformers accelerate omegaconf pandas Pillow
    conda deactivate
  fi
  done_ envs
else wants envs && skip "envs already done"; fi

# ---------------------------------------------------------------- assets
if wants assets && ! is_done assets; then
  log "assets (~25 GB)"
  eval "$(conda shell.bash hook)"; conda activate AEGIS
  bash setup/fetch_assets.sh
  conda deactivate
  done_ assets
else wants assets && skip "assets already done"; fi

# ---------------------------------------------------------------- verify
if wants verify; then
  log "verify"
  eval "$(conda shell.bash hook)"; conda activate AEGIS
  python setup/check_config.py
  python - <<'PY'
import os, pathlib, sys
root = pathlib.Path('.')
need = {
    'external/Diffusion-MU-Attack/src/execs/attack.py': 'attack harness',
    'external/Diffusion-MU-Attack/results/checkpoint-2800': 'WikiArt style classifier (Table 3)',
    'external/Diffusion-MU-Attack/prompts/church.csv': 'published prompt sets',
    'models/sd-v1-4-full-ema.ckpt': 'SD v1.4 checkpoint (training)',
    'data/imgs/coco_10k': 'COCO-10k reals (FID reference)',
}
missing = [(p, w) for p, w in need.items() if not (root / p).exists()]
for p, w in need.items():
    print(f"   {'ok     ' if (root/p).exists() else 'MISSING'} {w}")
if missing:
    print('\n   incomplete -- re-run: bash setup/bootstrap.sh --only assets', file=sys.stderr)
    sys.exit(1)
print('\n   ready. next: see REPRODUCE.md')
PY
fi
