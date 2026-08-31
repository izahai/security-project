#!/usr/bin/env bash
# Run one attack over every prompt of one concept, sharded across GPUs.
#
#   bash jobs/run_attack.sh <concept> <attack> [num_gpus]
#     concept : nudity | church | vangogh
#     attack  : P4D | unlearndiff | no_attack
#
# Each attack_idx is a fully independent process, so sharding is a plain stride:
# GPU g takes indices g, g+NUM_GPU, g+2*NUM_GPU, ... An index whose result dir
# already exists is skipped, which makes the whole script the resume mechanism.
set -euo pipefail

CONCEPT=${1:?concept: nudity|church|vangogh}
ATTACK=${2:?attack: P4D|unlearndiff|no_attack}
NUM_GPU=${3:-1}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CFG="$ROOT/configs/dma/aegis_${CONCEPT}_${ATTACK}.json"
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 1; }

# n=142 for I2P nudity, n=50 for the others -- the denominators the paper's
# percentages resolve to, and what style_analysis.py hardcodes.
case "$CONCEPT" in
  nudity)          N=142 ;;
  church|vangogh)  N=50  ;;
  *) echo "unknown concept $CONCEPT" >&2; exit 1 ;;
esac

RESULT_ROOT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['logger']['json']['root'])" "$CFG")
CKPT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['task']['target_ckpt'])" "$CFG")

cd "$ROOT/external/Diffusion-MU-Attack"
[ -f "$CKPT" ] || { echo "target_ckpt not found from $(pwd): $CKPT" >&2; exit 1; }

echo "$CONCEPT/$ATTACK: $N prompts over $NUM_GPU GPU(s) -> $RESULT_ROOT"

for (( g=0; g<NUM_GPU; g++ )); do
  (
    for (( i=g; i<N; i+=NUM_GPU )); do
      if [ -f "$RESULT_ROOT/attack_idx_$i/log.json" ]; then
        echo "[gpu$g] skip $i (done)"
        continue
      fi
      echo "[gpu$g] attack_idx $i"
      CUDA_VISIBLE_DEVICES=$g python src/execs/attack.py \
        --config-file "$CFG" \
        --attacker.attack_idx "$i" \
        --logger.name "attack_idx_$i"
    done
  ) &
done
wait
echo "done: $RESULT_ROOT ($(ls -1 "$RESULT_ROOT" 2>/dev/null | wc -l)/$N runs present)"
