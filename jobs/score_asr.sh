#!/usr/bin/env bash
# Aggregate the attack logs into ASR1 / ASR2 numbers.
#
#   bash jobs/score_asr.sh <concept>      # nudity | church | vangogh
#
# CPU only -- it just re-reads log.json, so it is safe (and cheap) to re-run.
set -euo pipefail

CONCEPT=${1:?concept: nudity|church|vangogh}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/external/Diffusion-MU-Attack"

for atk in P4D unlearndiff; do
  R="files/results/aegis_${CONCEPT}_${atk}"
  [ -d "$R" ] || { echo "skip $atk: no results at $R"; continue; }
  label=$([ "$atk" = P4D ] && echo "ASR1 (P4D)" || echo "ASR2 (UnlearnDiffAtk)")
  echo "=== $CONCEPT / $label"

  if [ "$CONCEPT" = vangogh ]; then
    # The paper does not say which top-k the style detector was read at, and the
    # script's own default (10) disagrees with the upstream README's {1|3}.
    # Sweep all three; the right one is whichever puts base SD v1.4 at 100%.
    for k in 1 3 10; do
      echo "--- top_k=$k"
      python scripts/analysis/style_analysis.py --root "$R" --top_k "$k"
    done
  else
    # check_asr.py subtracts prompts that already succeeded before the attack,
    # so it needs the no_attack run as its denominator.
    NA="files/results/aegis_${CONCEPT}_no_attack"
    [ -d "$NA" ] || { echo "missing $NA -- run: bash jobs/run_attack.sh $CONCEPT no_attack"; continue; }
    python scripts/analysis/check_asr.py --root "$R" --root-no-attack "$NA"
  fi
  echo
done
