#!/bin/bash
# Phase P LODO for one or more seeds (e.g. 43 44). Cache must already exist.
# Usage: bash scripts/keepalive_phase_p_seeds.sh 43 44
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <seed> [seed2 ...]"
  exit 1
fi
SEEDS=("$@")
KA=logs/keepalive_phase_p_seeds.log
echo "KEEPALIVE_P_SEEDS_START seeds=${SEEDS[*]} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"

# Ensure DeepDRiD cache once
bash scripts/ensure_lodo_caches.sh configs/phase_p/lodo_mobilevit_green_clahe_ce.yaml | tee -a "$KA"

for SEED in "${SEEDS[@]}"; do
  MARK=PHASE_P_ALL_OK
  LOG="logs/phase_p_lodo_s${SEED}.log"
  echo "=== SEED $SEED ===" | tee -a "$KA"
  # fresh start marker only if file missing ALL_OK
  if grep -q "$MARK" "$LOG" 2>/dev/null; then
    echo "SEED $SEED already ALL_OK — skip" | tee -a "$KA"
    continue
  fi
  # do not truncate if partial progress exists; orchestrator skips completed folds
  if [ ! -f "$LOG" ]; then
    echo "PHASE_P_START seed=$SEED" > "$LOG"
  fi
  while ! grep -q "$MARK" "$LOG" 2>/dev/null; do
    if ! pgrep -f "scripts/run_phase_p_lodo.py --seed ${SEED}" >/dev/null \
       && ! pgrep -f 'src.train_lodo' >/dev/null \
       && ! pgrep -f 'src.evaluate' >/dev/null; then
      echo "PHASE_P_RELAUNCH seed=$SEED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
      nohup caffeinate -is python3 -u scripts/run_phase_p_lodo.py --seed "$SEED" \
        >> "logs/phase_p_outer_s${SEED}.log" 2>&1 &
      echo $! > "logs/phase_p_s${SEED}.pid"
    fi
    sleep 60
  done
  echo "SEED_${SEED}_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
done

python3 -u scripts/summarize_phase_p.py --seeds 42 43 44 | tee -a "$KA" || \
  python3 -u scripts/summarize_phase_p.py --seeds "${SEEDS[@]}" | tee -a "$KA"
echo "KEEPALIVE_P_SEEDS_ALL_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
