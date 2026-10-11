#!/bin/bash
# Phase P — LODO with DeepDRiD cache fix. Quit Cursor/Chrome first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

SEED="${1:-42}"
MARK=PHASE_P_ALL_OK
LOG="logs/phase_p_lodo_s${SEED}.log"
KA=logs/keepalive_phase_p.log

echo "KEEPALIVE_PHASE_P_START seed=$SEED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"

# One-time: build missing DeepDRiD cache before training loops
if ! bash scripts/ensure_lodo_caches.sh configs/phase_p/lodo_mobilevit_green_clahe_ce.yaml; then
  echo "CACHE_BUILD_FAILED" | tee -a "$KA"
  exit 1
fi

while ! grep -q "$MARK" "$LOG" 2>/dev/null; do
  if ! pgrep -f 'scripts/run_phase_p_lodo.py' >/dev/null \
     && ! pgrep -f 'src.train_lodo' >/dev/null \
     && ! pgrep -f 'src.evaluate' >/dev/null; then
    echo "PHASE_P_RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
    nohup caffeinate -is python3 -u scripts/run_phase_p_lodo.py --seed "$SEED" \
      >> "logs/phase_p_outer_s${SEED}.log" 2>&1 &
    echo $! > "logs/phase_p_s${SEED}.pid"
  fi
  sleep 60
done
echo "KEEPALIVE_PHASE_P_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
