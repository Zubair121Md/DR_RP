#!/usr/bin/env bash
# Watchdog: keep Phase G/H hybrid 2×2 alive across jetsam kills.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

SEED="${1:-42}"
shift || true
CELLS="${*:-H1 H2 H3 H4}"

echo "HYBRID_WATCHDOG_START $(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$SEED cells=$CELLS" | tee -a logs/hybrid_watchdog.log

while true; do
  if grep -q "HYBRID_2X2_ALL_OK" logs/hybrid_2x2_phase_g.log 2>/dev/null; then
    echo "HYBRID_WATCHDOG_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/hybrid_watchdog.log
    exit 0
  fi

  alive=0
  if pgrep -f 'src.train --config configs/hybrid_runs' >/dev/null 2>&1; then
    alive=1
  elif pgrep -f 'scripts/run_hybrid_2x2.py' >/dev/null 2>&1; then
    alive=1
  fi

  if [[ "$alive" -eq 0 ]]; then
    echo "HYBRID_WATCHDOG_RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/hybrid_watchdog.log
    # shellcheck disable=SC2086
    nohup caffeinate -is env \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
      python3 -u scripts/run_hybrid_2x2.py --seed "$SEED" --cells $CELLS \
      >> logs/hybrid_2x2_stdout.log 2>&1 &
    echo $! > logs/hybrid_2x2.pid
    disown || true
    echo "HYBRID_WATCHDOG_STARTED pid=$(cat logs/hybrid_2x2.pid)" | tee -a logs/hybrid_watchdog.log
  fi

  sleep 90
done
