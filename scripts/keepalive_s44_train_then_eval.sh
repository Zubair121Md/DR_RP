#!/bin/bash
# Phase H seed 44: train H1–H4 with watchdog, then resilient external eval.
# Paste-friendly: run via nohup from Terminal.app (quit Cursor/Chrome for RAM).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

TRAIN_MARK=HYBRID_2X2_ALL_OK_SEEDS_44
TRAIN_LOG=logs/hybrid_2x2_phase_s44.log
EVAL_MARK=HYBRID_S44_EVAL_ALL_OK
EVAL_LOG=logs/hybrid_s44_eval_resilient.log
KA=logs/keepalive_s44.log

echo "KEEPALIVE_S44_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"

# --- Stage 1: train ---
while ! grep -q "$TRAIN_MARK" "$TRAIN_LOG" 2>/dev/null; do
  if ! pgrep -f 'scripts/watchdog_hybrid.py 44' >/dev/null \
     && ! pgrep -f 'scripts/run_hybrid_2x2.py' >/dev/null \
     && ! pgrep -f 'src.train --config configs/hybrid_runs' >/dev/null; then
    echo "TRAIN_RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
    nohup caffeinate -is python3 -u scripts/watchdog_hybrid.py 44 H1 H2 H3 H4 \
      >> logs/hybrid_watchdog_s44_outer.log 2>&1 &
    echo $! > logs/hybrid_watchdog_s44.pid
  fi
  sleep 60
done
echo "TRAIN_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"

# --- Stage 2: eval ---
while ! grep -q "$EVAL_MARK" "$EVAL_LOG" 2>/dev/null; do
  if ! pgrep -f 'scripts/eval_hybrid_s44_resilient.py' >/dev/null \
     && ! pgrep -f 'src.evaluate --config' >/dev/null; then
    echo "EVAL_RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
    nohup caffeinate -is python3 -u scripts/eval_hybrid_s44_resilient.py 20 \
      >> logs/hybrid_s44_eval_stdout.log 2>&1 &
    echo $! > logs/hybrid_s44_eval.pid
  fi
  sleep 45
done
echo "EVAL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
echo "KEEPALIVE_S44_ALL_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
