#!/bin/bash
# Phase O — train/eval Swin-Tiny (CNN vs Hybrid vs Transformer)
# Quit Cursor/Chrome on 8GB Mac. Paste in Terminal.app.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

MARK=PHASE_O_ALL_OK
LOG=logs/phase_o_swin.log
KA=logs/keepalive_phase_o.log

echo "KEEPALIVE_PHASE_O_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
while ! grep -q "$MARK" "$LOG" 2>/dev/null; do
  if ! pgrep -f 'scripts/run_phase_o.py' >/dev/null \
     && ! pgrep -f 'src.train --config configs/phase_o' >/dev/null \
     && ! pgrep -f 'src.evaluate' >/dev/null; then
    echo "PHASE_O_RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
    nohup caffeinate -is python3 -u scripts/run_phase_o.py --seeds 42 43 44 \
      >> logs/phase_o_outer.log 2>&1 &
    echo $! > logs/phase_o.pid
  fi
  sleep 60
done
echo "KEEPALIVE_PHASE_O_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$KA"
