#!/usr/bin/env bash
# Launch all 36 paper v2 trainings under caffeinate + watchdog (serial queue).
#
# Jobs:
#   matched B0 H1–H4 × seeds 42–44   (12)
#   hybrid   H1–H4 × seeds 45–47     (12)
#   matched B0 H1–H4 × seeds 45–47   (12)
#
# Usage:
#   bash scripts/run_paper_v2_36_caffeinated.sh
#
# Optional:
#   PAPER_V2_N_BOOT=20 bash scripts/run_paper_v2_36_caffeinated.sh
#
# Status:
#   tail -f logs/paper_v2_36.log
#   tail -f logs/paper_v2_36_watchdog.log
#   grep PAPER_V2_36_ALL_OK logs/paper_v2_36.log
#
# Stop (watchdog + children):
#   kill "$(cat logs/paper_v2_36_watchdog.pid)" 2>/dev/null || true
#   pkill -f 'scripts/run_paper_v2_36.py' || true
#   pkill -f 'src.train --config configs/paper_v2' || true
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs configs/paper_v2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTHONUNBUFFERED=1
export PAPER_V2_N_BOOT="${PAPER_V2_N_BOOT:-20}"

WD_PID_FILE=logs/paper_v2_36_watchdog.pid
WD_OUTER=logs/paper_v2_36_watchdog_outer.log

if [[ -f "$WD_PID_FILE" ]]; then
  old="$(cat "$WD_PID_FILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "Already running watchdog pid=$old"
    echo "  log: logs/paper_v2_36_watchdog.log"
    echo "  train log: logs/paper_v2_36.log"
    exit 0
  fi
fi

echo "=== generate configs/manifest (36 jobs) ==="
python3 -u scripts/generate_paper_v2_configs.py | tee -a logs/paper_v2_36_launch.log

echo "LAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ) n_boot=$PAPER_V2_N_BOOT" | tee -a logs/paper_v2_36_launch.log

# Outer caffeinate keeps the watchdog itself awake; watchdog relaunches
# the orchestrator under its own caffeinate -is.
nohup caffeinate -is python3 -u scripts/watchdog_paper_v2_36.py \
  >> "$WD_OUTER" 2>&1 &
echo $! > "$WD_PID_FILE"
disown || true

echo "STARTED watchdog pid=$(cat "$WD_PID_FILE")"
echo "  outer:  $WD_OUTER"
echo "  wd:     logs/paper_v2_36_watchdog.log"
echo "  train:  logs/paper_v2_36.log"
echo "  done when: grep -q PAPER_V2_36_ALL_OK logs/paper_v2_36.log"
echo
echo "NOTE: jobs run SERIALLY (one at a time). Parallel would jetsam on this machine."
echo "Estimated wall-clock ~7–11 days (see paper/TRAINING_RUNTIME_ESTIMATE.md)."
