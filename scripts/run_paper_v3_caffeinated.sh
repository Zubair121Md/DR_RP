#!/usr/bin/env bash
# Launch paper_v3 prioritized trainings under caffeinate + watchdog (serial).
#
# Groups (set PAPER_V3_GROUPS):
#   core     24  matched B0 H1–H4 × 42–44 + hybrid H1–H4 × 45–47
#   decomp    6  CLAHE unweighted CORAL + mild-weighted CE × 42–44
#   res448    6  hybrid H1/H2 at 448×448 × 42–44 (optional)
#
# Default (recommended package without 448):
#   PAPER_V3_GROUPS=core,decomp
#
# Usage (DO NOT run until you have confirmed the runtime estimate):
#   bash scripts/run_paper_v3_caffeinated.sh
#   PAPER_V3_GROUPS=core,decomp,res448 bash scripts/run_paper_v3_caffeinated.sh
#
# Status:
#   tail -f logs/paper_v3.log
#   tail -f logs/paper_v3_watchdog.log
#   grep PAPER_V3_ALL_OK logs/paper_v3.log
#
# Stop:
#   kill "$(cat logs/paper_v3_watchdog.pid)" 2>/dev/null || true
#   pkill -f 'scripts/run_paper_v3.py' || true
#   pkill -f 'src.train --config configs/paper_v3' || true
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs configs/paper_v3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTHONUNBUFFERED=1
export PAPER_V3_N_BOOT="${PAPER_V3_N_BOOT:-20}"
export PAPER_V3_GROUPS="${PAPER_V3_GROUPS:-core,decomp}"

WD_PID_FILE=logs/paper_v3_watchdog.pid
WD_OUTER=logs/paper_v3_watchdog_outer.log

if [[ -f "$WD_PID_FILE" ]]; then
  old="$(cat "$WD_PID_FILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "Already running watchdog pid=$old"
    echo "  log: logs/paper_v3_watchdog.log"
    echo "  train log: logs/paper_v3.log"
    exit 0
  fi
fi

echo "=== generate configs/manifest ==="
python3 -u scripts/generate_paper_v3_configs.py | tee -a logs/paper_v3_launch.log

echo "=== runtime estimate (printed; training not auto-confirmed) ==="
python3 -u scripts/print_paper_v3_runtime_estimate.py | tee -a logs/paper_v3_launch.log

echo "LAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ) groups=$PAPER_V3_GROUPS n_boot=$PAPER_V3_N_BOOT" \
  | tee -a logs/paper_v3_launch.log

nohup caffeinate -is python3 -u scripts/watchdog_paper_v3.py \
  >> "$WD_OUTER" 2>&1 &
echo $! > "$WD_PID_FILE"
disown || true

echo "STARTED watchdog pid=$(cat "$WD_PID_FILE")"
echo "  groups: $PAPER_V3_GROUPS"
echo "  outer:  $WD_OUTER"
echo "  wd:     logs/paper_v3_watchdog.log"
echo "  train:  logs/paper_v3.log"
echo "  done when: grep -q PAPER_V3_ALL_OK logs/paper_v3.log"
echo
echo "NOTE: jobs run SERIALLY. See paper/TRAINING_RUNTIME_ESTIMATE_V3.md."
