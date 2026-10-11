#!/usr/bin/env bash
# Caffeinated Phase G/H hybrid 2×2 trainer (frozen_protocol_v1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
SEED="${1:-42}"
CELLS="${2:-H1 H2 H3 H4}"
echo "LAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$SEED cells=$CELLS" | tee -a logs/hybrid_2x2_launch.log
# Cap MPS allocator to reduce jetsam on constrained Macs; batch≥16 still required.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
nohup caffeinate -is env PYTORCH_MPS_HIGH_WATERMARK_RATIO="$PYTORCH_MPS_HIGH_WATERMARK_RATIO" \
  python3 scripts/run_hybrid_2x2.py --seed "$SEED" --cells $CELLS \
  > logs/hybrid_2x2_stdout.log 2>&1 &
echo $! > logs/hybrid_2x2.pid
echo "STARTED pid=$(cat logs/hybrid_2x2.pid) watermark=$PYTORCH_MPS_HIGH_WATERMARK_RATIO"
