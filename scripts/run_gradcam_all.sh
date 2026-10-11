#!/bin/bash
# Phase N — Grad-CAM on all hybrid H1–H4 × seeds 42/43/44
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs paper/figures/gradcam
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
caffeinate -is python3 -u scripts/run_gradcam_all.py 2>&1 | tee logs/gradcam_all.log
echo "Open paper/figures/gradcam/INDEX.md"
