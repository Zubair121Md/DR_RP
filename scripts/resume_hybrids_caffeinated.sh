#!/bin/bash
# Detach fully from Cursor/agent shells so the pipeline survives session death.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

if [[ -f logs/resume.pid ]]; then
  old=$(cat logs/resume.pid || true)
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "Already running pid=$old"
    exit 0
  fi
fi

echo "LAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ) batch=1 device=cpu threads=2 pause=60s caffeinate=-is" | tee -a logs/resume_launch.log

(
  nohup caffeinate -is /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -u \
    scripts/resume_paper_pipeline.py \
    >> logs/resume_launch.log 2>&1 &
  echo $! > logs/resume.pid
)

echo "STARTED pid=$(cat logs/resume.pid)"
