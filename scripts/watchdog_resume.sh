#!/bin/bash
# Outer watchdog: keep resume_paper_pipeline alive until RESUME_PIPELINE_DONE.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "WATCHDOG_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/watchdog.log

while true; do
  if grep -q "RESUME_PIPELINE_DONE" logs/resume_pipeline_stdout.log 2>/dev/null; then
    echo "WATCHDOG_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/watchdog.log
    exit 0
  fi

  alive=0
  if [[ -f logs/resume.pid ]]; then
    pid=$(cat logs/resume.pid 2>/dev/null || true)
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  fi

  if [[ "$alive" -eq 0 ]]; then
    echo "WATCHDOG_RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/watchdog.log
    rm -f logs/resume.pid
    (
      nohup caffeinate -is /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -u \
        scripts/resume_paper_pipeline.py \
        >> logs/resume_launch.log 2>&1 &
      echo $! > logs/resume.pid
    )
    echo "WATCHDOG_STARTED pid=$(cat logs/resume.pid)" | tee -a logs/watchdog.log
  fi

  sleep 90
done
