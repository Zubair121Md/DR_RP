#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LOG=logs/hybrid_s42_eval_resilient.log
echo "KEEPALIVE_EVAL_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/keepalive_eval.log
while ! grep -q "HYBRID_S42_EVAL_ALL_OK" "$LOG" 2>/dev/null; do
  if ! pgrep -f 'scripts/eval_hybrid_s42_resilient.py' >/dev/null && ! pgrep -f 'src.evaluate --config' >/dev/null; then
    echo "KEEPALIVE_RELAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/keepalive_eval.log
    nohup caffeinate -is python3 -u scripts/eval_hybrid_s42_resilient.py 20 \
      >> logs/hybrid_s42_eval_stdout.log 2>&1 &
    echo $! > logs/hybrid_s42_eval.pid
  fi
  sleep 45
done
echo "KEEPALIVE_EVAL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/keepalive_eval.log
# then Phase H
nohup python3 -u scripts/watchdog_hybrid.py 43 H1 H2 H3 H4 >> logs/hybrid_watchdog_s43_outer.log 2>&1 &
echo $! > logs/hybrid_watchdog_s43.pid
echo "PHASE_H_STARTED pid=$(cat logs/hybrid_watchdog_s43.pid)" | tee -a logs/keepalive_eval.log
