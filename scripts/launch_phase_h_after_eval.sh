#!/bin/bash
cd "$(dirname "$0")/.."
LOG=logs/hybrid_s42_eval_resilient.log
echo "Waiting for HYBRID_S42_EVAL_ALL_OK..."
while ! grep -q "HYBRID_S42_EVAL_ALL_OK" "$LOG" 2>/dev/null; do
  sleep 60
  pgrep -f 'eval_hybrid_s42_resilient|src.evaluate' >/dev/null || {
    # eval died without OK — relaunch resilient once
    if ! grep -q "HYBRID_S42_EVAL_ALL_OK" "$LOG" 2>/dev/null; then
      echo "relaunch eval $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/phase_h_gate.log
      nohup caffeinate -is python3 -u scripts/eval_hybrid_s42_resilient.py 100 >> logs/hybrid_s42_eval_stdout.log 2>&1 &
      echo $! > logs/hybrid_s42_eval.pid
    fi
  }
done
echo "EVAL done — starting Phase H seed 43 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/phase_h_gate.log
nohup python3 -u scripts/watchdog_hybrid.py 43 H1 H2 H3 H4 >> logs/hybrid_watchdog_s43_outer.log 2>&1 &
echo $! > logs/hybrid_watchdog_s43.pid
wait
