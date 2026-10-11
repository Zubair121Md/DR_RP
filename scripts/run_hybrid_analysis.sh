#!/bin/bash
# Post-training analysis (Phases J/M/Q prep). Fast CPU — safe with Cursor open.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs experiments/results

echo "=== hybrid completeness check ==="
python3 - <<'PY'
from pathlib import Path
root = Path("results")
need = ["aptos_test","idrid_external","messidor2_external","deepdrid_external"]
miss=[]
for seed in (42,43,44):
  for h in range(1,5):
    rid=f"hybrid_h{h}_s{seed}_v1"
    for n in need:
      if not (root/rid/f"{n}_metrics.json").is_file():
        miss.append(f"{rid}/{n}")
print("missing", miss or "NONE")
raise SystemExit(1 if miss else 0)
PY

echo "=== aggregate seed tables + mild analysis ==="
python3 -u scripts/aggregate_hybrid_seeds.py | tee logs/hybrid_analysis.log

echo "=== optional: extend mild_stage_analysis CLI over hybrids ==="
python3 -u scripts/mild_stage_analysis.py \
  --runs \
    hybrid_h1_s42_v1 hybrid_h2_s42_v1 hybrid_h3_s42_v1 hybrid_h4_s42_v1 \
    hybrid_h1_s43_v1 hybrid_h2_s43_v1 hybrid_h3_s43_v1 hybrid_h4_s43_v1 \
    hybrid_h1_s44_v1 hybrid_h2_s44_v1 hybrid_h3_s44_v1 hybrid_h4_s44_v1 \
  --out results/mild_stage_analysis_hybrid_cli.json \
  | tee -a logs/hybrid_analysis.log

echo "ANALYSIS_ALL_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a logs/hybrid_analysis.log
echo "Open: experiments/results/hybrid_seed_summary.md"
