#!/bin/bash
# Phase M (paired stats) + Phase S (integrity)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs experiments/results paper/tables

echo "=== Phase M: paired stats ==="
python3 -u scripts/paired_stats.py | tee logs/paired_stats.log

echo "=== Phase S: integrity audit ==="
python3 -u scripts/integrity_audit.py | tee logs/integrity_audit.log

echo "M_AND_S_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
