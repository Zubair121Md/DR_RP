#!/bin/bash
# Phase Q/R — paper tables + figures (fast; safe with Cursor open)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs paper/tables paper/figures
python3 -u scripts/build_paper_artifacts.py | tee logs/paper_artifacts.log
echo "Open paper/tables/ and paper/figures/"
echo "PAPER_BUILD_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
