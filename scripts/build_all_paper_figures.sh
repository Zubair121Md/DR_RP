#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs paper/figures/pack
python3 -u scripts/build_all_paper_figures.py 2>&1 | tee logs/build_all_paper_figures.log
echo "Open paper/figures/pack/FIGURE_INDEX.md"
