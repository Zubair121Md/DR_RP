#!/bin/bash
# Ensure green_clahe caches exist for LODO (esp. DeepDRiD — was empty).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs datasets/cache/green_clahe/deepdrid

CFG="${1:-configs/phase_p/lodo_mobilevit_green_clahe_ce.yaml}"
N=$(find datasets/cache/green_clahe/deepdrid -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
echo "deepdrid green_clahe cache frames: $N"
if [ "$N" -lt 100 ]; then
  echo "Building DeepDRiD (+any missing) green_clahe cache — this can take a while..."
  python3 -u scripts/build_preprocess_cache.py \
    --config "$CFG" \
    --preprocess green_clahe \
    --cache-root datasets/cache \
    2>&1 | tee logs/build_deepdrid_cache.log
else
  echo "DeepDRiD cache looks OK (>=100 frames)"
fi
N2=$(find datasets/cache/green_clahe/deepdrid -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
echo "deepdrid cache after: $N2"
test "$N2" -ge 100
echo "LODO_CACHE_OK"
