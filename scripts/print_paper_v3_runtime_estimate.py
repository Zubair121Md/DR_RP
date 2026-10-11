#!/usr/bin/env python3
"""Print paper_v3 wall-clock estimate. Does not start training."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "configs" / "paper_v3" / "manifest_v3.json"

# Observed / conservative bands on this Mac (CPU, bs=4, freeze_bn).
# Hybrid MobileViT-S APTOS-only ≈ 4–6 h train + ~0.5–1.5 h 4-domain eval.
# Matched EfficientNet-B0 under same regime ≈ 3–5 h train + same eval.
# 448×448 is roughly 3–4× slower than 224 on CPU (FLOPs + no cache).
HOURS = {
    ("core", "matched_b0"): (3.5, 5.0),
    ("core", "hybrid"): (4.5, 6.5),
    ("decomp", "hybrid_decomp"): (4.5, 6.5),
    ("res448", "hybrid_448"): (14.0, 22.0),
}
EVAL_EXTRA = (0.5, 1.5)


def main() -> None:
    if not MAN.is_file():
        raise SystemExit(f"MISSING {MAN.relative_to(ROOT)} — run generate_paper_v3_configs.py first")
    data = json.loads(MAN.read_text())
    raw = os.environ.get("PAPER_V3_GROUPS", "core,decomp")
    groups = {g.strip() for g in raw.split(",") if g.strip()}
    jobs = [j for j in data["jobs"] if j["group"] in groups]

    lo = hi = 0.0
    by = {}
    for j in jobs:
        key = (j["group"], j["family"])
        band = HOURS.get(key)
        if band is None:
            raise SystemExit(f"No hour band for {key}")
        by.setdefault(key, 0)
        by[key] += 1
        lo += band[0] + EVAL_EXTRA[0]
        hi += band[1] + EVAL_EXTRA[1]

    print("PAPER_V3_RUNTIME_ESTIMATE")
    print(f"  groups_selected: {sorted(groups)}")
    print(f"  n_jobs: {len(jobs)}")
    for k, n in sorted(by.items()):
        print(f"    {k[0]}/{k[1]}: {n} × ~{HOURS[k][0]}–{HOURS[k][1]} h train + eval")
    print(f"  wall_clock_serial_hours: ~{lo:.0f}–{hi:.0f} h")
    print(f"  wall_clock_serial_days:  ~{lo/24:.1f}–{hi/24:.1f} days")
    print("  assumption: serial CPU; caffeinate; chunked train (rc 75)")
    print("  DO NOT LAUNCH until you explicitly confirm this estimate.")
    print("PAPER_V3_RUNTIME_ESTIMATE_OK")


if __name__ == "__main__":
    main()
