#!/usr/bin/env python3
"""Summarize Phase O: B0 vs MobileViT-H2 vs Swin-Tiny (green_clahe × CE)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
SPLITS = ("aptos_test", "idrid_external", "messidor2_external", "deepdrid_external")

# B0 green_clahe CE run ids from b0 summary
B0 = json.loads((ROOT / "results/b0_2x2_seed_summary.json").read_text())["cells"]["green_clahe_ce"]


def qwk(rid: str, split: str) -> float | None:
    p = ROOT / "results" / rid / f"{split}_metrics.json"
    if not p.is_file():
        return None
    return float(json.loads(p.read_text())["quadratic_weighted_kappa"])


def mean_std(xs: list[float]) -> str:
    a = np.asarray(xs, float)
    if a.size == 0:
        return "—"
    if a.size == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def main() -> None:
    rows = []
    # B0 — only APTOS/IDRiD in seed summary
    b0_aptos = [float(r["aptos_qwk"]) for r in B0["runs"]]
    b0_idrid = [float(r["idrid_qwk"]) for r in B0["runs"]]
    rows.append(("EfficientNet-B0", "APTOS", mean_std(b0_aptos)))
    rows.append(("EfficientNet-B0", "IDRiD", mean_std(b0_idrid)))
    rows.append(("EfficientNet-B0", "Messidor-2", "n/a in B0 seed table"))
    rows.append(("EfficientNet-B0", "DeepDRiD", "n/a in B0 seed table"))

    for family, tmpl in [
        ("MobileViT-S H2", "hybrid_h2_s{seed}_v1"),
        ("Swin-Tiny", "phase_o_swin_tiny_s{seed}_v1"),
    ]:
        for split in SPLITS:
            vals = []
            for s in SEEDS:
                v = qwk(tmpl.format(seed=s), split)
                if v is not None:
                    vals.append(v)
            rows.append((family, split, mean_std(vals)))

    lines = [
        "# Phase O — CNN vs Hybrid vs Transformer",
        "",
        "Matched recipe: **green_clahe × CE**. Seeds 42/43/44 where available.",
        "",
        "| Model | Split | QWK mean±std |",
        "|-------|-------|-------------:|",
    ]
    for m, sp, v in rows:
        lines.append(f"| {m} | {sp} | {v} |")
    lines += [
        "",
        "Swin runs: `results/phase_o_swin_tiny_s{42,43,44}_v1/`",
        "",
    ]
    out = ROOT / "experiments/results/phase_o_arch_comparison.md"
    out.write_text("\n".join(lines))
    print(out.read_text())
    print("SAVED", out)


if __name__ == "__main__":
    main()
