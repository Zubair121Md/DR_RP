#!/usr/bin/env python3
"""Summarize Phase P LODO — one seed or mean±std across seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HOLDOUTS = ("idrid", "messidor2", "deepdrid")
SPLIT = {
    "idrid": "idrid_external",
    "messidor2": "messidor2_external",
    "deepdrid": "deepdrid_external",
}


def qwk(rid: str, split: str) -> float | None:
    p = ROOT / "results" / rid / f"{split}_metrics.json"
    if not p.is_file():
        return None
    return float(json.loads(p.read_text())["quadratic_weighted_kappa"])


def fmt(xs: list[float]) -> str:
    a = np.asarray(xs, float)
    if a.size == 0:
        return "—"
    if a.size == 1:
        return f"{a[0]:.4f}"
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}"


def summarize_seed(seed: int) -> str:
    h2 = f"hybrid_h2_s{seed}_v1"
    lines = [
        f"# Phase P — LODO summary (seed {seed})",
        "",
        "Train = APTOS train + all externals except held-out. Val = APTOS val. Eval = held-out.",
        "",
        "| Holdout | LODO QWK on holdout | H2 (APTOS-only train) QWK on same |",
        "|---------|--------------------:|----------------------------------:|",
    ]
    for h in HOLDOUTS:
        rid = f"phase_p_lodo_holdout_{h}_s{seed}_v1"
        sp = SPLIT[h]
        a, b = qwk(rid, sp), qwk(h2, sp)
        if a is not None and b is not None:
            lines.append(f"| {h} | {a:.4f} | {b:.4f} |")
        else:
            lines.append(f"| {h} | {a if a is not None else '—'} | {b if b is not None else '—'} |")
    lines += ["", f"Runs: `results/phase_p_lodo_holdout_*_s{seed}_v1/`", ""]
    out = ROOT / "experiments/results" / f"phase_p_lodo_s{seed}.md"
    out.write_text("\n".join(lines))
    return str(out)


def summarize_multi(seeds: list[int]) -> str:
    lines = [
        f"# Phase P — LODO mean±std (seeds {seeds})",
        "",
        "| Holdout | LODO QWK | H2 APTOS-only QWK |",
        "|---------|---------:|------------------:|",
    ]
    payload = {"seeds": seeds, "holdouts": {}}
    for h in HOLDOUTS:
        sp = SPLIT[h]
        lodo_vals, h2_vals = [], []
        for s in seeds:
            a = qwk(f"phase_p_lodo_holdout_{h}_s{s}_v1", sp)
            b = qwk(f"hybrid_h2_s{s}_v1", sp)
            if a is not None:
                lodo_vals.append(a)
            if b is not None:
                h2_vals.append(b)
        lines.append(f"| {h} | {fmt(lodo_vals)} | {fmt(h2_vals)} |")
        payload["holdouts"][h] = {
            "lodo": lodo_vals,
            "h2": h2_vals,
            "lodo_mean": float(np.mean(lodo_vals)) if lodo_vals else None,
            "h2_mean": float(np.mean(h2_vals)) if h2_vals else None,
        }
    lines += ["", "Per-seed reports: `phase_p_lodo_s{42,43,44}.md`", ""]
    out = ROOT / "experiments/results" / "phase_p_lodo_seed_summary.md"
    out.write_text("\n".join(lines))
    (ROOT / "experiments/results" / "phase_p_lodo_seed_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()
    if args.seeds:
        for s in args.seeds:
            print(summarize_seed(s))
        print(summarize_multi(args.seeds))
        print((ROOT / "experiments/results/phase_p_lodo_seed_summary.md").read_text())
    else:
        seed = args.seed if args.seed is not None else 42
        p = summarize_seed(seed)
        print(Path(p).read_text())
        print("SAVED", p)


if __name__ == "__main__":
    main()
