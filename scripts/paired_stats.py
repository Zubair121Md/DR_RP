#!/usr/bin/env python3
"""Phase M: paired statistical tests over seeds (n=3).

Comparisons (seed-paired):
  B0 vs matched hybrid theme (APTOS / IDRiD QWK)
  Within-hybrid contrasts (H2−H1, H4−H3, H4−H2, H2−H3, …)

With only 3 seeds, Wilcoxon two-sided p cannot go below ~0.25 when all
signs agree — we still report it, plus exact paired sign-flip permutation
p and the mean±std of paired deltas. Do NOT overclaim “significance.”
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HYB = json.loads((ROOT / "experiments/results/hybrid_seed_summary.json").read_text())
B0 = json.loads((ROOT / "results/b0_2x2_seed_summary.json").read_text())
OUT_JSON = ROOT / "experiments/results/paired_stats.json"
OUT_MD = ROOT / "experiments/results/paired_stats.md"
TAB = ROOT / "paper" / "tables"
SEEDS = [42, 43, 44]


def wilcoxon_exact_twosided(diffs: np.ndarray) -> dict:
    """Exact Wilcoxon signed-rank for small n (no zeros assumed after drop)."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": None, "p_two_sided": None, "note": "all_zero_diffs"}
    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # average ties
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    w = min(w_pos, w_neg)
    # exact null: all 2^n sign assignments
    total = 0
    extreme = 0
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        total += 1
        signed = ranks * np.array(signs)
        wp = signed[signed > 0].sum()
        wn = -signed[signed < 0].sum()
        w_null = min(wp, wn)
        if w_null <= w + 1e-12:
            extreme += 1
    p = extreme / total
    return {"n": n, "W": w, "W_pos": w_pos, "W_neg": w_neg, "p_two_sided": p}


def sign_flip_mean_p(diffs: np.ndarray) -> float:
    """Two-sided permutation p on |mean| under random sign flips."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return float("nan")
    obs = abs(d.mean())
    total = 0
    extreme = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(d)):
        total += 1
        if abs((d * np.array(signs)).mean()) >= obs - 1e-15:
            extreme += 1
    return extreme / total


def summarize_pair(name: str, a: list[float], b: list[float], a_label: str, b_label: str) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    diffs = a - b  # positive => a better than b for QWK
    wil = wilcoxon_exact_twosided(diffs)
    return {
        "comparison": name,
        "a": a_label,
        "b": b_label,
        "seeds": SEEDS,
        "a_values": a.tolist(),
        "b_values": b.tolist(),
        "deltas_a_minus_b": diffs.tolist(),
        "mean_delta": float(diffs.mean()),
        "std_delta": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "wilcoxon": wil,
        "sign_flip_p_mean": sign_flip_mean_p(diffs),
        "all_same_sign": bool(np.all(diffs > 0) or np.all(diffs < 0) or np.all(diffs == 0)),
    }


def hybrid_metric(cell: str, seed: int, split: str, metric: str = "quadratic_weighted_kappa") -> float:
    rid = f"hybrid_{cell.lower()}_s{seed}_v1"
    return float(HYB["per_seed"][rid][split][metric])


def b0_metric(cell_key: str, seed: int, which: str) -> float:
    """which: aptos_qwk | idrid_qwk"""
    runs = B0["cells"][cell_key]["runs"]
    for r in runs:
        if int(r["seed"]) == seed:
            return float(r[which])
    raise KeyError(f"B0 {cell_key} seed {seed}")


def main() -> None:
    tests: list[dict] = []

    # --- B0 vs hybrid (matched theme; mild-CORAL vs CORAL caveat) ---
    pairs_b0 = [
        ("B0 rgb_ce vs H1 APTOS QWK", "rgb_ce", "H1", "aptos_qwk", "aptos_test"),
        ("B0 rgb_ce vs H1 IDRiD QWK", "rgb_ce", "H1", "idrid_qwk", "idrid_external"),
        ("B0 green_clahe_ce vs H2 APTOS QWK", "green_clahe_ce", "H2", "aptos_qwk", "aptos_test"),
        ("B0 green_clahe_ce vs H2 IDRiD QWK", "green_clahe_ce", "H2", "idrid_qwk", "idrid_external"),
        ("B0 rgb_coral vs H3 APTOS QWK", "rgb_coral", "H3", "aptos_qwk", "aptos_test"),
        ("B0 rgb_coral vs H3 IDRiD QWK", "rgb_coral", "H3", "idrid_qwk", "idrid_external"),
        ("B0 green_clahe_coral vs H4 APTOS QWK", "green_clahe_coral", "H4", "aptos_qwk", "aptos_test"),
        ("B0 green_clahe_coral vs H4 IDRiD QWK", "green_clahe_coral", "H4", "idrid_qwk", "idrid_external"),
    ]
    for name, b0k, hc, b0m, hs in pairs_b0:
        a = [b0_metric(b0k, s, b0m) for s in SEEDS]
        b = [hybrid_metric(hc, s, hs) for s in SEEDS]
        tests.append(summarize_pair(name, a, b, f"B0:{b0k}", f"Hybrid:{hc}"))

    # --- Within hybrid ---
    within = [
        ("H2−H1 APTOS (CLAHE under CE)", "H2", "H1", "aptos_test"),
        ("H2−H1 IDRiD (CLAHE under CE)", "H2", "H1", "idrid_external"),
        ("H2−H1 DeepDRiD (CLAHE under CE)", "H2", "H1", "deepdrid_external"),
        ("H4−H3 APTOS (CLAHE under mildCORAL)", "H4", "H3", "aptos_test"),
        ("H4−H3 IDRiD (CLAHE under mildCORAL)", "H4", "H3", "idrid_external"),
        ("H4−H2 APTOS (mildCORAL under CLAHE)", "H4", "H2", "aptos_test"),
        ("H4−H2 IDRiD (mildCORAL under CLAHE)", "H4", "H2", "idrid_external"),
        ("H3−H1 APTOS (mildCORAL under RGB)", "H3", "H1", "aptos_test"),
        ("H2−H3 DeepDRiD", "H2", "H3", "deepdrid_external"),
        ("H2−H4 Messidor-2", "H2", "H4", "messidor2_external"),
    ]
    for name, ca, cb, split in within:
        a = [hybrid_metric(ca, s, split) for s in SEEDS]
        b = [hybrid_metric(cb, s, split) for s in SEEDS]
        tests.append(summarize_pair(name, a, b, ca, cb))

    # Mild / ECE contrasts on APTOS
    for metric, label in [("mild_dr_recall", "mild recall"), ("ece", "ECE")]:
        a = [hybrid_metric("H3", s, "aptos_test", metric) for s in SEEDS]
        b = [hybrid_metric("H1", s, "aptos_test", metric) for s in SEEDS]
        tests.append(summarize_pair(f"H3−H1 APTOS {label}", a, b, "H3", "H1"))
        a = [hybrid_metric("H4", s, "aptos_test", metric) for s in SEEDS]
        b = [hybrid_metric("H2", s, "aptos_test", metric) for s in SEEDS]
        tests.append(summarize_pair(f"H4−H2 APTOS {label}", a, b, "H4", "H2"))

    payload = {
        "n_seeds": 3,
        "seeds": SEEDS,
        "method": {
            "pairing": "same random seed",
            "wilcoxon": "exact signed-rank via full 2^n enumeration",
            "sign_flip_p": "exact two-sided permutation on |mean delta|",
            "caveat": (
                "n=3 → even if every seed favors the same side, two-sided p is typically 0.25. "
                "Report mean±std of paired deltas; do not claim p<0.05 from these tests alone. "
                "B0 coral ≠ hybrid mild-CORAL — interpret H3/H4 vs B0 coral cautiously."
            ),
        },
        "tests": tests,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # Markdown
    lines = [
        "# Paired statistics (Phase M)",
        "",
        f"Seeds: {SEEDS}. Exact Wilcoxon + sign-flip permutation on paired deltas.",
        "",
        f"**Caveat:** {payload['method']['caveat']}",
        "",
        "| Comparison | mean Δ (A−B) | std Δ | Wilcoxon p | sign-flip p | same-sign? |",
        "|------------|-------------:|------:|-----------:|------------:|:----------:|",
    ]
    for t in tests:
        p_w = t["wilcoxon"]["p_two_sided"]
        p_w_s = "—" if p_w is None else f"{p_w:.3f}"
        lines.append(
            f"| {t['comparison']} | {t['mean_delta']:+.4f} | {t['std_delta']:.4f} | "
            f"{p_w_s} | {t['sign_flip_p_mean']:.3f} | {'Y' if t['all_same_sign'] else 'N'} |"
        )
    lines += ["", f"JSON: `{OUT_JSON.relative_to(ROOT)}`", ""]
    OUT_MD.write_text("\n".join(lines))

    # Paper table (subset of headline tests)
    TAB.mkdir(parents=True, exist_ok=True)
    headline = [t for t in tests if any(k in t["comparison"] for k in (
        "B0 green_clahe_ce vs H2",
        "B0 green_clahe_coral vs H4",
        "H2−H1 APTOS",
        "H2−H1 DeepDRiD",
        "H4−H3 IDRiD",
        "H4−H2 APTOS",
        "H4−H2 IDRiD",
        "H3−H1 APTOS mild",
        "H4−H2 APTOS ECE",
    ))]
    headers = ["Comparison", "mean_delta", "std_delta", "wilcoxon_p", "sign_flip_p"]
    rows = []
    for t in headline:
        p_w = t["wilcoxon"]["p_two_sided"]
        rows.append([
            t["comparison"],
            f"{t['mean_delta']:+.4f}",
            f"{t['std_delta']:.4f}",
            "—" if p_w is None else f"{p_w:.3f}",
            f"{t['sign_flip_p_mean']:.3f}",
        ])
    with (TAB / "table_paired_stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    md = ["# Headline paired tests (n=3 seeds)", "", "| " + " | ".join(headers) + " |",
          "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md += ["", "Full report: `experiments/results/paired_stats.md`", ""]
    (TAB / "table_paired_stats.md").write_text("\n".join(md))

    print(OUT_MD.read_text())
    print("PAIRED_STATS_OK")


if __name__ == "__main__":
    main()
