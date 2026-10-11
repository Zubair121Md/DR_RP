#!/usr/bin/env python3
"""
Build paper/tables/* from hybrid v1 (seeds 42–44) + paper_v3 runs.

Never hand-types numbers. Prints every source path read.
Does not overwrite checkpoint/result dirs.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = (
    ("aptos_test", "APTOS"),
    ("idrid_external", "IDRiD"),
    ("messidor2_external", "Messidor-2"),
    ("deepdrid_external", "DeepDRiD"),
)
CELLS = ("H1", "H2", "H3", "H4")
SETUPS = {
    "H1": "RGB × CE",
    "H2": "CLAHE × CE",
    "H3": "RGB × MW threshold",
    "H4": "CLAHE × MW threshold",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"MISSING {path.relative_to(ROOT)}")
    print("READ", path.relative_to(ROOT), flush=True)
    return json.loads(path.read_text())


def mean_sd(xs: list[float], d: int = 3) -> str:
    if len(xs) == 1:
        return f"{xs[0]:.{d}f}"
    return f"{st.mean(xs):.{d}f}±{st.stdev(xs):.{d}f}"


def signed_mean_sd(xs: list[float], d: int = 3) -> str:
    m, s = st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:+.{d}f}±{s:.{d}f}"


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    path = OUT / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("WROTE", path.relative_to(ROOT), "rows", len(rows), flush=True)


def hybrid_run(cell: str, seed: int) -> str:
    if seed in (42, 43, 44):
        return f"hybrid_{cell.lower()}_s{seed}_v1"
    if seed in (45, 46, 47):
        return f"hybrid_{cell.lower()}_s{seed}_v3"
    raise ValueError(seed)


def met(run: str, split: str) -> dict:
    return read_json(ROOT / "results" / run / f"{split}_metrics.json")


def build_hybrid_n6() -> None:
    """Primary hybrid QWK / mild / ECE with seeds 42–47."""
    qwk_rows, qwk_long = [], []
    mild_rows, mild_long = [], []
    ece_rows = []
    for cell in CELLS:
        qwk_vals = {lab: [] for _, lab in DOMAINS}
        mild = {k: [] for k in ("mild_dr_f1", "mild_dr_recall", "mild_dr_precision", "ece", "ece_temperature_scaled", "accuracy", "macro_f1")}
        ece_dom = {lab: [] for _, lab in DOMAINS}
        for seed in (42, 43, 44, 45, 46, 47):
            run = hybrid_run(cell, seed)
            for split, lab in DOMAINS:
                m = met(run, split)
                q = float(m["quadratic_weighted_kappa"])
                qwk_vals[lab].append(q)
                ece_dom[lab].append(float(m["ece"]))
                qwk_long.append([cell, seed, lab, f"{q:.6f}", f"results/{run}/{split}_metrics.json"])
            m0 = met(run, "aptos_test")
            for k in mild:
                mild[k].append(float(m0[k]))
                mild_long.append([cell, seed, k, f"{float(m0[k]):.6f}", f"results/{run}/aptos_test_metrics.json"])
        qwk_rows.append([cell, SETUPS[cell]] + [mean_sd(qwk_vals[lab]) for _, lab in DOMAINS])
        mild_rows.append([
            cell,
            mean_sd(mild["mild_dr_precision"]),
            mean_sd(mild["mild_dr_recall"]),
            mean_sd(mild["mild_dr_f1"]),
            mean_sd(mild["macro_f1"]),
            mean_sd(mild["accuracy"]),
            mean_sd(mild["ece"]),
            mean_sd(mild["ece_temperature_scaled"]),
        ])
        ece_rows.append([cell] + [mean_sd(ece_dom[lab]) for _, lab in DOMAINS])
    write_csv("table_v3_hybrid_qwk_n6.csv", ["Cell", "Setup", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD"], qwk_rows)
    write_csv("table_v3_hybrid_qwk_n6_per_seed.csv", ["cell", "seed", "domain", "qwk", "source"], qwk_long)
    write_csv(
        "table_v3_hybrid_mild_n6.csv",
        ["Cell", "Mild precision", "Mild recall", "Mild F1", "Macro-F1", "Accuracy", "ECE", "ECE@T"],
        mild_rows,
    )
    write_csv("table_v3_hybrid_ece_n6.csv", ["Cell", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD"], ece_rows)
    write_csv("table_v3_hybrid_mild_n6_long.csv", ["cell", "seed", "metric", "value", "source"], mild_long)


def build_matched_b0() -> None:
    rows, long_rows, paired = [], [], []
    for cell in CELLS:
        b0_vals = {lab: [] for _, lab in DOMAINS}
        hy_vals = {lab: [] for _, lab in DOMAINS}
        for seed in (42, 43, 44):
            b0 = f"matched_b0_{cell.lower()}_s{seed}_v3"
            hy = f"hybrid_{cell.lower()}_s{seed}_v1"
            for split, lab in DOMAINS:
                qb = float(met(b0, split)["quadratic_weighted_kappa"])
                qh = float(met(hy, split)["quadratic_weighted_kappa"])
                b0_vals[lab].append(qb)
                hy_vals[lab].append(qh)
                long_rows.append([cell, seed, lab, "matched_b0", f"{qb:.6f}", f"results/{b0}/{split}_metrics.json"])
                long_rows.append([cell, seed, lab, "hybrid", f"{qh:.6f}", f"results/{hy}/{split}_metrics.json"])
        rows.append(
            [cell, SETUPS[cell]]
            + [mean_sd(b0_vals[lab]) for _, lab in DOMAINS]
            + [mean_sd(hy_vals[lab]) for _, lab in DOMAINS]
        )
        for _, lab in DOMAINS:
            deltas = [a - b for a, b in zip(b0_vals[lab], hy_vals[lab])]
            paired.append([cell, lab, signed_mean_sd(deltas), sum(d > 0 for d in deltas), 3])
    write_csv(
        "table_v3_matched_b0_vs_hybrid.csv",
        ["Cell", "Setup", "B0 APTOS", "B0 IDRiD", "B0 Messidor-2", "B0 DeepDRiD",
         "Hybrid APTOS", "Hybrid IDRiD", "Hybrid Messidor-2", "Hybrid DeepDRiD"],
        rows,
    )
    write_csv(
        "table_v3_matched_b0_vs_hybrid_per_seed.csv",
        ["cell", "seed", "domain", "model", "qwk", "source"],
        long_rows,
    )
    write_csv(
        "table_v3_matched_b0_paired_delta.csv",
        ["Cell", "Domain", "Δ QWK (B0−hybrid)", "n_pos", "n_seeds"],
        paired,
    )


def build_decomp() -> None:
    """CLAHE decomposition: H2, D1 (unweighted CORAL), D2 (mild-weighted CE), H4."""
    specs = [
        ("H2", "CLAHE × CE", "hybrid_h2_s{s}_v1"),
        ("D1", "CLAHE × unweighted threshold", "hybrid_d1_coral_clahe_s{s}_v3"),
        ("D2", "CLAHE × mild-weighted CE", "hybrid_d2_mildce_clahe_s{s}_v3"),
        ("H4", "CLAHE × MW threshold", "hybrid_h4_s{s}_v1"),
    ]
    rows, long_rows = [], []
    for cell, setup, tmpl in specs:
        vals = {lab: {"qwk": [], "mild": [], "ece": []} for _, lab in DOMAINS}
        for seed in (42, 43, 44):
            run = tmpl.format(s=seed)
            for split, lab in DOMAINS:
                m = met(run, split)
                q, mr, e = float(m["quadratic_weighted_kappa"]), float(m["mild_dr_recall"]), float(m["ece"])
                vals[lab]["qwk"].append(q)
                vals[lab]["mild"].append(mr)
                vals[lab]["ece"].append(e)
                long_rows.append([cell, seed, lab, f"{q:.6f}", f"{mr:.6f}", f"{e:.6f}", f"results/{run}/{split}_metrics.json"])
        row = [cell, setup]
        for _, lab in DOMAINS:
            row += [mean_sd(vals[lab]["qwk"]), mean_sd(vals[lab]["mild"]), mean_sd(vals[lab]["ece"])]
        rows.append(row)
    header = ["Cell", "Setup"]
    for _, lab in DOMAINS:
        header += [f"{lab} QWK", f"{lab} mild recall", f"{lab} ECE"]
    write_csv("table_v3_decomp_clahe.csv", header, rows)
    write_csv(
        "table_v3_decomp_clahe_per_seed.csv",
        ["cell", "seed", "domain", "qwk", "mild_recall", "ece", "source"],
        long_rows,
    )


def build_res448() -> None:
    rows, long_rows, paired = [], [], []
    for cell in ("H1", "H2"):
        v224 = {lab: [] for _, lab in DOMAINS}
        v448 = {lab: [] for _, lab in DOMAINS}
        mild224, mild448 = [], []
        for seed in (42, 43, 44):
            r224 = f"hybrid_{cell.lower()}_s{seed}_v1"
            r448 = f"hybrid_{cell.lower()}_448_s{seed}_v3"
            for split, lab in DOMAINS:
                q2 = float(met(r224, split)["quadratic_weighted_kappa"])
                q4 = float(met(r448, split)["quadratic_weighted_kappa"])
                v224[lab].append(q2)
                v448[lab].append(q4)
                long_rows.append([cell, seed, lab, 224, f"{q2:.6f}", f"results/{r224}/{split}_metrics.json"])
                long_rows.append([cell, seed, lab, 448, f"{q4:.6f}", f"results/{r448}/{split}_metrics.json"])
            mild224.append(float(met(r224, "aptos_test")["mild_dr_recall"]))
            mild448.append(float(met(r448, "aptos_test")["mild_dr_recall"]))
        rows.append(
            [cell, SETUPS[cell]]
            + [mean_sd(v224[lab]) for _, lab in DOMAINS]
            + [mean_sd(v448[lab]) for _, lab in DOMAINS]
            + [mean_sd(mild224), mean_sd(mild448)]
        )
        for _, lab in DOMAINS:
            deltas = [a - b for a, b in zip(v448[lab], v224[lab])]
            paired.append([cell, lab, signed_mean_sd(deltas), sum(d > 0 for d in deltas), 3])
    write_csv(
        "table_v3_res448_vs_224.csv",
        ["Cell", "Setup",
         "224 APTOS", "224 IDRiD", "224 Messidor-2", "224 DeepDRiD",
         "448 APTOS", "448 IDRiD", "448 Messidor-2", "448 DeepDRiD",
         "224 mild recall", "448 mild recall"],
        rows,
    )
    write_csv(
        "table_v3_res448_per_seed.csv",
        ["cell", "seed", "domain", "image_size", "qwk", "source"],
        long_rows,
    )
    write_csv(
        "table_v3_res448_paired_delta.csv",
        ["Cell", "Domain", "Δ QWK (448−224)", "n_pos", "n_seeds"],
        paired,
    )


def build_messidor_pred0_448() -> None:
    rows = []
    for cell in ("H1", "H2"):
        for size, tmpl in ((224, "hybrid_{c}_s{s}_v1"), (448, "hybrid_{c}_448_s{s}_v3")):
            fracs = []
            for seed in (42, 43, 44):
                run = tmpl.format(c=cell.lower(), s=seed)
                p = read_json(ROOT / "results" / run / "messidor2_external_predictions.json")
                pred = p.get("y_pred", p.get("predictions"))
                if pred and isinstance(pred[0], dict):
                    pred = [x.get("pred", x.get("y_pred")) for x in pred]
                fracs.append(sum(int(x) == 0 for x in pred) / len(pred))
                rows.append([cell, size, seed, f"{fracs[-1]:.6f}", len(pred), f"results/{run}/messidor2_external_predictions.json"])
    # also summary
    summary = []
    by = {}
    for cell, size, seed, frac, n, src in rows:
        by.setdefault((cell, size), []).append(float(frac))
    for (cell, size), xs in sorted(by.items()):
        summary.append([cell, size, mean_sd(xs), 3])
    write_csv("table_v3_messidor_pred0_res.csv", ["cell", "image_size", "seed", "pred_frac_grade0", "n", "source"], rows)
    write_csv("table_v3_messidor_pred0_res_summary.csv", ["cell", "image_size", "pred_frac_grade0", "n_seeds"], summary)


def main() -> None:
    build_hybrid_n6()
    build_matched_b0()
    build_decomp()
    build_res448()
    build_messidor_pred0_448()
    print("BUILD_PAPER_V3_TABLES_OK")


if __name__ == "__main__":
    main()
