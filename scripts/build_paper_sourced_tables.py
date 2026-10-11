#!/usr/bin/env python3
"""
Rebuild paper/tables summaries from archived metrics/predictions/efficiency JSON only.
Never hand-types numbers. Prints every source path read.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)
CELLS = ("H1", "H2", "H3", "H4")
DOMAINS = (
    ("aptos_test", "APTOS"),
    ("idrid_external", "IDRiD"),
    ("messidor2_external", "Messidor-2"),
    ("deepdrid_external", "DeepDRiD"),
)
SOURCES: list[str] = []


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"MISSING required input: {path.relative_to(ROOT)}")
    SOURCES.append(str(path.relative_to(ROOT)))
    print("READ", path.relative_to(ROOT), flush=True)
    return json.loads(path.read_text())


def rid(cell: str, seed: int) -> str:
    return f"hybrid_{cell.lower()}_s{seed}_v1"


def mean_std(xs: list[float]) -> str:
    a = np.asarray(xs, dtype=float)
    # Sample std over seeds (ddof=1), matching prior paper tables
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("WROTE", path.relative_to(ROOT), flush=True)


def write_md(path: Path, title: str, header: list[str], rows: list[list]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    path.write_text("\n".join(lines) + "\n")
    print("WROTE", path.relative_to(ROOT), flush=True)


def build_hybrid_qwk() -> None:
    header = ["Cell", "Setup", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
    setups = {
        "H1": "rgb × CE",
        "H2": "green_clahe × CE",
        "H3": "rgb × mild_CORAL",
        "H4": "green_clahe × mild_CORAL",
    }
    rows = []
    long_rows = []
    for cell in CELLS:
        vals = {}
        for split, label in DOMAINS:
            xs = []
            for seed in SEEDS:
                m = read_json(ROOT / "results" / rid(cell, seed) / f"{split}_metrics.json")
                q = float(m["quadratic_weighted_kappa"])
                xs.append(q)
                long_rows.append([cell, seed, label, f"{q:.6f}", str((ROOT / "results" / rid(cell, seed) / f"{split}_metrics.json").relative_to(ROOT))])
            vals[label] = mean_std(xs)
        rows.append([cell, setups[cell], vals["APTOS"], vals["IDRiD"], vals["Messidor-2"], vals["DeepDRiD"]])
    write_csv(OUT / "table_hybrid_qwk.csv", header, rows)
    write_md(OUT / "table_hybrid_qwk.md", "MobileViT-S hybrid QWK (mean±std, seeds 42/43/44)", header, rows)
    write_csv(OUT / "table_hybrid_qwk_per_seed.csv", ["cell", "seed", "domain", "qwk", "source"], long_rows)


def build_mild_calibration() -> None:
    header = ["Cell", "mild F1", "mild recall", "ECE", "ECE@T"]
    rows = []
    long_rows = []
    for cell in CELLS:
        f1s, recs, eces, ece_ts = [], [], [], []
        for seed in SEEDS:
            m = read_json(ROOT / "results" / rid(cell, seed) / "aptos_test_metrics.json")
            f1s.append(float(m["mild_dr_f1"]))
            recs.append(float(m["mild_dr_recall"]))
            eces.append(float(m["ece"]))
            ece_ts.append(float(m.get("ece_temperature_scaled", m.get("ece_at_t", float("nan")))))
            long_rows.append([cell, seed, m["mild_dr_f1"], m["mild_dr_recall"], m["ece"], m.get("ece_temperature_scaled")])
        rows.append([cell, mean_std(f1s), mean_std(recs), mean_std(eces), mean_std(ece_ts)])
    write_csv(OUT / "table_mild_calibration.csv", header, rows)
    write_md(OUT / "table_mild_calibration.md", "APTOS test mild metrics and ECE", header, rows)
    write_csv(OUT / "table_mild_calibration_per_seed.csv", ["cell", "seed", "mild_f1", "mild_recall", "ece", "ece_t"], long_rows)


def build_ece_domains() -> None:
    header = ["Cell", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
    rows = []
    for cell in CELLS:
        vals = []
        for split, _ in DOMAINS:
            xs = []
            for seed in SEEDS:
                m = read_json(ROOT / "results" / rid(cell, seed) / f"{split}_metrics.json")
                xs.append(float(m["ece"]))
            vals.append(mean_std(xs))
        rows.append([cell, *vals])
    write_csv(OUT / "table_ece_by_domain.csv", header, rows)
    write_md(OUT / "table_ece_by_domain.md", "ECE by domain (mean±std)", header, rows)


def build_efficiency() -> None:
    path = ROOT / "results" / "efficiency_models.json"
    data = read_json(path)
    header = ["backbone", "head", "params_m", "flops_g", "device", "batch1_ms_mean", "reps", "note"]
    rows = []
    for r in data["rows"]:
        rows.append([
            r["backbone"],
            r["head"],
            f"{float(r['params_m']):.3f}",
            f"{float(r['flops_g']):.3f}",
            r["device"],
            f"{float(r['batch1_ms_mean']):.3f}",
            int(r["reps"]),
            "from efficiency_models.json",
        ])
    # Swin not in file — explicit missing row
    rows.append(["swin_tiny", "ce", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "not measured in efficiency_models.json"])
    write_csv(OUT / "table_efficiency.csv", header, rows)
    write_md(OUT / "table_efficiency.md", "Inference-cost snapshot (params / FLOPs / batch-1 latency)", header, rows)


def build_messidor_sanity() -> None:
    header = [
        "cell", "seed", "n", "qwk",
        "true_n0", "true_n1", "true_n2", "true_n3", "true_n4",
        "pred_n0", "pred_n1", "pred_n2", "pred_n3", "pred_n4",
        "pred_frac_grade0", "source_metrics", "source_preds",
    ]
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            mpath = ROOT / "results" / rid(cell, seed) / "messidor2_external_metrics.json"
            ppath = ROOT / "results" / rid(cell, seed) / "messidor2_external_predictions.json"
            m = read_json(mpath)
            p = read_json(ppath)
            yt = [int(x) for x in p["y_true"]]
            yp = [int(x) for x in p["y_pred"]]
            assert len(yt) == len(yp)
            ct, cp = Counter(yt), Counter(yp)
            rows.append([
                cell, seed, len(yt), f"{float(m['quadratic_weighted_kappa']):.6f}",
                ct[0], ct[1], ct[2], ct[3], ct[4],
                cp[0], cp[1], cp[2], cp[3], cp[4],
                f"{cp[0] / len(yp):.6f}",
                str(mpath.relative_to(ROOT)),
                str(ppath.relative_to(ROOT)),
            ])
    write_csv(OUT / "table_messidor2_sanity.csv", header, rows)
    # compact markdown for paper
    md_rows = []
    for cell in CELLS:
        sub = [r for r in rows if r[0] == cell]
        pred0 = [float(r[14]) for r in sub]
        qwks = [float(r[3]) for r in sub]
        md_rows.append([
            cell,
            mean_std(qwks),
            mean_std(pred0),
            int(sub[0][4]), int(sub[0][5]), int(sub[0][6]), int(sub[0][7]), int(sub[0][8]),
        ])
    write_md(
        OUT / "table_messidor2_sanity.md",
        "Messidor-2 sanity (label counts fixed; pred grade-0 fraction mean±std)",
        ["Cell", "QWK", "pred_frac_grade0", "true_n0", "true_n1", "true_n2", "true_n3", "true_n4"],
        md_rows,
    )


def build_lodo_per_seed() -> None:
    header = ["holdout", "seed", "lodo_qwk", "h2_qwk", "delta", "lodo_source", "h2_source"]
    rows = []
    hold_map = {
        "IDRiD": ("idrid", "idrid_external"),
        "Messidor-2": ("messidor2", "messidor2_external"),
        "DeepDRiD": ("deepdrid", "deepdrid_external"),
    }
    for label, (tag, split) in hold_map.items():
        for seed in SEEDS:
            lodo_path = ROOT / "results" / f"phase_p_lodo_holdout_{tag}_s{seed}_v1" / f"{split}_metrics.json"
            h2_path = ROOT / "results" / rid("H2", seed) / f"{split}_metrics.json"
            lq = float(read_json(lodo_path)["quadratic_weighted_kappa"])
            hq = float(read_json(h2_path)["quadratic_weighted_kappa"])
            rows.append([label, seed, f"{lq:.6f}", f"{hq:.6f}", f"{lq - hq:.6f}",
                         str(lodo_path.relative_to(ROOT)), str(h2_path.relative_to(ROOT))])
    write_csv(OUT / "table_lodo_vs_h2_per_seed.csv", header, rows)
    # rebuild mean±std table
    mean_rows = []
    for label in hold_map:
        sub = [r for r in rows if r[0] == label]
        lqs = [float(r[2]) for r in sub]
        hqs = [float(r[3]) for r in sub]
        dqs = [float(r[4]) for r in sub]
        mean_rows.append([label, mean_std(lqs), mean_std(hqs), mean_std(dqs)])
    write_csv(OUT / "table_lodo_vs_h2.csv", ["Holdout", "LODO QWK", "H2 APTOS-only QWK", "Δ (LODO−H2)"], mean_rows)
    write_md(OUT / "table_lodo_vs_h2.md", "LODO vs H2 held-out QWK",
             ["Holdout", "LODO QWK", "H2 APTOS-only QWK", "Δ (LODO−H2)"], mean_rows)


def build_aptos_metrics_per_seed() -> None:
    keys = ["accuracy", "macro_f1", "quadratic_weighted_kappa", "mild_dr_precision", "mild_dr_recall", "mild_dr_f1"]
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            path = ROOT / "results" / rid(cell, seed) / "aptos_test_metrics.json"
            m = read_json(path)
            rows.append([cell, seed, *[f"{float(m[k]):.6f}" for k in keys], str(path.relative_to(ROOT))])
    write_csv(OUT / "table_aptos_metrics_per_seed.csv", ["cell", "seed", *keys, "source"], rows)


def build_val_history() -> None:
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            path = ROOT / "results" / rid(cell, seed) / "history.json"
            hist = read_json(path)
            for h in hist:
                rows.append([cell, seed, int(h["epoch"]), f"{float(h['val_qwk']):.6f}",
                             f"{float(h['train_loss']):.6f}", f"{float(h['val_loss']):.6f}",
                             str(path.relative_to(ROOT))])
    write_csv(OUT / "table_val_history_per_seed.csv",
              ["cell", "seed", "epoch", "val_qwk", "train_loss", "val_loss", "source"], rows)


def build_confusion_pooled() -> None:
    rows = []
    for cell in CELLS:
        for split, label in DOMAINS:
            total = np.zeros((5, 5), dtype=int)
            for seed in SEEDS:
                path = ROOT / "results" / rid(cell, seed) / f"{split}_metrics.json"
                total += np.asarray(read_json(path)["confusion_matrix"], dtype=int)
            for i in range(5):
                for j in range(5):
                    rows.append([cell, label, i, j, int(total[i, j])])
    write_csv(OUT / "table_confusion_pooled_seeds.csv", ["cell", "domain", "true", "pred", "count"], rows)


def main() -> None:
    build_aptos_metrics_per_seed()
    build_val_history()
    build_confusion_pooled()
    build_hybrid_qwk()
    build_mild_calibration()
    build_ece_domains()
    build_efficiency()
    build_messidor_sanity()
    build_lodo_per_seed()
    src = OUT / "table_finish_source_files.txt"
    src.write_text("\n".join(dict.fromkeys(SOURCES)) + "\n")
    print("WROTE", src.relative_to(ROOT), "n_unique_sources", len(set(SOURCES)))
    print("BUILD_PAPER_SOURCED_TABLES_OK")


if __name__ == "__main__":
    main()
