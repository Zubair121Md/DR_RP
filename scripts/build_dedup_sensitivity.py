#!/usr/bin/env python3
"""
Schedule-stratified APTOS deduplication sensitivity for the 24 primary MobileViT-S runs.

Removes the APTOS test images that have an exact duplicate in train or validation
(paper/tables/aptos_duplicate_pairs.csv) and recomputes, on the full and deduplicated
test sets: QWK, grade-1 recall, macro-F1, accuracy, referable-DR (grade >= 2)
sensitivity / specificity / AUC, and for the cross-entropy cells H1/H2 only ECE,
NLL and multiclass Brier score. Threshold-head cells get "n.i." for probability metrics.

Reads results/hybrid_<cell>_s<seed>_<v1|v3>/aptos_test_predictions.json.
Writes paper/tables/deduplication_sensitivity.csv (per run) and
paper/tables/deduplication_sensitivity_summary.csv (mean ± SD per cell, schedule group, subset).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.metrics import (  # noqa: E402
    accuracy,
    expected_calibration_error,
    macro_f1,
    mild_dr_recall,
    quadratic_weighted_kappa,
)
from build_paper_analysis_tables import binary_metrics, referable_prob  # noqa: E402

OUT = ROOT / "paper" / "tables"
CELLS = ["H1", "H2", "H3", "H4"]
SEEDS = [42, 43, 44, 45, 46, 47]
CE_CELLS = {"H1", "H2"}
NI = "n.i."
COLS = ["cell", "seed", "schedule_group", "evaluation_subset", "n_images", "qwk", "mild_recall",
        "macro_f1", "accuracy", "referable_sensitivity", "referable_specificity", "referable_auc",
        "ece", "nll", "brier"]
METRICS = COLS[5:]


def schedule_group(seed: int) -> str:
    return "42-44 truncated" if seed <= 44 else "45-47 full cosine"


def leaked_test_ids() -> set[str]:
    p = OUT / "aptos_duplicate_pairs.csv"
    print("READ", p.relative_to(ROOT))
    leak = set()
    with p.open() as f:
        for r in csv.DictReader(f):
            splits = {r["split_a"], r["split_b"]}
            if r["cross_split"] == "True" and "test" in splits and splits != {"test"}:
                leak.add(r["id_a"] if r["split_a"] == "test" else r["id_b"])
    return leak


def nll_brier(y: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    p = np.clip(probs, 1e-12, 1.0)
    nll = float(-np.mean(np.log(p[np.arange(len(y)), y])))
    onehot = np.eye(probs.shape[1])[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return nll, brier


def run_metrics(cell: str, pred: dict, idx: list[int]) -> dict:
    y = np.asarray(pred["y_true"], int)[idx]
    yhat = np.asarray(pred["y_pred"], int)[idx]
    sub = {k: [pred[k][i] for i in idx] for k in ("probs", "logits")}
    scores = referable_prob(sub, coral=cell not in CE_CELLS)
    bm = binary_metrics((y >= 2).astype(int), scores, (yhat >= 2).astype(int))
    row = {"n_images": len(idx),
           "qwk": quadratic_weighted_kappa(y, yhat, 5),
           "mild_recall": mild_dr_recall(y, yhat, 5),
           "macro_f1": macro_f1(y, yhat, 5),
           "accuracy": accuracy(y, yhat),
           "referable_sensitivity": bm["sensitivity"],
           "referable_specificity": bm["specificity"],
           "referable_auc": bm["auc"]}
    if cell in CE_CELLS:
        probs = np.asarray(sub["probs"], float)
        row["ece"] = expected_calibration_error(y, probs)
        row["nll"], row["brier"] = nll_brier(y, probs)
    else:
        row["ece"] = row["nll"] = row["brier"] = NI
    return row


def fmt(v) -> str:
    return v if isinstance(v, str) else f"{v:.6f}"


def main() -> None:
    leak = leaked_test_ids()
    print("test images removed:", len(leak))
    rows = []
    for cell in CELLS:
        for s in SEEDS:
            run = f"hybrid_{cell.lower()}_s{s}_{'v1' if s <= 44 else 'v3'}"
            pp = ROOT / "results" / run / "aptos_test_predictions.json"
            if not pp.is_file():
                raise SystemExit(f"MISSING input: {pp.relative_to(ROOT)}")
            print("READ", pp.relative_to(ROOT))
            pred = json.loads(pp.read_text())
            all_idx = list(range(len(pred["ids"])))
            keep_idx = [i for i, x in enumerate(pred["ids"]) if str(x) not in leak]
            for subset, idx in (("all test images", all_idx), ("deduplicated test images", keep_idx)):
                m = run_metrics(cell, pred, idx)
                rows.append({"cell": cell, "seed": s, "schedule_group": schedule_group(s),
                             "evaluation_subset": subset, **{k: fmt(v) if k != "n_images" else v
                                                            for k, v in m.items()}})

    with (OUT / "deduplication_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print("WROTE paper/tables/deduplication_sensitivity.csv", len(rows))

    summ = []
    for cell in CELLS:
        for grp in ("42-44 truncated", "45-47 full cosine"):
            for subset in ("all test images", "deduplicated test images"):
                rr = [r for r in rows if r["cell"] == cell and r["schedule_group"] == grp
                      and r["evaluation_subset"] == subset]
                out = {"cell": cell, "schedule_group": grp, "evaluation_subset": subset,
                       "n_images": rr[0]["n_images"]}
                for k in METRICS:
                    vals = [r[k] for r in rr]
                    out[k] = NI if NI in vals else f"{mean(map(float, vals)):.3f}±{stdev(map(float, vals)):.3f}"
                summ.append(out)
    with (OUT / "deduplication_sensitivity_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print("WROTE paper/tables/deduplication_sensitivity_summary.csv", len(summ))

    maxd = max(abs(float(a["qwk"]) - float(b["qwk"]))
               for a, b in zip(rows[0::2], rows[1::2]))
    print(f"max per-run |QWK change| = {maxd:.4f}")
    print("DEDUP_SENSITIVITY_OK")


if __name__ == "__main__":
    main()
