#!/usr/bin/env python3
"""Phase J/M/Q prep: aggregate hybrid H1–H4 × seeds 42/43/44 into paper-ready tables.

Reads results/hybrid_h{1-4}_s{42,43,44}_v1/*_metrics.json (+ predictions for mild).
Writes:
  experiments/results/hybrid_seed_summary.json
  experiments/results/hybrid_seed_summary.md
  results/mild_stage_analysis_hybrid.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
CELLS = {
    "H1": "hybrid_h1_s{seed}_v1",
    "H2": "hybrid_h2_s{seed}_v1",
    "H3": "hybrid_h3_s{seed}_v1",
    "H4": "hybrid_h4_s{seed}_v1",
}
SPLITS = (
    "aptos_test",
    "idrid_external",
    "messidor2_external",
    "deepdrid_external",
)
METRICS = (
    "quadratic_weighted_kappa",
    "accuracy",
    "macro_f1",
    "mae",
    "mild_dr_f1",
    "mild_dr_recall",
    "ece",
    "ece_temperature_scaled",
    "temperature",
)


def mean_std(xs: list[float]) -> dict:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "values": [float(x) for x in arr],
    }


def load_metrics(run_id: str, split: str) -> dict | None:
    p = ROOT / "results" / run_id / f"{split}_metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def mild_from_preds(run_id: str, split: str) -> dict | None:
    p = ROOT / "results" / run_id / f"{split}_predictions.json"
    if not p.is_file():
        return None
    pred = json.loads(p.read_text())
    yt = np.asarray(pred["y_true"], dtype=int)
    yp = np.asarray(pred["y_pred"], dtype=int)
    cm = np.zeros((5, 5), dtype=int)
    for t, pr in zip(yt, yp):
        cm[t, pr] += 1
    tp1 = int(cm[1, 1])
    fp1 = int(cm[:, 1].sum() - tp1)
    fn1 = int(cm[1, :].sum() - tp1)
    prec = tp1 / (tp1 + fp1) if (tp1 + fp1) else 0.0
    rec = tp1 / (tp1 + fn1) if (tp1 + fn1) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "grade1_precision": prec,
        "grade1_recall": rec,
        "grade1_f1": f1,
        "conf_0_1": int(cm[0, 1] + cm[1, 0]),
        "conf_1_2": int(cm[1, 2] + cm[2, 1]),
        "cm_0_2": cm[:3, :3].tolist(),
    }


def main() -> None:
    per_seed: dict = {}
    aggregate: dict = {}
    mild_runs: list = []
    missing: list[str] = []

    for cell, tmpl in CELLS.items():
        aggregate[cell] = {}
        for split in SPLITS:
            bucket = {m: [] for m in METRICS}
            for seed in SEEDS:
                run_id = tmpl.format(seed=seed)
                per_seed.setdefault(run_id, {})
                m = load_metrics(run_id, split)
                if m is None:
                    missing.append(f"{run_id}/{split}")
                    continue
                row = {k: m.get(k) for k in METRICS}
                per_seed[run_id][split] = row
                for k in METRICS:
                    if isinstance(m.get(k), (int, float)) and not (
                        isinstance(m[k], float) and math.isnan(m[k])
                    ):
                        bucket[k].append(float(m[k]))
                mild = mild_from_preds(run_id, split)
                if mild:
                    mild_runs.append({"run_id": run_id, "cell": cell, "seed": seed, "split": split, **mild})
            aggregate[cell][split] = {k: mean_std(v) for k, v in bucket.items()}

    out_json = {
        "protocol": "frozen_protocol_v1",
        "architecture": "mobilevit_s",
        "cells": {
            "H1": "rgb × CE",
            "H2": "green_clahe × CE",
            "H3": "rgb × mild_CORAL",
            "H4": "green_clahe × mild_CORAL",
        },
        "seeds": list(SEEDS),
        "missing": missing,
        "per_seed": per_seed,
        "mean_std": aggregate,
        "notes": [
            "Selection was APTOS val QWK; these are held-out / external numbers.",
            "Messidor-2 near-zero QWK is a documented failure mode — do not average away.",
            "Do not cite historical *_stable batch=1 MV runs.",
        ],
    }
    jpath = ROOT / "experiments" / "results" / "hybrid_seed_summary.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(out_json, indent=2))

    # Markdown table (QWK mean±std)
    lines = [
        "# Hybrid 2×2 — seed mean ± std (n=3: 42/43/44)",
        "",
        "Metric: quadratic weighted kappa. Mean ± sample std.",
        "",
        "| Cell | APTOS test | IDRiD | Messidor-2 | DeepDRiD |",
        "|------|-----------:|------:|-----------:|---------:|",
    ]
    for cell in CELLS:
        row = [cell]
        for split in SPLITS:
            st = aggregate[cell][split]["quadratic_weighted_kappa"]
            if st["mean"] is None:
                row.append("—")
            else:
                row.append(f"{st['mean']:.3f}±{st['std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Mild-DR F1 (APTOS test, mean±std)",
        "",
        "| Cell | mild_dr_f1 | mild_dr_recall | ECE | ECE@T |",
        "|------|-----------:|---------------:|----:|------:|",
    ]
    for cell in CELLS:
        a = aggregate[cell]["aptos_test"]
        def fmt(k):
            st = a[k]
            return "—" if st["mean"] is None else f"{st['mean']:.3f}±{st['std']:.3f}"
        lines.append(
            f"| {cell} | {fmt('mild_dr_f1')} | {fmt('mild_dr_recall')} | {fmt('ece')} | {fmt('ece_temperature_scaled')} |"
        )
    if missing:
        lines += ["", "## Missing", ""] + [f"- `{m}`" for m in missing]
    lines += ["", f"JSON: `{jpath.relative_to(ROOT)}`", ""]
    mdpath = ROOT / "experiments" / "results" / "hybrid_seed_summary.md"
    mdpath.write_text("\n".join(lines))

    mild_path = ROOT / "results" / "mild_stage_analysis_hybrid.json"
    mild_path.write_text(
        json.dumps(
            {
                "note": "Mild-stage analysis for hybrid H1–H4 × seeds 42/43/44",
                "n_entries": len(mild_runs),
                "runs": mild_runs,
            },
            indent=2,
        )
    )

    print(mdpath.read_text())
    print("SAVED", jpath)
    print("SAVED", mdpath)
    print("SAVED", mild_path)
    if missing:
        print("MISSING", len(missing))
        raise SystemExit(1)
    print("HYBRID_ANALYSIS_OK")


if __name__ == "__main__":
    main()
