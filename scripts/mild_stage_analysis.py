#!/usr/bin/env python3
"""
Mild-stage analysis (research question / Section 9.2).

For each eval_report predictions file, report:
  - Grade-1 precision / recall
  - Confusions among grades 0–2
  - Normal↔Mild and Mild↔Moderate error counts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

DEFAULT_RUNS = [
    "baseline_efficientnet_b0_rgb_20260918_170311",
    "baseline_efficientnet_b0_green_clahe_20260918_092447",
    "baseline_efficientnet_b0_green_clahe_coral_20260918_203508",
    "baseline_efficientnet_b0_rgb_coral_20260919_063242",
]


def analyze_split(y_true, y_pred, num_classes: int = 5) -> dict:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1

    # Grade 1
    tp1 = int(cm[1, 1])
    fp1 = int(cm[:, 1].sum() - tp1)
    fn1 = int(cm[1, :].sum() - tp1)
    prec1 = tp1 / (tp1 + fp1) if (tp1 + fp1) else 0.0
    rec1 = tp1 / (tp1 + fn1) if (tp1 + fn1) else 0.0

    cm02 = cm[:3, :3].tolist()
    return {
        "n": int(len(yt)),
        "n_grade1": int((yt == 1).sum()),
        "grade1_precision": float(prec1),
        "grade1_recall": float(rec1),
        "confusion_0_2": cm02,
        "normal_to_mild": int(cm[0, 1]),
        "mild_to_normal": int(cm[1, 0]),
        "mild_to_moderate": int(cm[1, 2]),
        "moderate_to_mild": int(cm[2, 1]),
        "full_confusion": cm.tolist(),
    }


def analyze_run(run_id: str) -> dict:
    root = PROJECT / "results" / run_id
    out = {"run_id": run_id}
    cfg = root / "config.json"
    if cfg.is_file():
        c = json.loads(cfg.read_text())
        out["preprocess"] = c.get("preprocessing", {}).get("name")
        out["loss"] = c.get("loss", {}).get("name")
        out["head"] = c.get("model", {}).get("head", "ce")
        out["seed"] = c.get("experiment", {}).get("seed")
    for split in ("aptos_test", "idrid_external"):
        pred_path = root / f"{split}_predictions.json"
        if not pred_path.is_file():
            continue
        pred = json.loads(pred_path.read_text())
        out[split] = analyze_split(pred["y_true"], pred["y_pred"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument(
        "--out",
        type=str,
        default="results/mild_stage_analysis_exp1_4.json",
    )
    args = parser.parse_args()

    report = {
        "note": "Mild-stage (grade-1) analysis for preprocess × ordinal-head B0 2×2",
        "runs": [analyze_run(r) for r in args.runs if (PROJECT / "results" / r).is_dir()],
    }
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("SAVED", out)


if __name__ == "__main__":
    main()
