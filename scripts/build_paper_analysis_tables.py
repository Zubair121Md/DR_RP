#!/usr/bin/env python3
"""
Build paper analysis tables A–E from archived hybrid predictions/metrics only.

Reads: results/hybrid_h{1-4}_s{42,43,44}_v1/{split}_{metrics,predictions}.json
Writes: paper/tables/*.csv|.md|.tex
Prints every source path read. Never invents missing values.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    expected_calibration_error,
    quadratic_weighted_kappa,
    temperature_scale_probs,
)

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44)
CELLS = ("H1", "H2", "H3", "H4")
SPLITS = (
    ("aptos_test", "APTOS"),
    ("idrid_external", "IDRiD"),
    ("messidor2_external", "Messidor-2"),
    ("deepdrid_external", "DeepDRiD"),
)
CONTRASTS = (
    ("CLAHE_under_CE", "H2", "H1"),
    ("CLAHE_under_CORAL", "H4", "H3"),
    ("loss_under_RGB", "H3", "H1"),
    ("loss_under_CLAHE", "H4", "H2"),
)
METRIC_KEYS = (
    "quadratic_weighted_kappa",
    "accuracy",
    "macro_f1",
    "mae",
    "mild_dr_recall",
    "mild_dr_f1",
    "ece",
    "ece_temperature_scaled",
)
BOOT_N = 2000
BOOT_RNG = 42
SOURCES: list[str] = []


def rid(cell: str, seed: int) -> str:
    return f"hybrid_{cell.lower()}_s{seed}_v1"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"MISSING required input: {path}")
    SOURCES.append(str(path.relative_to(ROOT)))
    return json.loads(path.read_text())


def load_metrics(cell: str, seed: int, split: str) -> dict:
    return read_json(ROOT / "results" / rid(cell, seed) / f"{split}_metrics.json")


def load_pred(cell: str, seed: int, split: str) -> dict:
    return read_json(ROOT / "results" / rid(cell, seed) / f"{split}_predictions.json")


def is_coral(cell: str) -> bool:
    return cell in ("H3", "H4")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty table: {path}")
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("WROTE", path.relative_to(ROOT))


def write_md(path: Path, title: str, rows: list[dict], cols: list[str] | None = None) -> None:
    cols = cols or list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    path.write_text("\n".join(lines))
    print("WROTE", path.relative_to(ROOT))


def write_tex(path: Path, caption: str, label: str, rows: list[dict], cols: list[str]) -> None:
    lines = [
        "% Auto-generated — do not hand-edit",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        safe = [str(r[c]).replace("%", r"\%").replace("±", r"$\pm$") for c in cols]
        lines.append(" & ".join(safe) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))
    print("WROTE", path.relative_to(ROOT))


def fmt_ms(vals: list[float]) -> str:
    a = np.asarray(vals, float)
    if a.size == 0:
        raise RuntimeError("empty values for mean±std")
    if a.size == 1:
        return f"{a[0]:.6f}"
    return f"{a.mean():.6f}±{a.std(ddof=1):.6f}"


# ---------------------------------------------------------------------------
# A. Pairwise contrasts
# ---------------------------------------------------------------------------
def table_A() -> None:
    rows = []
    for contrast, a_cell, b_cell in CONTRASTS:
        for split, split_lab in SPLITS:
            for metric in METRIC_KEYS:
                deltas = []
                per = {}
                for s in SEEDS:
                    va = load_metrics(a_cell, s, split)[metric]
                    vb = load_metrics(b_cell, s, split)[metric]
                    if va is None or vb is None:
                        raise RuntimeError(f"NULL metric {metric} {a_cell}/{b_cell} s{s} {split}")
                    d = float(va) - float(vb)
                    deltas.append(d)
                    per[f"delta_s{s}"] = f"{d:.6f}"
                    per[f"{a_cell}_s{s}"] = f"{float(va):.6f}"
                    per[f"{b_cell}_s{s}"] = f"{float(vb):.6f}"
                signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
                rows.append(
                    {
                        "contrast": contrast,
                        "cell_A": a_cell,
                        "cell_B": b_cell,
                        "domain": split_lab,
                        "split_key": split,
                        "metric": metric,
                        "mean_delta": f"{np.mean(deltas):.6f}",
                        "std_delta": f"{np.std(deltas, ddof=1):.6f}",
                        "mean_pm_std": fmt_ms(deltas),
                        "n_pos": sum(1 for x in signs if x > 0),
                        "n_neg": sum(1 for x in signs if x < 0),
                        "n_zero": sum(1 for x in signs if x == 0),
                        **per,
                    }
                )
    write_csv(OUT / "table_A_pairwise_contrasts.csv", rows)
    # compact md: mean±std only
    compact = [
        {
            "contrast": r["contrast"],
            "domain": r["domain"],
            "metric": r["metric"],
            "delta_mean_pm_std": r["mean_pm_std"],
            "signs_+/-/0": f"{r['n_pos']}/{r['n_neg']}/{r['n_zero']}",
        }
        for r in rows
    ]
    write_md(OUT / "table_A_pairwise_contrasts.md", "A. Pairwise hybrid contrasts (A−B)", compact)
    write_tex(
        OUT / "table_A_pairwise_contrasts.tex",
        "Pairwise hybrid contrasts (mean$\\pm$std of seed deltas).",
        "tab:A_contrasts",
        compact,
        list(compact[0].keys()),
    )


# ---------------------------------------------------------------------------
# B. Referable-DR metrics
# ---------------------------------------------------------------------------
def referable_prob(pred: dict, coral: bool) -> np.ndarray:
    logits = np.asarray(pred["logits"], dtype=float)
    probs = np.asarray(pred["probs"], dtype=float)
    if coral:
        if logits.shape[1] != 4:
            raise RuntimeError(f"CORAL logits expected width 4, got {logits.shape}")
        # P(y>1) = sigmoid(z_1); z index 1 is threshold y>1
        return 1.0 / (1.0 + np.exp(-logits[:, 1]))
    if probs.shape[1] != 5:
        raise RuntimeError(f"CE probs expected width 5, got {probs.shape}")
    return probs[:, 2] + probs[:, 3] + probs[:, 4]


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    # sensitivity / specificity from hard predictions; AUC from scores
    tp = int(((y_true == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_bin == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    # AUC
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        auc_note = "undefined_single_class"
    else:
        # Mann–Whitney / rank AUC without sklearn dependency
        pos = scores[y_true == 1]
        neg = scores[y_true == 0]
        # handle ties via midrank
        order = np.argsort(scores)
        ranks = np.empty_like(scores, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
        # average ranks for ties
        sorted_scores = scores[order]
        i = 0
        while i < len(sorted_scores):
            j = i
            while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            if j > i:
                avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
                ranks[order[i : j + 1]] = avg
            i = j + 1
        sum_ranks_pos = ranks[y_true == 1].sum()
        n_pos, n_neg = len(pos), len(neg)
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc_note = "ok"
    return {
        "sensitivity": sens,
        "specificity": spec,
        "auc": auc,
        "auc_note": auc_note,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def table_B() -> None:
    rows = []
    for cell in CELLS:
        coral = is_coral(cell)
        for split, split_lab in SPLITS:
            for s in SEEDS:
                pred = load_pred(cell, s, split)
                y = np.asarray(pred["y_true"], int)
                y_bin = (y >= 2).astype(int)
                yhat = np.asarray(pred["y_pred"], int)
                yhat_bin = (yhat >= 2).astype(int)
                scores = referable_prob(pred, coral)
                bm = binary_metrics(y_bin, scores, yhat_bin)
                rows.append(
                    {
                        "cell": cell,
                        "seed": s,
                        "domain": split_lab,
                        "split_key": split,
                        "head": "coral" if coral else "ce",
                        "n": len(y),
                        "n_referable": int(y_bin.sum()),
                        "sensitivity": f"{bm['sensitivity']:.6f}" if bm["sensitivity"] == bm["sensitivity"] else "NA",
                        "specificity": f"{bm['specificity']:.6f}" if bm["specificity"] == bm["specificity"] else "NA",
                        "auc": f"{bm['auc']:.6f}" if bm["auc"] == bm["auc"] else "NA",
                        "auc_note": bm["auc_note"],
                        "tp": bm["tp"],
                        "tn": bm["tn"],
                        "fp": bm["fp"],
                        "fn": bm["fn"],
                        "score_def": "sigmoid(z_1)" if coral else "sum(softmax p2..p4)",
                    }
                )
    write_csv(OUT / "table_B_referable_dr.csv", rows)
    write_md(OUT / "table_B_referable_dr.md", "B. Referable-DR (grade≥2) metrics", rows)
    write_tex(
        OUT / "table_B_referable_dr.tex",
        "Referable-DR sensitivity, specificity, and AUC by cell, domain, and seed.",
        "tab:B_referable",
        rows,
        ["cell", "seed", "domain", "sensitivity", "specificity", "auc", "auc_note"],
    )


# ---------------------------------------------------------------------------
# C. Image-level bootstrap QWK CI
# ---------------------------------------------------------------------------
def bootstrap_qwk_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, rng_seed: int) -> dict:
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    point = quadratic_weighted_kappa(y_true, y_pred)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = quadratic_weighted_kappa(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "qwk": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_mean": float(boots.mean()),
        "boot_std": float(boots.std(ddof=1)),
        "n_boot": n_boot,
        "rng_seed": rng_seed,
    }


def table_C() -> None:
    rows = []
    for cell in CELLS:
        for split, split_lab in SPLITS:
            for s in SEEDS:
                pred = load_pred(cell, s, split)
                yt = np.asarray(pred["y_true"], int)
                yp = np.asarray(pred["y_pred"], int)
                # Also verify against archived metrics QWK
                m = load_metrics(cell, s, split)
                archived = float(m["quadratic_weighted_kappa"])
                recomputed = float(quadratic_weighted_kappa(yt, yp))
                if abs(archived - recomputed) > 1e-6:
                    raise RuntimeError(
                        f"QWK mismatch {cell} s{s} {split}: archived={archived} recomputed={recomputed}"
                    )
                ci = bootstrap_qwk_ci(yt, yp, BOOT_N, BOOT_RNG)
                rows.append(
                    {
                        "cell": cell,
                        "seed": s,
                        "domain": split_lab,
                        "split_key": split,
                        "n": len(yt),
                        "qwk": f"{ci['qwk']:.6f}",
                        "ci95_low": f"{ci['ci_low']:.6f}",
                        "ci95_high": f"{ci['ci_high']:.6f}",
                        "boot_mean": f"{ci['boot_mean']:.6f}",
                        "boot_std": f"{ci['boot_std']:.6f}",
                        "n_boot": BOOT_N,
                        "rng_seed": BOOT_RNG,
                        "archived_qwk_match_1e-6": "yes",
                    }
                )
    write_csv(OUT / "table_C_qwk_bootstrap_ci.csv", rows)
    write_md(OUT / "table_C_qwk_bootstrap_ci.md", "C. Image-level bootstrap 95% CI for QWK", rows)
    write_tex(
        OUT / "table_C_qwk_bootstrap_ci.tex",
        f"QWK with image-level bootstrap 95\\% CI ({BOOT_N} resamples, RNG={BOOT_RNG}).",
        "tab:C_qwk_boot",
        rows,
        ["cell", "seed", "domain", "qwk", "ci95_low", "ci95_high"],
    )


# ---------------------------------------------------------------------------
# D. ECE / ECE@T all domains + temperature table
# ---------------------------------------------------------------------------
def table_D() -> None:
    rows = []
    claim_rows = []
    for cell in CELLS:
        coral = is_coral(cell)
        head = "coral" if coral else "ce"
        for split, split_lab in SPLITS:
            for s in SEEDS:
                pred = load_pred(cell, s, split)
                m = load_metrics(cell, s, split)
                yt = np.asarray(pred["y_true"], int)
                logits = np.asarray(pred["logits"], float)
                probs = np.asarray(pred["probs"], float)
                T = float(m["temperature"])
                ece_raw = expected_calibration_error(yt, probs, n_bins=15)
                probs_t = temperature_scale_probs(logits, T, head=head)
                ece_t = expected_calibration_error(yt, probs_t, n_bins=15)
                arch_ece = float(m["ece"])
                arch_ece_t = float(m["ece_temperature_scaled"])
                # Allow tiny float drift vs archived
                if abs(ece_raw - arch_ece) > 1e-4:
                    raise RuntimeError(
                        f"ECE drift {cell} s{s} {split}: recomputed={ece_raw} archived={arch_ece}"
                    )
                if abs(ece_t - arch_ece_t) > 1e-4:
                    raise RuntimeError(
                        f"ECE@T drift {cell} s{s} {split}: recomputed={ece_t} archived={arch_ece_t}"
                    )
                row = {
                    "cell": cell,
                    "seed": s,
                    "domain": split_lab,
                    "split_key": split,
                    "temperature_T": f"{T:.6f}",
                    "ece": f"{ece_raw:.6f}",
                    "ece_at_T": f"{ece_t:.6f}",
                    "delta_ece_T_minus_raw": f"{(ece_t - ece_raw):.6f}",
                    "archived_ece": f"{arch_ece:.6f}",
                    "archived_ece_at_T": f"{arch_ece_t:.6f}",
                }
                rows.append(row)
                claim_rows.append(row)
    write_csv(OUT / "table_D_ece_all_domains.csv", rows)
    write_md(OUT / "table_D_ece_all_domains.md", "D. ECE and ECE@T by cell × domain × seed", rows)
    write_tex(
        OUT / "table_D_ece_all_domains.tex",
        "ECE and temperature-scaled ECE (T fitted on APTOS val; stored per run).",
        "tab:D_ece",
        rows,
        ["cell", "seed", "domain", "temperature_T", "ece", "ece_at_T", "delta_ece_T_minus_raw"],
    )

    # Claim-support table: external only, mean±std of (ECE@T − ECE) and both levels
    summary = []
    for cell in CELLS:
        for split, split_lab in SPLITS:
            e = []
            et = []
            d = []
            Ts = []
            for s in SEEDS:
                m = load_metrics(cell, s, split)
                e.append(float(m["ece"]))
                et.append(float(m["ece_temperature_scaled"]))
                d.append(float(m["ece_temperature_scaled"]) - float(m["ece"]))
                Ts.append(float(m["temperature"]))
            summary.append(
                {
                    "cell": cell,
                    "domain": split_lab,
                    "is_external": "no" if split == "aptos_test" else "yes",
                    "ece_mean_pm_std": fmt_ms(e),
                    "ece_at_T_mean_pm_std": fmt_ms(et),
                    "delta_T_minus_raw_mean_pm_std": fmt_ms(d),
                    "T_mean_pm_std": fmt_ms(Ts),
                    "n_seeds_where_ece_at_T_ge_ece": sum(1 for a, b in zip(et, e) if a >= b),
                }
            )
    write_csv(OUT / "table_D_temperature_external_repair_check.csv", summary)
    write_md(
        OUT / "table_D_temperature_external_repair_check.md",
        "D. Numbers relevant to whether temperature scaling reduces ECE (no conclusions)",
        summary,
    )
    write_tex(
        OUT / "table_D_temperature_external_repair_check.tex",
        "ECE vs ECE@T by domain (mean$\\pm$std over seeds). No interpretive conclusions.",
        "tab:D_T_check",
        summary,
        list(summary[0].keys()),
    )


# ---------------------------------------------------------------------------
# E. H2 mild confusion + IDs
# ---------------------------------------------------------------------------
def table_E() -> None:
    rows_cm = []
    rows_ids = []
    recalls = []
    for s in SEEDS:
        pred = load_pred("H2", s, "aptos_test")
        m = load_metrics("H2", s, "aptos_test")
        yt = np.asarray(pred["y_true"], int)
        yp = np.asarray(pred["y_pred"], int)
        ids = list(pred["ids"])
        cm = np.asarray(m["confusion_matrix"], int)
        # flatten cm rows
        for i in range(5):
            rows_cm.append(
                {
                    "seed": s,
                    "true_grade": i,
                    **{f"pred_{j}": int(cm[i, j]) for j in range(5)},
                }
            )
        mild_mask = yt == 1
        n_mild = int(mild_mask.sum())
        correct_idx = [i for i in range(len(yt)) if mild_mask[i] and yp[i] == 1]
        correct_ids = [ids[i] for i in correct_idx]
        recall = len(correct_ids) / n_mild if n_mild else float("nan")
        recalls.append(recall)
        archived = float(m["mild_dr_recall"])
        if abs(recall - archived) > 1e-6:
            raise RuntimeError(f"mild recall mismatch seed {s}: {recall} vs archived {archived}")
        if n_mild != 30:
            raise RuntimeError(f"expected 30 mild on APTOS test, got {n_mild}")
        if len(correct_ids) != 5:
            raise RuntimeError(f"expected 5/30 mild correct for H2 s{s}, got {len(correct_ids)}")
        # verify logits differ across seeds for the same IDs (different checkpoints)
        rows_ids.append(
            {
                "seed": s,
                "n_mild": n_mild,
                "n_correct_mild": len(correct_ids),
                "mild_recall": f"{recall:.6f}",
                "correct_mild_ids": ";".join(correct_ids),
                "pred_on_mild": ";".join(str(int(yp[i])) for i in range(len(yt)) if mild_mask[i]),
            }
        )
    # Cross-seed logit comparison on shared mild IDs
    pred42 = load_pred("H2", 42, "aptos_test")
    pred43 = load_pred("H2", 43, "aptos_test")
    pred44 = load_pred("H2", 44, "aptos_test")
    id_to_logit = []
    for s, pred in [(42, pred42), (43, pred43), (44, pred44)]:
        mapping = {i: np.asarray(L, float) for i, L in zip(pred["ids"], pred["logits"])}
        id_to_logit.append((s, mapping))
    shared_mild = None
    mild_ids_by_seed = []
    for s, pred in [(42, pred42), (43, pred43), (44, pred44)]:
        yt = np.asarray(pred["y_true"], int)
        ids = list(pred["ids"])
        mild_ids = {ids[i] for i in range(len(ids)) if yt[i] == 1}
        mild_ids_by_seed.append(mild_ids)
    shared_mild = set.intersection(*mild_ids_by_seed)
    logit_diffs = []
    for mid in sorted(shared_mild):
        L42 = id_to_logit[0][1][mid]
        L43 = id_to_logit[1][1][mid]
        L44 = id_to_logit[2][1][mid]
        logit_diffs.append(
            {
                "id": mid,
                "max_abs_diff_42_43": f"{np.max(np.abs(L42 - L43)):.6f}",
                "max_abs_diff_42_44": f"{np.max(np.abs(L42 - L44)):.6f}",
                "identical_all_seeds": str(
                    np.allclose(L42, L43) and np.allclose(L42, L44)
                ),
            }
        )
    n_identical = sum(1 for r in logit_diffs if r["identical_all_seeds"] == "True")
    if n_identical == len(logit_diffs):
        raise RuntimeError(
            "All mild-image logits identical across seeds — suspicious evaluation reuse"
        )

    write_csv(OUT / "table_E_h2_aptos_confusion_per_seed.csv", rows_cm)
    write_csv(OUT / "table_E_h2_mild_correct_ids.csv", rows_ids)
    write_csv(OUT / "table_E_h2_mild_logit_seed_diffs.csv", logit_diffs)
    write_md(
        OUT / "table_E_h2_mild_check.md",
        "E. H2 APTOS mild recall check (5/30 each seed)",
        rows_ids,
    )
    write_tex(
        OUT / "table_E_h2_mild_correct_ids.tex",
        "H2 APTOS-test mild images predicted correctly (expect 5/30 per seed).",
        "tab:E_mild",
        rows_ids,
        ["seed", "n_mild", "n_correct_mild", "mild_recall", "correct_mild_ids"],
    )
    print(
        "E_CHECK: mild recalls",
        recalls,
        "identical_logit_mild_images",
        n_identical,
        "/",
        len(logit_diffs),
    )


def write_sources() -> None:
    uniq = sorted(set(SOURCES))
    path = OUT / "table_AE_source_files.txt"
    path.write_text("\n".join(uniq) + "\n")
    print("WROTE", path.relative_to(ROOT), "n_sources", len(uniq))


def main() -> None:
    print("BUILD_PAPER_ANALYSIS_TABLES start")
    table_A()
    table_B()
    table_C()
    table_D()
    table_E()
    write_sources()
    print("BUILD_PAPER_ANALYSIS_TABLES_OK")


if __name__ == "__main__":
    main()
