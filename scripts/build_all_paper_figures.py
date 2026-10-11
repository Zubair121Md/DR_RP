#!/usr/bin/env python3
"""
Build a full paper figure pack from existing artifacts.

Honest scope:
  - Uses our B0 + hybrid results only (no invented literature numbers).
  - Binary = referable DR (grades ≥2) vs non-referable (0–1), a common DR grouping.
  - Augmentations shown = those in frozen protocol (flip / rotate / color jitter).
  - Architecture = schematic of our MobileViT-S hybrid pipeline (not a paper scan).

Outputs → paper/figures/pack/ + FIGURE_INDEX.md
"""
from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torchvision import transforms as T

from src.preprocessing import apply_retina_frontend, make_cached_frame
from src.utils import load_config, resolve_data_roots

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.restricted import CHECKPOINTS, require_or_skip  # noqa: E402
OUT = ROOT / "paper" / "figures" / "pack"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)
CELLS = ("H1", "H2", "H3", "H4")
CELL_LAB = {
    "H1": "RGB×CE",
    "H2": "G+CLAHE×CE",
    "H3": "RGB×mildCORAL",
    "H4": "G+CLAHE×mildCORAL",
}
SPLITS_EVAL = ("aptos_test", "idrid_external", "messidor2_external", "deepdrid_external")
SPLIT_LAB = {
    "aptos_test": "APTOS test",
    "idrid_external": "IDRiD",
    "messidor2_external": "Messidor-2",
    "deepdrid_external": "DeepDRiD",
}


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("SAVED", path.relative_to(ROOT), flush=True)
    return path


def load_metrics(rid: str, split: str) -> dict:
    return json.loads((ROOT / "results" / rid / f"{split}_metrics.json").read_text())


def load_preds(rid: str, split: str) -> dict:
    return json.loads((ROOT / "results" / rid / f"{split}_predictions.json").read_text())


def load_history(rid: str) -> list:
    p = ROOT / "results" / rid / "history.json"
    return json.loads(p.read_text()) if p.is_file() else []


# ─── 01 workflow ─────────────────────────────────────────────────────────────
def fig_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.2, 1.1, 1.8, 1.2, "Datasets\nAPTOS / IDRiD\nMessidor / DeepDRiD"),
        (2.3, 1.1, 1.8, 1.2, "Preprocess\nRGB or\nGreen+CLAHE"),
        (4.4, 1.1, 1.8, 1.2, "Augment\n(train only)\nflip/rot/jitter"),
        (6.5, 1.1, 1.8, 1.2, "Backbone\nMobileViT-S\n(CNN–Transformer)"),
        (8.6, 1.1, 1.5, 1.2, "Head\nCE or\nmild-CORAL"),
        (10.4, 1.1, 1.4, 1.2, "Metrics\nQWK / F1\nECE / CAM"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                            facecolor="#E8F1F8", edgecolor="#1F4E79", lw=1.5))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 1.7), xytext=(x0, 1.7),
                    arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=1.4))
    ax.set_title("Study workflow (frozen_protocol_v1)", fontsize=12, pad=8)
    save(fig, "01_workflow_pipeline.png")


# ─── 02 architecture ─────────────────────────────────────────────────────────
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    layers = [
        (0.4, 2.0, 1.4, 1.5, "Input\n224×224×3\nRGB or G+CLAHE"),
        (2.1, 2.0, 1.6, 1.5, "MobileViT stem\n+ MV blocks\n(local CNN +\nglobal transformer)"),
        (4.0, 2.0, 1.5, 1.5, "Final conv\n640×7×7\n(+ Grad-CAM)"),
        (5.8, 2.0, 1.4, 1.5, "Global pool\nfeature\nvector"),
        (7.5, 3.1, 2.0, 1.2, "CE head\n5-way softmax"),
        (7.5, 1.0, 2.0, 1.2, "mild-CORAL head\n4 ordinal logits\n(+ mild weights)"),
    ]
    for x, y, w, h, t in layers:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                            facecolor="#F7F3E8", edgecolor="#5C4033", lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
    # arrows
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        x0 = layers[a][0] + layers[a][2]
        x1 = layers[b][0]
        ax.annotate("", xy=(x1, 2.75), xytext=(x0, 2.75),
                    arrowprops=dict(arrowstyle="->", color="#5C4033", lw=1.3))
    ax.annotate("", xy=(7.5, 3.7), xytext=(7.2, 2.75),
                arrowprops=dict(arrowstyle="->", color="#5C4033", lw=1.2))
    ax.annotate("", xy=(7.5, 1.6), xytext=(7.2, 2.75),
                arrowprops=dict(arrowstyle="->", color="#5C4033", lw=1.2))
    ax.text(5, 4.6, "Hybrid feature extraction model (MobileViT-S)", ha="center", fontsize=12)
    ax.text(5, 0.35, "H1/H2 use CE · H3/H4 use mild-aware CORAL · BN frozen on CPU training",
            ha="center", fontsize=8, style="italic")
    save(fig, "02_hybrid_architecture.png")


# ─── 03 class distributions ───────────────────────────────────────────────────
def _aptos_counts() -> dict[str, Counter]:
    root = ROOT / "datasets" / "archive"
    out = {}
    for split, f in [("train", "train_1.csv"), ("val", "valid.csv"), ("test", "test.csv")]:
        df = pd.read_csv(root / f)
        cols = {c.lower(): c for c in df.columns}
        y = df[cols["diagnosis"]].astype(int)
        out[split] = Counter(int(v) for v in y)
    return out


def fig_class_distributions() -> None:
    counts = _aptos_counts()
    grades = list(range(5))
    # multiclass
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, split in zip(axes, ["train", "val", "test"]):
        vals = [counts[split].get(g, 0) for g in grades]
        ax.bar([str(g) for g in grades], vals, color="#4C78A8")
        ax.set_title(f"APTOS {split} (n={sum(vals)})")
        ax.set_xlabel("ICDR grade")
        ax.set_ylabel("Count" if split == "train" else "")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.01, str(v), ha="center", fontsize=8)
    fig.suptitle("Multiclass class frequency (ICDR 0–4)", fontsize=12)
    fig.tight_layout()
    save(fig, "03_class_freq_multiclass_aptos.png")

    # binary referable
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(3)
    nonref = [counts[s][0] + counts[s][1] for s in ["train", "val", "test"]]
    ref = [sum(counts[s][g] for g in (2, 3, 4)) for s in ["train", "val", "test"]]
    ax.bar(x - 0.18, nonref, 0.36, label="Non-referable (0–1)", color="#72B7B2")
    ax.bar(x + 0.18, ref, 0.36, label="Referable (≥2)", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(["train", "val", "test"])
    ax.set_ylabel("Count")
    ax.set_title("Binary class frequency (referable DR grouping)")
    ax.legend()
    for i, (a, b) in enumerate(zip(nonref, ref)):
        ax.text(i - 0.18, a + 20, str(a), ha="center", fontsize=8)
        ax.text(i + 0.18, b + 20, str(b), ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "04_class_freq_binary_referable.png")


# ─── 04 dataset pie / stacked ─────────────────────────────────────────────────
def fig_dataset_overview() -> None:
    counts = _aptos_counts()
    # external sizes from metrics
    sizes = {
        "APTOS train": sum(counts["train"].values()),
        "APTOS val": sum(counts["val"].values()),
        "APTOS test": sum(counts["test"].values()),
        "IDRiD": load_metrics("hybrid_h2_s42_v1", "idrid_external")["n_samples"],
        "Messidor-2": load_metrics("hybrid_h2_s42_v1", "messidor2_external")["n_samples"],
        "DeepDRiD": load_metrics("hybrid_h2_s42_v1", "deepdrid_external")["n_samples"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].pie(list(sizes.values()), labels=list(sizes.keys()), autopct="%1.1f%%",
                textprops={"fontsize": 8}, startangle=90)
    axes[0].set_title("Image distribution across sets")
    axes[1].barh(list(sizes.keys())[::-1], list(sizes.values())[::-1], color="#4C78A8")
    axes[1].set_xlabel("Images")
    axes[1].set_title("Dataset sizes")
    for i, (k, v) in enumerate(list(sizes.items())[::-1]):
        axes[1].text(v + 30, i, str(v), va="center", fontsize=8)
    fig.suptitle("Dataset representation", fontsize=12)
    fig.tight_layout()
    save(fig, "05_dataset_image_distribution.png")


# ─── 05 preprocess before/after ───────────────────────────────────────────────
def fig_preprocess_panels() -> None:
    if not require_or_skip("figure pack 06 (preprocessing panels)", "APTOS"):
        return
    cfg = load_config(ROOT / "experiments/hybrid_h2_s42_v1/config.yaml")
    roots = resolve_data_roots(cfg, ROOT)
    # use APTOS test first mild-ish image from diagnostics if available, else from CSV
    diag = ROOT / "experiments/results/preprocessing_diagnostics"
    # rebuild a clean panel from one raw path
    ds_csv = pd.read_csv(roots["aptos"] / "test.csv")
    cols = {c.lower(): c for c in ds_csv.columns}
    # pick grade 2 example
    row = ds_csv[ds_csv[cols["diagnosis"]] == 2].iloc[0]
    iid = str(row[cols["id_code"]])
    img_dir = roots["aptos"] / "test_images" / "test_images"
    # resolve
    path = None
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_dir / f"{iid}{ext}"
        if p.is_file():
            path = p
            break
    if path is None:
        # fallback any diagnostic
        path = next((diag.glob("aptos_test_*_y1.png")), None)
        if path is None:
            return
        # diagnostics are already panels — copy pack
        for src in diag.glob("*.png"):
            shutil.copy2(src, OUT / f"06_preprocess_diag_{src.name}")
        return

    orig = Image.open(path).convert("RGB")
    rgb224 = make_cached_frame("rgb", orig, 224)
    g224 = make_cached_frame(
        "green_clahe", orig, 224, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)
    )
    # also show native resize only
    resized = orig.resize((224, 224))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(orig)
    axes[0].set_title("Raw fundus")
    axes[1].imshow(resized)
    axes[1].set_title("After resize 224²")
    axes[2].imshow(rgb224)
    axes[2].set_title("RGB pipeline")
    axes[3].imshow(g224)
    axes[3].set_title("Green+CLAHE pipeline")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Preprocessing example (APTOS test id={iid}, grade 2)", fontsize=11)
    fig.tight_layout()
    save(fig, "06_preprocess_before_after.png")

    # copy existing diagnostics into pack too
    for src in sorted(diag.glob("*.png")):
        shutil.copy2(src, OUT / f"06b_{src.name}")


# ─── 06 augmentations ─────────────────────────────────────────────────────────
def fig_augmentations() -> None:
    if not require_or_skip("figure pack 07 (augmentation examples)", "APTOS"):
        return
    cfg = load_config(ROOT / "experiments/hybrid_h2_s42_v1/config.yaml")
    roots = resolve_data_roots(cfg, ROOT)
    ds_csv = pd.read_csv(roots["aptos"] / "train_1.csv")
    cols = {c.lower(): c for c in ds_csv.columns}
    row = ds_csv[ds_csv[cols["diagnosis"]] == 2].iloc[3]
    iid = str(row[cols["id_code"]])
    img_dir = roots["aptos"] / "train_images" / "train_images"
    path = None
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_dir / f"{iid}{ext}"
        if p.is_file():
            path = p
            break
    if path is None:
        return
    base = make_cached_frame(
        "green_clahe",
        Image.open(path).convert("RGB"),
        224,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
    )
    t = T.ToTensor()
    x = t(base)
    torch.manual_seed(0)
    augs = [
        ("Original (no aug)", x),
        ("HFlip", T.RandomHorizontalFlip(p=1.0)(x.clone())),
        ("Rotate ±10°", T.RandomRotation(degrees=10)(x.clone())),
        ("ColorJitter", T.ColorJitter(0.1, 0.1)(x.clone())),
        ("Combined protocol", T.Compose([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(10),
            T.ColorJitter(0.1, 0.1),
        ])(x.clone())),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
    for ax, (title, ten) in zip(axes, augs):
        img = ten.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle("Train-time data augmentations (frozen protocol)", fontsize=11)
    fig.tight_layout()
    save(fig, "07_augmentation_examples.png")


# ─── 07 confusion matrices ────────────────────────────────────────────────────
def fig_confusion_matrices() -> None:
    # H2 and H4 seed42 on APTOS + IDRiD (+ Messidor for failure)
    specs = [
        ("hybrid_h2_s42_v1", "aptos_test", "H2 APTOS"),
        ("hybrid_h2_s42_v1", "idrid_external", "H2 IDRiD"),
        ("hybrid_h4_s42_v1", "aptos_test", "H4 APTOS"),
        ("hybrid_h4_s42_v1", "idrid_external", "H4 IDRiD"),
        ("hybrid_h2_s42_v1", "messidor2_external", "H2 Messidor-2"),
        ("hybrid_h2_s42_v1", "deepdrid_external", "H2 DeepDRiD"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, (rid, split, title) in zip(axes.ravel(), specs):
        cm = np.asarray(load_metrics(rid, split)["confusion_matrix"], dtype=float)
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm_n[i, j] > 0.5 else "black", fontsize=7)
    fig.suptitle("Confusion matrices (seed 42; row-normalized color, raw counts annotated)", fontsize=12)
    fig.tight_layout()
    save(fig, "08_confusion_matrices_h2_h4.png")

    # all hybrid cells APTOS s42
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, c in zip(axes, CELLS):
        rid = f"hybrid_{c.lower()}_s42_v1"
        cm = np.asarray(load_metrics(rid, "aptos_test")["confusion_matrix"], dtype=float)
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{c} {CELL_LAB[c]}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm_n[i, j] > 0.5 else "black")
    fig.suptitle("APTOS test confusion matrices — all hybrid cells (seed 42)", fontsize=12)
    fig.tight_layout()
    save(fig, "09_confusion_matrices_all_cells_aptos.png")


# ─── 08 ROC OvR ───────────────────────────────────────────────────────────────
def fig_roc_curves() -> None:
    # H2 s42 on each domain + all cells on APTOS
    def plot_ovr(ax, y_true, probs, title: str) -> None:
        y = np.asarray(y_true)
        p = np.asarray(probs)
        for g in range(5):
            yt = (y == g).astype(int)
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            fpr, tpr, _ = roc_curve(yt, p[:, g])
            ax.plot(fpr, tpr, lw=1.4, label=f"G{g} AUC={auc(fpr, tpr):.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=6, loc="lower right")

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    rid = "hybrid_h2_s42_v1"
    for ax, split in zip(axes.ravel(), SPLITS_EVAL):
        pred = load_preds(rid, split)
        plot_ovr(ax, pred["y_true"], pred["probs"], f"H2 · {SPLIT_LAB[split]}")
    fig.suptitle("One-vs-rest ROC (H2 seed 42; secondary to QWK)", fontsize=12)
    fig.tight_layout()
    save(fig, "10_roc_ovr_h2_all_domains.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, c in zip(axes.ravel(), CELLS):
        pred = load_preds(f"hybrid_{c.lower()}_s42_v1", "aptos_test")
        plot_ovr(ax, pred["y_true"], pred["probs"], f"{c} {CELL_LAB[c]} · APTOS")
    fig.suptitle("One-vs-rest ROC — hybrid cells on APTOS test (seed 42)", fontsize=12)
    fig.tight_layout()
    save(fig, "11_roc_ovr_all_cells_aptos.png")


# ─── 09 metric comparison bars ────────────────────────────────────────────────
def fig_metric_comparisons() -> None:
    # mean over seeds for Acc / macroF1 / QWK / mildF1 on APTOS
    metrics = ["accuracy", "macro_f1", "quadratic_weighted_kappa", "mild_dr_f1",
               "mild_dr_precision", "mild_dr_recall"]
    labels = ["Accuracy", "Macro-F1", "QWK", "Mild F1", "Mild Prec", "Mild Rec"]
    means = {c: [] for c in CELLS}
    stds = {c: [] for c in CELLS}
    for m in metrics:
        for c in CELLS:
            vals = [load_metrics(f"hybrid_{c.lower()}_s{s}_v1", "aptos_test")[m] for s in SEEDS]
            means[c].append(float(np.mean(vals)))
            stds[c].append(float(np.std(vals, ddof=1)))

    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, c in enumerate(CELLS):
        ax.bar(x + (i - 1.5) * width, means[c], width, yerr=stds[c], capsize=3,
               label=f"{c} {CELL_LAB[c]}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Hybrid cells — APTOS test metrics (mean±std, n=3 seeds)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "12_metrics_acc_f1_qwk_mild.png")

    # B0 vs Hybrid IDRiD/APTOS QWK from tables
    b0 = json.loads((ROOT / "results/b0_2x2_seed_summary.json").read_text())
    hyb = json.loads((ROOT / "experiments/results/hybrid_seed_summary.json").read_text())
    pairs = [
        ("RGB×CE", "rgb_ce", "H1"),
        ("G+CLAHE×CE", "green_clahe_ce", "H2"),
        ("RGB×CORAL*", "rgb_coral", "H3"),
        ("G+CLAHE×CORAL*", "green_clahe_coral", "H4"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric_key, split_h, title in [
        (axes[0], "aptos_qwk", "aptos_test", "APTOS QWK"),
        (axes[1], "idrid_qwk", "idrid_external", "IDRiD QWK"),
    ]:
        xb = np.arange(len(pairs))
        b0m = [b0["cells"][k]["summary"][metric_key]["mean"] for _, k, _ in pairs]
        b0s = [b0["cells"][k]["summary"][metric_key]["std"] for _, k, _ in pairs]
        hm = [hyb["mean_std"][h][split_h]["quadratic_weighted_kappa"]["mean"] for _, _, h in pairs]
        hs = [hyb["mean_std"][h][split_h]["quadratic_weighted_kappa"]["std"] for _, _, h in pairs]
        ax.bar(xb - 0.18, b0m, 0.36, yerr=b0s, capsize=3, label="EfficientNet-B0")
        ax.bar(xb + 0.18, hm, 0.36, yerr=hs, capsize=3, label="MobileViT-S hybrid")
        ax.set_xticks(xb)
        ax.set_xticklabels([p[0] for p in pairs], rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Comparison: B0 vs proposed hybrid (*B0 uses CORAL; hybrid H3/H4 mild-CORAL)", fontsize=11)
    fig.tight_layout()
    save(fig, "13_comparison_b0_vs_hybrid.png")


# ─── 10 per-class precision/recall/f1 ─────────────────────────────────────────
def fig_per_class_prf() -> None:
    rid = "hybrid_h2_s42_v1"
    m = load_metrics(rid, "aptos_test")
    grades = np.arange(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.25
    ax.bar(grades - w, m["per_class_precision"], w, label="Precision")
    ax.bar(grades, m["per_class_recall"], w, label="Recall")
    ax.bar(grades + w, m["per_class_f1"], w, label="F1")
    ax.set_xticks(grades)
    ax.set_xticklabels([f"G{g}" for g in grades])
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Precision / Recall / F1 — H2 APTOS test (seed 42)")
    ax.legend()
    fig.tight_layout()
    save(fig, "14_per_class_precision_recall_f1.png")


# ─── 11 training curves ───────────────────────────────────────────────────────
def fig_training_curves() -> None:
    # H2 all seeds accuracy + loss + qwk
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for s in SEEDS:
        hist = load_history(f"hybrid_h2_s{s}_v1")
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["val_accuracy"] for h in hist], label=f"seed {s}")
        axes[1].plot(ep, [h["train_loss"] for h in hist], label=f"train s{s}", alpha=0.7)
        axes[1].plot(ep, [h["val_loss"] for h in hist], "--", label=f"val s{s}", alpha=0.7)
        axes[2].plot(ep, [1 - h["val_accuracy"] for h in hist], label=f"seed {s}")
    axes[0].set_title("Val accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7)
    axes[1].set_title("Train/val loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=6, ncol=2)
    axes[2].set_title("Classification error (1−acc) over epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Error")
    axes[2].legend(fontsize=7)
    fig.suptitle("H2 (G+CLAHE×CE) training dynamics", fontsize=12)
    fig.tight_layout()
    save(fig, "15_training_curves_h2.png")

    # all cells seed 42 val QWK
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in CELLS:
        hist = load_history(f"hybrid_{c.lower()}_s42_v1")
        if not hist:
            continue
        ax.plot([h["epoch"] for h in hist], [h["val_qwk"] for h in hist], label=f"{c} {CELL_LAB[c]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val QWK")
    ax.set_title("Validation QWK improvement over epochs (seed 42)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "16_val_qwk_over_epochs_all_cells.png")


# ─── 12 efficiency ────────────────────────────────────────────────────────────
def fig_efficiency() -> None:
    rows = json.loads((ROOT / "results/efficiency_models.json").read_text())["rows"]
    labels = [f"{r['backbone']}\n{r['head']}" for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].bar(labels, [r["params_m"] for r in rows], color="#4C78A8")
    axes[0].set_title("Params (M)")
    axes[1].bar(labels, [r["flops_g"] for r in rows], color="#F58518")
    axes[1].set_title("FLOPs (G)")
    axes[2].bar(labels, [r["batch1_ms_mean"] for r in rows], color="#54A24B")
    axes[2].set_title("Latency batch-1 (ms)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Efficiency comparison (from efficiency_models.json)", fontsize=11)
    fig.tight_layout()
    save(fig, "17_efficiency_params_flops_latency.png")


# ─── 13 domain QWK heatmap ────────────────────────────────────────────────────
def fig_qwk_heatmap() -> None:
    hyb = json.loads((ROOT / "experiments/results/hybrid_seed_summary.json").read_text())
    mat = np.zeros((4, 4))
    for i, c in enumerate(CELLS):
        for j, split in enumerate(SPLITS_EVAL):
            mat[i, j] = hyb["mean_std"][c][split]["quadratic_weighted_kappa"]["mean"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels([SPLIT_LAB[s] for s in SPLITS_EVAL], rotation=15, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{c} {CELL_LAB[c]}" for c in CELLS])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Mean QWK heatmap (seeds 42/43/44)")
    fig.tight_layout()
    save(fig, "18_qwk_heatmap_domains.png")


# ─── index ────────────────────────────────────────────────────────────────────
def write_index(paths: list[str]) -> None:
    lines = [
        "# Paper figure pack — index",
        "",
        "Generated by `scripts/build_all_paper_figures.py`.",
        "",
        "## Included",
        "",
    ]
    for p in paths:
        lines.append(f"- `{p}`")
    lines += [
        "",
        "## Notes / honesty",
        "",
        "- **Binary distribution** = referable DR (grades ≥2) vs non-referable (0–1).",
        "- **ROC** is OvR multiclass (secondary); primary metric remains QWK.",
        "- **Literature SOTA table** is *not* auto-invented — use your own citations in text;",
        "  figure `13_comparison_b0_vs_hybrid.png` is the controlled internal comparison.",
        "- Augmentations match frozen protocol: HFlip 0.5, Rotate ±10°, ColorJitter 0.1.",
        "- Preprocess: RGB vs Green+CLAHE (no FOV crop in implementation).",
        "",
        "FIGURE_PACK_OK",
        "",
    ]
    (OUT / "FIGURE_INDEX.md").write_text("\n".join(lines))
    (ROOT / "paper" / "figures" / "FIGURE_INDEX.md").write_text(
        "# Figures\n\nSee [`pack/FIGURE_INDEX.md`](pack/FIGURE_INDEX.md) for the full pack.\n"
    )


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    generators = [
        fig_workflow,
        fig_architecture,
        fig_class_distributions,
        fig_dataset_overview,
        fig_preprocess_panels,
        fig_augmentations,
        fig_confusion_matrices,
        fig_roc_curves,
        fig_metric_comparisons,
        fig_per_class_prf,
        fig_training_curves,
        fig_efficiency,
        fig_qwk_heatmap,
    ]
    for fn in generators:
        print(f">>> {fn.__name__}", flush=True)
        try:
            fn()
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
            raise
    paths = sorted(str(p.relative_to(ROOT)) for p in OUT.glob("*.png"))
    write_index(paths)
    print("FIGURE_PACK_OK", len(paths), "pngs", flush=True)


if __name__ == "__main__":
    main()
