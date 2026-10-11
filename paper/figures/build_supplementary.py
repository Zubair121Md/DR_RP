#!/usr/bin/env python3
"""
Supplementary figures S1–S15 from paper/tables CSVs and saved predictions only.

Style: no in-figure titles; panel letters; Okabe–Ito / viridis; fonts >= 8 pt;
width 6.9 in (double column) or 3.3 in; seed std error bars with per-seed points;
PDF (vector) + PNG 600 dpi under paper/figures/supp/.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.restricted import CHECKPOINTS, require_or_skip  # noqa: E402
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures" / "supp"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "paper" / "tables"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
CELLS = ["H1", "H2", "H3", "H4"]
CELL_COLOR = dict(zip(CELLS, OKABE[:4]))
CELL_LABEL = {
    "H1": "H1 RGB, CE",
    "H2": "H2 CLAHE, CE",
    "H3": "H3 RGB, MW-threshold",
    "H4": "H4 CLAHE, MW-threshold",
}
DOMAINS = ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
SPLIT = {"APTOS": "aptos_test", "IDRiD": "idrid_external", "Messidor-2": "messidor2_external",
         "DeepDRiD": "deepdrid_external"}
SEEDS = [42, 43, 44]
SEEDS6 = [42, 43, 44, 45, 46, 47]


def run_dir(cell: str, seed: int) -> Path:
    return ROOT / "results" / f"hybrid_{cell.lower()}_s{seed}_{'v1' if seed <= 44 else 'v3'}"


def read_csv(name: str) -> list[dict]:
    path = TABLES / name
    if not path.is_file():
        raise SystemExit(f"MISSING {path.relative_to(ROOT)}")
    print("READ", path.relative_to(ROOT))
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"MISSING {path.relative_to(ROOT)}")
    print("READ", path.relative_to(ROOT))
    return json.loads(path.read_text())


# Supplementary figures are numbered in the order the Version B manuscript first cites them.
NUMBER_BY_TOPIC = {
    "grade_distribution": 1,
    "validation_qwk_curves": 2,
    "efficiency": 3,
    "augmentation": 4,
    "architecture": 5,
    "dataset_examples": 6,
    "qwk_bootstrap_ci": 7,
    "domain_shift_drop": 8,
    "aptos_metrics": 9,
    "temperature_scaling": 10,
    "confusion_aptos": 11,
    "confusion_idrid": 12,
    "confusion_messidor2": 13,
    "confusion_deepdrid": 14,
    "messidor_prediction_distribution": 15,
    "referable_dr": 16,
    "matched_b0": 17,
    "lodo": 18,
    "gradcam_mild": 19,
    "decomp_clahe": 20,
    "res448": 21,
}


def save(fig: plt.Figure, stem: str) -> None:
    topic = stem.split("_", 1)[1]
    stem = f"figS{NUMBER_BY_TOPIC[topic]:02d}_{topic}"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("WROTE", p.relative_to(ROOT))
    plt.close(fig)


def seed_dots(ax, x: float, ys: list[float], color: str, spread: float = 0.08) -> None:
    ys = np.asarray(ys, dtype=float)
    offs = (np.arange(len(ys)) - (len(ys) - 1) / 2) * spread
    ax.scatter(x + offs, ys, s=12, color=color, edgecolors="white", linewidths=0.3, zorder=3)
    ax.errorbar(x, ys.mean(), yerr=ys.std(ddof=1) if len(ys) > 1 else 0, fmt="o", color=color,
                markersize=4, capsize=2, elinewidth=0.8, zorder=2)


def cell_legend(ax, loc="best") -> None:
    ax.legend(handles=[mpatches.Patch(color=CELL_COLOR[c], label=CELL_LABEL[c]) for c in CELLS],
              frameon=False, loc=loc)


# S1 ---------------------------------------------------------------------------
def s01_architecture() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.1, 1.0, 1.6, "Input\n224×224×3\nRGB or CLAHE"),
        (2.1, 1.0, 1.9, "MobileViT-S\nconvolution +\ntransformer blocks"),
        (4.4, 1.0, 1.6, "Final conv\n(Grad-CAM\nlayer)"),
        (6.4, 1.0, 1.4, "Global\naverage\npool"),
    ]
    for x, y, w, t in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, 1.0, boxstyle="round,pad=0.03",
                                             facecolor="#EEEEEE", edgecolor="#333333", lw=0.8))
        ax.text(x + w / 2, y + 0.5, t, ha="center", va="center")
    for x0, x1 in [(1.7, 2.1), (4.0, 4.4), (6.0, 6.4)]:
        ax.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5), arrowprops=dict(arrowstyle="->", lw=0.9))
    heads = [(8.2, 1.9, "CE head\n5 logits", OKABE[4]), (8.2, 0.1, "Threshold head\n4 logits", OKABE[1])]
    for x, y, t, c in heads:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), 1.7, 0.9, boxstyle="round,pad=0.03",
                                             facecolor=c, edgecolor="#333333", lw=0.8, alpha=0.55))
        ax.text(x + 0.85, y + 0.45, t, ha="center", va="center")
        ax.annotate("", xy=(x, y + 0.45), xytext=(7.8, 1.5), arrowprops=dict(arrowstyle="->", lw=0.9))
    save(fig, "figS01_architecture")


# S2 ---------------------------------------------------------------------------
def s02_grade_distribution() -> None:
    manifest = read_csv("dataset_manifest.csv")
    rows: list[tuple[str, list[int]]] = []
    for split in ("train", "val", "test"):
        r = next(m for m in manifest if m["dataset"] == "APTOS" and m["split"] == split)
        rows.append((f"APTOS {split}", [int(r[f"n{g}"]) for g in range(5)]))
    for label, split in [("IDRiD", "idrid_external"), ("Messidor-2", "messidor2_external"),
                         ("DeepDRiD", "deepdrid_external")]:
        p = read_json(ROOT / "results" / "hybrid_h2_s42_v1" / f"{split}_predictions.json")
        yt = np.asarray(p["y_true"], dtype=int)
        rows.append((label, [int((yt == g).sum()) for g in range(5)]))
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    grade_colors = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    for i, (name, counts) in enumerate(rows):
        tot = sum(counts)
        left = 0.0
        for g, n in enumerate(counts):
            frac = n / tot
            ax.barh(i, frac, left=left, color=grade_colors[g], edgecolor="white", lw=0.5)
            if frac > 0.06:
                ax.text(left + frac / 2, i, str(n), ha="center", va="center",
                        color="white" if g < 3 else "black")
            left += frac
        ax.text(1.01, i, f"n={tot}", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of images")
    ax.legend(handles=[mpatches.Patch(color=grade_colors[g], label=f"Grade {g}") for g in range(5)],
              ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, "figS02_grade_distribution")


# S3 ---------------------------------------------------------------------------
def s03_augmentation() -> None:
    if not require_or_skip("Figure S4 (augmentation examples)", "APTOS"):
        return
    import yaml

    from src.preprocessing import CachedPreprocess, make_cached_frame

    cfg = yaml.safe_load((ROOT / "configs/paper/fig02_preprocess.yaml").read_text())
    print("READ configs/paper/fig02_preprocess.yaml")
    iid = cfg["selection"]["image_id"]
    hits = list((ROOT / "datasets/archive").rglob(f"{iid}.png"))
    if not hits:
        raise SystemExit(f"MISSING APTOS image {iid}")
    print("READ", hits[0].relative_to(ROOT))
    raw = Image.open(hits[0]).convert("RGB")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    aug = CachedPreprocess(train=True)
    torch.manual_seed(42)
    fig, axes = plt.subplots(2, 5, figsize=(6.9, 3.0))
    for r, (prep, label) in enumerate([("rgb", "RGB"), ("green_clahe", "CLAHE")]):
        frame = make_cached_frame(prep, raw, 224)
        axes[r, 0].imshow(frame)
        axes[r, 0].set_ylabel(label)
        for c in range(1, 5):
            x = aug(frame).numpy().transpose(1, 2, 0) * std + mean
            axes[r, c].imshow(np.clip(x, 0, 1))
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, t in enumerate(["(a) Cached input", "(b) Draw 1", "(c) Draw 2", "(d) Draw 3", "(e) Draw 4"]):
        axes[0, c].set_title(t)
    fig.tight_layout()
    save(fig, "figS03_augmentation")


# S4 ---------------------------------------------------------------------------
def s04_val_curves() -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.6), sharey=True)
    for ax, cell, letter in zip(axes, CELLS, "abcd"):
        for seed in SEEDS6:
            hist = read_json(run_dir(cell, seed) / "history.json")
            ep = [int(h["epoch"]) for h in hist]
            q = [float(h["val_qwk"]) for h in hist]
            ax.plot(ep, q, color=CELL_COLOR[cell], lw=0.9, alpha=0.8, ls="-" if seed <= 44 else "--")
            best = int(np.argmax(q))
            ax.scatter([ep[best]], [q[best]], color=CELL_COLOR[cell], s=14, zorder=3,
                       edgecolors="black", linewidths=0.4)
        ax.set_xlabel(f"Epoch\n({letter}) {CELL_LABEL[cell]}")
    axes[0].set_ylabel("APTOS validation QWK")
    fig.legend(handles=[Line2D([], [], color="#333333", ls="-", label="Seeds 42–44"),
                        Line2D([], [], color="#333333", ls="--", label="Seeds 45–47")],
               ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "figS04_validation_qwk_curves")


# S5 ---------------------------------------------------------------------------
def s05_aptos_metrics() -> None:
    rows = read_csv("master_results_per_run.csv")
    metrics = [("APTOS_accuracy", "Accuracy"), ("APTOS_macro_F1", "Macro-F1"), ("QWK_APTOS", "QWK"),
               ("APTOS_mild_precision", "Mild precision"), ("APTOS_mild_recall", "Mild recall"),
               ("APTOS_mild_F1", "Mild F1")]
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    for mi, (key, _) in enumerate(metrics):
        for ci, cell in enumerate(CELLS):
            ys = [float(r[key]) for r in rows if r["cell"] == cell]
            seed_dots(ax, mi + (ci - 1.5) * 0.18, ys, CELL_COLOR[cell], spread=0.04)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("APTOS test score")
    ax.set_ylim(0, 1)
    cell_legend(ax, loc="lower left")
    save(fig, "figS05_aptos_metrics")


# S6 ---------------------------------------------------------------------------
def s06_domain_drop() -> None:
    rows = read_csv("table_v3_hybrid_qwk_n6_per_seed.csv")
    q = {(r["cell"], int(r["seed"]), r["domain"]): float(r["qwk"]) for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), sharey=True)
    for ax, dom, letter in zip(axes, DOMAINS[1:], "abc"):
        for ci, cell in enumerate(CELLS):
            drops = [q[(cell, s, "APTOS")] - q[(cell, s, dom)] for s in SEEDS6]
            seed_dots(ax, ci, drops, CELL_COLOR[cell])
        ax.set_xticks(range(4))
        ax.set_xticklabels(CELLS)
        ax.set_xlabel(f"({letter}) APTOS minus {dom}")
    axes[0].set_ylabel("QWK drop")
    fig.tight_layout()
    save(fig, "figS06_domain_shift_drop")


# S7 ---------------------------------------------------------------------------
def s07_bootstrap_ci() -> None:
    rows = read_csv("table_C_qwk_bootstrap_ci.csv")
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.4), sharey=True)
    for ax, dom, letter in zip(axes, DOMAINS, "abcd"):
        for ci, cell in enumerate(CELLS):
            sub = sorted((r for r in rows if r["cell"] == cell and r["domain"] == dom), key=lambda r: int(r["seed"]))
            for k, r in enumerate(sub):
                x = ci + (k - 1) * 0.2
                lo, hi, pt = float(r["ci95_low"]), float(r["ci95_high"]), float(r["qwk"])
                ax.plot([x, x], [lo, hi], color=CELL_COLOR[cell], lw=1.0)
                ax.scatter([x], [pt], color=CELL_COLOR[cell], s=10, zorder=3)
        ax.axhline(0, color=OKABE[7], lw=0.6, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(CELLS)
        ax.set_xlabel(f"({letter}) {dom}")
    axes[0].set_ylabel("QWK with 95% bootstrap CI")
    fig.tight_layout()
    save(fig, "figS07_qwk_bootstrap_ci")


# S8 ---------------------------------------------------------------------------
def s08_referable() -> None:
    rows = read_csv("table_B_referable_dr.csv")
    metrics = [("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("auc", "AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.0), sharey=True)
    for ax, (key, name), letter in zip(axes, metrics, "abc"):
        for di, dom in enumerate(DOMAINS):
            for ci, cell in enumerate(CELLS):
                ys = [float(r[key]) for r in rows if r["cell"] == cell and r["domain"] == dom
                      and r[key] not in ("", "nan")]
                if ys:
                    seed_dots(ax, di + (ci - 1.5) * 0.18, ys, CELL_COLOR[cell], spread=0.04)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"], rotation=30, ha="right")
        ax.set_xlabel(f"({letter}) {name}")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Referable DR (grade ≥ 2)")
    fig.legend(handles=[mpatches.Patch(color=CELL_COLOR[c], label=CELL_LABEL[c]) for c in CELLS],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "figS08_referable_dr")


# S9 ---------------------------------------------------------------------------
def s09_temperature() -> None:
    ce_cells = ["H1", "H2"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.4), sharey=True)
    reports = {(c, s): read_json(run_dir(c, s) / "eval_report.json") for c in ce_cells for s in SEEDS6}
    for ax, dom, letter in zip(axes, DOMAINS, "abcd"):
        for ci, cell in enumerate(ce_cells):
            for k, s in enumerate(SEEDS6):
                m = reports[(cell, s)][SPLIT[dom]]["metrics"]
                off = (k - 2.5) * 0.04
                x0, x1 = ci - 0.2 + off, ci + 0.2 + off
                ax.plot([x0, x1], [m["ece"], m["ece_temperature_scaled"]], color=CELL_COLOR[cell], lw=0.8)
                ax.scatter([x0], [m["ece"]], color=CELL_COLOR[cell], s=10, marker="o", zorder=3)
                ax.scatter([x1], [m["ece_temperature_scaled"]], color=CELL_COLOR[cell], s=12, marker="^", zorder=3)
        ax.set_xticks(range(len(ce_cells)))
        ax.set_xticklabels(ce_cells)
        ax.set_xlabel(f"({letter}) {dom}")
    axes[0].set_ylabel("ECE (circle) → ECE after T (triangle)")
    fig.tight_layout()
    save(fig, "figS09_temperature_scaling")


# S10–S13 ----------------------------------------------------------------------
def _cm(ax, cm: np.ndarray, label: str, fs: float = 8) -> None:
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0) * 100
    ax.imshow(pct, cmap="viridis", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{pct[i, j]:.0f}%\n{int(cm[i, j])}", ha="center", va="center", fontsize=fs,
                    color="black" if pct[i, j] > 55 else "white")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlabel(f"Predicted grade\n{label}", fontsize=fs + 1)
    ax.set_ylabel("True grade", fontsize=fs + 1)


SCHEDULE_GROUPS = (("seeds 42–44, truncated", (42, 43, 44)), ("seeds 45–47, full cosine", (45, 46, 47)))


def s10_to_s13_confusion() -> None:
    """One figure per test set: rows are schedule groups (summed over the three seeds within a group), columns cells."""
    for k, dom in enumerate(DOMAINS):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6))
        letters = iter("abcdefgh")
        for r, (glabel, seeds) in enumerate(SCHEDULE_GROUPS):
            for c, cell in enumerate(CELLS):
                cm = np.zeros((5, 5), dtype=int)
                for s in seeds:
                    cm += np.asarray(read_json(run_dir(cell, s) / f"{SPLIT[dom]}_metrics.json")["confusion_matrix"], dtype=int)
                _cm(axes[r, c], cm, f"({next(letters)}) {cell}, {glabel}", fs=5.5)
        fig.tight_layout()
        save(fig, f"figS{10 + k:02d}_confusion_{dom.lower().replace('-', '')}")


# S14 --------------------------------------------------------------------------
def s14_messidor_predictions() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    grade_colors = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    labels = []
    ref = None
    bars = []
    for cell in CELLS:
        for s in SEEDS6:
            cm = np.asarray(read_json(run_dir(cell, s) / "messidor2_external_metrics.json")["confusion_matrix"], dtype=int)
            if ref is None:
                ref = cm.sum(axis=1).tolist()
            elif cm.sum(axis=1).tolist() != ref:
                raise SystemExit("Messidor-2 reference labels differ across runs")
            bars.append((f"{cell} seed {s}", cm.sum(axis=0).tolist()))
    bars.insert(0, ("Reference labels", ref))
    for i, (name, counts) in enumerate(bars):
        tot = sum(counts)
        left = 0.0
        for g, n in enumerate(counts):
            ax.barh(i, n / tot, left=left, color=grade_colors[g], edgecolor="white", lw=0.4)
            left += n / tot
        labels.append(name)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of Messidor-2 images")
    ax.legend(handles=[mpatches.Patch(color=grade_colors[g], label=f"Grade {g}") for g in range(5)],
              ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.set_size_inches(6.9, 5.4)
    save(fig, "figS14_messidor_prediction_distribution")


# S15 --------------------------------------------------------------------------
def s15_efficiency() -> None:
    rows = [r for r in read_csv("table_efficiency.csv") if r["params_m"] != "MISSING"]
    names = {"efficientnet_b0": "B0", "mobilevit_s": "MViT-S"}
    labels = [f"{names.get(r['backbone'], r['backbone'])}\n{r['head'].upper()}" for r in rows]
    colors = [OKABE[0] if r["backbone"] == "efficientnet_b0" else OKABE[1] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
    fig.legend(handles=[mpatches.Patch(color=OKABE[0], label="EfficientNet-B0 (B0)"),
                        mpatches.Patch(color=OKABE[1], label="MobileViT-S (MViT-S)")],
               ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    for ax, key, ylab, letter in zip(axes, ["params_m", "flops_g", "batch1_ms_mean"],
                                     ["Parameters (M)", "GFLOPs", "Batch-1 latency (ms)"], "abc"):
        ax.bar(range(len(rows)), [float(r[key]) for r in rows], color=colors)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_xlabel(f"({letter})")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "figS15_efficiency")


def copy_main(stem_main: str, topic: str) -> None:
    """Copy a main-text figure into the supplement under its Version B number."""
    for ext in ("pdf", "png"):
        src = ROOT / "paper" / "figures" / "main" / f"{stem_main}.{ext}"
        if not src.is_file():
            if stem_main in ("fig03_datasets", "fig10_gradcam_mild_panel"):
                require_or_skip(f"Figure S{NUMBER_BY_TOPIC[topic]} (copy of {stem_main})", extra=(src,))
                return
            raise SystemExit(f"MISSING {src.relative_to(ROOT)}")
        print("READ", src.relative_to(ROOT))
        dst = OUT / f"figS{NUMBER_BY_TOPIC[topic]:02d}_{topic}.{ext}"
        dst.write_bytes(src.read_bytes())
        print("WROTE", dst.relative_to(ROOT))


def main() -> None:
    s01_architecture()
    s02_grade_distribution()
    s03_augmentation()
    s04_val_curves()
    s05_aptos_metrics()
    s06_domain_drop()
    s07_bootstrap_ci()
    s08_referable()
    s09_temperature()
    s10_to_s13_confusion()
    s14_messidor_predictions()
    s15_efficiency()
    # Figures moved out of the Version B main text.
    copy_main("fig03_datasets", "dataset_examples")
    copy_main("fig12_matched_b0_vs_hybrid", "matched_b0")
    copy_main("fig09_lodo_vs_h2", "lodo")
    copy_main("fig10_gradcam_mild_panel", "gradcam_mild")
    copy_main("fig13_decomp_clahe", "decomp_clahe")
    copy_main("fig14_res448_vs_224", "res448")
    print("BUILD_SUPPLEMENTARY_OK")


if __name__ == "__main__":
    main()
