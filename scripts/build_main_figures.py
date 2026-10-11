#!/usr/bin/env python3
"""Build main-text figures for the revised manuscript (no fabricated metrics)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.restricted import CHECKPOINTS, require_or_skip  # noqa: E402
OUT = ROOT / "paper" / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)


def save(fig, name: str) -> Path:
    p = OUT / name
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("SAVED", p.relative_to(ROOT))
    return p


def qwk(rid: str, split: str) -> float:
    return float(
        json.loads((ROOT / "results" / rid / f"{split}_metrics.json").read_text())[
            "quadratic_weighted_kappa"
        ]
    )


def metrics(rid: str, split: str) -> dict:
    return json.loads((ROOT / "results" / rid / f"{split}_metrics.json").read_text())


def predictions(rid: str, split: str) -> dict:
    return json.loads((ROOT / "results" / rid / f"{split}_predictions.json").read_text())


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Study overview: locked train/selection rule and evaluation branches", fontsize=12, pad=8)

    def box(x, y, w, h, text, fc="#EEF3F8", ec="#2F4A6D"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                             facecolor=fc, edgecolor=ec, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, wrap=True)

    box(0.2, 3.4, 2.2, 1.2, "APTOS train\n(n=2930)\nICDR 0–4")
    box(2.8, 3.4, 2.6, 1.2, "2×2 hybrids × 3 seeds\nRGB/CLAHE × CE/mild-CORAL\nMobileViT-S")
    box(5.8, 3.4, 2.4, 1.2, "Select checkpoint\non APTOS val QWK\n(never test/external)")
    box(8.6, 3.4, 3.0, 1.2, "Evaluate\nAPTOS test + IDRiD\n+ Messidor-2 + DeepDRiD")

    for x0, x1 in [(2.4, 2.8), (5.4, 5.8), (8.2, 8.6)]:
        ax.annotate("", xy=(x1, 4.0), xytext=(x0, 4.0),
                    arrowprops=dict(arrowstyle="->", color="#2F4A6D", lw=1.4))

    box(0.2, 1.5, 3.4, 1.3, "CNN baseline\nEfficientNet-B0\nmatched preprocess×loss\n(APTOS + IDRiD seed table)", fc="#E8F5E9", ec="#2E7D32")
    box(4.0, 1.5, 3.6, 1.3, "Transformer ablation\nSwin-Tiny\ngreen-CLAHE × CE\nseeds 42/43/44", fc="#FFF3E0", ec="#EF6C00")
    box(8.0, 1.5, 3.6, 1.3, "LODO ablation (optional)\nAPTOS + non-held-out externals\nval=APTOS; eval held-out\n(not primary train rule)", fc="#FCE4EC", ec="#C2185B")

    ax.annotate("", xy=(1.9, 2.8), xytext=(1.9, 3.4),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
    ax.annotate("", xy=(5.8, 2.8), xytext=(7.0, 3.4),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
    ax.annotate("", xy=(9.8, 2.8), xytext=(10.0, 3.4),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
    ax.text(6.0, 0.55, "Primary claim path is the top row (APTOS-only training). Lower boxes are controlled comparisons.",
            ha="center", fontsize=8, style="italic", color="#444")
    save(fig, "fig01_study_pipeline.png")


def fig_preprocess_pipeline() -> None:
    """Original → green → CLAHE → stack, with mild crop if available."""
    if not require_or_skip("legacy preprocessing panel (fig02_preprocess_steps)", "APTOS"):
        return
    # Prefer a known mild APTOS test image from Grad-CAM set
    candidates = [
        ROOT / "datasets/archive/test_images/test_images/ead23cc922ed.png",
        ROOT / "datasets/archive/test_images/ead23cc922ed.png",
    ]
    # also search
    found = None
    for c in candidates:
        if c.is_file():
            found = c
            break
    if found is None:
        hits = list((ROOT / "datasets/archive").rglob("ead23cc922ed.png"))
        found = hits[0] if hits else None
    if found is None:
        # fallback any test image
        hits = list((ROOT / "datasets/archive").rglob("*.png"))[:5]
        found = hits[0] if hits else None
    if found is None:
        print("SKIP preprocess fig — no image")
        return

    import cv2
    rgb = np.array(Image.open(found).convert("RGB"))
    green = rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(green)
    stacked = np.stack([enh, enh, enh], axis=-1)

    # centre crop zoom
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    s = min(h, w) // 5
    zoom = rgb[cy - s : cy + s, cx - s : cx + s]

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.2))
    titles = ["Original RGB", "Green channel", "CLAHE on green", "3-channel stack", "Centre crop (RGB)"]
    imgs = [rgb, green, enh, stacked, zoom]
    cmaps = [None, "gray", "gray", None, None]
    for ax, im, t, cm in zip(axes, imgs, titles, cmaps):
        if im.ndim == 2:
            ax.imshow(im, cmap=cm)
        else:
            ax.imshow(im)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Green-CLAHE frontend (clip=2.0, 8×8). Example: {found.stem} (APTOS mild case when available).", fontsize=10)
    fig.tight_layout()
    save(fig, "fig02_preprocess_steps.png")


def fig_dataset_gallery() -> None:
    """One sample per dataset if files exist."""
    if not require_or_skip("legacy dataset gallery (fig03_dataset_gallery)", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD"):
        return
    picks = []
    # APTOS
    for pat in ["datasets/archive/test_images/test_images/*.png", "datasets/archive/test_images/*.png"]:
        hits = list(ROOT.glob(pat))
        if hits:
            picks.append(("APTOS", hits[0]))
            break
    # IDRiD
    hits = list((ROOT / "datasets/idrid").rglob("IDRiD_021*.jpg")) + list((ROOT / "datasets/idrid").rglob("IDRiD_021*.png"))
    if not hits:
        hits = list((ROOT / "datasets/idrid/B. Disease Grading/1. Original Images").rglob("*.jpg"))[:1]
    if hits:
        picks.append(("IDRiD", hits[0]))
    # Messidor
    hits = list((ROOT / "Messidor ").rglob("*.png") )[:1] + list((ROOT / "Messidor ").rglob("*.JPG"))[:1]
    if hits:
        picks.append(("Messidor-2", hits[0]))
    # DeepDRiD
    hits = list((ROOT / "datasets/DeepDRiD/regular_fundus_images").rglob("*.jpg"))[:1]
    if hits:
        picks.append(("DeepDRiD", hits[0]))

    if len(picks) < 2:
        print("SKIP dataset gallery — insufficient images")
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 3.2))
    if len(picks) == 1:
        axes = [axes]
    for ax, (name, path) in zip(axes, picks):
        im = Image.open(path).convert("RGB")
        ax.imshow(im)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle("Domain appearance differs across evaluation sets (illustrative samples).", fontsize=10)
    fig.tight_layout()
    save(fig, "fig03_dataset_gallery.png")


def fig_coral_vs_ce() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("CE head (5 logits)", fontsize=11)
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 2.2), 2.2, 2.0, boxstyle="round,pad=0.05",
                                         facecolor="#E3F2FD", edgecolor="#1565C0"))
    ax.text(1.6, 3.2, "Backbone\nfeatures", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(3.5, 3.2), xytext=(2.8, 3.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    for i, g in enumerate(["P0", "P1", "P2", "P3", "P4"]):
        y = 4.6 - i * 0.7
        ax.add_patch(mpatches.Rectangle((3.6, y - 0.25), 1.8, 0.5, facecolor="#BBDEFB", edgecolor="#1565C0"))
        ax.text(4.5, y, g, ha="center", va="center", fontsize=9)
    ax.text(6.8, 3.2, "softmax\n→ argmax grade", ha="center", va="center", fontsize=9)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("CORAL / mild-CORAL head (K−1=4 thresholds)", fontsize=11)
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 2.2), 2.2, 2.0, boxstyle="round,pad=0.05",
                                         facecolor="#FFF3E0", edgecolor="#EF6C00"))
    ax.text(1.6, 3.2, "Backbone\nfeatures", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(3.5, 3.2), xytext=(2.8, 3.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    for i, lab in enumerate(["1{y>0}", "1{y>1}", "1{y>2}", "1{y>3}"]):
        y = 4.4 - i * 0.75
        ax.add_patch(mpatches.Rectangle((3.6, y - 0.28), 2.2, 0.55, facecolor="#FFE0B2", edgecolor="#EF6C00"))
        ax.text(4.7, y, lab, ha="center", va="center", fontsize=8.5)
    ax.text(7.3, 4.0, "Example grade 2\n→ bits 1,1,0,0", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#FFECB3", edgecolor="#EF6C00"))
    ax.text(7.3, 2.2, "decode:\nsum(σ(z_k)>0.5)", ha="center", va="center", fontsize=9)
    ax.text(5.0, 0.45, "Mild-aware CORAL upweights thresholds k=0,1 (weights 2,2,1,1) and mild samples (y=1).",
            ha="center", fontsize=8, style="italic", transform=ax.transData)
    fig.tight_layout()
    save(fig, "fig04_ce_vs_coral_heads.png")


def _plot_cm(ax, cm, title: str) -> None:
    cm = np.asarray(cm, dtype=float)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)


def fig_confusion_main() -> None:
    # Use seed 42 as representative; note in caption that patterns are consistent
    panels = [
        ("hybrid_h2_s42_v1", "aptos_test", "H2 APTOS test (seed 42)"),
        ("hybrid_h4_s42_v1", "aptos_test", "H4 APTOS test (seed 42)"),
    ]
    # B0 green_clahe ce — find a run with confusion
    # Prefer the seed-42 green_clahe×CE B0 run used in the seed summary
    b0_rid = "baseline_efficientnet_b0_green_clahe_20260918_092447"
    b0_path = ROOT / "results" / b0_rid / "aptos_test_metrics.json"
    b0 = None
    if b0_path.is_file():
        m = json.loads(b0_path.read_text())
        if "confusion_matrix" in m:
            b0 = (b0_rid, m["confusion_matrix"])
    fig, axes = plt.subplots(1, 3 if b0 else 2, figsize=(12, 3.8))
    for ax, (rid, sp, title) in zip(axes, panels):
        m = metrics(rid, sp)
        _plot_cm(ax, m["confusion_matrix"], title)
    if b0:
        _plot_cm(axes[2], b0[1], "EfficientNet-B0 green-CLAHE×CE\nAPTOS test (seed 42)")
    fig.suptitle("Confusion matrices on APTOS test (rows = true grade).", fontsize=11)
    fig.tight_layout()
    save(fig, "fig05_confusion_aptos_h2_h4_b0.png")


def fig_messidor_diagnostic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, cell, title in [
        (axes[0], "h2", "H2 on Messidor-2 (seed 42)"),
        (axes[1], "h4", "H4 on Messidor-2 (seed 42)"),
    ]:
        m = metrics(f"hybrid_{cell}_s42_v1", "messidor2_external")
        pred = predictions(f"hybrid_{cell}_s42_v1", "messidor2_external")
        _plot_cm(ax, m["confusion_matrix"], title)
        hist = Counter(pred["y_pred"])
        ax.text(0.5, -0.22, "pred counts: " + ", ".join(f"{k}:{hist[k]}" for k in range(5) if hist.get(k)),
                transform=ax.transAxes, ha="center", fontsize=8)
    fig.suptitle("Messidor-2 failure mode: H2 collapses toward grade 0; H4 spreads but remains poorly ordered.", fontsize=10)
    fig.tight_layout()
    save(fig, "fig06_messidor_confusion.png")


def reliability_curve(y_true, probs, n_bins=15):
    yt = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == yt).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        xs.append(conf[mask].mean())
        ys.append(correct[mask].mean())
        ns.append(mask.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def fig_reliability() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, cell, color in [(axes[0], "h2", "#1565C0"), (axes[1], "h4", "#C62828")]:
        pred = predictions(f"hybrid_{cell}_s42_v1", "aptos_test")
        xs, ys, ns = reliability_curve(pred["y_true"], pred["probs"])
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(xs, ys, "-o", color=color, ms=4)
        ece = metrics(f"hybrid_{cell}_s42_v1", "aptos_test")["ece"]
        ax.set_title(f"{cell.upper()} APTOS test (seed 42)\nECE={ece:.3f}", fontsize=10)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Reliability diagrams (max-probability confidence, 15 bins).", fontsize=11)
    fig.tight_layout()
    save(fig, "fig07_reliability_h2_h4.png")


def fig_gradcam_mild() -> None:
    """Compose mild Grad-CAM panels for H1/H2/H4 if present."""
    mild_id = "ead23cc922ed"
    paths = []
    for cell in ["h1", "h2", "h4"]:
        p = ROOT / f"paper/figures/gradcam/hybrid_{cell}_s42_v1/aptos_test_g1_{mild_id}.png"
        if p.is_file():
            paths.append((cell.upper(), p))
    compare = ROOT / "paper/figures/gradcam/compare/aptos_test_grade1_all_runs.png"
    if not paths and not compare.is_file():
        print("SKIP gradcam fig")
        return
    if compare.is_file() and len(paths) < 2:
        im = Image.open(compare)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.imshow(im)
        ax.axis("off")
        ax.set_title("Grad-CAM across hybrid runs on a fixed APTOS mild (grade 1) case.", fontsize=10)
        save(fig, "fig08_gradcam_mild.png")
        return
    fig, axes = plt.subplots(1, len(paths), figsize=(3.6 * len(paths), 3.5))
    if len(paths) == 1:
        axes = [axes]
    for ax, (lab, p) in zip(axes, paths):
        ax.imshow(Image.open(p))
        ax.set_title(f"{lab} · mild {mild_id}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Grad-CAM on the same APTOS mild image (seed 42). Inspect whether heat overlaps lesions vs disc/borders.", fontsize=9)
    fig.tight_layout()
    save(fig, "fig08_gradcam_mild.png")


def fig_lodo_bars() -> None:
    holdouts = [
        ("idrid", "idrid_external", "IDRiD"),
        ("messidor2", "messidor2_external", "Messidor-2"),
        ("deepdrid", "deepdrid_external", "DeepDRiD"),
    ]
    lodo_m, lodo_s, h2_m, h2_s, labels = [], [], [], [], []
    for h, sp, lab in holdouts:
        labels.append(lab)
        a = [qwk(f"phase_p_lodo_holdout_{h}_s{s}_v1", sp) for s in SEEDS]
        b = [qwk(f"hybrid_h2_s{s}_v1", sp) for s in SEEDS]
        lodo_m.append(np.mean(a)); lodo_s.append(np.std(a, ddof=1))
        h2_m.append(np.mean(b)); h2_s.append(np.std(b, ddof=1))
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.bar(x - w / 2, h2_m, w, yerr=h2_s, capsize=3, label="H2 APTOS-only", color="#4C78A8")
    ax.bar(x + w / 2, lodo_m, w, yerr=lodo_s, capsize=3, label="LODO multi-source", color="#F58518")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out QWK"); ax.set_ylim(0, 1.0)
    ax.set_title("LODO vs APTOS-only H2 (mean±std, n=3 seeds)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig09_lodo_vs_h2.png")


def fig_qwk_heatmap() -> None:
    cells = ["H1", "H2", "H3", "H4"]
    splits = [
        ("aptos_test", "APTOS"),
        ("idrid_external", "IDRiD"),
        ("messidor2_external", "Messidor-2"),
        ("deepdrid_external", "DeepDRiD"),
    ]
    mat = np.zeros((4, 4))
    for i, c in enumerate(cells):
        for j, (sp, _) in enumerate(splits):
            vals = [qwk(f"hybrid_{c.lower()}_s{s}_v1", sp) for s in SEEDS]
            mat[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([s[1] for s in splits])
    ax.set_yticks(range(4)); ax.set_yticklabels(cells)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Hybrid mean QWK by cell and domain (n=3)")
    fig.tight_layout()
    save(fig, "fig10_hybrid_qwk_heatmap.png")


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})
    fig_pipeline()
    fig_preprocess_pipeline()
    fig_dataset_gallery()
    fig_coral_vs_ce()
    fig_confusion_main()
    fig_messidor_diagnostic()
    fig_reliability()
    fig_gradcam_mild()
    fig_lodo_bars()
    fig_qwk_heatmap()
    # index
    files = sorted(p.name for p in OUT.glob("*.png"))
    (OUT / "INDEX.md").write_text(
        "# Main-text figures\n\n" + "\n".join(f"- `{f}`" for f in files) + "\n"
    )
    print("MAIN_FIGURES_OK", len(files))


if __name__ == "__main__":
    main()
