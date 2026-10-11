#!/usr/bin/env python3
"""
Build manuscript figures 2–9 from paper/tables CSVs + saved predictions.
Style: no in-figure titles; panel letters; Okabe–Ito; >=8 pt; 3.3/6.9 in;
PDF vector + PNG 600 dpi. Fail loudly if required inputs missing.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.restricted import CHECKPOINTS, require_or_skip  # noqa: E402
OUT = ROOT / "paper" / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "grey": "#999999",
}
CELL_COLOR = {"H1": OKABE["blue"], "H2": OKABE["orange"], "H3": OKABE["green"], "H4": OKABE["pink"]}


def save(fig: plt.Figure, stem: str) -> None:
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", pdf.relative_to(ROOT))
    print("WROTE", png.relative_to(ROOT))


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"MISSING {path.relative_to(ROOT)}")
    print("READ", path.relative_to(ROOT))
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_mean_std(s: str) -> tuple[float, float]:
    s = s.replace("−", "-").strip()
    if "±" not in s:
        return float(s), 0.0
    a, b = s.split("±")
    return float(a), float(b)


def fig02() -> None:
    if not require_or_skip("Figure 2 (fig02_preprocess)", "APTOS"):
        return
    cfg_path = ROOT / "configs/paper/fig02_preprocess.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    print("READ", cfg_path.relative_to(ROOT))
    image_id = cfg["selection"]["image_id"]
    true_grade = int(cfg["selection"]["true_grade"])
    crop_cfg = cfg["selection"]["crop"]
    hits = list((ROOT / "datasets/archive").rglob(f"{image_id}.png"))
    if not hits:
        raise SystemExit(f"MISSING image for fig02 id={image_id}")
    path = hits[0]
    print("READ", path.relative_to(ROOT))
    import cv2

    rgb = np.array(Image.open(path).convert("RGB"))
    green = rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(green)
    h, w = rgb.shape[:2]
    s = int(min(h, w) * float(crop_cfg["size_frac"]))
    cx = int(w * float(crop_cfg["cx_frac"]))
    cy = int(h * float(crop_cfg["cy_frac"]))
    x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
    x1, y1 = min(w, x0 + s), min(h, y0 + s)

    def resize224(arr: np.ndarray) -> np.ndarray:
        im = Image.fromarray(arr if arr.ndim == 3 else np.stack([arr] * 3, axis=-1))
        return np.array(im.resize((224, 224), Image.BILINEAR))

    rgb224 = resize224(rgb)
    clahe224 = resize224(np.stack([enh, enh, enh], axis=-1))
    zoom_rgb = rgb[y0:y1, x0:x1]
    zoom_clahe = np.stack([enh, enh, enh], axis=-1)[y0:y1, x0:x1]

    boxed = rgb.copy()
    cv2.rectangle(boxed, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), max(2, min(h, w) // 200))

    fig = plt.figure(figsize=(6.9, 3.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.1, 1.0], hspace=0.25, wspace=0.15)
    panels = [
        (0, 0, boxed, f"(a) Original\ngrade {true_grade}", None),
        (0, 1, green, "(b) Green", "gray"),
        (0, 2, enh, "(c) CLAHE", "gray"),
        (0, 3, rgb224, "(d) RGB 224", None),
        (0, 4, clahe224, "(e) CLAHE 224", None),
        (1, 1, zoom_rgb, "(f) Zoom RGB", None),
        (1, 3, zoom_clahe, "(g) Zoom CLAHE", None),
    ]
    for r, c, im, lab, cmap in panels:
        ax = fig.add_subplot(gs[r, c])
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap)
        else:
            ax.imshow(im)
        ax.set_xlabel(lab, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    # empty spacer cells
    for r, c in [(1, 0), (1, 2), (1, 4)]:
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    save(fig, "fig02_preprocess")
    # also legacy name referenced by manuscript
    for ext in ("png", "pdf"):
        src = OUT / f"fig02_preprocess.{ext}"
        dst = OUT / f"fig02_preprocess_steps.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())


def _square_crop_fov(im: Image.Image) -> Image.Image:
    """Crop to approximate FOV by thresholding dark borders, then centre square."""
    arr = np.array(im.convert("RGB"))
    gray = arr.mean(axis=2)
    mask = gray > 15
    ys, xs = np.where(mask)
    if len(xs) < 100:
        w, h = im.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return im.crop((left, top, left + s, top + s))
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = im.crop((x0, y0, x1 + 1, y1 + 1))
    w, h = crop.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return crop.crop((left, top, left + s, top + s))


def fig03() -> None:
    if not require_or_skip("dataset example gallery (fig03_datasets, Figure S6)", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD"):
        return
    cfg = yaml.safe_load((ROOT / "configs/paper/fig03_datasets.yaml").read_text())
    print("READ configs/paper/fig03_datasets.yaml")
    rng = random.Random(int(cfg["selection"]["rule_seed"]))
    grades = [int(g) for g in cfg["selection"]["grades"]]

    # Build candidate pools from available labels
    pools: dict[str, dict[int, list[tuple[Path, int, tuple[int, int]]]]] = defaultdict(lambda: defaultdict(list))

    # APTOS test
    import pandas as pd

    aptos_csv = ROOT / "datasets/archive/test.csv"
    if not aptos_csv.is_file():
        raise SystemExit("MISSING datasets/archive/test.csv")
    print("READ", aptos_csv.relative_to(ROOT))
    df = pd.read_csv(aptos_csv)
    id_col = "id_code" if "id_code" in df.columns else df.columns[0]
    lab_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]
    for _, row in df.iterrows():
        g = int(row[lab_col])
        stem = str(row[id_col])
        hits = list((ROOT / "datasets/archive").rglob(f"{stem}.png"))
        if hits:
            im = Image.open(hits[0])
            pools["APTOS"][g].append((hits[0], g, im.size))

    # IDRiD
    for split_csv, img_dir in [
        (
            ROOT / "datasets/idrid/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
            ROOT / "datasets/idrid/B. Disease Grading/1. Original Images/a. Training Set",
        ),
        (
            ROOT / "datasets/idrid/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
            ROOT / "datasets/idrid/B. Disease Grading/1. Original Images/b. Testing Set",
        ),
    ]:
        if not split_csv.is_file():
            continue
        print("READ", split_csv.relative_to(ROOT))
        dfi = pd.read_csv(split_csv)
        ic = dfi.columns[0]
        lc = [c for c in dfi.columns if "grade" in c.lower() or "retinopathy" in c.lower()]
        lc = lc[0] if lc else dfi.columns[1]
        for _, row in dfi.iterrows():
            stem = str(row[ic])
            g = int(row[lc])
            for ext in (".jpg", ".png", ".JPG"):
                p = img_dir / f"{stem}{ext}"
                if p.is_file():
                    im = Image.open(p)
                    pools["IDRiD"][g].append((p, g, im.size))
                    break

    # Messidor-2
    mess_csv = ROOT / "Messidor " / "messidor_data.csv"
    if mess_csv.is_file():
        print("READ", mess_csv.relative_to(ROOT))
        dfm = pd.read_csv(mess_csv)
        for _, row in dfm.iterrows():
            if int(row.get("adjudicated_gradable", 1)) != 1:
                continue
            g = int(row["adjudicated_dr_grade"])
            name = str(row["image_id"])
            hits = list((ROOT / "Messidor ").rglob(name + ".*"))
            if not hits:
                hits = list((ROOT / "Messidor ").rglob(name))
            if hits:
                im = Image.open(hits[0])
                pools["Messidor-2"][g].append((hits[0], g, im.size))

    # DeepDRiD regular fundus (same label rule as src/dataset.py)
    deep_root = ROOT / "datasets/DeepDRiD" / "regular_fundus_images"
    for csvp in [
        deep_root / "regular-fundus-training" / "regular-fundus-training.csv",
        deep_root / "regular-fundus-validation" / "regular-fundus-validation.csv",
    ]:
        if not csvp.is_file():
            continue
        print("READ", csvp.relative_to(ROOT))
        dfd = pd.read_csv(csvp)
        base = csvp.parent
        for _, row in dfd.iterrows():
            image_id = str(row["image_id"])
            if "_l" in image_id:
                g_raw = row.get("left_eye_DR_Level")
            elif "_r" in image_id:
                g_raw = row.get("right_eye_DR_Level")
            else:
                continue
            if pd.isna(g_raw):
                continue
            g = int(g_raw)
            patient = str(row.get("patient_id", ""))
            rel = str(row.get("image_path", "")).replace("\\", "/").lstrip("/")
            candidates = []
            if rel:
                candidates.append(deep_root / Path(rel).name)
                candidates.append(base / Path(rel).name)
                parts = Path(rel).parts
                if len(parts) >= 2:
                    candidates.append(base / parts[-2] / parts[-1])
                    candidates.append(base / "Images" / parts[-2] / parts[-1])
            if patient:
                candidates.append(base / "Images" / patient / f"{image_id}.jpg")
                candidates.append(base / patient / f"{image_id}.jpg")
            candidates.append(base / f"{image_id}.jpg")
            for p in candidates:
                if p.is_file():
                    im = Image.open(p)
                    pools["DeepDRiD"][g].append((p, g, im.size))
                    break
            else:
                hits = list(base.rglob(f"{image_id}.jpg"))
                if hits:
                    im = Image.open(hits[0])
                    pools["DeepDRiD"][g].append((hits[0], g, im.size))
    if not pools["DeepDRiD"]:
        raise SystemExit("MISSING DeepDRiD grade-labelled images for fig03")

    picks = []
    selection_log = []
    for ds in ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]:
        for g in grades:
            cands = pools[ds].get(g, [])
            if not cands:
                raise SystemExit(f"MISSING fig03 candidates dataset={ds} grade={g}")
            choice = rng.choice(cands)
            picks.append((ds, g, choice))
            selection_log.append({"dataset": ds, "grade": g, "path": str(choice[0].relative_to(ROOT)), "native_wh": list(choice[2])})

    (OUT / "fig03_datasets_selection.json").write_text(json.dumps(selection_log, indent=2))
    print("WROTE", (OUT / "fig03_datasets_selection.json").relative_to(ROOT))

    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.6))
    letters = "abcdefgh"
    for i, (ds, g, (path, _, size)) in enumerate(picks):
        r, c = divmod(i, 4)
        ax = axes[r][c]
        im = _square_crop_fov(Image.open(path).convert("RGB"))
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"({letters[i]}) {ds}\ngrade {g} · {size[0]}×{size[1]}", fontsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    fig.tight_layout()
    save(fig, "fig03_datasets")
    for ext in ("png", "pdf"):
        src = OUT / f"fig03_datasets.{ext}"
        dst = OUT / f"fig03_dataset_gallery.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())


def fig04() -> None:
    """Cross-entropy vs threshold head, including the rank-consistency failure."""
    fig = plt.figure(figsize=(6.9, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.7], hspace=0.35, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    ax = ax_a
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.5, "(a) Cross-entropy head", fontsize=9, fontweight="bold")
    ax.add_patch(mpatches.FancyBboxPatch((0.4, 2.0), 2.0, 2.2, boxstyle="round,pad=0.04",
                                         facecolor="#EEEEEE", edgecolor="#333"))
    ax.text(1.4, 3.1, "Backbone", ha="center", fontsize=8)
    for i, g in enumerate(["z₀", "z₁", "z₂", "z₃", "z₄"]):
        y = 4.6 - i * 0.7
        ax.add_patch(mpatches.Rectangle((3.2, y - 0.22), 1.6, 0.44, facecolor=OKABE["sky"], edgecolor="#333", lw=0.6))
        ax.text(4.0, y, g, ha="center", va="center", fontsize=8)
    ax.text(6.8, 3.2, "softmax\npred = argmax\nvalid class\nprobabilities", ha="center", fontsize=8)

    ax = ax_b
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.5, "(b) Threshold head (MW-threshold)", fontsize=9, fontweight="bold")
    ax.add_patch(mpatches.FancyBboxPatch((0.4, 2.0), 2.0, 2.2, boxstyle="round,pad=0.04",
                                         facecolor="#EEEEEE", edgecolor="#333"))
    ax.text(1.4, 3.1, "Backbone", ha="center", fontsize=8)
    for i, (lab, w) in enumerate([("z₀: y>0", "×2"), ("z₁: y>1", "×2"), ("z₂: y>2", "×1"), ("z₃: y>3", "×1")]):
        y = 4.4 - i * 0.75
        ax.add_patch(mpatches.Rectangle((3.0, y - 0.25), 2.4, 0.5, facecolor=OKABE["orange"], edgecolor="#333", lw=0.6))
        ax.text(4.2, y, f"{lab}  {w}", ha="center", va="center", fontsize=8)
    ax.text(7.4, 3.8, "decoded grade =\n# thresholds with\nσ(z) > 0.5", ha="center", fontsize=8)
    ax.text(7.4, 1.7, "mild sample\nweight = 2\nif y=1", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=OKABE["vermillion"]))

    ax = ax_c
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.text(0.2, 3.0, "(c) Rank consistency of the threshold head", fontsize=9, fontweight="bold")
    # Ideal ordered cumulatives.
    ax.plot([1.0, 2.2, 3.4, 4.6], [2.4, 1.9, 1.3, 0.7], "-o", color=OKABE["green"], ms=4, lw=1.2)
    ax.text(2.8, 2.55, "Intended: σ(z₀) ≥ σ(z₁) ≥ σ(z₂) ≥ σ(z₃)", ha="center", color=OKABE["green"], fontsize=8)
    # Observed non-monotonic example (schematic).
    ax.plot([5.4, 6.6, 7.8, 9.0], [1.6, 2.3, 1.1, 1.8], "-o", color=OKABE["vermillion"], ms=4, lw=1.2)
    ax.text(7.2, 2.55, "Observed: non-monotonic cumulatives", ha="center", color=OKABE["vermillion"], fontsize=8)
    ax.text(5.0, 0.35,
            "Decoded grades remain usable for QWK and recall. Derived class probabilities require clipping "
            "and renormalisation and are not interpreted as calibrated graded beliefs; ECE is reported for "
            "H1/H2 only.",
            ha="center", va="center", fontsize=7.5, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "fig04_ce_vs_coral_heads")
    # Version B main-text figure 3 uses this panel set.
    for ext in ("png", "pdf"):
        src = OUT / f"fig04_ce_vs_coral_heads.{ext}"
        (OUT / f"fig03_threshold_head.{ext}").write_bytes(src.read_bytes())


def fig05() -> None:
    rows = read_csv(ROOT / "paper/tables/table_hybrid_qwk_per_seed.csv")
    domains = ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
    cells = ["H1", "H2", "H3", "H4"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.6), sharey=True)
    for ax, dom, letter in zip(axes, domains, "abcd"):
        for i, cell in enumerate(cells):
            ys = [float(r["qwk"]) for r in rows if r["cell"] == cell and r["domain"] == dom]
            xs = np.full(len(ys), i) + (np.arange(len(ys)) - 1) * 0.08
            ax.scatter(xs, ys, s=18, color=CELL_COLOR[cell], zorder=3, edgecolors="white", linewidths=0.3)
            ax.errorbar(i, float(np.mean(ys)), yerr=float(np.std(ys, ddof=1)), fmt="o", color=CELL_COLOR[cell],
                        markersize=5, capsize=2, elinewidth=0.8, zorder=2)
        ax.axhline(0, color=OKABE["grey"], lw=0.6, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(cells, fontsize=8)
        ax.set_xlabel(f"({letter}) {dom}", fontsize=8)
        ax.tick_params(labelsize=8)
        if letter == "a":
            ax.set_ylabel("QWK", fontsize=8)
    fig.tight_layout()
    save(fig, "fig05_qwk_dots")
    # manuscript still references heatmap name — write alias
    for ext in ("png", "pdf"):
        src = OUT / f"fig05_qwk_dots.{ext}"
        dst = OUT / f"fig10_hybrid_qwk_heatmap.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())


def _reliability(y_true, probs, n_bins=15):
    yt = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == yt).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_n = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        if not np.any(mask):
            bin_acc.append(np.nan)
            bin_conf.append((lo + hi) / 2)
            bin_n.append(0)
            continue
        bin_acc.append(correct[mask].mean())
        bin_conf.append(conf[mask].mean())
        bin_n.append(int(mask.sum()))
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_n)


def fig06() -> None:
    # Cross-entropy cells only: the threshold head's class probabilities are not interpretable.
    specs = [
        ("H1", "aptos_test", "APTOS"),
        ("H2", "aptos_test", "APTOS"),
        ("H1", "messidor2_external", "Messidor-2"),
        ("H2", "messidor2_external", "Messidor-2"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.8), gridspec_kw={"height_ratios": [2.2, 0.9]})
    letters = "abcd"
    for col, (cell, split, dom) in enumerate(specs):
        yt_all, pr_all = [], []
        eces = []
        for seed in (42, 43, 44, 45, 46, 47):
            run = f"hybrid_{cell.lower()}_s{seed}_{'v1' if seed <= 44 else 'v3'}"
            pred_path = ROOT / "results" / run / f"{split}_predictions.json"
            met_path = ROOT / "results" / run / f"{split}_metrics.json"
            if not pred_path.is_file():
                raise SystemExit(f"MISSING {pred_path.relative_to(ROOT)}")
            print("READ", pred_path.relative_to(ROOT))
            p = json.loads(pred_path.read_text())
            m = json.loads(met_path.read_text())
            print("READ", met_path.relative_to(ROOT))
            yt_all.extend(p["y_true"])
            pr_all.extend(p["probs"])
            eces.append(float(m["ece"]))
        xs, ys, ns = _reliability(yt_all, pr_all)
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], "--", color=OKABE["grey"], lw=0.8)
        ok = ~np.isnan(ys)
        ax.plot(xs[ok], ys[ok], "-o", color=CELL_COLOR[cell], ms=3, lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"({letters[col]}) {cell} · {dom}\nECE {np.mean(eces):.3f}±{np.std(eces, ddof=1):.3f}", fontsize=8)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=8)
        ax2 = axes[1, col]
        ax2.bar(np.linspace(0.5 / 15, 1 - 0.5 / 15, 15), ns, width=1 / 15 * 0.9, color=CELL_COLOR[cell], alpha=0.8)
        ax2.set_xlim(0, 1)
        ax2.tick_params(labelsize=8)
        ax2.set_xlabel("Confidence", fontsize=8)
        if col == 0:
            ax2.set_ylabel("Count", fontsize=8)
    fig.tight_layout()
    save(fig, "fig06_reliability")
    for ext in ("png", "pdf"):
        src = OUT / f"fig06_reliability.{ext}"
        dst = OUT / f"fig07_reliability_h2_h4.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())


def _cm_plot(ax, cm, letter_title: str) -> None:
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) * 100
    im = ax.imshow(pct, cmap="viridis", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            val = int(cm[i, j])
            txt = f"{pct[i, j]:.0f}%\n({val})"
            color = "black" if pct[i, j] > 55 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_ylabel("True grade", fontsize=8)
    ax.set_xlabel(f"Predicted grade\n{letter_title}", fontsize=8)
    ax.tick_params(labelsize=8)


def fig07() -> None:
    panels = [
        ("hybrid_h2_s42_v1", "aptos_test", "(a) MobileViT-S H2 (CLAHE, CE)"),
        ("hybrid_h4_s42_v1", "aptos_test", "(b) MobileViT-S H4 (CLAHE, MW-threshold)"),
        ("matched_b0_h2_s42_v3", "aptos_test", "(c) EfficientNet-B0 (CLAHE, CE)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9))
    for ax, (run, split, title) in zip(axes, panels):
        path = ROOT / "results" / run / f"{split}_metrics.json"
        if not path.is_file():
            raise SystemExit(f"MISSING {path.relative_to(ROOT)}")
        print("READ", path.relative_to(ROOT))
        m = json.loads(path.read_text())
        _cm_plot(ax, m["confusion_matrix"], title)
    fig.tight_layout()
    save(fig, "fig07_confusion_aptos")
    for ext in ("png", "pdf"):
        src = OUT / f"fig07_confusion_aptos.{ext}"
        dst = OUT / f"fig05_confusion_aptos_h2_h4_b0.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())


def fig08() -> None:
    panels = [
        ("hybrid_h2_s42_v1", "(a) H2 (CLAHE, CE)"),
        ("hybrid_h4_s42_v1", "(b) H4 (CLAHE, MW-threshold)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    for ax, (run, title) in zip(axes, panels):
        path = ROOT / "results" / run / "messidor2_external_metrics.json"
        print("READ", path.relative_to(ROOT))
        m = json.loads(path.read_text())
        _cm_plot(ax, m["confusion_matrix"], title)
    fig.tight_layout()
    save(fig, "fig08_messidor_confusion")
    for ext in ("png", "pdf"):
        src = OUT / f"fig08_messidor_confusion.{ext}"
        dst = OUT / f"fig06_messidor_confusion.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())


def fig09() -> None:
    rows = read_csv(ROOT / "paper/tables/table_lodo_vs_h2_per_seed.csv")
    holdouts = ["IDRiD", "Messidor-2", "DeepDRiD"]
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    for i, h in enumerate(holdouts):
        sub = [r for r in rows if r["holdout"] == h]
        for r in sub:
            ax.plot([i - 0.15, i + 0.15], [float(r["h2_qwk"]), float(r["lodo_qwk"])],
                    color=OKABE["grey"], lw=0.8, zorder=1)
            ax.scatter([i - 0.15], [float(r["h2_qwk"])], color=OKABE["blue"], s=22, zorder=2)
            ax.scatter([i + 0.15], [float(r["lodo_qwk"])], color=OKABE["vermillion"], s=22, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(holdouts, fontsize=8)
    ax.set_ylabel("QWK", fontsize=8)
    ax.legend(
        handles=[
            mpatches.Patch(color=OKABE["blue"], label="H2 APTOS-only"),
            mpatches.Patch(color=OKABE["vermillion"], label="LODO"),
        ],
        fontsize=8,
        frameon=False,
        loc="lower right",
    )
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig09_lodo_vs_h2")


def main() -> None:
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    fig08()
    fig09()
    print("BUILD_FIGS_2_TO_9_OK")


if __name__ == "__main__":
    main()
