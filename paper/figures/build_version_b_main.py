#!/usr/bin/env python3
"""
Version B main result figures (PDF + PNG 600 dpi) under paper/figures/main/.

Figure 4: schedule-stratified QWK (panels H1–H4; triangles 42–44, circles 45–47).
Figure 5: schedule-stratified grade-1 recall (same marker convention).
Figure 6: H1/H2 calibration only — ECE before and after temperature scaling,
          with schedule-group markers; no threshold-head calibration.

All numbers are read from paper/tables CSVs. No pooled six-seed headline mean.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "paper" / "tables"

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "grey": "#999999",
}
CELL_COLOR = {"H1": OKABE["blue"], "H2": OKABE["orange"], "H3": OKABE["green"], "H4": OKABE["pink"]}
CELLS = ["H1", "H2", "H3", "H4"]
DOMAINS = ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
SEEDS_TRUNC = (42, 43, 44)
SEEDS_FULL = (45, 46, 47)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_csv(name: str) -> list[dict[str, str]]:
    p = TABLES / name
    if not p.is_file():
        raise SystemExit(f"MISSING {p.relative_to(ROOT)}")
    print("READ", p.relative_to(ROOT))
    with p.open() as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, stem: str) -> None:
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print("WROTE", path.relative_to(ROOT))
    plt.close(fig)


def schedule_legend(fig, y: float = -0.02) -> None:
    handles = [
        plt.Line2D([0], [0], marker="^", color="k", linestyle="none", markersize=6,
                   label="Seeds 42–44 (truncated schedule)"),
        plt.Line2D([0], [0], marker="o", color="k", linestyle="none", markersize=6,
                   label="Seeds 45–47 (full cosine schedule)"),
        plt.Line2D([0], [0], color=OKABE["grey"], lw=1.6, solid_capstyle="butt",
                   label="Schedule-group mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, y), fontsize=8)


def plot_schedule_stratified(ax, x: float, ys_trunc: list[float], ys_full: list[float],
                             color: str, spread: float = 0.07) -> None:
    """Individual seeds by schedule group; group means as short horizontal bars. No pooled mean."""
    yt = np.asarray(ys_trunc, dtype=float)
    yf = np.asarray(ys_full, dtype=float)
    if len(yt) != 3 or len(yf) != 3:
        raise SystemExit(f"Expected 3 seeds per schedule group, got {len(yt)}/{len(yf)}")
    xt = x - 0.14 + (np.arange(3) - 1) * spread
    xf = x + 0.14 + (np.arange(3) - 1) * spread
    ax.scatter(xt, yt, s=28, marker="^", color=color, edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(xf, yf, s=28, marker="o", color=color, edgecolors="white", linewidths=0.4, zorder=3)
    ax.plot([x - 0.22, x - 0.06], [yt.mean(), yt.mean()], color=color, lw=1.8, solid_capstyle="butt", zorder=2)
    ax.plot([x + 0.06, x + 0.22], [yf.mean(), yf.mean()], color=color, lw=1.8, solid_capstyle="butt", zorder=2)


def fig04_schedule_qwk() -> None:
    rows = read_csv("master_results_per_run.csv")
    by = {(r["cell"], int(r["seed"])): r for r in rows}
    for c in CELLS:
        for s in SEEDS_TRUNC + SEEDS_FULL:
            if (c, s) not in by:
                raise SystemExit(f"MISSING master row {c} seed {s}")

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), sharey=True)
    for ax, cell, letter in zip(axes.ravel(), CELLS, "abcd"):
        color = CELL_COLOR[cell]
        for di, dom in enumerate(DOMAINS):
            col = f"QWK_{dom}"
            yt = [float(by[(cell, s)][col]) for s in SEEDS_TRUNC]
            yf = [float(by[(cell, s)][col]) for s in SEEDS_FULL]
            plot_schedule_stratified(ax, di, yt, yf, color)
        ax.axhline(0, color=OKABE["grey"], lw=0.6, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"], rotation=18, ha="right")
        ax.set_xlabel(f"({letter}) {cell}")
        if letter in "ac":
            ax.set_ylabel("QWK")
        ax.set_ylim(-0.25, 1.05)
    schedule_legend(fig, y=-0.01)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig04_schedule_qwk")
    # Keep manuscript-friendly aliases used by the rebuild chain.
    for ext in ("png", "pdf"):
        src = OUT / f"fig04_schedule_qwk.{ext}"
        (OUT / f"fig11_hybrid_qwk_n6.{ext}").write_bytes(src.read_bytes())


def fig05_schedule_g1() -> None:
    rows = read_csv("external_grade1_recall_per_run.csv")
    by = {(r["cell"], int(r["seed"])): r for r in rows}
    for c in CELLS:
        for s in SEEDS_TRUNC + SEEDS_FULL:
            if (c, s) not in by:
                raise SystemExit(f"MISSING grade-1 row {c} seed {s}")

    # Confirm the fixed grade-1 denominators that appear in the caption.
    n_by_dom = {}
    for dom in DOMAINS:
        ns = {int(r[f"{dom}_grade1_n"]) for r in rows}
        if len(ns) != 1:
            raise SystemExit(f"grade-1 n differs for {dom}: {ns}")
        n_by_dom[dom] = ns.pop()
    print("grade-1 denominators", n_by_dom)

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), sharey=True)
    for ax, cell, letter in zip(axes.ravel(), CELLS, "abcd"):
        color = CELL_COLOR[cell]
        for di, dom in enumerate(DOMAINS):
            col = f"{dom}_grade1_recall"
            yt = [float(by[(cell, s)][col]) for s in SEEDS_TRUNC]
            yf = [float(by[(cell, s)][col]) for s in SEEDS_FULL]
            plot_schedule_stratified(ax, di, yt, yf, color)
        ax.set_xticks(range(4))
        labels = [f"{d}\n(n={n_by_dom[d]})" for d in DOMAINS]
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel(f"({letter}) {cell}")
        if letter in "ac":
            ax.set_ylabel("Grade-1 recall")
        ax.set_ylim(-0.05, 1.05)
    schedule_legend(fig, y=-0.01)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig05_schedule_mild_recall")


def fig06_ce_calibration() -> None:
    """ECE before → after temperature scaling for H1/H2 only, schedule-stratified."""
    rows = read_csv("ece_ce_cells_n6_per_seed.csv")
    by = {(r["cell"], int(r["seed"]), r["domain"]): r for r in rows}
    for cell in ("H1", "H2"):
        for s in SEEDS_TRUNC + SEEDS_FULL:
            for d in DOMAINS:
                if (cell, s, d) not in by:
                    raise SystemExit(f"MISSING ECE row {cell} {s} {d}")

    fig, axes = plt.subplots(2, 4, figsize=(6.9, 4.4), sharey=True)
    for row_i, cell in enumerate(("H1", "H2")):
        color = CELL_COLOR[cell]
        for col_i, (dom, letter) in enumerate(zip(DOMAINS, "abcdefgh"[row_i * 4:(row_i + 1) * 4])):
            ax = axes[row_i, col_i]
            for group, seeds, marker, x0 in (
                ("trunc", SEEDS_TRUNC, "^", 0.0),
                ("full", SEEDS_FULL, "o", 1.0),
            ):
                for j, s in enumerate(seeds):
                    r = by[(cell, s, dom)]
                    e0 = float(r["ece"])
                    e1 = float(r["ece_T"])
                    x = x0 + (j - 1) * 0.12
                    ax.plot([x, x], [e0, e1], color=color, lw=0.8, alpha=0.7, zorder=2)
                    ax.scatter([x], [e0], s=26, marker=marker, color=color, edgecolors="white",
                               linewidths=0.4, zorder=3)
                    ax.scatter([x], [e1], s=26, marker=marker, facecolors="white", edgecolors=color,
                               linewidths=1.0, zorder=4)
            # Group means for before (filled bar) and after (open bar).
            for seeds, x0 in ((SEEDS_TRUNC, 0.0), (SEEDS_FULL, 1.0)):
                m0 = np.mean([float(by[(cell, s, dom)]["ece"]) for s in seeds])
                m1 = np.mean([float(by[(cell, s, dom)]["ece_T"]) for s in seeds])
                ax.plot([x0 - 0.18, x0 + 0.18], [m0, m0], color=color, lw=1.6, zorder=2)
                ax.plot([x0 - 0.18, x0 + 0.18], [m1, m1], color=color, lw=1.6, ls=":", zorder=2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["42–44", "45–47"], fontsize=7)
            ax.set_xlabel(f"({letter}) {cell} · {dom}")
            if col_i == 0:
                ax.set_ylabel("ECE")
            ax.set_ylim(0, 0.65)
            ax.axhline(0, color=OKABE["grey"], lw=0.4)

    handles = [
        plt.Line2D([0], [0], marker="^", color="k", linestyle="none", markersize=6,
                   label="Seeds 42–44 (truncated)"),
        plt.Line2D([0], [0], marker="o", color="k", linestyle="none", markersize=6,
                   label="Seeds 45–47 (full cosine)"),
        plt.Line2D([0], [0], marker="o", color="k", linestyle="none", markersize=6,
                   markerfacecolor="k", label="ECE before T"),
        plt.Line2D([0], [0], marker="o", color="k", linestyle="none", markersize=6,
                   markerfacecolor="white", markeredgecolor="k", label="ECE after T"),
        plt.Line2D([0], [0], color=OKABE["grey"], lw=1.6, label="Group mean (before)"),
        plt.Line2D([0], [0], color=OKABE["grey"], lw=1.6, ls=":", label="Group mean (after)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, "fig06_ce_calibration")
    for ext in ("png", "pdf"):
        src = OUT / f"fig06_ce_calibration.{ext}"
        (OUT / f"fig06_reliability.{ext}").write_bytes(src.read_bytes())


def main() -> None:
    fig04_schedule_qwk()
    fig05_schedule_g1()
    fig06_ce_calibration()
    print("BUILD_VERSION_B_MAIN_OK")


if __name__ == "__main__":
    main()
