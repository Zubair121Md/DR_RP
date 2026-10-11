#!/usr/bin/env python3
"""Build paper_v3 result figures (PDF + PNG 600 dpi) under paper/figures/main/."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "paper" / "tables"

OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
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


def parse_pm(s: str) -> tuple[float, float]:
    if "±" in s:
        a, b = s.split("±")
        return float(a), float(b)
    return float(s), 0.0


def fig11_qwk_n6() -> None:
    rows = read_csv("table_v3_hybrid_qwk_n6_per_seed.csv")
    domains = ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
    cells = ["H1", "H2", "H3", "H4"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.6), sharey=True)
    for ax, dom, letter in zip(axes, domains, "abcd"):
        for i, cell in enumerate(cells):
            ys = [float(r["qwk"]) for r in rows if r["cell"] == cell and r["domain"] == dom]
            xs = np.linspace(i - 0.12, i + 0.12, len(ys))
            ax.scatter(xs, ys, s=12, color=OKABE[i], zorder=3)
            ax.errorbar(i, np.mean(ys), yerr=np.std(ys, ddof=1), fmt="o", color=OKABE[i],
                        markersize=5, capsize=2, zorder=2)
        ax.axhline(0, color="#888888", lw=0.6, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(cells)
        ax.set_xlabel(f"({letter}) {dom}")
        if letter == "a":
            ax.set_ylabel("QWK")
    fig.tight_layout()
    save(fig, "fig11_hybrid_qwk_n6")


def fig12_matched_b0() -> None:
    rows = read_csv("table_v3_matched_b0_vs_hybrid_per_seed.csv")
    domains = ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
    cells = ["H1", "H2", "H3", "H4"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.7), sharey=True)
    for ax, dom, letter in zip(axes, domains, "abcd"):
        for i, cell in enumerate(cells):
            b0 = [float(r["qwk"]) for r in rows if r["cell"] == cell and r["domain"] == dom and r["model"] == "matched_b0"]
            hy = [float(r["qwk"]) for r in rows if r["cell"] == cell and r["domain"] == dom and r["model"] == "hybrid"]
            ax.scatter([i - 0.12] * 3, b0, s=14, color=OKABE[0], zorder=3)
            ax.scatter([i + 0.12] * 3, hy, s=14, color=OKABE[1], zorder=3)
            ax.errorbar(i - 0.12, np.mean(b0), yerr=np.std(b0, ddof=1), fmt="o", color=OKABE[0], markersize=4, capsize=2)
            ax.errorbar(i + 0.12, np.mean(hy), yerr=np.std(hy, ddof=1), fmt="o", color=OKABE[1], markersize=4, capsize=2)
        ax.axhline(0, color="#888888", lw=0.6, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(cells)
        ax.set_xlabel(f"({letter}) {dom}")
        if letter == "a":
            ax.set_ylabel("QWK")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=OKABE[0], label="Matched EfficientNet-B0"),
                        Patch(color=OKABE[1], label="MobileViT-S")],
               ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig12_matched_b0_vs_hybrid")


def fig13_decomp() -> None:
    rows = read_csv("table_v3_decomp_clahe_per_seed.csv")
    cells = ["H2", "D1", "D2", "H4"]
    labels = ["H2", "D1", "D2", "H4"]
    metrics = [("APTOS", "qwk", "APTOS QWK"), ("APTOS", "mild_recall", "APTOS mild recall"),
               ("APTOS", "ece", "APTOS ECE (CE only)"), ("IDRiD", "qwk", "IDRiD QWK")]
    threshold_cells = {"D1", "H4"}
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.7))
    for ax, (dom, key, title), letter in zip(axes, metrics, "abcd"):
        for i, cell in enumerate(cells):
            if key == "ece" and cell in threshold_cells:
                ax.text(i, 0.5, "n.i.", ha="center", va="center", transform=ax.get_xaxis_transform(), color="#666666")
                continue
            ys = [float(r[key]) for r in rows if r["cell"] == cell and r["domain"] == dom]
            xs = np.linspace(i - 0.1, i + 0.1, len(ys))
            ax.scatter(xs, ys, s=12, color=OKABE[i], zorder=3)
            ax.errorbar(i, np.mean(ys), yerr=np.std(ys, ddof=1), fmt="o", color=OKABE[i],
                        markersize=5, capsize=2)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel(f"({letter}) {title}")
    fig.tight_layout()
    save(fig, "fig13_decomp_clahe")


def fig14_res448() -> None:
    rows = read_csv("table_v3_res448_per_seed.csv")
    domains = ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.6), sharey=True)
    for ax, dom, letter in zip(axes, domains, "abcd"):
        for i, cell in enumerate(("H1", "H2")):
            y224 = [float(r["qwk"]) for r in rows if r["cell"] == cell and r["domain"] == dom and r["image_size"] == "224"]
            y448 = [float(r["qwk"]) for r in rows if r["cell"] == cell and r["domain"] == dom and r["image_size"] == "448"]
            for a, b in zip(y224, y448):
                ax.plot([2 * i - 0.15, 2 * i + 0.15], [a, b], color="#bbbbbb", lw=0.8, zorder=1)
            ax.scatter([2 * i - 0.15] * 3, y224, s=14, color=OKABE[0], zorder=3, label="224" if i == 0 and letter == "a" else None)
            ax.scatter([2 * i + 0.15] * 3, y448, s=14, color=OKABE[1], zorder=3, label="448" if i == 0 and letter == "a" else None)
        ax.axhline(0, color="#888888", lw=0.6, ls="--")
        ax.set_xticks([0, 2])
        ax.set_xticklabels(["H1", "H2"])
        ax.set_xlabel(f"({letter}) {dom}")
        if letter == "a":
            ax.set_ylabel("QWK")
            ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig14_res448_vs_224")


def main() -> None:
    fig11_qwk_n6()
    fig12_matched_b0()
    fig13_decomp()
    fig14_res448()
    print("BUILD_FIGS_V3_OK")


if __name__ == "__main__":
    main()
