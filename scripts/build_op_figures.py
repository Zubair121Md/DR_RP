#!/usr/bin/env python3
"""Figures for Phase O (arch compare) + Phase P (LODO). Multi-seed mean±std."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures" / "pack"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)


def save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("SAVED", p.relative_to(ROOT), flush=True)


def qwk(rid: str, split: str) -> float:
    return float(
        json.loads((ROOT / "results" / rid / f"{split}_metrics.json").read_text())[
            "quadratic_weighted_kappa"
        ]
    )


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def fig_lodo_bars() -> None:
    holdouts = [
        ("idrid", "idrid_external", "IDRiD"),
        ("messidor2", "messidor2_external", "Messidor-2"),
        ("deepdrid", "deepdrid_external", "DeepDRiD"),
    ]
    lodo_m, lodo_s, h2_m, h2_s, labels = [], [], [], [], []
    for h, sp, lab in holdouts:
        labels.append(lab)
        lm, ls = mean_std([qwk(f"phase_p_lodo_holdout_{h}_s{s}_v1", sp) for s in SEEDS])
        hm, hs = mean_std([qwk(f"hybrid_h2_s{s}_v1", sp) for s in SEEDS])
        lodo_m.append(lm)
        lodo_s.append(ls)
        h2_m.append(hm)
        h2_s.append(hs)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, h2_m, w, yerr=h2_s, capsize=3, label="H2 APTOS-only train", color="#4C78A8")
    ax.bar(x + w / 2, lodo_m, w, yerr=lodo_s, capsize=3, label="LODO multi-source train", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("QWK on held-out set")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase P — LODO vs APTOS-only hybrid H2 (mean±std, n=3)")
    ax.legend()
    for i, (a, b, sa, sb) in enumerate(zip(h2_m, lodo_m, h2_s, lodo_s)):
        ax.text(i - w / 2, a + sa + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + sb + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "19_lodo_vs_h2_holdout_qwk.png")


def fig_lodo_heatmap() -> None:
    """Mean QWK across seeds for each LODO holdout model on all domains."""
    holdouts = ["idrid", "messidor2", "deepdrid"]
    splits = [
        ("aptos_test", "APTOS"),
        ("idrid_external", "IDRiD"),
        ("messidor2_external", "Messidor-2"),
        ("deepdrid_external", "DeepDRiD"),
    ]
    mat = np.zeros((len(holdouts), len(splits)))
    for i, h in enumerate(holdouts):
        for j, (sp, _) in enumerate(splits):
            vals = [qwk(f"phase_p_lodo_holdout_{h}_s{s}_v1", sp) for s in SEEDS]
            mat[i, j] = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s[1] for s in splits])
    ax.set_yticks(range(len(holdouts)))
    ax.set_yticklabels([f"holdout={h}" for h in holdouts])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            held = (holdouts[i] == "idrid" and splits[j][0].startswith("idrid")) or (
                holdouts[i] == "messidor2" and splits[j][0].startswith("messidor")
            ) or (holdouts[i] == "deepdrid" and splits[j][0].startswith("deepdrid"))
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if held else "normal",
                color="black",
            )
            if held:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="k", lw=2)
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("LODO models — mean QWK across seeds (bold+box = held-out)")
    fig.tight_layout()
    save(fig, "20_lodo_qwk_heatmap.png")


def fig_phase_o_arch() -> None:
    seeds = SEEDS
    splits = [
        ("aptos_test", "APTOS"),
        ("idrid_external", "IDRiD"),
        ("messidor2_external", "Messidor-2"),
        ("deepdrid_external", "DeepDRiD"),
    ]
    families = [
        ("MobileViT-S H2", "hybrid_h2_s{s}_v1"),
        ("Swin-Tiny", "phase_o_swin_tiny_s{s}_v1"),
    ]
    b0 = json.loads((ROOT / "results/b0_2x2_seed_summary.json").read_text())["cells"][
        "green_clahe_ce"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, sp_key, title, b0_key in [
        (axes[0], "aptos_test", "APTOS test QWK", "aptos_qwk"),
        (axes[1], "idrid_external", "IDRiD QWK", "idrid_qwk"),
    ]:
        means, stds, labs = [], [], []
        vals = [float(r[b0_key]) for r in b0["runs"]]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
        labs.append("EfficientNet-B0")
        for name, tmpl in families:
            vals = [qwk(tmpl.format(s=s), sp_key) for s in seeds]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1))
            labs.append(name)
        x = np.arange(len(labs))
        ax.bar(x, means, yerr=stds, capsize=4, color=["#54A24B", "#4C78A8", "#E45756"])
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=12, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    fig.suptitle("Phase O — CNN vs Hybrid vs Transformer (green_clahe × CE, n=3)", fontsize=11)
    fig.tight_layout()
    save(fig, "21_phase_o_arch_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(splits))
    w = 0.35
    h2m, h2s, swm, sws = [], [], [], []
    for sp, _ in splits:
        a = [qwk(f"hybrid_h2_s{s}_v1", sp) for s in seeds]
        b = [qwk(f"phase_o_swin_tiny_s{s}_v1", sp) for s in seeds]
        h2m.append(np.mean(a))
        h2s.append(np.std(a, ddof=1))
        swm.append(np.mean(b))
        sws.append(np.std(b, ddof=1))
    ax.bar(x - w / 2, h2m, w, yerr=h2s, capsize=3, label="MobileViT-S H2")
    ax.bar(x + w / 2, swm, w, yerr=sws, capsize=3, label="Swin-Tiny")
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in splits])
    ax.set_ylabel("QWK")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("H2 vs Swin-Tiny across domains (mean±std, n=3)")
    ax.legend()
    fig.tight_layout()
    save(fig, "22_h2_vs_swin_all_domains.png")


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white"})
    fig_lodo_bars()
    fig_lodo_heatmap()
    fig_phase_o_arch()
    idx = OUT / "FIGURE_INDEX.md"
    extra = [
        "",
        "## Phase O / P additions",
        "",
        "- `19_lodo_vs_h2_holdout_qwk.png` (mean±std n=3)",
        "- `20_lodo_qwk_heatmap.png` (mean across seeds)",
        "- `21_phase_o_arch_comparison.png`",
        "- `22_h2_vs_swin_all_domains.png`",
        "",
    ]
    if idx.is_file():
        text = idx.read_text()
        if "19_lodo" not in text:
            idx.write_text(text.rstrip() + "\n" + "\n".join(extra))
        else:
            # refresh note lines if present
            pass
    print("OP_FIGURES_OK")


if __name__ == "__main__":
    main()
