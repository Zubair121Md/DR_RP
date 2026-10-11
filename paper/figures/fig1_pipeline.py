#!/usr/bin/env python3
"""
Figure 1: study design (vector PDF + 600 dpi PNG).

Image counts come from paper/tables/dataset_manifest.csv; the evaluation coverage of the
matched EfficientNet-B0 run and the LODO training sizes come from results/. Stops if a
required count is missing. Writes paper/figures/fig1_pipeline.{pdf,png}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "paper/tables/dataset_manifest.csv"
OUT_DIR = ROOT / "paper/figures"
B0_RUN = "matched_b0_h2_s42_v3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
})

# Light tints of the Okabe-Ito palette, one per column role.
FILL = {"train": "#CFE3F2", "model": "#EEEEEE", "select": "#F8E3B8", "eval": "#EFD5E4"}
EDGE = {"train": "#0072B2", "model": "#555555", "select": "#E69F00", "eval": "#CC79A7"}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def load_manifest() -> dict[tuple[str, str], int]:
    if not MANIFEST.is_file():
        print("MISSING", rel(MANIFEST))
        raise SystemExit(1)
    print("READ", rel(MANIFEST))
    out = {}
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            if row["n_images"]:
                out[(row["dataset"], row["split"])] = int(row["n_images"])
    return out


def need(m: dict, dataset: str, split: str) -> int:
    if (dataset, split) not in m:
        raise SystemExit(f"MISSING count for {dataset}/{split} in {rel(MANIFEST)}")
    return m[(dataset, split)]


def b0_coverage() -> list[tuple[str, bool]]:
    out = []
    for fname, lab in (("aptos_test_metrics.json", "APTOS test"), ("idrid_external_metrics.json", "IDRiD"),
                       ("messidor2_external_metrics.json", "Messidor-2"),
                       ("deepdrid_external_metrics.json", "DeepDRiD")):
        p = ROOT / "results" / B0_RUN / fname
        print("READ" if p.is_file() else "ABSENT", rel(p))
        out.append((lab, p.is_file()))
    return out


def lodo_sizes() -> list[int]:
    sizes = []
    for h in ("idrid", "messidor2", "deepdrid"):
        p = ROOT / "results" / f"phase_p_lodo_holdout_{h}_s42_v1" / "train_summary.json"
        if not p.is_file():
            print("MISSING", rel(p))
            raise SystemExit(1)
        print("READ", rel(p))
        sizes.append(int(json.loads(p.read_text())["train_size"]))
    return sizes


def box(ax, x, y, w, h, text, role, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=FILL[role], ec=EDGE[role], lw=0.9, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, linespacing=1.25)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333", mutation_scale=8))


def main() -> None:
    m = load_manifest()
    n_train = need(m, "APTOS", "train")
    n_val = need(m, "APTOS", "val")
    n_test = need(m, "APTOS", "test")
    n_idrid = need(m, "IDRiD", "external_all")
    n_mes = need(m, "Messidor-2", "external_gradable")
    n_dd = need(m, "DeepDRiD", "external_eval")
    cov = b0_coverage()
    lsz = lodo_sizes()

    ext_all = (f"APTOS test ({n_test:,})\nIDRiD ({n_idrid:,})\n"
               f"Messidor-2 ({n_mes:,})\nDeepDRiD ({n_dd:,})")
    b0_eval = "\n".join(lab if ok else f"{lab}: not available" for lab, ok in cov)

    rows = [
        ("Primary\n(APTOS only)",
         f"APTOS train\n({n_train:,} images)",
         "MobileViT-S\nH1–H4: RGB or CLAHE\n× CE or MW-threshold\nSeeds 42–44 / 45–47\n(schedule-stratified)",
         f"Best epoch by\nAPTOS val QWK\n({n_val} images)\nT fitted on val", ext_all, False),
        ("CNN\nreference",
         f"APTOS train\n({n_train:,} images)",
         "EfficientNet-B0\nH1–H4, same regime\n3 seeds (42–44)\n(exploratory)",
         "Best epoch by\nAPTOS val QWK", b0_eval, False),
        ("Transformer\nreference",
         f"APTOS train\n({n_train:,} images)",
         "Swin-Tiny\nCLAHE × CE\n3 seeds (42–44)\n(exploratory)",
         "Best epoch by\nAPTOS val QWK", ext_all, False),
        ("Leave-one-\ndataset-out",
         f"APTOS train +\n2 external sets\n({min(lsz):,}–{max(lsz):,} images)",
         "MobileViT-S\nCLAHE × CE\n3 seeds (42–44)\n(exploratory)",
         "Best epoch by\nAPTOS val QWK", "Held-out external set\n(IDRiD, Messidor-2\nor DeepDRiD)", True),
    ]

    fig, ax = plt.subplots(figsize=(6.9, 5.3))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(-0.35, 6.5)
    ax.axis("off")

    label_w = 1.15
    col_x = [1.25, 3.45, 5.65, 7.85]
    col_w = 1.95
    box_h = 1.25
    gap = 0.28
    heads = ["Training data", "Model", "Selection", "Evaluation"]
    for x, h in zip(col_x, heads):
        ax.text(x + col_w / 2, 6.3, h, ha="center", va="center", fontsize=9, fontweight="bold")

    y = 6.0 - box_h
    roles = ["train", "model", "select", "eval"]
    for label, *cells, dashed in rows:
        ax.text(label_w / 2 + 0.02, y + box_h / 2, label, ha="center", va="center", fontsize=8.5, fontweight="bold")
        for k, (x, text, role) in enumerate(zip(col_x, cells, roles)):
            box(ax, x, y, col_w, box_h, text, role, dashed=dashed and role in ("train", "eval"))
            if k < 3:
                arrow(ax, x + col_w + 0.02, col_x[k + 1] - 0.02, y + box_h / 2)
        y -= box_h + gap

    # Protocol-deviation disclosure (Version B).
    note = ("Protocol audit. Seeds 42–44 of the primary MobileViT-S cells received a truncated "
            "learning-rate schedule because scheduler state was not retained during mid-epoch "
            "checkpoint resumes; seeds 45–47 received the intended full cosine schedule. "
            "The two groups are reported separately and are not treated as exchangeable replicates. "
            "Threshold-head probabilities were non-monotonic; calibration is restricted to H1/H2.")
    ax.add_patch(FancyBboxPatch((0.15, -0.25), 9.7, 0.85, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc="#FFF4E5", ec="#E69F00", lw=0.9))
    ax.text(5.0, 0.18, note, ha="center", va="center", fontsize=7.2, wrap=True,
            bbox=dict(boxstyle="square,pad=0", fc="none", ec="none"), linespacing=1.25)

    fig.tight_layout(pad=0.2)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        out = OUT_DIR / f"fig1_pipeline.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print("WROTE", rel(out))
    plt.close(fig)
    print("FIG1_OK")


if __name__ == "__main__":
    main()
