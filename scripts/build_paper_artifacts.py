#!/usr/bin/env python3
"""Phase Q/R: build paper tables + figures from frozen hybrid/B0 JSON artifacts.

Outputs:
  paper/tables/table_hybrid_qwk.{md,csv,tex}
  paper/tables/table_hybrid_vs_b0_idrid.{md,csv,tex}
  paper/tables/table_mild_calibration.{md,csv,tex}
  paper/figures/fig_hybrid_qwk_by_domain.png
  paper/figures/fig_domain_shift_delta.png
  paper/CLAIM_NOTES.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HYB = json.loads((ROOT / "experiments/results/hybrid_seed_summary.json").read_text())
B0 = json.loads((ROOT / "results/b0_2x2_seed_summary.json").read_text())
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CELLS = ["H1", "H2", "H3", "H4"]
CELL_LABEL = HYB["cells"]
SPLITS = [
    ("aptos_test", "APTOS"),
    ("idrid_external", "IDRiD"),
    ("messidor2_external", "Messidor-2"),
    ("deepdrid_external", "DeepDRiD"),
]


def fmt_ms(st: dict, digits: int = 3) -> str:
    if not st or st.get("mean") is None:
        return "—"
    return f"{st['mean']:.{digits}f}±{st['std']:.{digits}f}"


def write_md_csv_tex(stem: str, headers: list[str], rows: list[list[str]], caption: str) -> None:
    # markdown
    md = [f"# {caption}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    (TAB / f"{stem}.md").write_text("\n".join(md))
    # csv
    with (TAB / f"{stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    # latex
    cols = "l" + "c" * (len(headers) - 1)
    lines = [
        "% Auto-generated — do not hand-edit; regenerate via scripts/build_paper_artifacts.py",
        f"% {caption}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{stem}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        safe = [c.replace("%", r"\%").replace("±", r"$\pm$").replace("−", "-") for c in r]
        lines.append(" & ".join(safe) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / f"{stem}.tex").write_text("\n".join(lines))


def table_hybrid_qwk() -> None:
    headers = ["Cell", "Setup"] + [s[1] for s in SPLITS]
    rows = []
    for c in CELLS:
        row = [c, CELL_LABEL[c]]
        for sk, _ in SPLITS:
            row.append(fmt_ms(HYB["mean_std"][c][sk]["quadratic_weighted_kappa"]))
        rows.append(row)
    write_md_csv_tex(
        "table_hybrid_qwk",
        headers,
        rows,
        "MobileViT-S hybrid 2×2 QWK (mean±std over seeds 42/43/44)",
    )


def table_mild_cal() -> None:
    headers = ["Cell", "mild F1", "mild recall", "ECE", "ECE@T"]
    rows = []
    for c in CELLS:
        a = HYB["mean_std"][c]["aptos_test"]
        rows.append(
            [
                c,
                fmt_ms(a["mild_dr_f1"]),
                fmt_ms(a["mild_dr_recall"]),
                fmt_ms(a["ece"]),
                fmt_ms(a["ece_temperature_scaled"]),
            ]
        )
    write_md_csv_tex(
        "table_mild_calibration",
        headers,
        rows,
        "APTOS test mild-DR and calibration (hybrid, mean±std)",
    )


def table_hybrid_vs_b0() -> None:
    """Compare overlapping preprocess×loss themes on APTOS + IDRiD only (B0 summary scope)."""
    # Map B0 cells to hybrid themes
    mapping = [
        ("B0 RGB×CE", "rgb_ce", "H1", "Hybrid RGB×CE"),
        ("B0 G+CLAHE×CE", "green_clahe_ce", "H2", "Hybrid G+CLAHE×CE"),
        ("B0 RGB×CORAL", "rgb_coral", "H3", "Hybrid RGB×mildCORAL"),
        ("B0 G+CLAHE×CORAL", "green_clahe_coral", "H4", "Hybrid G+CLAHE×mildCORAL"),
    ]
    headers = ["Model", "APTOS QWK", "IDRiD QWK"]
    rows = []
    for b0_lab, b0_key, h_key, h_lab in mapping:
        b = B0["cells"].get(b0_key, {}).get("summary", {})
        rows.append(
            [
                b0_lab,
                fmt_ms(b.get("aptos_qwk", {})),
                fmt_ms(b.get("idrid_qwk", {})),
            ]
        )
        rows.append(
            [
                h_lab,
                fmt_ms(HYB["mean_std"][h_key]["aptos_test"]["quadratic_weighted_kappa"]),
                fmt_ms(HYB["mean_std"][h_key]["idrid_external"]["quadratic_weighted_kappa"]),
            ]
        )
    write_md_csv_tex(
        "table_hybrid_vs_b0_idrid",
        headers,
        rows,
        "B0 vs MobileViT-S (APTOS/IDRiD QWK; B0 uses CORAL where hybrid uses mild-CORAL)",
    )


def fig_qwk_bars() -> None:
    x = np.arange(len(CELLS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (sk, lab) in enumerate(SPLITS):
        means = [HYB["mean_std"][c][sk]["quadratic_weighted_kappa"]["mean"] for c in CELLS]
        stds = [HYB["mean_std"][c][sk]["quadratic_weighted_kappa"]["std"] for c in CELLS]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{CELL_LABEL[c]}" for c in CELLS], fontsize=8)
    ax.set_ylabel("QWK")
    ax.set_title("Hybrid 2×2 — QWK by domain (n=3 seeds)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.2, 1.0)
    fig.tight_layout()
    out = FIG / "fig_hybrid_qwk_by_domain.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_domain_delta() -> None:
    """APTOS − external QWK drop (larger = worse shift)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(CELLS))
    width = 0.25
    for i, (sk, lab) in enumerate(SPLITS[1:]):
        deltas = []
        for c in CELLS:
            a = HYB["mean_std"][c]["aptos_test"]["quadratic_weighted_kappa"]["mean"]
            e = HYB["mean_std"][c][sk]["quadratic_weighted_kappa"]["mean"]
            deltas.append(a - e)
        ax.bar(x + (i - 1) * width, deltas, width, label=f"APTOS−{lab}")
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS)
    ax.set_ylabel("QWK drop (APTOS − external)")
    ax.set_title("Domain shift magnitude (hybrid mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_domain_shift_delta.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def claim_notes() -> None:
    notes = """# Claim notes (research integrity)

## Safe to claim (with seed mean±std)
- MobileViT-S hybrid **H2 (green_clahe × CE)** has strongest **APTOS test** QWK among the 2×2 (~0.88).
- **H2** also leads **DeepDRiD** among hybrids.
- **H4 (green_clahe × mild_CORAL)** leads **IDRiD** among hybrids (~0.57 mean) but with large seed variance.
- CE hybrids are better calibrated (lower ECE) than mild-CORAL hybrids on APTOS.

## Must report as negative / failure
- **Messidor-2 QWK ≈ 0** (sometimes negative) for all hybrid cells — do **not** claim Messidor generalization.
- Mild-CORAL raises mild recall on APTOS but **hurts ECE** and can collapse external QWK (esp. H3 DeepDRiD).

## Do not cite
- Any `*_stable` MobileViT run with `batch_size=1` (BN invalid).
- Selection metrics from APTOS test / externals (selection was APTOS val only).

## B0 vs hybrid caveat
- B0 seed table uses standard **CORAL**; hybrid H3/H4 use **mild-aware CORAL**. Compare carefully in text.
- B0 summary may lack Messidor/DeepDRiD seed aggregates — prefer hybrid tables for those domains.
"""
    (ROOT / "paper" / "CLAIM_NOTES.md").write_text(notes)


def main() -> None:
    table_hybrid_qwk()
    table_mild_cal()
    table_hybrid_vs_b0()
    fig_qwk_bars()
    fig_domain_delta()
    claim_notes()
    print("PAPER_ARTIFACTS_OK")
    for p in sorted(TAB.glob("table_*")):
        print(" ", p.relative_to(ROOT))
    for p in sorted(FIG.glob("fig_*")):
        print(" ", p.relative_to(ROOT))
    print(" ", "paper/CLAIM_NOTES.md")


if __name__ == "__main__":
    main()
