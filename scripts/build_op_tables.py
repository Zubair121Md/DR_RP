#!/usr/bin/env python3
"""Export Phase O / P tables into paper/tables/."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)
HOLDOUTS = (
    ("idrid", "idrid_external", "IDRiD"),
    ("messidor2", "messidor2_external", "Messidor-2"),
    ("deepdrid", "deepdrid_external", "DeepDRiD"),
)


def qwk(rid: str, split: str) -> float:
    return float(
        json.loads((ROOT / "results" / rid / f"{split}_metrics.json").read_text())[
            "quadratic_weighted_kappa"
        ]
    )


def fmt(xs: list[float]) -> str:
    a = np.asarray(xs, float)
    if a.size == 0:
        return "—"
    if a.size == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def write_md_csv_tex(stem: str, headers: list[str], rows: list[list[str]], caption: str) -> None:
    md = [f"# {caption}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    (TAB / f"{stem}.md").write_text("\n".join(md))
    with (TAB / f"{stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    cols = "l" + "c" * (len(headers) - 1)
    lines = [
        "% Auto-generated — scripts/build_op_tables.py",
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
        safe = [c.replace("%", r"\%").replace("±", r"$\pm$") for c in r]
        lines.append(" & ".join(safe) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / f"{stem}.tex").write_text("\n".join(lines))
    print("SAVED", stem)


def table_lodo() -> None:
    headers = ["Holdout", "LODO QWK", "H2 APTOS-only QWK", "Δ (LODO−H2)"]
    rows = []
    for h, sp, lab in HOLDOUTS:
        lodo = [qwk(f"phase_p_lodo_holdout_{h}_s{s}_v1", sp) for s in SEEDS]
        h2 = [qwk(f"hybrid_h2_s{s}_v1", sp) for s in SEEDS]
        delta = [a - b for a, b in zip(lodo, h2)]
        rows.append([lab, fmt(lodo), fmt(h2), fmt(delta)])
    write_md_csv_tex(
        "table_lodo_vs_h2",
        headers,
        rows,
        "Phase P LODO vs APTOS-only H2 on held-out sets (mean±std, seeds 42/43/44)",
    )


def table_phase_o() -> None:
    splits = [
        ("aptos_test", "APTOS"),
        ("idrid_external", "IDRiD"),
        ("messidor2_external", "Messidor-2"),
        ("deepdrid_external", "DeepDRiD"),
    ]
    headers = ["Model"] + [s[1] for s in splits]
    rows = []

    # B0: only aptos/idrid from seed summary
    b0 = json.loads((ROOT / "results/b0_2x2_seed_summary.json").read_text())["cells"][
        "green_clahe_ce"
    ]
    b0_aptos = [float(r["aptos_qwk"]) for r in b0["runs"]]
    b0_idrid = [float(r["idrid_qwk"]) for r in b0["runs"]]
    rows.append(["EfficientNet-B0", fmt(b0_aptos), fmt(b0_idrid), "—", "—"])

    for name, tmpl in [
        ("MobileViT-S H2", "hybrid_h2_s{s}_v1"),
        ("Swin-Tiny", "phase_o_swin_tiny_s{s}_v1"),
    ]:
        row = [name]
        for sp, _ in splits:
            row.append(fmt([qwk(tmpl.format(s=s), sp) for s in SEEDS]))
        rows.append(row)

    write_md_csv_tex(
        "table_phase_o_arch",
        headers,
        rows,
        "Phase O — CNN vs Hybrid vs Transformer, green_clahe × CE (mean±std, n=3)",
    )


def main() -> None:
    table_lodo()
    table_phase_o()
    print("OP_TABLES_OK")


if __name__ == "__main__":
    main()
