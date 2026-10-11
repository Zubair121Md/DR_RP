#!/usr/bin/env python3
"""
Build every figure and the manuscript from paper/tables CSVs and saved predictions only.

Outputs: PDF (vector) + PNG 600 dpi under paper/figures/main/ and paper/figures/supp/,
then paper/MANUSCRIPT.md and paper/MANUSCRIPT.docx, then runs paper/check_manuscript.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main() -> None:
    required = [
        ROOT / "paper/tables/table_hybrid_qwk.csv",
        ROOT / "paper/tables/table_hybrid_qwk_per_seed.csv",
        ROOT / "paper/tables/table_mild_calibration.csv",
        ROOT / "paper/tables/table_lodo_vs_h2_per_seed.csv",
        ROOT / "paper/tables/dataset_manifest.csv",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        print("Missing tables — regenerating sourced tables…")
        run([sys.executable, "-u", "scripts/build_paper_sourced_tables.py"])

    if any((ROOT / "results").glob("matched_b0_*_v3")):
        run([sys.executable, "-u", "scripts/build_paper_v3_tables.py"])
    for script in ("scripts/build_submission_tables.py", "scripts/verify_coral.py",
                   "scripts/audit_aptos_split_duplicates.py", "scripts/record_environment.py",
                   "scripts/count_norm_layers.py"):
        run([sys.executable, "-u", script])

    run([sys.executable, "-u", str(ROOT / "paper/figures/fig1_pipeline.py")])
    for ext in ("png", "pdf"):
        src = ROOT / "paper/figures" / f"fig1_pipeline.{ext}"
        dst = OUT / f"fig1_pipeline.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())
            (OUT / f"fig01_study_pipeline.{ext}").write_bytes(src.read_bytes())
            print("COPIED", dst.relative_to(ROOT))

    run([sys.executable, "-u", str(ROOT / "paper/figures/build_figs_2_to_9.py")])
    run([sys.executable, "-u", str(ROOT / "scripts/build_gradcam_mild_panel.py")])
    if (ROOT / "paper/tables/table_v3_hybrid_qwk_n6_per_seed.csv").is_file():
        run([sys.executable, "-u", str(ROOT / "paper/figures/build_figs_v3.py")])
    # Version B primary result figures (schedule-stratified QWK / mild recall / H1–H2 ECE).
    run([sys.executable, "-u", str(ROOT / "paper/figures/build_version_b_main.py")])
    run([sys.executable, "-u", str(ROOT / "paper/figures/build_supplementary.py")])
    run([sys.executable, "-u", str(ROOT / "scripts/build_manuscript_md.py")])
    run([sys.executable, "-u", str(ROOT / "scripts/build_manuscript_docx.py")])
    run([sys.executable, "-u", str(ROOT / "paper/check_manuscript.py")])
    print("MAKE_ALL_OK")


if __name__ == "__main__":
    main()
