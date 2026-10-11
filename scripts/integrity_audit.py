#!/usr/bin/env python3
"""Phase S: research-integrity audit + final checklist.

Checks protocol-valid hybrid artifacts, bans *_stable batch=1 cites,
verifies paper tables/figures exist, and writes paper/INTEGRITY_AUDIT.md.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "INTEGRITY_AUDIT.md"
SEEDS = (42, 43, 44)
CELLS = ("H1", "H2", "H3", "H4")
NEEDED = (
    "aptos_test_metrics.json",
    "idrid_external_metrics.json",
    "messidor2_external_metrics.json",
    "deepdrid_external_metrics.json",
)


def check(ok: bool, msg: str) -> tuple[str, bool]:
    return (("PASS" if ok else "FAIL"), ok), msg  # type: ignore


def main() -> None:
    rows: list[tuple[str, bool, str]] = []

    def add(ok: bool, msg: str) -> None:
        rows.append(("PASS" if ok else "FAIL", ok, msg))

    # 1) Hybrid completeness
    miss = []
    for s in SEEDS:
        for c in CELLS:
            rid = f"hybrid_{c.lower()}_s{s}_v1"
            ckpt = ROOT / "checkpoints" / rid / "best.pt"
            if not ckpt.is_file():
                miss.append(f"ckpt:{rid}")
            for n in NEEDED:
                if not (ROOT / "results" / rid / n).is_file():
                    miss.append(f"{rid}/{n}")
    add(not miss, f"All 12 hybrids have best.pt + 4 metrics ({len(miss)} missing)")

    # 2) Ban stable MV cites in paper/ (allow audit/claim docs that forbid them)
    banned_hits = []
    allow_name = {"INTEGRITY_AUDIT.md", "CLAIM_NOTES.md"}
    for p in (ROOT / "paper").rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".md", ".tex", ".csv", ".json"}:
            continue
        if p.name in allow_name:
            continue
        text = p.read_text(errors="ignore")
        if "_stable" in text:
            banned_hits.append(str(p.relative_to(ROOT)))
    add(not banned_hits, f"paper/ does not cite *_stable runs ({banned_hits or 'clean'})")

    # 3) Config BN / device sanity on hybrids
    bad_bs = []
    for s in SEEDS:
        for c in CELLS:
            rid = f"hybrid_{c.lower()}_s{s}_v1"
            cfgp = ROOT / "experiments" / rid / "config.yaml"
            if not cfgp.is_file():
                bad_bs.append(rid)
                continue
            txt = cfgp.read_text()
            # v1 recipe used batch_size: 4 with freeze_bn
            if "freeze_bn: true" not in txt and "freeze_bn: True" not in txt:
                bad_bs.append(f"{rid}:no_freeze_bn")
            if "device: cpu" not in txt:
                bad_bs.append(f"{rid}:not_cpu")
    add(not bad_bs, f"Hybrid configs CPU + freeze_bn ({bad_bs or 'ok'})")

    # 4) Analysis artifacts
    required = [
        ROOT / "experiments/results/hybrid_seed_summary.json",
        ROOT / "experiments/results/hybrid_seed_summary.md",
        ROOT / "experiments/results/paired_stats.json",
        ROOT / "experiments/results/paired_stats.md",
        ROOT / "results/efficiency_models.json",
        ROOT / "paper/tables/table_hybrid_qwk.md",
        ROOT / "paper/tables/table_mild_calibration.md",
        ROOT / "paper/tables/table_hybrid_vs_b0_idrid.md",
        ROOT / "paper/tables/table_paired_stats.md",
        ROOT / "paper/figures/fig_hybrid_qwk_by_domain.png",
        ROOT / "paper/figures/fig_domain_shift_delta.png",
        ROOT / "paper/figures/gradcam/INDEX.md",
        ROOT / "paper/CLAIM_NOTES.md",
    ]
    missing_art = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    add(not missing_art, f"Required paper/analysis artifacts present ({missing_art or 'ok'})")

    # 5) Grad-CAM coverage
    man = ROOT / "paper/figures/gradcam/manifest.json"
    if man.is_file():
        n = json.loads(man.read_text()).get("n", 0)
        add(n >= 120, f"Grad-CAM manifest n={n} (expect ≥120)")
    else:
        add(False, "Grad-CAM manifest missing")

    # 6) Selection rule reminder (APTOS val only) — check claim notes mention it
    claim = (ROOT / "paper/CLAIM_NOTES.md").read_text() if (ROOT / "paper/CLAIM_NOTES.md").is_file() else ""
    add(
        "selection" in claim.lower() or "APTOS val" in claim or "Do not cite" in claim,
        "CLAIM_NOTES documents selection / cite bans",
    )

    # 7) Messidor failure acknowledged
    add(
        "Messidor" in claim,
        "CLAIM_NOTES acknowledges Messidor-2 failure mode",
    )

    n_pass = sum(1 for _, ok, _ in rows if ok)
    n_fail = sum(1 for _, ok, _ in rows if not ok)
    all_ok = n_fail == 0

    lines = [
        "# Phase S — Research integrity audit",
        "",
        f"**Result:** {'PASS' if all_ok else 'FAIL'} ({n_pass} passed, {n_fail} failed)",
        "",
        "| Status | Check |",
        "|--------|-------|",
    ]
    for status, _, msg in rows:
        lines.append(f"| {status} | {msg} |")
    lines += [
        "",
        "## Writing checklist (manual)",
        "",
        "- [ ] Cite only `hybrid_h*_s{42,43,44}_v1` and protocol-valid B0 runs",
        "- [ ] Never cite MobileViT `*_stable` (batch=1 / BN-invalid)",
        "- [ ] Selection metric = APTOS **val** QWK only",
        "- [ ] Report Messidor-2 hybrid collapse as a negative result",
        "- [ ] B0 coral ≠ hybrid mild-CORAL — caveat in text",
        "- [ ] Paired tests: with n=3, report mean Δ; do not invent p<0.05",
        "- [ ] Tables from `paper/tables/`; figures from `paper/figures/`",
        "- [ ] Grad-CAM: prefer `compare/` mild-grade panels",
        "",
        "## Sources of truth",
        "",
        "- Protocol: `configs/research/frozen_protocol.yaml`",
        "- Seed means: `experiments/results/hybrid_seed_summary.md`",
        "- Paired stats: `experiments/results/paired_stats.md`",
        "- Claims: `paper/CLAIM_NOTES.md`",
        "",
    ]
    if all_ok:
        lines.append("INTEGRITY_AUDIT_PASS")
    else:
        lines.append("INTEGRITY_AUDIT_FAIL")
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
