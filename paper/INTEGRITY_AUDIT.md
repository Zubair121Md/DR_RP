# Phase S — Research integrity audit

**Result:** PASS (7 passed, 0 failed)

| Status | Check |
|--------|-------|
| PASS | All 12 hybrids have best.pt + 4 metrics (0 missing) |
| PASS | paper/ does not cite *_stable runs (clean) |
| PASS | Hybrid configs CPU + freeze_bn (ok) |
| PASS | Required paper/analysis artifacts present (ok) |
| PASS | Grad-CAM manifest n=120 (expect ≥120) |
| PASS | CLAIM_NOTES documents selection / cite bans |
| PASS | CLAIM_NOTES acknowledges Messidor-2 failure mode |

## Writing checklist (manual)

- [x] Cite only `hybrid_h*_s{42,43,44}_v1` and protocol-valid B0 runs *(wired in `paper/MANUSCRIPT.md`)*
- [x] Never cite MobileViT `*_stable` (batch=1 / BN-invalid)
- [x] Selection metric = APTOS **val** QWK only
- [x] Report Messidor-2 hybrid collapse as a negative result
- [x] B0 coral ≠ hybrid mild-CORAL — caveat in text
- [x] Paired tests: with n=3, report mean Δ; do not invent p<0.05
- [x] Tables from `paper/tables/` (incl. O/P); figures from `paper/figures/pack/` 01–22
- [x] Related Work + References from `Book 4.xlsx` (`paper/MANUSCRIPT.md`, `paper/REFERENCES.md`)
- [ ] Grad-CAM: prefer `compare/` mild-grade panels *(author pick for PDF)*
- [ ] Author metadata, acknowledgements, competing interests, CRediT
- [ ] Journal template + final proofread before submission

## Sources of truth

- Protocol: `configs/research/frozen_protocol.yaml`
- Seed means: `experiments/results/hybrid_seed_summary.md`
- Paired stats: `experiments/results/paired_stats.md`
- Claims: `paper/CLAIM_NOTES.md`

INTEGRITY_AUDIT_PASS
