# Table-by-table audit

**Audit date:** 2026-10-11  
**Primary sources:** `paper/tables/*.csv`, built `paper/MANUSCRIPT.md`

| Table | Title OK | Columns/rows OK | Schedule split | Values vs CSV | Footnote / xref | Issues | Status |
|---|---|---|---|---|---|---|---|
| Table 1 | Yes | Roles + grade counts | N/A | Match `dataset_manifest.csv` / `aptos_split_audit.csv` | OK | None | Pass |
| Table 2 | Yes | Training settings | Seeds/schedule rows | Match configs + `environment.csv` / schedule audit | OK | Swin last-epoch LR truncated — disclosed | Pass |
| Table 3 | Yes | H1–H4 × 2 schedule groups × 4 datasets QWK | Yes | Match `table_v3_hybrid_qwk_n6_per_seed.csv` | Explicitly not pooled | Results prose previously contradicted IDRiD direction — **fixed** | Pass after fix |
| Table 4 | Yes | Mild + overall APTOS | Yes | Match APTOS metric summaries | n=30 mild | — | Pass |
| Table 5 | Yes | Grade-1 recall all domains | Yes | Match `external_grade1_recall_*` | n noted | — | Pass |
| Table 6 | Yes | ECE before→after; H1/H2 only | Yes | Match `ece_ce_cells_n6_per_seed.csv` | Threshold excluded | — | Pass |
| Table 7 | Yes | Matched B0 vs MobileViT-S | Schedule column present | Match `table_v3_matched_b0_vs_hybrid.csv` | Secondary exploratory | — | Pass |
| Table 8 | Yes | Descriptive paired diffs | Within-group | Match paired tables | No confirmatory Holm headline | — | Pass |
| Table 9 | Yes | Swin vs MobileViT-S | Truncated seeds | Match `table_phase_o_arch.csv` | Secondary | — | Pass |
| Table 10 | Yes | LODO vs H2 | Seeds 42–44 | Match `table_lodo_vs_h2*` | Secondary | — | Pass |
| Table 11 | Yes | Loss decomposition | Seeds 42–44 | Match `table_v3_decomp_clahe.csv` | ECE CE only | — | Pass |
| Table 12 | Yes | 448 vs 224 | Noted schedule caveat | Match `table_v3_res448_vs_224.csv` | Secondary | — | Pass |
| Table S1 | Yes | APTOS partition audit | N/A | Match `aptos_split_audit.csv` | MD5 exact dups | Cross-label 12 pairs | Pass |
| Table S2 | Yes | Schedule regime audit | Yes | Match `training_schedule_audit.csv` | — | — | Pass |
| Table S3 | Yes | Threshold-head verification | N/A | Match coral verification | — | — | Pass |
| Table S4 | Yes | Per-run primary metrics | Per seed | Match `master_results_per_run.csv` | ECE CE only | — | Pass |
| Table S5 | Yes | Per-seed external / related | Per seed | Match source tables | — | Spot-check OK | Pass |

## Decimal places

Primary QWK/recall/ECE reported to three decimals with ± sample SD over three seeds per schedule group — consistent across Tables 3–6.

## Action items

1. Keep Table 3 prose aligned with schedule-specific IDRiD direction (done).  
2. Optional: add schedule-stratified dedup means as Table Sx (CSV prepared: `deduplication_sensitivity.csv`).
