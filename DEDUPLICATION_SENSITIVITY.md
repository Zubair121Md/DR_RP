# Deduplication sensitivity audit

**Audit date:** 2026-10-11  
**Script:** `scripts/build_dedup_sensitivity.py` (recomputes every metric from `results/hybrid_<cell>_s<seed>_<v1|v3>/aptos_test_predictions.json`)  
**Outputs:**
- `paper/tables/deduplication_sensitivity.csv` — 48 rows (4 cells × 6 seeds × 2 subsets)
- `paper/tables/deduplication_sensitivity_summary.csv` — 16 rows, mean ± sample SD per cell × schedule group × subset (Supplementary Table S6)
- `deduplication_sensitivity.csv` (repository root) — copy of the summary in the requested column order

## Leakage facts (source: `paper/tables/aptos_split_audit.csv`, `aptos_duplicate_pairs.csv`)

| Claim | Source value | Match |
|---|---|---|
| 134 exact duplicate pairs | 134 | Yes |
| 24 APTOS test images with an exact duplicate in train or validation | 24 | Yes |
| 12 cross-partition duplicate pairs with different grades | 12 | Yes |
| Removing affected test images leaves 342 images | `n_kept` = 342 | Yes |

## Metric coverage

All requested columns are now computed on both the full (366) and the deduplicated (342) test set:
`cell, schedule_group, evaluation_subset, n_images, qwk, mild_recall, macro_f1, accuracy, referable_sensitivity, referable_specificity, referable_auc, ece, nll, brier`.

- ECE, NLL and Brier are reported for H1/H2 only; H3/H4 are `n.i.` (non-monotonic threshold head, Section 3.5).
- Referable DR is grade ≥ 2. The score is p2 + p3 + p4 for cross-entropy models and σ(z₁) for threshold heads.
- Full-set values reproduce Tables 4 and 6 of the manuscript.
- Per-run QWK in the new file is identical to the older `aptos_dedup_sensitivity.csv` (max |difference| = 0).

## Result (schedule-stratified, as reported in Section 4.5)

- The largest change in a schedule-group mean QWK is 0.007, and the largest change for a single run is 0.015.
- H2, truncated schedule: 0.876 ± 0.013 → 0.873 ± 0.013.
- H2, full cosine: 0.836 ± 0.009 → 0.829 ± 0.011.
- The earlier wording, "at most 0.005", came from a pooled six-seed mean and has been replaced with the schedule-stratified figures.

## Statement in the paper

Exact-duplicate leakage exists in the distributed third-party APTOS partition. Duplicates were retained so the partition matches its distributed form, and the sensitivity analysis quantifies the numerical effect rather than claiming that leakage is absent. Near-duplicates were not assessed.
