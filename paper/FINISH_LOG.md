# Paper finish log (2026-09-28)

## Done (no new training)

1. **Canonical RQ + novelty** locked in `paper/RESEARCH_QUESTION.md`, `paper/CLAIM_NOTES.md`, and `paper/MANUSCRIPT.md`.
2. **Sourced tables** rebuilt from metrics/predictions JSON:
   - `table_hybrid_qwk.csv` (+ per-seed)
   - `table_mild_calibration.csv` (+ per-seed)
   - `table_ece_by_domain.csv`
   - `table_efficiency.csv` (Swin marked MISSING, not estimated)
   - `table_messidor2_sanity.csv` (label counts + pred grade-0 collapse)
   - `table_lodo_vs_h2.csv` (+ per-seed)
3. **Figures 1–10** as PDF + PNG 600 dpi under `paper/figures/main/` via `make_all.py` / builders.
4. **Manuscript numbers** synced from CSVs (`scripts/sync_manuscript_numbers.py`).
5. **`paper/check_manuscript.py` → CHECK_MANUSCRIPT_OK** (docx regenerated from md).
6. **`configs/paper_v2` 36-job queue remains deferred.**

## Assumptions

- Seed mean±std uses sample std (`ddof=1`).
- Fig 3 DeepDRiD sampling uses regular-fundus train+val CSVs with left/right eye grade columns (same rule as `src/dataset.py`).
- Fig 6 reliability pools seeds 42–44; ECE annotation is mean±std of per-seed metric ECE.
- Efficiency snapshot is MPS batch-1 from `results/efficiency_models.json` only.
- Archived B0 confusion in Fig 7 uses run `baseline_efficientnet_b0_green_clahe_20260918_092447`.

## Manuscript build (humanised text, all figures embedded)

- Prose lives in `paper/MANUSCRIPT_TEMPLATE.md`; every result number is a `{{placeholder}}`.
- `scripts/build_manuscript_md.py` fills placeholders from `paper/tables/*.csv` and `results/*/train_summary.json`,
  asserts every qualitative claim in the prose (fails on any false claim), writes `paper/MANUSCRIPT.md`
  and the value ledger `paper/tables/manuscript_values.csv`.
- `scripts/build_manuscript_docx.py` writes `paper/MANUSCRIPT.docx`: real Word tables, 10 main + 15 supplementary
  figures embedded with captions, 3 equations rendered to `paper/figures/equations/`, page numbers.
- `paper/check_manuscript.py` reads the built docx (never regenerates it) and checks numbers, captions, citation
  order, reference coverage, placeholders and raw paths. Optional argument: path to another docx.
- Supplementary figures are numbered in first-citation order (S1 grade distribution … S15 Messidor-2 predictions).
- Not embedded (stale or superseded): old `paper/figures/pack/*`, old Grad-CAM compare grids (CORAL panels blank),
  `main/fig10_hybrid_qwk_heatmap.*`, `main/fig08_gradcam_mild.png`, and legacy-name copies in `main/`.

### Corrections made while rewriting

- Temperature scaling: previous text said it "does not repair external ECE". Per table D it lowers H2 ECE on all
  four sets in every seed; it raises H3/H4 ECE on Messidor-2 and DeepDRiD in every seed. H3's fitted T is 0.05 in
  all seeds, the lower clamp of `fit_temperature` in `src/metrics.py`.
- Calibration sentence: CE cells beat CORAL cells on APTOS and IDRiD; on DeepDRiD only H2 is lowest (H1 > H4).
- Selected epochs are taken from `history.json` (first maximum of val QWK); `train_summary.json` reports
  `best_epoch: 20` for every hybrid run although `best_selection_score` matches the history maximum.

## How to regenerate

```bash
python3 scripts/build_paper_sourced_tables.py
python3 paper/figures/make_all.py      # figures, manuscript md + docx, and the check
```

## Paper v3 results integrated (2026-10-04)

- All 36 paper_v3 jobs complete (core+decomp+res448); optional GPU BN not run.
- Primary hybrid tables now use six seeds (42–47).
- Matched B0, mild-CORAL decomposition (D1/D2), and 448×448 results added to manuscript + Figures 5/7/12/13.
- `CHECK_MANUSCRIPT_OK` after rebuild.
