# Project audit report — Version B exploratory manuscript

**Manuscript title:** Green-Channel CLAHE and a Mild-Weighted Threshold Objective for Cross-Dataset Diabetic Retinopathy Grading with MobileViT-S: An Exploratory Evaluation  
**Audit date:** 2026-10-11  
**Scope:** Manuscript, tables, figures, references, provenance, deduplication, reproducibility, Zenodo/submission readiness  
**Constraint observed:** No scientific claim changes beyond what stored results support. No results invented. Audit completed before manuscript edits.

---

## 1. Executive summary

The Version B manuscript is largely coherent as an **exploratory, schedule-stratified** study: transparency note, H1–H4 definitions, non-CORAL threshold-head disclosure, and H1/H2-only calibration are present. `paper/check_manuscript.py` previously reported `CHECK_MANUSCRIPT_OK` after the six-figure rebuild.

The most serious scientific inconsistency is a **false IDRiD CLAHE direction claim** in Results §4.2: the prose says CLAHE (H2) produced a higher mean IDRiD QWK than RGB (H1) in *both* schedule groups, but Table 3 and `table_v3_hybrid_qwk_n6_per_seed.csv` show the opposite for the truncated group (H2 0.364 ± 0.148 vs H1 0.395 ± 0.147) and the claimed direction only for the full-cosine group.

Secondary blockers for journal submission are **outdated README**, **missing LICENSE / Zenodo / CITATION**, **incomplete APTOS third-party provenance** (no archive URL/hash/download date in manuscript or [25]), **deduplication sensitivity not schedule-stratified and missing ECE/NLL/Brier/referable columns**, and several **recent / preprint / niche-venue references** that need accurate labelling and claim-support checks ([6], [10], [12], [13], [19]–[24]).

Publication readiness is **moderate for an exploratory methods/medical-AI article** after the IDRiD sentence fix, reference cleanup, provenance pack, and Zenodo code release — not yet submission-ready as of this audit.

---

## 2. Critical issues

| ID | Issue | Evidence | Required action |
|---|---|---|---|
| C1 | Results §4.2 claims CLAHE > RGB on IDRiD in **both** schedule groups; truncated group shows H2 < H1 | `MANUSCRIPT.md` L181 vs Table 3 / `table_v3_hybrid_qwk_n6_per_seed.csv` | Replace with schedule-specific wording (see §4 of master prompt) |
| C2 | APTOS cited as Kaggle competition only [25]; manuscript uses third-party pre-partitioned archive without URL, hash, or download date | §3.1 APTOS paragraph; References [25] | Document exact third-party source or mark TODO; expand [25] |
| C3 | README still describes baseline EfficientNet-B0 / Experiment 1 only; contradicts Version B manuscript | Root `README.md` | Rewrite README for Version B exploratory study |
| C4 | No LICENSE file; Zenodo release and journal data/code statements incomplete | Repo root | Add LICENSE + Zenodo metadata + code availability DOI placeholder |

---

## 3. High-priority issues

| ID | Issue | Notes |
|---|---|---|
| H1 | Deduplication sensitivity CSV lacks schedule stratification and ECE/NLL/Brier/referable metrics | `aptos_dedup_sensitivity.csv` has cell/seed/qwk/mild only |
| H2 | Manuscript reports pooled H2 dedup means (0.856 ± 0.024 / 0.851 ± 0.027) across six seeds without schedule split | §3.1 / §4.5; OK as sensitivity footnote if labelled; prefer schedule-stratified table |
| H3 | Cross-split label-disagreeing duplicates (12) mentioned in Table S1 audit CSV but under-emphasised in main prose | `aptos_split_audit.csv` |
| H4 | Reference [10] is ACSSE EfficientNet-*application* paper, not Tan & Le; weak venue for “EfficientNet often proposed” claim | Prefer stronger EfficientNet/DR citations; keep Tan & Le as architecture [35] |
| H5 | References [6], [20], [24] are arXiv; [24] labelled only as arXiv URL — OK if explicitly “arXiv preprint” | Label preprints clearly |
| H6 | Messidor-2 labels: images from ADCIS Messidor-2; adjudicated grades from Google Brain / Krause-panel lineage [29] — provenance must state both | `dataset_manifest.csv` points at `Messidor /messidor_data.csv` |
| H7 | Stale `environment.csv` says “version control, none” despite `.git` present | Update environment record |
| H8 | Cover letter, blinded manuscript, submission pack, Zenodo JSON/CFF absent | Create under `submission/` and root |

---

## 4. Medium-priority issues

| ID | Issue |
|---|---|
| M1 | Wording: “universally superior”, “H2 improved on IDRiD”, “clean causal estimates” — mostly hedged correctly; tighten “improved” at Table 12 prose |
| M2 | “rank-consistent CORAL models / CORAL” used in negation — acceptable; avoid implying trained H3/H4 are CORAL |
| M3 | IDRiD “pooled” train+test into external set — correct logistical meaning; distinct from forbidden six-seed pooling |
| M4 | Supplementary figure folder contains duplicate/stale filenames (e.g. `figS06_qwk_bootstrap_ci.png` beside `figS06_dataset_examples.png`) | Manuscript embeds correct paths; clean aliases |
| M5 | Deduplication does not report NLL/Brier; may need TODO if not recomputed |
| M6 | Several 2025–2026 papers ([5], [12], [13], [19], [21]–[23]) need DOI-resolution confirmation before submission |
| M7 | Code availability “on reasonable request” only — journals increasingly require public repo + DOI |
| M8 | No `environment.yml`; `requirements.txt` uses loose `>=` pins, not exact versions from `environment.csv` |

---

## 5. Low-priority issues

| ID | Issue |
|---|---|
| L1 | Keywords include “calibration” and “ordinal regression” — OK if threshold ordinality is carefully worded |
| L2 | Figure captions slightly longer than some journal limits |
| L3 | Confusion matrices S11–S14 sum seeds 42–47 (pooled counts) — acceptable for confusion display if text notes summation ≠ homogeneous replicates |
| L4 | `Book 4.xlsx` in repo root is unrelated clutter |
| L5 | macOS `26.3` / Python 3.13.2 is unusual; document clearly for reproducibility |

---

## 6. Files checked

| Path | Role | Status |
|---|---|---|
| `paper/MANUSCRIPT.md` | Built manuscript | Checked |
| `paper/MANUSCRIPT_TEMPLATE.md` | Editable source | Checked |
| `paper/MANUSCRIPT.docx` | Word build | Present (not re-diffed line-by-line) |
| `paper/check_manuscript.py` | Number/citation checker | Known OK previously |
| `paper/REFERENCES.md` | Reference notes | Present |
| `paper/tables/*.csv` | Metric sources | Sampled + key files fully read |
| `paper/tables/table_v3_hybrid_qwk_n6_per_seed.csv` | Per-seed QWK | Verified IDRiD H1/H2 |
| `paper/tables/aptos_dedup_sensitivity.csv` | Dedup sensitivity | Checked columns |
| `paper/tables/aptos_duplicate_pairs.csv` | 134 pairs | Checked |
| `paper/tables/aptos_split_audit.csv` | Split audit | Checked |
| `paper/tables/dataset_manifest.csv` | Dataset counts | Checked |
| `paper/tables/environment.csv` | Env record | Checked |
| `paper/figures/main/fig01–fig06*` | Main figures | Present |
| `paper/figures/supp/figS01–figS21*` | Supp figures | Present (some stale aliases) |
| `README.md` | Project README | Outdated |
| `requirements.txt` | Deps | Loose pins |
| `LICENSE` | License | **Missing** |
| `.zenodo.json` / `CITATION.cff` | Release metadata | **Missing** |
| `docs/` / `submission/` / `data_manifest/` | Submission pack | **Missing** |
| `configs/`, `src/`, `scripts/`, `results/`, `experiments/` | Code/artifacts | Present |
| `datasets/` | Local data | Present (not redistributed) |

---

## 7. Inconsistencies found

| ID | Location | Claim / state | Source truth | Severity |
|---|---|---|---|---|
| I1 | Results 4.2 | CLAHE higher IDRiD QWK in both groups | Truncated: H2 0.364 < H1 0.395; full: H2 0.336 > H1 0.136 | Critical |
| I2 | Abstract Results | Does not repeat false both-groups IDRiD claim | Abstract OK on this point; Results wrong | Critical (Results) |
| I3 | Ref [25] | Kaggle APTOS only | Study uses third-party partition files `train_1.csv` / `valid.csv` / `test.csv` | High |
| I4 | README title/scope | Baseline EfficientNet APTOS→IDRiD | Manuscript is MobileViT-S 2×2 + 3 externals | High |
| I5 | environment.csv | “no git repository” | `.git` exists | Medium |
| I6 | Dedup narrative | Six-seed H2 mean without schedule split | Primary analysis elsewhere schedule-stratified | Medium |
| I7 | Fig S11–S14 | “summed over seeds 42–47” | Summation pools schedule groups visually | Low–Medium |
| I8 | Ref [10] | Cited with EfficientNet lightweight claim | ACSSE application paper; architecture is [35] | Medium |

---

## 8. TODO items

| TODO ID | Item | Blocking? |
|---|---|---|
| T1 | Fix IDRiD CLAHE sentence in template + rebuild md/docx | Yes |
| T2 | Record APTOS third-party archive URL, download date, SHA-256 (or mark unavailable) | Yes |
| T3 | Expand Messidor-2 provenance: ADCIS images + Google adjudicated grades file | Yes |
| T4 | Build schedule-stratified `deduplication_sensitivity.csv` with requested columns; NLL/Brier TODO if not computable from stored logits | Yes |
| T5 | Rewrite README; add LICENSE, requirements pins, environment.yml | Yes |
| T6 | Create Zenodo/CITATION/docs/submission pack | Yes |
| T7 | Reference audit + preprint labels + DOI checks for [5][6][10][12][13][19]–[29] | Yes |
| T8 | Clean stale supplementary figure aliases | No |
| T9 | Obtain Zenodo DOI after GitHub release (cannot invent DOI) | Yes for final submission |
| T10 | Verify journal APC/indexing from official pages before selecting target | Yes |

---

## 9. Publication-readiness score

**62 / 100**

| Component | Score (0–10) | Comment |
|---|---|---|
| Version B framing / transparency | 9 | Strong |
| Numerical integrity (checker) | 7 | Checker OK; one prose contradiction |
| Tables / figures | 8 | Schedule-stratified main set good |
| References | 5 | Preprints/niche venues need care |
| Provenance / dedup | 5 | Partial; missing archive hash |
| Reproducibility packaging | 4 | Code exists; release tooling missing |
| Journal packaging | 2 | Submission files absent |

---

## 10. Recommended journal tier

**Tier: Mid exploratory / medical-AI methods venues (Scopus-indexed preferred), not top clinical ophthalmology.**

Rationale: exploratory scope, protocol deviation, non-rank-consistent threshold head, third-party APTOS partition, and CPU/batch-4 regime make *Nature Medicine / Ophthalmology / Lancet Digital Health* inappropriate. Better fit: medical imaging / biomedical engineering / applied AI journals that accept transparent negative/mixed external-validation findings.

Concrete shortlist will be provided in `JOURNAL_SHORTLIST.md` with APC routes verified from official pages where possible; unverified APC claims will be marked TODO.

---

*End of audit-only report. Subsequent files and fixes follow this audit.*

## 11. Resolution status (updated 2026-10-11, after fixes)

Everything above is the pre-fix audit snapshot and is kept unchanged. This section records how each finding was resolved.

| ID | Status | Evidence |
|---|---|---|
| I1 / T1 | Fixed | §4.2 now states H2 ≤ H1 on IDRiD (truncated) and H2 > H1 (full cosine); `NUMBERS_AUDIT.csv` semantic row |
| I3 / T2 | Fixed | [25] cites the Kaggle partition (MariaHerreroT, version 3); archive SHA-256 is in `data_manifest/aptos_archive_hash.csv` |
| I4 / T5 | Fixed | README rewritten for Version B; LICENSE, pinned `requirements.txt` and `environment.yml` added |
| I5 | Not an error | First git commit (2026-10-10) is after every training run, so “no git repository” was true when the environment was recorded |
| I6 / T4 | Fixed | §4.5 is schedule-stratified (group ≤ 0.007, run ≤ 0.015); Supplementary Table S6; NLL and Brier computed |
| I7 | Fixed | Figures S11–S14 regenerated with one row per schedule group (seeds summed only within a group); captions updated |
| I8 | Partly fixed | [10] title, DOI and venue verified; a stronger EfficientNet-DR citation is an optional author decision |
| T3 | Fixed | §3.1 names ADCIS images [27,28] and Google Brain adjudicated grades [29] |
| T6 | Fixed | `.zenodo.json`, `CITATION.cff`, `docs/`, `submission/` (built by `scripts/build_submission_package.py`) |
| T7 | Fixed | `scripts/audit_references.py`: every listed DOI and arXiv ID resolves with a matching title |
| T8 | Open (optional) | Stale supplementary figure aliases |
| T9 | Open | Zenodo DOI requires making the GitHub repo public and creating release `v1.0.0-exploratory` |
| T10 | Open | APC, scope and indexing to be re-verified on submission day |

The numbers audit (`scripts/audit_numbers.py`) traced 380 numeric tokens. All 204 cells of Tables 3–6 and 8 were recomputed from raw per-run JSON with zero mismatches. The remaining tokens were matched to source CSVs or to placeholders with named sources. **Updated readiness score: 80 / 100.** The remaining gap is the public code release with a DOI, plus same-day journal checks.

## 12. Second external review (2026-10-11)

| Point | Verdict after checking the data | Action |
|---|---|---|
| Table 5 H3 IDRiD 0.333 ± 0.577 | Confirmed from saved predictions: seed 42 recalled 25/25 by predicting grade 1 for 305/516 IDRiD images; seeds 43–44 recalled 0/25 | Sentence in §4.3 and notes in the Table 5 and Table 8 captions (all values from placeholders; claim-checked) |
| Table 8 inherits that spread | Confirmed (H3−H1 +0.107 ± 0.780; H4−H3 +0.173 ± 0.441) | Table 8 caption note |
| §4.6 pooled seeds 42–47 | Confirmed | Messidor-2 grade-0 shares now reported per schedule group |
| Table 9 “three backbones”, four rows | Confirmed | Caption states four rows (MobileViT-S H2 in both schedule groups) |
| Table 9 caption attached to decomposition tables | Not reproduced: in the built .docx every caption appears once, directly above its own table | None needed; likely a text-extraction artefact |
| 448×448 mild-recall claim | Confirmed overstatement (0.467 ± 0.033 vs 0.422 ± 0.019) | §4.7 and Discussion reworded; claim check (difference < 0.05) |
| D2 wording | Confirmed | “relative to truncated-schedule H2”, modest vs full-cosine H2 |
| CLAHE’s most repeated result | Confirmed: H2−H1 QWK positive 3/3 on APTOS and DeepDRiD in both groups | Stated in Abstract, §4.2 and Discussion; claim-checked |
| RGB collapse | Confirmed (H1 full cosine: external QWK ≤ 0.136, zero external grade-1 recall, 100% grade 0 on Messidor-2; matched B0 far higher) | New paragraph in §4.2 and sentence in Discussion |
| Figures S11–S14 pooling | Confirmed | Regenerated per schedule group |
| References lacking volume/pages | Confirmed for [8], [9], [14], [17], [18], [22], [23], [31], [35]–[38] | Filled from Crossref / PMLR / arXiv records; Swin [36] pages corrected to the DOI version (9992–10002). [10] has no pages in Crossref |
| Stray “&” in references | Not an error: “&” belongs to the journal names in [23] and [27] and is stored as a valid XML escape | None |
| Missing archive hashes | Confirmed | IDRiD zip SHA-256, DeepDRiD commit, and manifest digests for all four image folders (`data_manifest/`) |
