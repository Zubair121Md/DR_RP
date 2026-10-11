# Manuscript consistency report

**Source of truth for edits:** `paper/MANUSCRIPT_TEMPLATE.md` (rebuild `MANUSCRIPT.md` / `.docx` after changes).  
**Audit date:** 2026-10-11

## Summary

Version B framing is consistent (exploratory, schedule-stratified, threshold-head not CORAL, ECE for H1/H2 only). One critical Results prose error on IDRiD CLAHE direction was identified and corrected in the template. Remaining gaps are provenance/reference packaging, not internal H1–H4 definition drift.

## Consistency table

| Location | Issue | Current text | Required correction | Priority | Status |
|---|---|---|---|---|---|
| Title / Abstract / Keywords | Title matches Version B exploratory framing | Full title ends with “An Exploratory Evaluation” | Keep; update root README to match | High | Open (README) |
| Results §4.2 | False both-groups IDRiD CLAHE claim | “CLAHE produced a higher mean IDRiD QWK than RGB in both groups (0.364… vs 0.395…; 0.336… vs 0.136…)” | Schedule-specific: trunc H2 did not exceed H1; full H2 exceeded H1 | Critical | Fixed in template |
| Abstract Results | Does not claim both-groups IDRiD CLAHE gain | Dataset/schedule-specific H2 APTOS/DeepDRiD numbers | Keep; ensure no reintroduction of both-groups claim | Critical | OK |
| Research question | Exploratory APTOS-only MobileViT-S | Matches Methods/Results | Keep | — | OK |
| Novelty statement | Schedule-stratified + non-monotonic thresholds | Matches transparency note | Keep | — | OK |
| “exploratory evaluation” | Used throughout abstract/methods/discussion | Consistent | Keep | — | OK |
| Seeds 42–44 / 45–47 | Separated in Tables 3–6 and Figs 4–6 | Primary tables schedule-stratified | Keep; avoid pooled confirmatory claims | — | OK |
| Dedup §3.1 / §4.5 | H2 means over six seeds without schedule split | “0.856 ± 0.024 … 0.851 ± 0.027” | Prefer schedule-stratified sensitivity table or label as pooled sensitivity only | Medium | Partial (CSV created) |
| H1–H4 definitions | Table in §3.6 | RGB/CLAHE × CE/MW threshold | Keep consistent wording (“MW threshold objective”) | — | OK |
| Table 1 counts | Match `dataset_manifest.csv` | 2930/366/366; 516; 1744; 1589 | Keep | — | OK |
| Fig/Table numbering | Main Figs 1–6; Tables 1–12; S1–S21 | Checker previously OK | Rebuild after edits | High | Rebuild pending |
| Abbreviations | DR, CLAHE, QWK, ECE, ICDR, MW defined | First-use generally OK | Spot-check remaining | Low | OK |
| “superior” / “improved” / “causal” | Hedged but present | “universally superior”, “H2 improved”, “causal estimates” | Softened in template | Medium | Fixed in template |
| “rank-consistent CORAL” | Used in negation only | “do not describe … as rank-consistent CORAL” | Keep negation; do not affirm | — | OK |
| Ref [6][20][24] | arXiv | Not labelled preprint | Label “arXiv preprint” | High | Fixed in template |
| Ref [25] | Kaggle only | Missing third-party partition disclosure in ref | [25] now cites the Kaggle partition (MariaHerreroT, version 3) used; archive SHA-256 recorded | High | Fixed |
| Messidor-2 §3.1 | “Adjudicated DR grades [29]” | Underspecifies image vs label sources | Clarify ADCIS images + Google adjudicated grades | High | Fixed in template |
| Code availability | Reasonable request only | No Zenodo DOI | Add DOI when obtained | High | Open |
| Root README | Baseline EfficientNet experiment | Does not match manuscript | Rewrite | Critical | Open |

## Cross-reference check (high level)

| Item | Cited in text | Caption present | Notes |
|---|---|---|---|
| Figures 1–6 | Yes | Yes | Main Version B set |
| Tables 1–12 | Yes | Yes | Schedule-stratified primary tables 3–6 |
| Figures S1–S21 | Yes | Yes | Stale alias files may coexist on disk |
| Tables S1–S5 | Yes | Yes | — |

## Numerical agreement (selected)

Abstract H2 APTOS/DeepDRiD and Table 3 IDRiD cells match `table_v3_hybrid_qwk_n6_per_seed.csv` (see `NUMBERS_AUDIT.csv`). The only flagged mismatch in that audit is the pre-fix Results prose claim.
