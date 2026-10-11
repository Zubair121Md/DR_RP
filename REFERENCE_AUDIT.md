# Reference audit ([1]–[38])

**Audit date:** 2026-10-11  
**Detail CSV:** `REFERENCE_AUDIT.csv`

## Summary

- All 38 references are cited in the manuscript body (per prior `CHECK_MANUSCRIPT_OK`).
- No uncited reference gaps detected in the numbered list.
- Preprints **[6], [20], [24]** are now labelled *arXiv preprint* in the template.
- **[25]** now cites the Kaggle partition actually used (MariaHerreroT, “APTOS-2019 dataset”, version 3, https://www.kaggle.com/datasets/mariaherrerot/aptos2019) and, within the same entry, the original APTOS 2019 competition. Archive SHA-256 is in `data_manifest/aptos_archive_hash.csv`.
- **Live resolution (`scripts/audit_references.py`):** every listed DOI ([1]–[16], [18]–[24], [26]–[30]) and the arXiv ID for [34] was resolved against Crossref / arXiv; all titles match (SequenceMatcher ≥ 0.85) and Crossref years and containers agree. [10]'s venue was corrected to *Applied Computer Science and Software Engineering* to match Crossref.
- **[27]–[29]** Messidor chain clarified in Methods: ADCIS Messidor-2 images + Google adjudicated grades linked to Krause et al. protocol.

## Priority corrections applied

| Ref | Action |
|---|---|
| [6] | Label as arXiv preprint |
| [20] | Label as arXiv preprint |
| [24] | Label as arXiv preprint |
| [25] | Disclose third-party partition usage |
| Methods Messidor | Clarify image vs label sources |

## Priority corrections still TODO (do not invent)

| Ref | Issue | TODO |
|---|---|---|
| [10] | Small-venue EfficientNet-B0 *application* paper; DOI and title verified, but a stronger peer-reviewed EfficientNet-DR citation would be preferable | Optional (author decision) |
| [17], [31], [35]–[38] | Conference papers listed without DOI/pages | Optional: add proceedings pages/DOI if the target journal's style requires them |

## Claim-support notes

| Claim area | Supporting refs | Caution |
|---|---|---|
| CORAL rank consistency | [15] | Do not imply H3/H4 are CORAL |
| Calibration / temperature | [38] | CE models only |
| Messidor adjudicated labels | [29] + Google grades release | Not official ADCIS ground truth |
| APTOS | [25] + provenance doc | Partition-specific results |

## Duplicate / missing check

- No duplicate DOIs detected in the reference list.
- Numbering 1–38 contiguous.
