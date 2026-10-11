# Submission go / no-go checklist

**Date:** 2026-10-11  
**Overall recommendation:** **CONDITIONAL GO.** The manuscript, audits and submission pack are ready. Two items must be done by the author before upload: a public code release with a Zenodo DOI, and same-day verification of the chosen journal.

## Checklist

- [x] Title updated (Version B exploratory).
- [x] Abstract updated (schedule-stratified; CE-only calibration).
- [x] Research question updated.
- [x] Novelty statement updated.
- [x] Schedule groups separated (Tables 3–6, Figs 4–6).
- [x] No pooled six-seed claims as a confirmatory headline.
- [x] No CORAL calibration claims (ECE for H1/H2 only).
- [x] Contradictory IDRiD CLAHE sentence corrected and manuscript rebuilt (`CHECK_MANUSCRIPT_OK`).
- [x] Deduplication sensitivity completed (all requested columns incl. NLL/Brier; Supplementary Table S6).
- [x] Dataset provenance completed for all four datasets (archives, DeepDRiD commit, per-file manifests).
- [x] Reference audit completed, with live DOI/arXiv resolution of every listed identifier.
- [x] Numbers audit completed (`NUMBERS_AUDIT.csv`; 0 mismatches, 0 untraced).
- [x] Figure audit completed.
- [x] Table audit completed.
- [x] Code pushed to GitHub (https://github.com/Zubair121Md/DR_RP, currently **private**), with a fresh history containing no dataset pixels.
- [x] GitHub release `v1.0.0-exploratory` created.
- [ ] Repository made public.
- [ ] Zenodo DOI obtained and inserted into Code availability, README, `CITATION.cff` and the cover letter.
- [x] Cover letter prepared (`submission/02_Cover_Letter.docx`).
- [x] Blinded manuscript prepared (`submission/07_Blinded_Manuscript.docx`; identity tokens and docx metadata checked by script).
- [ ] Journal APC verified on submission day.
- [ ] Journal scope verified for chosen target.
- [ ] Journal indexing verified for chosen target.

## Go criteria

All boxes checked **and** the chosen journal's official APC, scope and indexing pages re-verified on the day of submission.

## Remaining blockers (author actions)

1. Make the GitHub repo public, enable the Zenodo GitHub integration, and re-publish (or create a new) release so Zenodo archives it; record the DOI.
2. Replace "will be made public and archived on Zenodo" in Code availability with the DOI, restore or regenerate the fundus figures locally (see README), then rerun `scripts/build_manuscript_md.py`, `scripts/build_manuscript_docx.py`, `paper/check_manuscript.py` and `scripts/build_submission_package.py`.
3. Verify the target journal's APC, scope and indexing on its official pages.
