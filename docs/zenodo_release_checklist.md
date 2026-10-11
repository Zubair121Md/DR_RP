# Zenodo release checklist

1. Create public GitHub repository (code + docs; **no fundus images**).
2. Push code, configs, scripts, metric tables, allowed prediction files, documentation.
3. Create GitHub release tagged `v1.0.0-exploratory`.
4. Connect the GitHub repository to Zenodo.
5. Create a Zenodo deposit from the GitHub tag (`.zenodo.json` present).
6. Obtain Zenodo DOI.
7. Add DOI badge to `README.md`.
8. Add DOI to manuscript Code availability (edit `paper/MANUSCRIPT_TEMPLATE.md` then rebuild).
9. Add DOI to cover letter.
10. Archive the exact commit hash in `ZENODO_RELEASE_CHECKLIST.md` / release notes.
11. Verify release contents against `REPRODUCIBILITY_CHECKLIST.md`.
12. Verify MIT license compatibility and dataset redistribution restrictions.

**Current status:** Packaging files created; GitHub URL recorded (repo private); DOI is **TODO**.
