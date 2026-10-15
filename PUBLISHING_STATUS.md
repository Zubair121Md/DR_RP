# Publishing status (15 October 2026)

## Done on this machine

| Stage | Status |
|---|---|
| 0. Claim freeze / rebuild | `BUILD_MANUSCRIPT_MD_OK`, all `CLAIM_OK`, `CHECK_MANUSCRIPT_OK` |
| 1. Commit + push | `d31e88a` on `main` at https://github.com/Zubair121Md/DR_RP |
| 2. Zenodo DOI | **Blocked on your login** — see `AUTHOR_ACTIONS_NOW.md` §1. Source zip ready: `dist/DR_RP-v1.0.0-exploratory-source.zip` |
| 3. Figures + docx + submission pack | Built locally in `paper/MANUSCRIPT.docx` and `submission/` (git-ignored; contains fundus figures) |
| 4. Integrity checks | Manuscript checker passed; restricted fundus files remain outside git |
| 5. Journal verification | PLOS ONE primary — see `JOURNAL_VERIFICATION.md` |
| 6. Preprint file | `preprint/MANUSCRIPT.docx` ready to upload |
| 7. Journal portal submit | **Blocked on your login** — see `AUTHOR_ACTIONS_NOW.md` §3 |

## GitHub

- Visibility: **public**
- Release tag: `v1.0.0-exploratory`
- Fresh history: no dataset fundus pixels in `git rev-list --all --objects`
- Licence: MIT

## Local-only (do not commit)

- `paper/MANUSCRIPT.docx`
- `submission/`
- Restricted PNGs/PDFs under `paper/figures/` (preprocess, dataset examples, Grad-CAM, augmentation)
- `dist/*.zip`
- Backup of old history + moved files: `~/Desktop/dr-research-restricted-local/`

## After you finish AUTHOR_ACTIONS_NOW §1

```bash
export ZENODO_DOI=10.5281/zenodo.XXXXXXX
python3 scripts/insert_zenodo_doi.py
# then ask the agent to commit + push + retag
```
