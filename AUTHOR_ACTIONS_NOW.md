# Author actions that still need a human click

Everything that can be automated from this machine is done (see `PUBLISHING_STATUS.md`).
The steps below require **your** login. Do them in this order. Estimated time: 30–60 minutes
for Zenodo + preprint; journal submission is a separate session (1–2 hours).

---

## 1. Zenodo DOI (blocks final Code-availability wording)

1. Open https://zenodo.org and sign in with **GitHub** (account that owns `Zubair121Md`).
2. Go to https://zenodo.org/account/settings/github/ and flip **ON** for `Zubair121Md/DR_RP`.
3. Open https://github.com/Zubair121Md/DR_RP/releases/tag/v1.0.0-exploratory  
   — or create a new release `v1.0.1-exploratory` from current `main` (preferred so the
   archived commit includes the latest caption/claim-check fixes).
4. Zenodo will create a deposit automatically. Copy:
   - **Version DOI** (this release), e.g. `10.5281/zenodo.xxxxxxx`
   - **Concept DOI** (all versions)
5. Paste both into a reply in this chat **or** run:

```bash
export ZENODO_DOI=10.5281/zenodo.XXXXXXX   # version DOI
python3 scripts/insert_zenodo_doi.py
```

That script updates `paper/MANUSCRIPT_TEMPLATE.md`, `README.md`, `CITATION.cff`,
`ZENODO_RELEASE_CHECKLIST.md`, rebuilds the manuscript and submission pack.

**Manual upload fallback** (if GitHub integration fails): upload
`dist/DR_RP-v1.0.0-exploratory-source.zip` at https://zenodo.org/deposit/new
using metadata from `.zenodo.json`. Prefer the GitHub integration.

---

## 2. Preprint (recommended same day as journal submit)

**medRxiv** (clinical / medical AI framing): https://www.medrxiv.org/submit-a-manuscript  
**arXiv** (cs.CV or eess.IV): https://arxiv.org/submit  

Upload file ready locally:

- `preprint/MANUSCRIPT.docx` (copy of the built manuscript; convert to PDF in Word/Pages if the server requires PDF)

Title, abstract and author metadata match the manuscript. State:

> Code: https://github.com/Zubair121Md/DR_RP (release v1.0.0-exploratory); Zenodo DOI: \<insert after step 1\>.

PLOS ONE allows preprints. After the preprint DOI appears, add it to the cover letter.

---

## 3. Submit to PLOS ONE

Portal: https://www.editorialmanager.com/pone/  

Use the local pack (git-ignored, not on GitHub):

| File | Path |
|---|---|
| Main manuscript | `submission/01_Main_Manuscript.docx` |
| Cover letter | `submission/02_Cover_Letter.docx` |
| Main figures | `submission/03_Figures/Figure_1…6.png` (or PDF) |
| Supplementary | `submission/04_Supplementary_Material/` |
| Data/code statement | `submission/05_Data_Code_Availability.txt` |
| Declarations | `submission/06_Declarations.txt` |
| Blinded MS | not required for PLOS ONE (single-blind) — keep for other journals |

**At submission, before payment screen:**

- Article type: Research Article  
- Subject areas: ophthalmology / medical imaging / machine learning (pick closest)  
- **Publication Fee Assistance** or Research4Life: apply **now** if you cannot pay the APC  
  (editors do not see fee info). Official fees: https://plos.org/fees/  
- Data availability: GitHub URL + Zenodo DOI; datasets under provider licences  
- Competing interests: none  
- Financial disclosure: no external funding  
- Ethics: not required (public de-identified datasets)  
- Preprint: yes / URL if posted  

**Do not invent a Zenodo DOI in the portal.** If step 1 is unfinished, write
“GitHub release v1.0.0-exploratory; Zenodo DOI to be added before acceptance.”

---

## 4. Optional: ORCID

Add ORCID to Zenodo creators and to the PLOS author profile so citations link cleanly.
Update `.zenodo.json` `"orcid"` field when you have it.
