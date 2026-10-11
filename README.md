# Green-Channel CLAHE and Mild-Weighted Threshold Objective for Cross-Dataset DR Grading (MobileViT-S)

**Exploratory evaluation (Version B).** APTOS-only training of MobileViT-S with a 2×2 design:

| Cell | Input | Loss |
|------|--------|------|
| H1 | RGB | Cross-entropy |
| H2 | Green-channel CLAHE | Cross-entropy |
| H3 | RGB | Mild-weighted shared-weight threshold objective |
| H4 | Green-channel CLAHE | Mild-weighted shared-weight threshold objective |

External evaluation: **IDRiD**, **Messidor-2**, **DeepDRiD**. Checkpoints selected by APTOS validation QWK only.

## Critical transparency

1. **Seeds 42–44** received a **truncated** learning-rate schedule (checkpoint-resume scheduler defect).  
2. **Seeds 45–47** received the intended **full cosine** schedule.  
3. These groups are **not** exchangeable six-seed replicates.  
4. The threshold head is **not** rank-consistent CORAL; **ECE is reported only for H1/H2**.

Manuscript: `paper/MANUSCRIPT.md` / `paper/MANUSCRIPT.docx`  
Audit pack: `REVIEW_REPORT.md`, `MANUSCRIPT_CONSISTENCY_REPORT.md`, `DATASET_PROVENANCE.md`, …

## Environment (as recorded)

From `paper/tables/environment.csv` (training machine):

- Python **3.13.2**
- PyTorch **2.7.1**, torchvision **0.22.1**, timm **1.0.29**
- Apple M2 CPU training / MPS evaluation; batch size 4; BatchNorm frozen

Install (approximate; prefer pinned `environment.yml` when recreating):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce tables and figures

```bash
python3 paper/figures/make_all.py
# or stepwise:
python3 scripts/build_manuscript_md.py
python3 scripts/build_manuscript_docx.py
python3 paper/check_manuscript.py
```

## Data

Datasets are **not** redistributed. See `DATASET_PROVENANCE.md` and `data_manifest/` (archive SHA-256s, the DeepDRiD git commit, and per-file hashes of every image used).
The APTOS partition used here is the third-party Kaggle “APTOS-2019 dataset” (version 3), https://www.kaggle.com/datasets/mariaherrerot/aptos2019.

### Figures that contain fundus images

The repository contains **no dataset pixels**. The following outputs are generated locally only and are git-ignored:
Figure 2 (preprocessing), Supplementary Figures S4 (augmentation), S6 (dataset examples) and S19 (Grad-CAM),
all Grad-CAM overlays, preprocessing diagnostics, and the manuscript `.docx` files and `submission/` package that embed them.
With the datasets in place, `python3 paper/figures/make_all.py` regenerates them; without the datasets the figure scripts
print a `SKIP …` message and continue. Building the `.docx` requires those figures.

Per-run metric files and predictions under `results/` and model checkpoints are also not in the repository.

## Licence

- **Code, configuration files, scripts and documentation:** MIT Licence (`LICENSE`). You may reuse, modify and redistribute them, including commercially, provided the copyright and licence notice are kept.
- **Derived tables and non-image figures** in `paper/tables/` and `paper/figures/` (metrics, charts, diagrams): also released under the MIT Licence. They are derived from the datasets' labels and model outputs, so please also cite the dataset providers.
- **Dataset images and labels:** not included and not covered by this licence. They remain under their providers' terms: IDRiD CC BY 4.0; DeepDRiD CC BY-SA 4.0; Messidor-2 ADCIS terms (adjudicated grades: Krause et al.); the APTOS partition is listed with licence “Unknown” on Kaggle and derives from the APTOS 2019 competition data.

## Citation

See `CITATION.cff`. GitHub release: `v1.0.0-exploratory`. Zenodo DOI: **TODO** (requires a public repository and the Zenodo GitHub integration).

## Contact

Zubair Md Ishaq — zubairishaq12@gmail.com
