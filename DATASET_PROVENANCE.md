# Dataset provenance audit

**Audit date:** 2026-10-11  
**Rule:** Do not invent download dates, archive URLs, or checksums. Missing fields are labelled **TODO**.

Label-file hashes recorded in `data_manifest/`.

---

## APTOS 2019 (third-party pre-partitioned derivative)

| Field | Value |
|---|---|
| Official dataset name | APTOS 2019 Blindness Detection (Asia Pacific Tele-Ophthalmology Society) |
| Official paper / competition | Kaggle competition page (cited within [25]) |
| Official competition URL | https://www.kaggle.com/c/aptos2019-blindness-detection |
| **Partition used in this study** | **Third-party pre-partitioned derivative**, not an official competition train/validation/test split |
| Partition source | Kaggle dataset “APTOS-2019 dataset” by MariaHerreroT, version 3: https://www.kaggle.com/datasets/mariaherrerot/aptos2019 [25] |
| Local root | `datasets/archive/` |
| Label files | `train_1.csv`, `valid.csv`, `test.csv` |
| Image folders | `train_images/`, `val_images/`, `test_images/` |
| Label mapping | ICDR diagnosis column → grades 0–4 |
| Inclusion | All labelled images in the three CSV files |
| Exclusion | None beyond the distributed files |
| Counts | Train 2,930; Val 366; Test 366; Total 3,662 |
| Adjudication | Competition labels (single-source as distributed); not re-adjudicated here |
| Patient/eye metadata | Not used / not consistently available in this derivative |
| License / redistribution | Kaggle page lists the licence as “Unknown”; underlying images follow the APTOS 2019 competition rules. **Do not upload images to Zenodo** |
| Results caveat | **Results are specific to this partition only** |

### APTOS archive record

Written by `scripts/hash_dataset_archives.py` to `data_manifest/aptos_archive_hash.csv`.

| Item | Value |
|---|---|
| Archive URL | https://www.kaggle.com/datasets/mariaherrerot/aptos2019 (version 3) |
| Archive filename | `archive.zip` (local: `datasets/archive.zip`) |
| Archive SHA-256 | `ffdf1300357080f0d493d2bdb2857511fedd827e6ee4021a8c4966459ac874dc` |
| Size | 8,602,506,535 bytes; 3,662 image members |
| Download date | 2026-09-16 (local file modification date; the exact download timestamp was not recorded separately) |
| CSV member dates inside the zip | `train_1.csv`, `valid.csv`, `test.csv`: 2021-11-17 |

### APTOS label-file SHA-256 (local copies)

See `data_manifest/aptos_label_file_hashes.csv` (computed 2026-10-11 from local files).

---

## IDRiD

| Field | Value |
|---|---|
| Official name | Indian Diabetic Retinopathy Image Dataset (IDRiD) |
| Official paper | Porwal et al., *Data*, 2018 [26] |
| Official download | IEEE DataPort / dataset site linked from [26] |
| Local path | `datasets/idrid/` |
| Label files | Official Disease Grading training and testing CSVs |
| Mapping | Official ICDR grades 0–4 |
| Inclusion | Training (413) + test (103) **pooled as one external set** (516) because IDRiD is never used for training |
| Exclusion | None beyond official disease-grading set |
| Adjudication | Official IDRiD disease-grading labels |
| Patient/eye metadata | Available in IDRiD release; not used beyond grading |
| License | CC BY 4.0 (`LICENSE.txt` / `CC-BY-4.0.txt` shipped in the archive); attribution to Porwal et al. required |
| Archive hash | `B. Disease Grading.zip` SHA-256 `8a9f4752…3e4d8c` (212,405,123 bytes; local date 2026-09-16); full values in `data_manifest/external_dataset_hashes.csv` |
| Files used | 516 images; per-file SHA-256 in `data_manifest/idrid_file_manifest.csv` |

---

## Messidor-2

| Field | Value |
|---|---|
| Official image collection | Messidor-2 (ADCIS) — 874 examinations / 1,748 images; **no official DR ground truth from ADCIS** |
| Official image URL | https://www.adcis.net/en/third-party/messidor2/ |
| Related Messidor paper | Decencière et al. [27]; Abràmoff et al. context [28] |
| **Labels used here** | Adjudicated five-level DR + gradability from the Google Brain Messidor-2 DR grades release associated with the Krause et al. adjudication protocol [29] |
| Local label files | `Messidor /messidor_data.csv`, `Messidor /messidor-2.csv` |
| Inclusion | Images with `adjudicated_gradable=1` → **1,744** images |
| Exclusion | Non-gradable per adjudicated flag |
| Label type | Multi-grader adjudication (per Krause protocol / Google grades), not ADCIS official GT |
| Patient/eye metadata | Pairing spreadsheet exists for examinations; evaluation here is image-level |
| License | Messidor-2 download requires ADCIS access form; labels have separate Google/Kaggle terms |
| Archive hash | Original ADCIS download archive not retained. The extracted `IMAGES/` folder (1,748 files) is identified by manifest digest `991d2971…9858c4` (local date 2026-09-19); per-file SHA-256 in `data_manifest/messidor2_file_manifest.csv`. Label SHA-256 in `data_manifest/messidor_label_file_hashes.csv` |

---

## DeepDRiD

| Field | Value |
|---|---|
| Official name | DeepDRiD challenge dataset |
| Official paper | Liu et al., *Patterns*, 2022 [30] |
| Local path | `datasets/DeepDRiD/regular_fundus_images/` |
| Labels | Regular fundus training/validation CSVs (left/right eye grades) |
| Inclusion | Regular fundus graded images used in eval → **1,589** (`dataset_manifest.csv`) |
| Exclusion | Challenge subsets not used in this external eval |
| Adjudication | Challenge-provided grades |
| Source version | git clone of https://github.com/deepdrdoc/DeepDRiD at commit `56d8af71319803150d0e7b90533e0918c7bf85ee` (2022-07-27) |
| Files | `regular_fundus_images/` manifest digest `b279c727…42d2dd` (2,000 image files; local date 2026-09-19); per-file SHA-256 in `data_manifest/deepdrid_file_manifest.csv` |
| License | CC BY-SA 4.0 (`LICENSE` in the repository); do not upload images to Zenodo |

---

## Cross-cutting notes

1. No dataset images should be packaged in the Zenodo code release.  
2. Predictions/logits/metrics may be released if compatible with dataset licenses (confirm before upload).  
3. All four datasets now have a hash record written by `scripts/hash_dataset_archives.py`: original archives (APTOS, IDRiD), a git commit (DeepDRiD), and manifest digests of the extracted image folders (all four). The only gap is the original Messidor-2 download archive, which was not retained; the per-file manifest identifies the exact images used.
