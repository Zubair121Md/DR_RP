# Data manifest (no images)

This folder stores checksums and provenance notes for the dataset archives and label files used in the study. Fundus images are not packaged.

| File | Content |
|---|---|
| `aptos_archive_hash.csv` | SHA-256, size, local date and CSV member dates of the APTOS partition `archive.zip` from https://www.kaggle.com/datasets/mariaherrerot/aptos2019 (version 3); written by `scripts/hash_dataset_archives.py` |
| `aptos_label_file_hashes.csv` | SHA-256 of local `datasets/archive/{train_1,valid,test}.csv` |
| `messidor_label_file_hashes.csv` | SHA-256 of local Messidor label CSVs |
| `external_dataset_hashes.csv` | IDRiD original zip SHA-256, DeepDRiD git commit, and manifest digests of every extracted image folder |
| `<dataset>_file_manifest.csv` | Relative path, size and SHA-256 of every image file read by the code (no pixels) |

All files are written by `scripts/hash_dataset_archives.py`. A manifest digest is the SHA-256 of the sorted `relative_path,sha256` lines of a folder; it identifies the exact files used but is not an original-archive hash. The original Messidor-2 download archive was not retained.
