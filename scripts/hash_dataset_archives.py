#!/usr/bin/env python3
"""
Record dataset provenance without redistributing images.

  data_manifest/aptos_archive_hash.csv       SHA-256, size and dates of the APTOS Kaggle archive
  data_manifest/external_dataset_hashes.csv  IDRiD original zip archives, the DeepDRiD git commit, and a manifest
                                             digest of every extracted image folder actually read by the code
  data_manifest/<dataset>_file_manifest.csv  relative path, size and SHA-256 of every image file (no pixels)

A manifest digest is the SHA-256 of the sorted "relative_path,sha256" lines of a folder. It identifies the exact files
used; it is not the hash of an original download archive.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data_manifest"
APTOS_ARCHIVE = ROOT / "datasets" / "archive.zip"
APTOS_URL = "https://www.kaggle.com/datasets/mariaherrerot/aptos2019"
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

ARCHIVES = [
    ("IDRiD", ROOT / "datasets/idrid/B. Disease Grading.zip", "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"),
    ("IDRiD", ROOT / "datasets/idrid/A. Segmentation.zip", "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"),
    ("IDRiD", ROOT / "datasets/idrid/C. Localization.zip", "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"),
]
FOLDERS = [
    ("APTOS", ROOT / "datasets/archive", APTOS_URL),
    ("IDRiD", ROOT / "datasets/idrid/B. Disease Grading", ARCHIVES[0][2]),
    ("Messidor-2", ROOT / "Messidor " / "IMAGES", "https://www.adcis.net/en/third-party/messidor2/"),
    ("DeepDRiD", ROOT / "datasets/DeepDRiD/regular_fundus_images", "https://github.com/deepdrdoc/DeepDRiD"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mdate(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).date().isoformat()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def aptos_archive() -> None:
    if not APTOS_ARCHIVE.is_file():
        print("SKIP APTOS archive: not present locally", rel(APTOS_ARCHIVE))
        return
    print("READ", rel(APTOS_ARCHIVE))
    with zipfile.ZipFile(APTOS_ARCHIVE) as z:
        csvs = {i.filename: dt.datetime(*i.date_time).date().isoformat() for i in z.infolist() if i.filename.endswith(".csv")}
        n_img = sum(1 for i in z.infolist() if i.filename.lower().endswith(IMG_EXT))
    out = MAN / "aptos_archive_hash.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "source_url", "sha256", "size_bytes", "local_mtime", "n_image_members", "csv_members_and_zip_dates"])
        w.writerow([APTOS_ARCHIVE.name, APTOS_URL, sha256(APTOS_ARCHIVE), APTOS_ARCHIVE.stat().st_size, mdate(APTOS_ARCHIVE),
                    n_img, "; ".join(f"{k}={v}" for k, v in sorted(csvs.items()))])
    print("WROTE", rel(out))


def main() -> None:
    aptos_archive()
    rows = []
    for ds, p, url in ARCHIVES:
        if not p.is_file():
            print("SKIP", ds, "archive not present:", rel(p))
            continue
        print("READ", rel(p))
        rows.append([ds, "original zip archive", rel(p), url, sha256(p), p.stat().st_size, mdate(p), ""])

    deep = ROOT / "datasets/DeepDRiD"
    if (deep / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(deep), "rev-parse", "HEAD"], text=True).strip()
        when = subprocess.check_output(["git", "-C", str(deep), "log", "-1", "--format=%cs"], text=True).strip()
        remote = subprocess.check_output(["git", "-C", str(deep), "remote", "get-url", "origin"], text=True).strip()
        rows.append(["DeepDRiD", "git clone (commit)", rel(deep), remote, commit, "", when, "commit date"])

    for ds, folder, url in FOLDERS:
        if not folder.is_dir():
            print("SKIP", ds, "folder not present:", rel(folder))
            continue
        files = sorted(q for q in folder.rglob("*") if q.is_file() and q.suffix.lower() in IMG_EXT)
        print("READ", rel(folder), len(files), "image files")
        man = []
        for q in files:
            man.append((str(q.relative_to(folder)), q.stat().st_size, sha256(q)))
        digest = hashlib.sha256("\n".join(f"{a},{c}" for a, _, c in man).encode()).hexdigest()
        stem = ds.lower().replace("-", "")
        out = MAN / f"{stem}_file_manifest.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relative_path", "size_bytes", "sha256"])
            w.writerows(man)
        print("WROTE", rel(out))
        rows.append([ds, "extracted image folder (manifest digest)", rel(folder), url, digest,
                     sum(b for _, b, _ in man), mdate(folder), f"{len(man)} image files; per-file hashes in {out.name}"])

    out = MAN / "external_dataset_hashes.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "kind", "local_path", "official_source", "sha256_or_commit", "size_bytes", "local_date", "notes"])
        w.writerows(rows)
    print("WROTE", rel(out), len(rows), "rows")


if __name__ == "__main__":
    main()
