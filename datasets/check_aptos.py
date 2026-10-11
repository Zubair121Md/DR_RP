#!/usr/bin/env python3
"""
Verify APTOS retinal images and label CSVs are correctly connected.

Does not modify or move any dataset files. Does not train a model.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

# Project root = directory containing this script
ROOT = Path(__file__).resolve().parent

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

# Expected layout under archive/ (discovered from this project)
SPLIT_SPECS = [
    {
        "name": "train",
        "csv_candidates": ["train_1.csv", "train.csv"],
        "image_dir_candidates": [
            "train_images/train_images",
            "train_images",
            "train",
        ],
    },
    {
        "name": "validation",
        "csv_candidates": ["valid.csv", "val.csv", "validation.csv"],
        "image_dir_candidates": [
            "val_images/val_images",
            "val_images",
            "valid_images/valid_images",
            "valid_images",
            "validation_images",
            "val",
            "valid",
        ],
    },
    {
        "name": "test",
        "csv_candidates": ["test.csv"],
        "image_dir_candidates": [
            "test_images/test_images",
            "test_images",
            "test",
        ],
    },
]


def find_data_root(root: Path) -> Path:
    """Prefer archive/ if present; otherwise use project root."""
    archive = root / "archive"
    if archive.is_dir():
        return archive
    return root


def resolve_existing(base: Path, relative_candidates: list[str]) -> Path | None:
    for rel in relative_candidates:
        path = base / rel
        if path.exists():
            return path
    return None


def list_images(folder: Path) -> dict[str, Path]:
    """Map stem (id_code) -> image path. If multiple extensions share a stem, keep all via list later."""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in folder.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            by_stem[path.stem].append(path)

    # Prefer a single path per id; if duplicates by stem, keep first and note later
    resolved: dict[str, Path] = {}
    for stem, paths in by_stem.items():
        resolved[stem] = sorted(paths)[0]
    return resolved


def load_labels(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load id_code / diagnosis labels; return dataframe and warnings."""
    warnings: list[str] = []
    df = pd.read_csv(csv_path)

    # Drop fully empty rows (common with trailing blank lines)
    before = len(df)
    df = df.dropna(how="all")
    dropped_empty = before - len(df)
    if dropped_empty:
        warnings.append(f"{dropped_empty} fully empty row(s) skipped")

    cols_lower = {c.lower().strip(): c for c in df.columns}
    if "id_code" not in cols_lower or "diagnosis" not in cols_lower:
        raise ValueError(
            f"{csv_path.name}: expected columns id_code and diagnosis, "
            f"found {list(df.columns)}"
        )

    id_col = cols_lower["id_code"]
    diag_col = cols_lower["diagnosis"]
    df = df[[id_col, diag_col]].rename(columns={id_col: "id_code", diag_col: "diagnosis"})
    df["id_code"] = df["id_code"].astype(str).str.strip()
    # Drop rows where id_code is empty / nan string
    empty_ids = df["id_code"].isin(["", "nan", "None"])
    if empty_ids.any():
        warnings.append(f"{int(empty_ids.sum())} row(s) with empty id_code dropped")
        df = df.loc[~empty_ids].copy()

    return df, warnings


def check_images_openable(paths: list[Path], max_failures_to_list: int = 20) -> tuple[int, int, list[str]]:
    """Try opening each image with Pillow. Returns (ok, fail, failure messages)."""
    ok = 0
    fail = 0
    failures: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                img.verify()
            # verify() can leave the file in a bad state; reopen for a light load
            with Image.open(path) as img:
                img.load()
            ok += 1
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            fail += 1
            if len(failures) < max_failures_to_list:
                failures.append(f"{path.name}: {exc}")
    return ok, fail, failures


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    data_root = find_data_root(ROOT)
    print_section("APTOS dataset integrity check")
    print(f"Project root : {ROOT}")
    print(f"Data root    : {data_root}")

    # Discover splits
    splits: list[dict] = []
    discovery_issues: list[str] = []

    print_section("1–2. Label files and image folders")
    for spec in SPLIT_SPECS:
        csv_path = resolve_existing(data_root, spec["csv_candidates"])
        img_dir = resolve_existing(data_root, spec["image_dir_candidates"])

        print(f"\n[{spec['name']}]")
        if csv_path:
            print(f"  labels : {csv_path.relative_to(ROOT)}")
        else:
            print(f"  labels : NOT FOUND (tried {spec['csv_candidates']})")
            discovery_issues.append(f"Missing label CSV for {spec['name']}")

        if img_dir:
            print(f"  images : {img_dir.relative_to(ROOT)}")
        else:
            print(f"  images : NOT FOUND (tried {spec['image_dir_candidates']})")
            discovery_issues.append(f"Missing image folder for {spec['name']}")

        if csv_path and img_dir:
            splits.append({"name": spec["name"], "csv": csv_path, "images": img_dir})

    if not splits:
        print("\nERROR: Could not locate any complete train/val/test split.")
        print("Inspected data root contents:")
        for p in sorted(data_root.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            print(f"  [{kind}] {p.name}")
        return 1

    # Per-split analysis
    global_ext_counter: Counter[str] = Counter()
    overall = {
        "label_rows": 0,
        "images_on_disk": 0,
        "missing_images": 0,
        "orphan_images": 0,
        "duplicate_ids": 0,
        "unreadable": 0,
        "class_counts": Counter(),
    }
    split_summaries: list[dict] = []

    for split in splits:
        name = split["name"]
        print_section(f"Split: {name}")

        df, warnings = load_labels(split["csv"])
        for w in warnings:
            print(f"  WARNING: {w}")

        images_by_id = list_images(split["images"])
        image_paths = list(images_by_id.values())
        ext_counter = Counter(p.suffix.lower() for p in image_paths)
        global_ext_counter.update(ext_counter)

        # Duplicate IDs in labels
        dup_mask = df["id_code"].duplicated(keep=False)
        n_dup_rows = int(dup_mask.sum())
        n_dup_ids = int(df.loc[dup_mask, "id_code"].nunique()) if n_dup_rows else 0
        duplicate_id_list = (
            sorted(df.loc[dup_mask, "id_code"].unique().tolist()) if n_dup_rows else []
        )

        label_ids = set(df["id_code"])
        image_ids = set(images_by_id.keys())

        missing = sorted(label_ids - image_ids)
        orphans = sorted(image_ids - label_ids)

        # Class counts (all label rows, including duplicates if any)
        class_counts = Counter()
        invalid_diag = 0
        for val in df["diagnosis"]:
            try:
                # Allow 0, 1, 2, 3, 4 even if stored as float (e.g. 2.0)
                if pd.isna(val):
                    invalid_diag += 1
                    continue
                code = int(val)
                class_counts[code] += 1
            except (TypeError, ValueError):
                invalid_diag += 1

        print(f"  Label rows              : {len(df)}")
        print(f"  Unique id_code in labels: {df['id_code'].nunique()}")
        print(f"  Images on disk          : {len(images_by_id)}")
        print(f"  Extensions              : {dict(sorted(ext_counter.items())) or 'none'}")
        print(f"  Duplicate id_code rows  : {n_dup_rows} (unique duplicated IDs: {n_dup_ids})")
        if duplicate_id_list:
            preview = ", ".join(duplicate_id_list[:10])
            more = "" if len(duplicate_id_list) <= 10 else f" ... (+{len(duplicate_id_list) - 10} more)"
            print(f"    duplicated IDs        : {preview}{more}")

        print("  Diagnosis class counts:")
        for c in range(5):
            print(f"    class {c}: {class_counts.get(c, 0)}")
        other_classes = {k: v for k, v in class_counts.items() if k not in range(5)}
        if other_classes:
            print(f"    other classes         : {other_classes}")
        if invalid_diag:
            print(f"    invalid/missing diagnosis values: {invalid_diag}")

        print(f"  Labels with missing image: {len(missing)}")
        if missing:
            preview = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
            print(f"    missing id_codes      : {preview}{more}")

        print(f"  Images with no label     : {len(orphans)}")
        if orphans:
            preview = ", ".join(orphans[:10])
            more = "" if len(orphans) <= 10 else f" ... (+{len(orphans) - 10} more)"
            print(f"    orphan image stems    : {preview}{more}")

        print("  Checking whether images can be opened...")
        ok, fail, fail_msgs = check_images_openable(image_paths)
        print(f"  Readable images          : {ok}")
        print(f"  Unreadable images        : {fail}")
        for msg in fail_msgs:
            print(f"    - {msg}")
        if fail > len(fail_msgs):
            print(f"    ... (+{fail - len(fail_msgs)} more)")

        overall["label_rows"] += len(df)
        overall["images_on_disk"] += len(images_by_id)
        overall["missing_images"] += len(missing)
        overall["orphan_images"] += len(orphans)
        overall["duplicate_ids"] += n_dup_ids
        overall["unreadable"] += fail
        overall["class_counts"].update(class_counts)

        split_summaries.append(
            {
                "name": name,
                "labels": len(df),
                "images": len(images_by_id),
                "missing": len(missing),
                "orphans": len(orphans),
                "dup_ids": n_dup_ids,
                "unreadable": fail,
                "ok": (
                    len(missing) == 0
                    and len(orphans) == 0
                    and n_dup_ids == 0
                    and fail == 0
                    and invalid_diag == 0
                ),
            }
        )

    # Cross-split duplicate IDs
    print_section("Cross-split ID overlap")
    ids_by_split: dict[str, set[str]] = {}
    for split in splits:
        df, _ = load_labels(split["csv"])
        ids_by_split[split["name"]] = set(df["id_code"])

    names = list(ids_by_split.keys())
    any_overlap = False
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sorted(ids_by_split[a] & ids_by_split[b])
            print(f"  {a} ∩ {b}: {len(overlap)} shared id_code(s)")
            if overlap:
                any_overlap = True
                preview = ", ".join(overlap[:10])
                more = "" if len(overlap) <= 10 else f" ... (+{len(overlap) - 10} more)"
                print(f"    examples: {preview}{more}")
    if not any_overlap:
        print("  No overlapping id_code values across splits.")

    # Final summary
    print_section("SUMMARY")
    if discovery_issues:
        print("Discovery issues:")
        for issue in discovery_issues:
            print(f"  - {issue}")

    print("\nPer-split:")
    print(f"  {'split':<12} {'labels':>8} {'images':>8} {'missing':>8} {'orphans':>8} {'dup_ids':>8} {'bad_img':>8} {'status':>8}")
    for s in split_summaries:
        status = "OK" if s["ok"] else "ISSUES"
        print(
            f"  {s['name']:<12} {s['labels']:>8} {s['images']:>8} {s['missing']:>8} "
            f"{s['orphans']:>8} {s['dup_ids']:>8} {s['unreadable']:>8} {status:>8}"
        )

    print("\nOverall diagnosis counts (all label rows):")
    for c in range(5):
        print(f"  class {c}: {overall['class_counts'].get(c, 0)}")

    print("\nImage extensions used:")
    if global_ext_counter:
        for ext, count in sorted(global_ext_counter.items()):
            print(f"  {ext}: {count}")
    else:
        print("  (none found)")

    print("\nTotals:")
    print(f"  Label rows              : {overall['label_rows']}")
    print(f"  Images on disk          : {overall['images_on_disk']}")
    print(f"  Missing images          : {overall['missing_images']}")
    print(f"  Orphan images           : {overall['orphan_images']}")
    print(f"  Duplicate IDs (in-split): {overall['duplicate_ids']}")
    print(f"  Unreadable images       : {overall['unreadable']}")

    all_ok = (
        not discovery_issues
        and all(s["ok"] for s in split_summaries)
        and not any_overlap
    )
    print()
    if all_ok:
        print("RESULT: PASS — labels and images look correctly connected.")
        return 0

    print("RESULT: ISSUES FOUND — see details above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
