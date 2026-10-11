#!/usr/bin/env python3
"""
Verify the IDRiD Disease Grading dataset (external DR test set).

Does not modify, rename, move, or delete any IDRiD files.
Does not train a model.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

# APTOS uses the International Clinical Diabetic Retinopathy (ICDR) 0–4 scale.
APTOS_GRADE_SCHEME = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

# IDRiD Disease Grading uses the same ICDR 0–4 retinopathy grades
# (plus a separate macular edema risk column that APTOS labels do not use).
IDRID_GRADE_NOTES = (
    "IDRiD 'Retinopathy grade' uses integers 0–4 on the International Clinical "
    "Diabetic Retinopathy severity scale, matching APTOS diagnosis classes. "
    "IDRiD also provides 'Risk of macular edema' (0–2), which is separate from "
    "the APTOS 0–4 DR grade and is reported here for completeness only."
)


def find_idrid_root(root: Path) -> Path | None:
    candidates = [
        root / "idrid",
        root / "IDRiD",
        root / "IDRID",
        root,
    ]
    for path in candidates:
        if not path.is_dir():
            continue
        # Prefer a folder that contains Disease Grading
        if (path / "B. Disease Grading").is_dir():
            return path
        # Or is itself the Disease Grading folder
        if (path / "1. Original Images").is_dir() and (path / "2. Groundtruths").is_dir():
            return path
    return None


def find_disease_grading(idrid_root: Path) -> Path | None:
    direct = idrid_root / "B. Disease Grading"
    if direct.is_dir():
        return direct
    if (idrid_root / "1. Original Images").is_dir():
        return idrid_root
    # Fuzzy search
    for path in idrid_root.rglob("*"):
        if path.is_dir() and "disease grading" in path.name.lower():
            return path
    return None


def resolve_split_dirs(grading_root: Path) -> list[dict]:
    """Locate train/test image folders and label CSVs under Disease Grading."""
    images_root = grading_root / "1. Original Images"
    labels_root = grading_root / "2. Groundtruths"

    splits: list[dict] = []

    train_img_candidates = [
        images_root / "a. Training Set",
        images_root / "Training Set",
        images_root / "train",
    ]
    test_img_candidates = [
        images_root / "b. Testing Set",
        images_root / "Testing Set",
        images_root / "test",
    ]

    train_csv_candidates = []
    test_csv_candidates = []
    if labels_root.is_dir():
        for csv_path in sorted(labels_root.glob("*.csv")):
            name_l = csv_path.name.lower()
            if "train" in name_l:
                train_csv_candidates.append(csv_path)
            elif "test" in name_l:
                test_csv_candidates.append(csv_path)

    def first_existing(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    train_img = first_existing(train_img_candidates)
    test_img = first_existing(test_img_candidates)
    train_csv = train_csv_candidates[0] if train_csv_candidates else None
    test_csv = test_csv_candidates[0] if test_csv_candidates else None

    if train_img or train_csv:
        splits.append({"name": "training", "images": train_img, "csv": train_csv})
    if test_img or test_csv:
        splits.append({"name": "testing", "images": test_img, "csv": test_csv})

    return splits


def list_images(folder: Path) -> dict[str, Path]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in folder.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            by_stem[path.stem].append(path)
    return {stem: sorted(paths)[0] for stem, paths in by_stem.items()}


def load_labels(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")

    # Normalize column names (strip trailing spaces common in IDRiD CSVs)
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}

    image_col = None
    for key in ("image name", "image_name", "id_code", "image id", "image"):
        if key in cols_lower:
            image_col = cols_lower[key]
            break

    grade_col = None
    for key in ("retinopathy grade", "diagnosis", "dr grade", "grade"):
        if key in cols_lower:
            grade_col = cols_lower[key]
            break

    if image_col is None or grade_col is None:
        raise ValueError(
            f"{csv_path.name}: could not find image-name / retinopathy-grade columns. "
            f"Found: {list(df.columns)}"
        )

    dme_col = None
    for key in ("risk of macular edema", "macular edema", "dme"):
        if key in cols_lower:
            dme_col = cols_lower[key]
            break

    keep = [image_col, grade_col] + ([dme_col] if dme_col else [])
    out = df[keep].copy()
    rename = {image_col: "image_name", grade_col: "retinopathy_grade"}
    if dme_col:
        rename[dme_col] = "risk_of_macular_edema"
    out = out.rename(columns=rename)

    # Drop junk columns / rows where image name is empty
    out["image_name"] = out["image_name"].astype(str).str.strip()
    # Strip extension if present in labels
    out["image_id"] = out["image_name"].str.replace(
        r"\.(png|jpg|jpeg|tif|tiff|bmp)$", "", regex=True, case=False
    )
    empty = out["image_id"].isin(["", "nan", "None"])
    if empty.any():
        warnings.append(f"{int(empty.sum())} row(s) with empty image name dropped")
        out = out.loc[~empty].copy()

    # Ignore trailing documentation-like empty unnamed columns already excluded
    extra_unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if extra_unnamed:
        warnings.append(
            f"ignored {len(extra_unnamed)} Unnamed:* column(s) in CSV "
            "(present in official IDRiD training labels file)"
        )

    return out, warnings


def check_images_openable(
    paths: list[Path], max_failures_to_list: int = 20
) -> tuple[int, int, list[str]]:
    ok = 0
    fail = 0
    failures: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                img.verify()
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
    print_section("IDRiD dataset integrity check")
    print(f"Project root: {ROOT}")

    idrid_root = find_idrid_root(ROOT)
    if idrid_root is None:
        print("ERROR: Could not find an IDRiD folder.")
        print("Looked for: idrid/, IDRiD/, or Disease Grading under the project root.")
        return 1

    print(f"IDRiD root  : {idrid_root}")

    # High-level package contents
    print_section("Observed IDRiD package layout")
    top_items = sorted(p.name for p in idrid_root.iterdir() if not p.name.startswith("."))
    for name in top_items:
        path = idrid_root / name
        kind = "dir " if path.is_dir() else "file"
        print(f"  [{kind}] {name}")

    grading_root = find_disease_grading(idrid_root)
    if grading_root is None:
        print("\nERROR: Could not locate 'B. Disease Grading' (DR grading subset).")
        print("Segmentation / localization folders alone are not enough for DR grade checks.")
        return 1

    print(f"\nUsing Disease Grading root: {grading_root.relative_to(ROOT)}")
    print(
        "Note: A. Segmentation and C. Localization are separate IDRiD tasks "
        "and are not used for APTOS-style 0–4 DR grade verification."
    )

    splits = resolve_split_dirs(grading_root)
    print_section("1–4. Images, labels, format, filename pattern")

    if not splits:
        print("ERROR: No train/test image folders or label CSVs found under Disease Grading.")
        return 1

    for split in splits:
        print(f"\n[{split['name']}]")
        if split["images"]:
            print(f"  images : {split['images'].relative_to(ROOT)}")
        else:
            print("  images : NOT FOUND")
        if split["csv"]:
            print(f"  labels : {split['csv'].relative_to(ROOT)}")
            print("  format : CSV")
        else:
            print("  labels : NOT FOUND")

    # Sample filename pattern from first available image folder
    sample_folder = next((s["images"] for s in splits if s["images"]), None)
    if sample_folder:
        samples = sorted(p.name for p in sample_folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)[:5]
        print("\nExample image filenames:")
        for name in samples:
            print(f"  {name}")
        print("Label IDs in CSV typically match the filename stem (e.g. IDRiD_001 for IDRiD_001.jpg).")

    overall = {
        "label_rows": 0,
        "images_on_disk": 0,
        "missing": 0,
        "orphans": 0,
        "dup_ids": 0,
        "unreadable": 0,
        "grades": Counter(),
        "dme": Counter(),
        "ext": Counter(),
    }
    split_summaries: list[dict] = []
    ids_by_split: dict[str, set[str]] = {}
    grade_values_seen: set[int] = set()
    issues: list[str] = []

    for split in splits:
        name = split["name"]
        print_section(f"Split: {name}")

        if not split["images"] or not split["csv"]:
            msg = f"{name}: missing images folder and/or label CSV"
            print(f"  ERROR: {msg}")
            issues.append(msg)
            continue

        df, warnings = load_labels(split["csv"])
        for w in warnings:
            print(f"  WARNING: {w}")

        images_by_id = list_images(split["images"])
        image_paths = list(images_by_id.values())
        ext_counter = Counter(p.suffix.lower() for p in image_paths)

        dup_mask = df["image_id"].duplicated(keep=False)
        n_dup_rows = int(dup_mask.sum())
        n_dup_ids = int(df.loc[dup_mask, "image_id"].nunique()) if n_dup_rows else 0
        dup_list = sorted(df.loc[dup_mask, "image_id"].unique().tolist()) if n_dup_rows else []

        label_ids = set(df["image_id"])
        image_ids = set(images_by_id.keys())
        missing = sorted(label_ids - image_ids)
        orphans = sorted(image_ids - label_ids)

        grade_counts: Counter[int] = Counter()
        invalid_grades = 0
        for val in df["retinopathy_grade"]:
            try:
                if pd.isna(val):
                    invalid_grades += 1
                    continue
                code = int(val)
                grade_counts[code] += 1
                grade_values_seen.add(code)
            except (TypeError, ValueError):
                invalid_grades += 1

        dme_counts: Counter[int] = Counter()
        if "risk_of_macular_edema" in df.columns:
            for val in df["risk_of_macular_edema"]:
                try:
                    if pd.isna(val):
                        continue
                    dme_counts[int(val)] += 1
                except (TypeError, ValueError):
                    pass

        print(f"  Label rows               : {len(df)}")
        print(f"  Unique image IDs (labels): {df['image_id'].nunique()}")
        print(f"  Images on disk           : {len(images_by_id)}")
        print(f"  Extensions               : {dict(sorted(ext_counter.items())) or 'none'}")
        print(f"  Duplicate image IDs      : {n_dup_ids}")
        if dup_list:
            preview = ", ".join(dup_list[:10])
            more = "" if len(dup_list) <= 10 else f" ... (+{len(dup_list) - 10} more)"
            print(f"    duplicated IDs         : {preview}{more}")

        print("  Retinopathy grade counts (DR):")
        for c in range(5):
            print(f"    class {c}: {grade_counts.get(c, 0)}")
        other = {k: v for k, v in grade_counts.items() if k not in range(5)}
        if other:
            print(f"    other grades           : {other}")
        if invalid_grades:
            print(f"    invalid/missing grades : {invalid_grades}")

        if dme_counts:
            print("  Risk of macular edema counts (not used by APTOS):")
            for k in sorted(dme_counts):
                print(f"    risk {k}: {dme_counts[k]}")

        print(f"  Labels missing image file: {len(missing)}")
        if missing:
            preview = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
            print(f"    missing IDs            : {preview}{more}")

        print(f"  Images without label     : {len(orphans)}")
        if orphans:
            preview = ", ".join(orphans[:10])
            more = "" if len(orphans) <= 10 else f" ... (+{len(orphans) - 10} more)"
            print(f"    orphan stems           : {preview}{more}")

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
        overall["missing"] += len(missing)
        overall["orphans"] += len(orphans)
        overall["dup_ids"] += n_dup_ids
        overall["unreadable"] += fail
        overall["grades"].update(grade_counts)
        overall["dme"].update(dme_counts)
        overall["ext"].update(ext_counter)
        ids_by_split[name] = label_ids

        split_ok = (
            len(missing) == 0
            and len(orphans) == 0
            and n_dup_ids == 0
            and fail == 0
            and invalid_grades == 0
            and not other
        )
        if not split_ok:
            issues.append(f"{name}: integrity issues found")

        split_summaries.append(
            {
                "name": name,
                "labels": len(df),
                "images": len(images_by_id),
                "missing": len(missing),
                "orphans": len(orphans),
                "dup_ids": n_dup_ids,
                "unreadable": fail,
                "ok": split_ok,
            }
        )

    # Cross-split name reuse (common/official in IDRiD: same IDRiD_XXX in train and test folders)
    print_section("Cross-split image ID overlap")
    names = list(ids_by_split.keys())
    any_overlap = False
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sorted(ids_by_split[a] & ids_by_split[b])
            print(f"  {a} ∩ {b}: {len(overlap)} shared image ID string(s)")
            if overlap:
                any_overlap = True
                preview = ", ".join(overlap[:10])
                more = "" if len(overlap) <= 10 else f" ... (+{len(overlap) - 10} more)"
                print(f"    examples: {preview}{more}")
    if any_overlap:
        print(
            "  NOTE: Shared ID strings across train/test are expected in IDRiD. "
            "They refer to different files in different folders; always key by "
            "(split, image_id), not image_id alone."
        )
    else:
        print("  No overlapping image ID strings across splits.")

    # Grading scheme comparison vs APTOS
    print_section("12. DR grading scheme vs APTOS (0–4)")
    print(IDRID_GRADE_NOTES)
    print("\nAPTOS / ICDR reference:")
    for k, v in APTOS_GRADE_SCHEME.items():
        print(f"  {k}: {v}")

    print(f"\nIDRiD retinopathy grades observed: {sorted(grade_values_seen) if grade_values_seen else 'none'}")
    scheme_ok = grade_values_seen.issubset(set(range(5))) and len(grade_values_seen) > 0
    if scheme_ok and grade_values_seen == set(range(5)):
        print(
            "SCHEME CHECK: PASS — IDRiD retinopathy grades are integers 0–4, "
            "compatible with the APTOS diagnosis scheme."
        )
    elif scheme_ok:
        missing_classes = sorted(set(range(5)) - grade_values_seen)
        print(
            "SCHEME CHECK: PASS with note — grades are within 0–4 (APTOS-compatible), "
            f"but these classes have zero samples: {missing_classes}."
        )
    else:
        print(
            "SCHEME CHECK: FAIL — observed grades are not confined to APTOS-compatible 0–4."
        )
        issues.append("retinopathy grades not compatible with APTOS 0–4 scheme")

    # Final summary
    print_section("SUMMARY")
    print("Per-split:")
    print(
        f"  {'split':<12} {'labels':>8} {'images':>8} {'missing':>8} "
        f"{'orphans':>8} {'dup_ids':>8} {'bad_img':>8} {'status':>8}"
    )
    for s in split_summaries:
        status = "OK" if s["ok"] else "ISSUES"
        print(
            f"  {s['name']:<12} {s['labels']:>8} {s['images']:>8} {s['missing']:>8} "
            f"{s['orphans']:>8} {s['dup_ids']:>8} {s['unreadable']:>8} {status:>8}"
        )

    print("\nOverall retinopathy grade counts:")
    for c in range(5):
        print(f"  class {c}: {overall['grades'].get(c, 0)}")

    if overall["dme"]:
        print("\nOverall macular edema risk counts (extra IDRiD label, not APTOS):")
        for k in sorted(overall["dme"]):
            print(f"  risk {k}: {overall['dme'][k]}")

    print("\nImage extensions used:")
    for ext, count in sorted(overall["ext"].items()):
        print(f"  {ext}: {count}")

    print("\nTotals:")
    print(f"  Labeled rows             : {overall['label_rows']}")
    print(f"  Retinal images on disk   : {overall['images_on_disk']}")
    print(f"  Missing images           : {overall['missing']}")
    print(f"  Orphan images            : {overall['orphans']}")
    print(f"  Duplicate IDs (in-split) : {overall['dup_ids']}")
    print(f"  Unreadable images        : {overall['unreadable']}")

    all_ok = (
        not issues
        and split_summaries
        and all(s["ok"] for s in split_summaries)
        and scheme_ok
    )
    print()
    if all_ok:
        print(
            "RESULT: PASS — IDRiD Disease Grading images and DR labels look correctly connected "
            "and use an APTOS-compatible 0–4 retinopathy scale."
        )
        return 0

    print("RESULT: ISSUES FOUND — see details above.")
    for issue in issues:
        print(f"  - {issue}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
