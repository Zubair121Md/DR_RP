#!/usr/bin/env python3
"""
PHASE E — Split / leakage audit (read-only).

Outputs:
  experiments/results/split_audit.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.dataset import build_dataset  # noqa: E402
from src.utils import load_config, resolve_data_roots  # noqa: E402


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def aptos_csv_audit() -> dict:
    root = PROJECT / "datasets" / "archive"
    splits = {}
    for name, csv_name in [("train", "train_1.csv"), ("val", "valid.csv"), ("test", "test.csv")]:
        df = pd.read_csv(root / csv_name).dropna(how="all")
        ids = df["id_code"].astype(str).str.strip()
        splits[name] = {
            "n_rows": int(len(df)),
            "n_unique_ids": int(ids.nunique()),
            "dup_ids": int(ids.duplicated().sum()),
            "class_counts": {str(k): int(v) for k, v in Counter(df["diagnosis"].astype(int)).items()},
            "ids": set(ids.tolist()),
        }
    overlaps = {
        "train∩val": len(splits["train"]["ids"] & splits["val"]["ids"]),
        "train∩test": len(splits["train"]["ids"] & splits["test"]["ids"]),
        "val∩test": len(splits["val"]["ids"] & splits["test"]["ids"]),
    }
    # drop heavy id sets from JSON
    light = {k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in splits.items()}
    return {"splits": light, "id_overlaps": overlaps}


def hash_sample(ds_name: str, kwargs: dict, n: int = 64) -> dict:
    ds = build_dataset(ds_name, transform=None, **kwargs)
    hashes = []
    labels = []
    ids = []
    for i in range(min(n, len(ds))):
        s = ds[i]
        p = Path(s["path"])
        hashes.append(file_md5(p))
        labels.append(int(s["label"]))
        ids.append(s["id"])
    # within-sample duplicates
    c = Counter(hashes)
    dups = sum(1 for v in c.values() if v > 1)
    return {
        "n_scanned": len(hashes),
        "n_unique_hashes": len(c),
        "duplicate_hash_groups": int(dups),
        "class_counts_scanned": {str(k): int(v) for k, v in Counter(labels).items()},
        "n_dataset": len(ds),
    }


def cross_dataset_hash_probe(cfg_roots: dict, n_each: int = 40) -> dict:
    """Compare MD5 of first n frames across datasets (cheap contamination probe)."""
    sets = {}
    for tag, name, kw in [
        ("aptos_train", "aptos", {"root": cfg_roots["aptos"], "split": "train"}),
        ("aptos_test", "aptos", {"root": cfg_roots["aptos"], "split": "test"}),
        ("idrid", "idrid", {"root": cfg_roots["idrid"], "split": "all"}),
        ("messidor2", "messidor2", {"root": cfg_roots["messidor2"], "split": "all"}),
        ("deepdrid", "deepdrid", {"root": cfg_roots["deepdrid"], "split": "all"}),
    ]:
        ds = build_dataset(name, transform=None, **kw)
        hs = set()
        for i in range(min(n_each, len(ds))):
            hs.add(file_md5(Path(ds[i]["path"])))
        sets[tag] = hs
    overlaps = {}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlaps[f"{a}∩{b}"] = len(sets[a] & sets[b])
    return {"n_each": n_each, "overlaps": overlaps}


def selection_rules() -> dict:
    return {
        "checkpoint_selection": "APTOS validation QWK only (src/train.py)",
        "temperature_fit": "APTOS validation only (src/evaluate.py)",
        "externals_in_train": False,
        "test_in_selection": False,
        "hyperparam_tuning_on_externals": "Not observed in code; mild-CORAL weights are fixed defaults",
    }


def main() -> None:
    cfg = load_config(PROJECT / "configs/baseline_efficientnet_b0_green_clahe_mild_coral_local.yaml")
    roots = resolve_data_roots(cfg, PROJECT)

    report = {
        "note": "PHASE E split/leakage audit — read-only",
        "aptos_csv": aptos_csv_audit(),
        "selection_rules": selection_rules(),
        "dataset_hash_samples": {},
        "issues": [],
        "warnings": [],
        "status": "PASS",
    }

    for tag, name, kw in [
        ("aptos_train", "aptos", {"root": roots["aptos"], "split": "train"}),
        ("aptos_val", "aptos", {"root": roots["aptos"], "split": "val"}),
        ("aptos_test", "aptos", {"root": roots["aptos"], "split": "test"}),
        ("idrid", "idrid", {"root": roots["idrid"], "split": "all"}),
        ("messidor2", "messidor2", {"root": roots["messidor2"], "split": "all"}),
        ("deepdrid", "deepdrid", {"root": roots["deepdrid"], "split": "all"}),
    ]:
        try:
            report["dataset_hash_samples"][tag] = hash_sample(name, kw, n=80)
            print("OK", tag, report["dataset_hash_samples"][tag]["n_dataset"])
        except Exception as exc:
            report["dataset_hash_samples"][tag] = {"error": str(exc)}
            report["warnings"].append(f"{tag}: {exc}")

    report["cross_dataset_md5_probe"] = cross_dataset_hash_probe(roots, n_each=50)

    ov = report["aptos_csv"]["id_overlaps"]
    if any(v > 0 for v in ov.values()):
        report["issues"].append(f"APTOS ID overlap: {ov}")
        report["status"] = "BLOCKED"

    cross = report["cross_dataset_md5_probe"]["overlaps"]
    if any(v > 0 for v in cross.values()):
        report["warnings"].append(
            f"MD5 overlaps in {50}-image probe (may be coincidence or contamination): {cross}"
        )

    # DeepDRiD patient-level: same patient can contribute multiple images — document
    report["warnings"].append(
        "DeepDRiD has patient_id; evaluation uses image-level metrics. "
        "Patient-level clustering not applied (document as limitation)."
    )
    report["warnings"].append(
        "APTOS/IDRiD/Messidor-2: no patient IDs in provided labels — patient leakage cannot be audited."
    )

    if report["issues"]:
        report["status"] = "BLOCKED"
    elif report["warnings"]:
        report["status"] = "PASS_WITH_WARNINGS"

    out = PROJECT / "experiments" / "results" / "split_audit.json"
    out.write_text(json.dumps(report, indent=2))
    print("SAVED", out, "STATUS", report["status"])


if __name__ == "__main__":
    main()
