#!/usr/bin/env python3
"""APTOS split provenance and exact-duplicate audit (read-only).

1. Confirms train_1.csv / valid.csv / test.csv are disjoint by id; per-split grade counts.
2. Exact duplicates by (a) MD5 of the original image file and (b) MD5 of the decoded cached
   224x224 RGB pixel array (catches identical content saved with different encodings).
3. Sensitivity: APTOS test QWK and mild recall of every MobileViT-S H1-H4 run (seeds 42-47)
   recomputed after removing test images that have an exact duplicate in train or validation.

Near-duplicate (perceptual) detection is not attempted: global hashes of fundus photographs are
dominated by the circular mask and gave no usable separation.

Writes paper/tables/aptos_split_audit.csv, aptos_duplicate_pairs.csv, aptos_dedup_sensitivity.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import mild_dr_recall, quadratic_weighted_kappa  # noqa: E402

APT = ROOT / "datasets" / "archive"
CACHE = ROOT / "datasets" / "cache" / "rgb" / "aptos"
OUT = ROOT / "paper" / "tables"
SPLITS = {"train": ("train_1.csv", "train_images/train_images"),
          "val": ("valid.csv", "val_images/val_images"),
          "test": ("test.csv", "test_images/test_images")}
CELLS = ["H1", "H2", "H3", "H4"]
SEEDS = [42, 43, 44, 45, 46, 47]


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_pixels(p: Path) -> str:
    with Image.open(p) as im:
        return hashlib.md5(np.asarray(im.convert("RGB")).tobytes()).hexdigest()


def find_image(folder: Path, image_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        p = folder / f"{image_id}{ext}"
        if p.is_file():
            return p
    raise SystemExit(f"MISSING input: image {image_id} in {folder.relative_to(ROOT)}")


def main() -> None:
    recs = []
    for split, (csvn, imgd) in SPLITS.items():
        cp = APT / csvn
        if not cp.is_file():
            raise SystemExit(f"MISSING input: {cp.relative_to(ROOT)}")
        print("READ", cp.relative_to(ROOT))
        df = pd.read_csv(cp).dropna(how="all")
        for _, r in df.iterrows():
            iid = str(r["id_code"]).strip()
            cache = CACHE / split / f"{iid}.png"
            if not cache.is_file():
                raise SystemExit(f"MISSING input: {cache.relative_to(ROOT)}")
            recs.append({"split": split, "id": iid, "label": int(r["diagnosis"]),
                         "file": find_image(APT / imgd, iid), "cache": cache})
    print("READ", len(recs), "original images under datasets/archive and cached frames under",
          CACHE.relative_to(ROOT))

    ids = defaultdict(set)
    for r in recs:
        ids[r["split"]].add(r["id"])
    audit = []
    for s in SPLITS:
        lab = [r["label"] for r in recs if r["split"] == s]
        audit.append({"item": f"{s} images", "value": len(lab)})
        for g in range(5):
            audit.append({"item": f"{s} grade {g}", "value": lab.count(g)})
    audit.append({"item": "total images", "value": len(recs)})
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        audit.append({"item": f"id overlap {a}-{b}", "value": len(ids[a] & ids[b])})

    for r in recs:
        r["md5_file"] = md5_file(r["file"])
        r["md5_pix"] = md5_pixels(r["cache"])

    rows = []
    seen = set()
    for key in ("md5_file", "md5_pix"):
        groups = defaultdict(list)
        for r in recs:
            groups[r[key]].append(r)
        for grp in groups.values():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    pk = tuple(sorted([(a["split"], a["id"]), (b["split"], b["id"])]))
                    if pk in seen:
                        continue
                    seen.add(pk)
                    rows.append({"detected_by": "file bytes" if key == "md5_file" else "decoded 224x224 pixels",
                                 "split_a": a["split"], "id_a": a["id"], "label_a": a["label"],
                                 "split_b": b["split"], "id_b": b["id"], "label_b": b["label"],
                                 "cross_split": a["split"] != b["split"], "labels_agree": a["label"] == b["label"]})
    audit.append({"item": "exact duplicate pairs, all", "value": len(rows)})
    audit.append({"item": "exact duplicate pairs found only by decoded pixels",
                  "value": sum(r["detected_by"] != "file bytes" for r in rows)})
    for a, b in (("train", "train"), ("train", "val"), ("train", "test"), ("val", "test"), ("val", "val"), ("test", "test")):
        audit.append({"item": f"exact duplicate pairs {a}-{b}",
                      "value": sum(1 for r in rows if sorted([r["split_a"], r["split_b"]]) == sorted([a, b]))})
    audit.append({"item": "cross-split exact pairs with different labels",
                  "value": sum(1 for r in rows if r["cross_split"] and not r["labels_agree"])})
    leak = {(r["id_a"] if r["split_a"] == "test" else r["id_b"]) for r in rows
            if r["cross_split"] and "test" in (r["split_a"], r["split_b"]) and {r["split_a"], r["split_b"]} != {"test"}}
    leak_val = {(r["id_a"] if r["split_a"] == "val" else r["id_b"]) for r in rows
                if r["cross_split"] and {r["split_a"], r["split_b"]} == {"train", "val"}}
    audit.append({"item": "test images with an exact duplicate in train or validation", "value": len(leak)})
    audit.append({"item": "validation images with an exact duplicate in train", "value": len(leak_val)})

    # sensitivity on saved predictions
    sens = []
    for c in CELLS:
        for s in SEEDS:
            run = f"hybrid_{c.lower()}_s{s}_{'v1' if s <= 44 else 'v3'}"
            pp = ROOT / "results" / run / "aptos_test_predictions.json"
            if not pp.is_file():
                raise SystemExit(f"MISSING input: {pp.relative_to(ROOT)}")
            print("READ", pp.relative_to(ROOT))
            d = json.loads(pp.read_text())
            keep = [i for i, x in enumerate(d["ids"]) if str(x) not in leak]
            yt = [d["y_true"][i] for i in keep]
            yp = [d["y_pred"][i] for i in keep]
            sens.append({"cell": c, "seed": s, "n_all": len(d["ids"]), "n_kept": len(keep),
                         "qwk_all": round(quadratic_weighted_kappa(d["y_true"], d["y_pred"], 5), 6),
                         "qwk_dedup": round(quadratic_weighted_kappa(yt, yp, 5), 6),
                         "mild_recall_all": round(mild_dr_recall(d["y_true"], d["y_pred"], 5), 6),
                         "mild_recall_dedup": round(mild_dr_recall(yt, yp, 5), 6),
                         "n_mild_kept": sum(1 for y in yt if y == 1)})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("aptos_split_audit.csv", audit), ("aptos_duplicate_pairs.csv", rows),
                       ("aptos_dedup_sensitivity.csv", sens)):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print("WROTE", f"paper/tables/{name}")
    for a in audit:
        print(f"{a['item']}: {a['value']}")
    for c in CELLS:
        a = [r["qwk_all"] for r in sens if r["cell"] == c]
        b = [r["qwk_dedup"] for r in sens if r["cell"] == c]
        print(c, f"QWK all {mean(a):.3f}±{stdev(a):.3f} dedup {mean(b):.3f}±{stdev(b):.3f}")
    print("APTOS_AUDIT_OK")


if __name__ == "__main__":
    main()
