#!/usr/bin/env python3
"""Generate configs + manifest for the 36 paper v2 trainings.

Jobs (exactly 36; never drop/add by result):
  1. Matched EfficientNet-B0: H1–H4 × seeds 42–44  → 12
  2. Hybrid MobileViT-S:      H1–H4 × seeds 45–47  → 12
  3. Matched EfficientNet-B0: H1–H4 × seeds 45–47  → 12

All use hybrid-matched regime: CPU, batch_size=4, freeze_bn=True,
same mild-CORAL weights, interrupt chunking. Run IDs end in _v2.
Archived B0 / hybrid_v1 dirs are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_CFG = ROOT / "configs" / "paper_v2"
OUT_MANIFEST = ROOT / "configs" / "paper_v2" / "manifest_36.json"

CELLS = {
    "H1": {"preprocess": "rgb", "loss": "cross_entropy", "head": "ce"},
    "H2": {"preprocess": "green_clahe", "loss": "cross_entropy", "head": "ce"},
    "H3": {"preprocess": "rgb", "loss": "mild_aware_coral", "head": "coral"},
    "H4": {"preprocess": "green_clahe", "loss": "mild_aware_coral", "head": "coral"},
}

DATA_BLOCK = {
    "datasets_root": "datasets",
    "aptos": {
        "root": "archive",
        "train_csv": "train_1.csv",
        "val_csv": "valid.csv",
        "test_csv": "test.csv",
        "train_images": "train_images/train_images",
        "val_images": "val_images/val_images",
        "test_images": "test_images/test_images",
    },
    "idrid": {
        "root": "idrid/B. Disease Grading",
        "train_csv": "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        "test_csv": "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
        "train_images": "1. Original Images/a. Training Set",
        "test_images": "1. Original Images/b. Testing Set",
    },
    "messidor2": {
        "root": "../Messidor ",
        "labels_csv": "messidor_data.csv",
        "images_dir": "IMAGES",
    },
    "deepdrid": {"root": "DeepDRiD"},
    "num_classes": 5,
    "class_names": [0, 1, 2, 3, 4],
}

EVAL_BLOCK = {
    "datasets": [
        {"name": "aptos_test", "dataset": "aptos", "split": "test"},
        {"name": "idrid_external", "dataset": "idrid", "split": "all"},
        {"name": "messidor2_external", "dataset": "messidor2", "split": "all"},
        {"name": "deepdrid_external", "dataset": "deepdrid", "split": "all"},
    ]
}

TRAIN_BLOCK = {
    "train_dataset": "aptos",
    "epochs": 20,
    "batch_size": 4,
    "num_workers": 0,
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "amp": False,
    "selection_metric": "quadratic_weighted_kappa",
    "interrupt_every": 20,
    "device": "cpu",
    "freeze_bn": True,
    "max_steps_per_launch": 60,
}


def _preprocess(name: str) -> dict:
    block = {
        "name": name,
        "image_size": 224,
        "cache_root": "datasets/cache",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    if name == "green_clahe":
        block["clahe_clip_limit"] = 2.0
        block["clahe_tile_grid_size"] = [8, 8]
    return block


def _loss(name: str) -> dict:
    if name == "cross_entropy":
        return {"name": "cross_entropy"}
    return {
        "name": "mild_aware_coral",
        "mild_sample_weight": 2.0,
        "threshold_weights": [2.0, 2.0, 1.0, 1.0],
    }


def _model(arch: str, head: str) -> dict:
    if arch == "mobilevit_s":
        return {
            "name": "mobilevit_s",
            "source": "timm",
            "pretrained": True,
            "head": head,
            "num_classes": 5,
        }
    return {
        "name": "efficientnet_b0",
        "source": "torchvision",
        "pretrained": True,
        "head": head,
        "num_classes": 5,
    }


def build_job(family: str, cell: str, seed: int) -> dict:
    spec = CELLS[cell]
    arch = "mobilevit_s" if family == "hybrid" else "efficientnet_b0"
    if family == "hybrid":
        run_id = f"hybrid_{cell.lower()}_s{seed}_v2"
        exp_name = f"hybrid_mobilevit_s_{spec['preprocess']}_{spec['loss']}_s{seed}_v2"
        desc = (
            f"Paper v2 hybrid {cell}: MobileViT-S {spec['preprocess']} × {spec['loss']} "
            f"seed {seed} (CPU bs=4 freeze_bn=True; matched to primary hybrids)."
        )
    else:
        run_id = f"matched_b0_{cell.lower()}_s{seed}_v2"
        exp_name = f"matched_efficientnet_b0_{spec['preprocess']}_{spec['loss']}_s{seed}_v2"
        desc = (
            f"Paper v2 matched B0 {cell}: EfficientNet-B0 {spec['preprocess']} × {spec['loss']} "
            f"seed {seed} (CPU bs=4 freeze_bn=True; archived B0 untouched)."
        )
    cfg = {
        "experiment": {
            "name": exp_name,
            "description": desc,
            "seed": int(seed),
            "protocol": "frozen_protocol_v1",
            "cell": cell,
            "family": family,
            "version": "v2",
        },
        "data": deepcopy(DATA_BLOCK),
        "preprocessing": _preprocess(spec["preprocess"]),
        "model": _model(arch, spec["head"]),
        "loss": _loss(spec["loss"]),
        "train": deepcopy(TRAIN_BLOCK),
        "evaluate": deepcopy(EVAL_BLOCK),
    }
    return {
        "run_id": run_id,
        "family": family,
        "cell": cell,
        "seed": int(seed),
        "config_rel": f"configs/paper_v2/{run_id}.yaml",
        "config": cfg,
    }


def all_jobs() -> list[dict]:
    jobs: list[dict] = []
    # 12 matched B0 × seeds 42–44
    for seed in (42, 43, 44):
        for cell in ("H1", "H2", "H3", "H4"):
            jobs.append(build_job("matched_b0", cell, seed))
    # 12 hybrid × seeds 45–47
    for seed in (45, 46, 47):
        for cell in ("H1", "H2", "H3", "H4"):
            jobs.append(build_job("hybrid", cell, seed))
    # 12 matched B0 × seeds 45–47
    for seed in (45, 46, 47):
        for cell in ("H1", "H2", "H3", "H4"):
            jobs.append(build_job("matched_b0", cell, seed))
    if len(jobs) != 36:
        raise SystemExit(f"Expected 36 jobs, got {len(jobs)}")
    return jobs


def config_hash(cfg: dict) -> str:
    blob = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def main() -> None:
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs()
    manifest = []
    for job in jobs:
        cfg = job["config"]
        path = ROOT / job["config_rel"]
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        h = config_hash(cfg)
        entry = {
            "run_id": job["run_id"],
            "family": job["family"],
            "cell": job["cell"],
            "seed": job["seed"],
            "config_rel": job["config_rel"],
            "config_hash": h,
            "batch_size": cfg["train"]["batch_size"],
            "device": cfg["train"]["device"],
            "freeze_bn": cfg["train"]["freeze_bn"],
        }
        manifest.append(entry)
        print(f"WROTE {path.relative_to(ROOT)} hash={h}")
    OUT_MANIFEST.write_text(json.dumps({"n_jobs": len(manifest), "jobs": manifest}, indent=2))
    print(f"MANIFEST {OUT_MANIFEST.relative_to(ROOT)} n={len(manifest)}")


if __name__ == "__main__":
    main()
