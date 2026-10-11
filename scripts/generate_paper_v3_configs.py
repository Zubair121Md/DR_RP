#!/usr/bin/env python3
"""Generate configs + manifest for the prioritized paper_v3 training package.

Groups (launch subsets with PAPER_V3_GROUPS):
  core   (24): matched EfficientNet-B0 H1–H4 × seeds 42–44
               + hybrid MobileViT-S H1–H4 × seeds 45–47
  decomp (6):  CLAHE decomposition cells × seeds 42–44
               - D1: green_clahe × unweighted CORAL
               - D2: green_clahe × mild-weighted CE
  res448 (6):  hybrid H1/H2 at 448×448 × seeds 42–44 (optional)

All new run IDs end in _v3. Never overwrites hybrid_*_v1, archived B0, or paper_v2 dirs.
Default regime: CPU, batch_size=4, freeze_bn=True (matched to primary hybrids).
448 jobs omit preprocess cache (online resize to 448).
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_CFG = ROOT / "configs" / "paper_v3"
OUT_MANIFEST = OUT_CFG / "manifest_v3.json"

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
    "max_steps_per_launch": 800,
}


def _preprocess(name: str, image_size: int = 224, use_cache: bool = True) -> dict:
    block: dict = {
        "name": name,
        "image_size": int(image_size),
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    if use_cache:
        block["cache_root"] = "datasets/cache"
    if name == "green_clahe":
        block["clahe_clip_limit"] = 2.0
        block["clahe_tile_grid_size"] = [8, 8]
    return block


def _loss(name: str) -> dict:
    if name == "cross_entropy":
        return {"name": "cross_entropy"}
    if name == "coral":
        return {"name": "coral"}
    if name == "mild_weighted_ce":
        return {"name": "mild_weighted_ce", "mild_sample_weight": 2.0}
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


def _job(
    *,
    group: str,
    family: str,
    cell: str,
    seed: int,
    arch: str,
    preprocess: str,
    loss: str,
    head: str,
    run_id: str,
    exp_name: str,
    desc: str,
    image_size: int = 224,
    use_cache: bool = True,
) -> dict:
    cfg = {
        "experiment": {
            "name": exp_name,
            "description": desc,
            "seed": int(seed),
            "protocol": "frozen_protocol_v1",
            "cell": cell,
            "family": family,
            "group": group,
            "version": "v3",
        },
        "data": deepcopy(DATA_BLOCK),
        "preprocessing": _preprocess(preprocess, image_size=image_size, use_cache=use_cache),
        "model": _model(arch, head),
        "loss": _loss(loss),
        "train": deepcopy(TRAIN_BLOCK),
        "evaluate": deepcopy(EVAL_BLOCK),
    }
    return {
        "run_id": run_id,
        "group": group,
        "family": family,
        "cell": cell,
        "seed": int(seed),
        "config_rel": f"configs/paper_v3/{run_id}.yaml",
        "config": cfg,
    }


def all_jobs() -> list[dict]:
    jobs: list[dict] = []

    # --- core: matched B0 × 42–44 ---
    for seed in (42, 43, 44):
        for cell in ("H1", "H2", "H3", "H4"):
            spec = CELLS[cell]
            jobs.append(
                _job(
                    group="core",
                    family="matched_b0",
                    cell=cell,
                    seed=seed,
                    arch="efficientnet_b0",
                    preprocess=spec["preprocess"],
                    loss=spec["loss"],
                    head=spec["head"],
                    run_id=f"matched_b0_{cell.lower()}_s{seed}_v3",
                    exp_name=f"matched_efficientnet_b0_{spec['preprocess']}_{spec['loss']}_s{seed}_v3",
                    desc=(
                        f"Paper v3 matched B0 {cell}: EfficientNet-B0 {spec['preprocess']} × "
                        f"{spec['loss']} seed {seed} (CPU bs=4 freeze_bn=True; archived B0 untouched)."
                    ),
                )
            )

    # --- core: hybrid seeds 45–47 ---
    for seed in (45, 46, 47):
        for cell in ("H1", "H2", "H3", "H4"):
            spec = CELLS[cell]
            jobs.append(
                _job(
                    group="core",
                    family="hybrid",
                    cell=cell,
                    seed=seed,
                    arch="mobilevit_s",
                    preprocess=spec["preprocess"],
                    loss=spec["loss"],
                    head=spec["head"],
                    run_id=f"hybrid_{cell.lower()}_s{seed}_v3",
                    exp_name=f"hybrid_mobilevit_s_{spec['preprocess']}_{spec['loss']}_s{seed}_v3",
                    desc=(
                        f"Paper v3 hybrid {cell}: MobileViT-S {spec['preprocess']} × "
                        f"{spec['loss']} seed {seed} (extra seed; CPU bs=4 freeze_bn=True)."
                    ),
                )
            )

    # --- decomp under CLAHE × 42–44 ---
    for seed in (42, 43, 44):
        jobs.append(
            _job(
                group="decomp",
                family="hybrid_decomp",
                cell="D1",
                seed=seed,
                arch="mobilevit_s",
                preprocess="green_clahe",
                loss="coral",
                head="coral",
                run_id=f"hybrid_d1_coral_clahe_s{seed}_v3",
                exp_name=f"hybrid_mobilevit_s_green_clahe_coral_s{seed}_v3",
                desc=(
                    f"Paper v3 decomp D1: MobileViT-S green_clahe × unweighted CORAL "
                    f"seed {seed} (ordinal head without mild weights)."
                ),
            )
        )
        jobs.append(
            _job(
                group="decomp",
                family="hybrid_decomp",
                cell="D2",
                seed=seed,
                arch="mobilevit_s",
                preprocess="green_clahe",
                loss="mild_weighted_ce",
                head="ce",
                run_id=f"hybrid_d2_mildce_clahe_s{seed}_v3",
                exp_name=f"hybrid_mobilevit_s_green_clahe_mild_weighted_ce_s{seed}_v3",
                desc=(
                    f"Paper v3 decomp D2: MobileViT-S green_clahe × mild-weighted CE "
                    f"seed {seed} (CE head with mild sample weight 2; no ordinal head)."
                ),
            )
        )

    # --- optional 448×448 H1 vs H2 × 42–44 ---
    for seed in (42, 43, 44):
        for cell in ("H1", "H2"):
            spec = CELLS[cell]
            jobs.append(
                _job(
                    group="res448",
                    family="hybrid_448",
                    cell=cell,
                    seed=seed,
                    arch="mobilevit_s",
                    preprocess=spec["preprocess"],
                    loss=spec["loss"],
                    head=spec["head"],
                    run_id=f"hybrid_{cell.lower()}_448_s{seed}_v3",
                    exp_name=f"hybrid_mobilevit_s_{spec['preprocess']}_{spec['loss']}_448_s{seed}_v3",
                    desc=(
                        f"Paper v3 resolution {cell}: MobileViT-S {spec['preprocess']} × "
                        f"{spec['loss']} at 448×448 seed {seed} (no 224 cache; online resize)."
                    ),
                    image_size=448,
                    use_cache=False,
                )
            )

    expected = 24 + 6 + 6
    if len(jobs) != expected:
        raise SystemExit(f"Expected {expected} jobs, got {len(jobs)}")
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
            "group": job["group"],
            "family": job["family"],
            "cell": job["cell"],
            "seed": job["seed"],
            "config_rel": job["config_rel"],
            "config_hash": h,
            "batch_size": cfg["train"]["batch_size"],
            "device": cfg["train"]["device"],
            "freeze_bn": cfg["train"]["freeze_bn"],
            "image_size": cfg["preprocessing"]["image_size"],
        }
        manifest.append(entry)
        print(f"WROTE {path.relative_to(ROOT)} group={job['group']} hash={h}")

    counts = {}
    for e in manifest:
        counts[e["group"]] = counts.get(e["group"], 0) + 1
    OUT_MANIFEST.write_text(
        json.dumps({"n_jobs": len(manifest), "groups": counts, "jobs": manifest}, indent=2)
    )
    print(f"MANIFEST {OUT_MANIFEST.relative_to(ROOT)} n={len(manifest)} groups={counts}")


if __name__ == "__main__":
    main()
