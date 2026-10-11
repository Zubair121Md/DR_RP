#!/usr/bin/env python3
"""
Optional sensitivity: one Normal-BN + larger-batch GPU check for MobileViT-S H2.

Not part of the default paper_v3 queue. Writes configs/paper_v3_optional/*.yaml only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "paper_v3_optional"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "hybrid_h2_gpu_bn_s42_v3_opt"
    cfg = {
        "experiment": {
            "name": "hybrid_mobilevit_s_green_clahe_ce_gpu_bn_s42_v3_opt",
            "description": (
                "Optional sensitivity: MobileViT-S H2 green_clahe × CE, seed 42, "
                "device=mps, batch_size=16, freeze_bn=False (not in core v3 queue)."
            ),
            "seed": 42,
            "protocol": "frozen_protocol_v1",
            "cell": "H2",
            "family": "hybrid_gpu_bn",
            "group": "optional_gpu_bn",
            "version": "v3_opt",
        },
        "data": {
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
        },
        "preprocessing": {
            "name": "green_clahe",
            "image_size": 224,
            "cache_root": "datasets/cache",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8],
        },
        "model": {
            "name": "mobilevit_s",
            "source": "timm",
            "pretrained": True,
            "head": "ce",
            "num_classes": 5,
        },
        "loss": {"name": "cross_entropy"},
        "train": {
            "train_dataset": "aptos",
            "epochs": 20,
            "batch_size": 16,
            "num_workers": 0,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "amp": False,
            "selection_metric": "quadratic_weighted_kappa",
            "interrupt_every": 0,
            "device": "mps",
            "freeze_bn": False,
            "max_steps_per_launch": 0,
        },
        "evaluate": {
            "datasets": [
                {"name": "aptos_test", "dataset": "aptos", "split": "test"},
                {"name": "idrid_external", "dataset": "idrid", "split": "all"},
                {"name": "messidor2_external", "dataset": "messidor2", "split": "all"},
                {"name": "deepdrid_external", "dataset": "deepdrid", "split": "all"},
            ]
        },
    }
    path = OUT / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    h = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:16]
    print(f"WROTE {path.relative_to(ROOT)} hash={h}")
    print()
    print("Manual launch (after you confirm; not auto-started):")
    print(f"  python3 -u -m src.train --config {path.relative_to(ROOT)} --run-id {run_id}")
    print(
        "  python3 -u -m src.evaluate "
        f"--config {path.relative_to(ROOT)} "
        f"--checkpoint checkpoints/{run_id}/best.pt "
        f"--results-dir results/{run_id} --n-boot 20"
    )
    print()
    print("Estimate: ~1–3 h on MPS for train+eval (much faster than CPU).")
    print("OPTIONAL_GPU_BN_CONFIG_OK")


if __name__ == "__main__":
    main()
