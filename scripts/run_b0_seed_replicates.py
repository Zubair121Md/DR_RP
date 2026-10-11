#!/usr/bin/env python3
"""
Run B0 2×2 seed replicates (research question: preprocess × ordinal head).

Cells (same as Exp1–4):
  rgb + CE, green_clahe + CE, green_clahe + CORAL, rgb + CORAL

Existing seed-42 runs are kept; this trains seeds 43/44 (default) with cache.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent

CELLS = [
    {
        "key": "rgb_ce",
        "base": "configs/baseline_efficientnet_b0_rgb_local.yaml",
        "label": "RGB+CE",
    },
    {
        "key": "green_clahe_ce",
        "base": "configs/baseline_efficientnet_b0_green_clahe_local.yaml",
        "label": "G+CLAHE+CE",
    },
    {
        "key": "green_clahe_coral",
        "base": "configs/baseline_efficientnet_b0_green_clahe_coral_local.yaml",
        "label": "G+CLAHE+CORAL",
    },
    {
        "key": "rgb_coral",
        "base": "configs/baseline_efficientnet_b0_rgb_coral_local.yaml",
        "label": "RGB+CORAL",
    },
]


def write_seed_config(base_rel: str, seed: int, key: str) -> Path:
    base_path = PROJECT / base_rel
    cfg = yaml.safe_load(base_path.read_text())
    cfg["experiment"]["seed"] = int(seed)
    # Keep base name + seed tag for readable run_ids
    root_name = str(cfg["experiment"]["name"]).rstrip("_")
    # Avoid double-tagging if already seeded
    if f"_s{seed}" not in root_name:
        cfg["experiment"]["name"] = f"{root_name}_s{seed}"
    cfg["experiment"]["description"] = (
        str(cfg["experiment"].get("description", "")).strip()
        + f" Seed replicate s={seed} for B0 2×2 (preprocess × ordinal head)."
    )
    # Ensure cache is on for deployable-compute path
    cfg.setdefault("preprocessing", {})["cache_root"] = "datasets/cache"
    out = PROJECT / "configs" / "seed_runs" / f"{key}_s{seed}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def run_one(cfg_path: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n===== TRAIN {cfg_path.name} =====\n")
        log.flush()
        train = subprocess.run(
            ["python3", "-u", "-m", "src.train", "--config", str(cfg_path)],
            cwd=PROJECT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if train.returncode != 0:
            raise RuntimeError(f"train failed ({train.returncode}) for {cfg_path}")

        # Newest checkpoint matching experiment name prefix
        cfg = yaml.safe_load(cfg_path.read_text())
        prefix = cfg["experiment"]["name"]
        ckpt_dirs = sorted(
            (PROJECT / "checkpoints").glob(f"{prefix}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not ckpt_dirs or not (ckpt_dirs[0] / "best.pt").is_file():
            raise FileNotFoundError(f"No checkpoint for {prefix}")
        ckpt = ckpt_dirs[0] / "best.pt"
        log.write(f"===== EVAL {ckpt} =====\n")
        log.flush()
        ev = subprocess.run(
            [
                "python3",
                "-u",
                "-m",
                "src.evaluate",
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(ckpt),
                "--n-boot",
                "1000",
            ],
            cwd=PROJECT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if ev.returncode != 0:
            raise RuntimeError(f"eval failed ({ev.returncode}) for {ckpt}")
        return ckpt_dirs[0].name


def summarize(seeds: list[int]) -> Path:
    """Aggregate seed-42 anchors + new replicates into mean±std table."""
    # Anchor seed-42 run ids (from completed MPS ablation)
    anchors = {
        "rgb_ce": "baseline_efficientnet_b0_rgb_20260918_170311",
        "green_clahe_ce": "baseline_efficientnet_b0_green_clahe_20260918_092447",
        "green_clahe_coral": "baseline_efficientnet_b0_green_clahe_coral_20260918_203508",
        "rgb_coral": "baseline_efficientnet_b0_rgb_coral_20260919_063242",
    }
    cells = {}
    for cell in CELLS:
        key = cell["key"]
        runs = []
        # seed 42 anchor
        runs.append(("42", anchors[key]))
        for seed in seeds:
            prefix = yaml.safe_load((PROJECT / cell["base"]).read_text())["experiment"]["name"]
            # seeded configs rename to {name}_s{seed}
            # find matching result dirs
            matches = sorted(
                (PROJECT / "results").glob(f"{prefix}_s{seed}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not matches:
                # fallback: any dir containing key pattern and seed tag in config
                for p in sorted((PROJECT / "results").glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
                    cfgp = p / "config.json"
                    if not cfgp.is_file():
                        continue
                    c = json.loads(cfgp.read_text())
                    if int(c.get("experiment", {}).get("seed", -1)) != seed:
                        continue
                    prep = c.get("preprocessing", {}).get("name")
                    loss = c.get("loss", {}).get("name")
                    head = c.get("model", {}).get("head", "ce")
                    want_prep = "rgb" if key.startswith("rgb") else "green_clahe"
                    want_coral = key.endswith("coral")
                    if prep != want_prep:
                        continue
                    if want_coral and loss != "coral" and head != "coral":
                        continue
                    if not want_coral and loss == "coral":
                        continue
                    matches = [p]
                    break
            if matches:
                runs.append((str(seed), matches[0].name))

        metrics = []
        for seed, run_id in runs:
            er = PROJECT / "results" / run_id / "eval_report.json"
            if not er.is_file():
                continue
            e = json.loads(er.read_text())
            metrics.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "aptos_qwk": e["aptos_test"]["metrics"]["quadratic_weighted_kappa"],
                    "aptos_mild": e["aptos_test"]["metrics"]["mild_dr_recall"],
                    "aptos_ece": e["aptos_test"]["metrics"].get("ece"),
                    "idrid_qwk": e["idrid_external"]["metrics"]["quadratic_weighted_kappa"],
                    "idrid_mild": e["idrid_external"]["metrics"]["mild_dr_recall"],
                    "idrid_ece": e["idrid_external"]["metrics"].get("ece"),
                }
            )
        cells[key] = {"label": cell["label"], "runs": metrics}
        if len(metrics) >= 2:
            import statistics as stats

            def mean_std(vals):
                vals = [v for v in vals if v is not None]
                if not vals:
                    return None
                if len(vals) == 1:
                    return {"mean": vals[0], "std": 0.0, "n": 1}
                return {
                    "mean": float(stats.mean(vals)),
                    "std": float(stats.stdev(vals)),
                    "n": len(vals),
                }

            cells[key]["summary"] = {
                "aptos_qwk": mean_std([m["aptos_qwk"] for m in metrics]),
                "idrid_qwk": mean_std([m["idrid_qwk"] for m in metrics]),
                "idrid_mild": mean_std([m["idrid_mild"] for m in metrics]),
                "aptos_ece": mean_std([m["aptos_ece"] for m in metrics]),
            }

    out = PROJECT / "results" / "b0_2x2_seed_summary.json"
    out.write_text(
        json.dumps(
            {
                "note": "B0 2×2 seed replicates for preprocess × ordinal-head RQ",
                "seeds_requested": seeds,
                "cells": cells,
            },
            indent=2,
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument(
        "--cells",
        nargs="+",
        default=[c["key"] for c in CELLS],
        help="Subset of cell keys to run",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        out = summarize(args.seeds)
        print("SUMMARY", out)
        return

    log = PROJECT / "logs" / "b0_seed_replicates.log"
    completed = []
    for seed in args.seeds:
        for cell in CELLS:
            if cell["key"] not in args.cells:
                continue
            cfg = write_seed_config(cell["base"], seed, cell["key"])
            print(f"START {cell['label']} seed={seed} cfg={cfg.name}", flush=True)
            run_id = run_one(cfg, log)
            completed.append({"cell": cell["key"], "seed": seed, "run_id": run_id})
            print(f"DONE {cell['label']} seed={seed} -> {run_id}", flush=True)

    out = summarize(args.seeds)
    manifest = PROJECT / "results" / "b0_seed_replicates_manifest.json"
    manifest.write_text(json.dumps({"completed": completed, "summary": str(out)}, indent=2))
    print("MANIFEST", manifest)
    print("SUMMARY", out)


if __name__ == "__main__":
    main()
