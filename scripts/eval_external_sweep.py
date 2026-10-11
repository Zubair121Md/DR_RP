#!/usr/bin/env python3
"""
Re-evaluate all checkpoints with full external suite:
APTOS test, IDRiD, Messidor-2, DeepDRiD (+ logits, grade-1 F1, temp-scaled ECE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.evaluate import evaluate  # noqa: E402
from src.utils import load_config, save_json  # noqa: E402


def _find_config_for_ckpt(ckpt: Path, project: Path) -> Path | None:
    # Prefer sibling experiment config
    run_id = ckpt.parent.name
    exp_yaml = project / "experiments" / run_id / "config.yaml"
    if exp_yaml.is_file():
        return exp_yaml
    exp_json = project / "experiments" / run_id / "config.json"
    if exp_json.is_file():
        return exp_json
    # Fall back: match by experiment name prefix in configs/
    name = run_id.rsplit("_", 2)[0] if "_" in run_id else run_id
    for folder in (project / "configs", project / "configs" / "seed_runs", project / "configs" / "hybrid_runs"):
        if not folder.is_dir():
            continue
        for p in folder.glob("*.yaml"):
            if name.startswith(p.stem) or p.stem in run_id:
                return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--only", type=str, nargs="*", default=None, help="Substring filters on run_id")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoints_root)
    if not ckpt_root.is_absolute():
        ckpt_root = PROJECT / ckpt_root

    ckpts = sorted(ckpt_root.glob("*/best.pt"))
    if args.only:
        ckpts = [c for c in ckpts if any(s in c.parent.name for s in args.only)]
    if args.limit > 0:
        ckpts = ckpts[: args.limit]

    summary = []
    for i, ckpt in enumerate(ckpts, 1):
        cfg_path = _find_config_for_ckpt(ckpt, PROJECT)
        if cfg_path is None:
            print(f"[{i}/{len(ckpts)}] SKIP (no config): {ckpt}")
            continue
        print(f"\n[{i}/{len(ckpts)}] {ckpt.parent.name}")
        print(f"  config: {cfg_path}")
        if cfg_path.suffix == ".json":
            config = json.loads(cfg_path.read_text())
        else:
            config = load_config(cfg_path)
        # Ensure messidor/deepdrid roots exist in older configs
        data = config.setdefault("data", {})
        data.setdefault(
            "messidor2",
            {"root": str((PROJECT / "Messidor ").resolve()), "labels_csv": "messidor_data.csv", "images_dir": "IMAGES"},
        )
        data.setdefault("deepdrid", {"root": "DeepDRiD"})
        # Prefer online Messidor if cache not built yet: leave cache_root as-is;
        # evaluate.py falls back when messidor cache missing.
        try:
            report = evaluate(
                config,
                checkpoint=ckpt,
                project_root=PROJECT,
                n_boot=args.n_boot,
            )
            row = {
                "run_id": report.get("run_id"),
                "checkpoint": str(ckpt),
                "temperature": report.get("temperature"),
            }
            for key in ("aptos_test", "idrid_external", "messidor2_external", "deepdrid_external"):
                block = report.get(key) or {}
                m = block.get("metrics") or {}
                if m:
                    row[key] = {
                        "n": block.get("n"),
                        "qwk": m.get("quadratic_weighted_kappa"),
                        "macro_f1": m.get("macro_f1"),
                        "mild_f1": m.get("mild_dr_f1"),
                        "mild_recall": m.get("mild_dr_recall"),
                        "ece": m.get("ece"),
                        "ece_t": m.get("ece_temperature_scaled"),
                    }
                elif "error" in block:
                    row[key] = {"error": block["error"]}
            summary.append(row)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            summary.append({"run_id": ckpt.parent.name, "error": str(exc)})

    out = PROJECT / "results" / "external_eval_sweep.json"
    save_json(out, summary)
    print(f"\nSWEEP_DONE n={len(summary)} → {out}")


if __name__ == "__main__":
    main()
