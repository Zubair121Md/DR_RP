#!/usr/bin/env python3
"""Phase P orchestrator: LODO train + eval for each holdout fold."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
HOLDOUTS = ("idrid", "messidor2", "deepdrid")
HOLD_METRICS = {
    "idrid": "idrid_external_metrics.json",
    "messidor2": "messidor2_external_metrics.json",
    "deepdrid": "deepdrid_external_metrics.json",
}


def run(cmd: list[str], log: Path) -> int:
    print(">>>", " ".join(cmd), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write("\n>>> " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"EXIT:{p.returncode}\n")
    print("EXIT", p.returncode, flush=True)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdouts", nargs="+", default=list(HOLDOUTS))
    ap.add_argument("--config", default="configs/phase_p/lodo_mobilevit_green_clahe_ce.yaml")
    args = ap.parse_args()
    log = ROOT / "logs" / f"phase_p_lodo_s{args.seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"PHASE_P_START seed={args.seed} holdouts={args.holdouts}\n")

    for holdout in args.holdouts:
        rid = f"phase_p_lodo_holdout_{holdout}_s{args.seed}_v1"
        summary = ROOT / "results" / rid / "train_summary.json"
        if not summary.is_file():
            while not summary.is_file():
                rc = run(
                    [
                        PY, "-u", "-m", "src.train_lodo",
                        "--config", args.config,
                        "--holdout", holdout,
                        "--run-id", rid,
                        "--seed", str(args.seed),
                    ],
                    log,
                )
                if rc not in (0, 75):
                    raise SystemExit(f"LODO train failed {rid} rc={rc}")
                if rc == 0:
                    break
                time.sleep(5)

        # Full external eval (all sets) for the checkpoint
        out = ROOT / "results" / rid
        needed = [
            "aptos_test_metrics.json",
            "idrid_external_metrics.json",
            "messidor2_external_metrics.json",
            "deepdrid_external_metrics.json",
        ]
        if all((out / n).is_file() for n in needed):
            print(f"SKIP eval {rid}", flush=True)
            continue
        ckpt = ROOT / "checkpoints" / rid / "best.pt"
        cfg = ROOT / "experiments" / rid / "config.yaml"
        if not cfg.is_file():
            # write from json
            cfg_json = ROOT / "experiments" / rid / "config.json"
            cfg = ROOT / args.config
        for attempt in range(1, 8):
            if all((out / n).is_file() for n in needed):
                break
            rc = run(
                [
                    PY, "-u", "-m", "src.evaluate",
                    "--config", str(cfg),
                    "--checkpoint", str(ckpt),
                    "--n-boot", "20",
                    "--results-dir", str(out),
                ],
                ROOT / "logs" / f"phase_p_eval_{holdout}_s{args.seed}.log",
            )
            if rc == 0 and all((out / n).is_file() for n in needed):
                break
            time.sleep(20)
        else:
            raise SystemExit(f"eval failed {rid}")

    run([PY, "-u", "scripts/summarize_phase_p.py", "--seed", str(args.seed)], log)
    print("PHASE_P_ALL_OK", flush=True)
    with log.open("a") as f:
        f.write("PHASE_P_ALL_OK\n")
    # mean±std table when seeds 42/43/44 are all complete
    if all(
        (ROOT / "results" / f"phase_p_lodo_holdout_{h}_s{s}_v1" / HOLD_METRICS[h]).is_file()
        for s in (42, 43, 44)
        for h in HOLDOUTS
    ):
        run([PY, "-u", "scripts/summarize_phase_p.py", "--seeds", "42", "43", "44"], log)


if __name__ == "__main__":
    main()
