#!/usr/bin/env python3
"""Phase O — CNN vs Hybrid vs Transformer (matched green_clahe × CE recipe).

Trains Swin-Tiny seeds; compares to existing B0 (G+CLAHE×CE) and Hybrid H2.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SEEDS_DEFAULT = [42, 43, 44]


def run(cmd: list[str], log: Path) -> int:
    print(">>>", " ".join(cmd), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write("\n>>> " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"EXIT:{p.returncode}\n")
    print(f"EXIT:{p.returncode}", flush=True)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    args = ap.parse_args()
    log = ROOT / "logs" / "phase_o_swin.log"
    log.write_text(f"PHASE_O_START seeds={args.seeds}\n")

    for seed in args.seeds:
        cfg = f"configs/phase_o/swin_tiny_green_clahe_ce_s{seed}.yaml"
        rid = f"phase_o_swin_tiny_s{seed}_v1"
        summary = ROOT / "results" / rid / "train_summary.json"
        if summary.is_file():
            print(f"SKIP train {rid}", flush=True)
        else:
            while not summary.is_file():
                rc = run([PY, "-m", "src.train", "--config", cfg, "--run-id", rid], log)
                if rc not in (0, 75):
                    raise SystemExit(f"train failed {rid} rc={rc}")
                if rc == 0 and summary.is_file():
                    break
                if rc == 0 and not summary.is_file():
                    # unexpected
                    break
                time.sleep(5)

        # eval
        needed = [
            "aptos_test_metrics.json",
            "idrid_external_metrics.json",
            "messidor2_external_metrics.json",
            "deepdrid_external_metrics.json",
        ]
        out = ROOT / "results" / rid
        if all((out / n).is_file() for n in needed):
            print(f"SKIP eval {rid}", flush=True)
            continue
        ckpt = ROOT / "checkpoints" / rid / "best.pt"
        exp_cfg = ROOT / "experiments" / rid / "config.yaml"
        if not exp_cfg.is_file():
            exp_cfg = ROOT / cfg  # fallback
        for attempt in range(1, 8):
            if all((out / n).is_file() for n in needed):
                break
            rc = run(
                [
                    PY, "-u", "-m", "src.evaluate",
                    "--config", str(exp_cfg if exp_cfg.is_file() else ROOT / cfg),
                    "--checkpoint", str(ckpt),
                    "--n-boot", "20",
                    "--results-dir", str(out),
                ],
                ROOT / "logs" / f"phase_o_eval_s{seed}.log",
            )
            if rc == 0 and all((out / n).is_file() for n in needed):
                break
            time.sleep(20)
        else:
            raise SystemExit(f"eval failed {rid}")

    # summary vs B0 / H2
    rc = run([PY, "-u", "scripts/summarize_phase_o.py"], log)
    if rc != 0:
        raise SystemExit(rc)
    print("PHASE_O_ALL_OK", flush=True)
    with log.open("a") as f:
        f.write("PHASE_O_ALL_OK\n")


if __name__ == "__main__":
    main()
