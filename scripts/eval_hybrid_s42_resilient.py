#!/usr/bin/env python3
"""Phase I: evaluate hybrid H1-H4 one checkpoint at a time with retries."""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CELLS = ["hybrid_h1_s42_v1", "hybrid_h2_s42_v1", "hybrid_h3_s42_v1", "hybrid_h4_s42_v1"]
# Exact basenames produced by src.evaluate (save_json f"{name}_metrics.json")
NEEDED = [
    "aptos_test_metrics.json",
    "idrid_external_metrics.json",
    "messidor2_external_metrics.json",
    "deepdrid_external_metrics.json",
]
LOG = ROOT / "logs" / "hybrid_s42_eval_resilient.log"

def complete(run_id: str) -> bool:
    d = ROOT / "results" / run_id
    return all((d / n).is_file() for n in NEEDED)

def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as log:
        log.write(f"\nRESILIENT_EVAL_START n_boot={n_boot}\n")
    for run_id in CELLS:
        if complete(run_id):
            print(f"SKIP complete {run_id}", flush=True)
            continue
        ckpt = ROOT / "checkpoints" / run_id / "best.pt"
        cfg = ROOT / "experiments" / run_id / "config.yaml"
        if not ckpt.is_file() or not cfg.is_file():
            print(f"MISSING {run_id}", flush=True)
            continue
        ok = False
        for attempt in range(1, 8):
            if complete(run_id):
                ok = True
                break
            print(f"EVAL {run_id} attempt={attempt}", flush=True)
            env = os.environ.copy()
            env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1"})
            with LOG.open("a") as log:
                log.write(f"EVAL {run_id} attempt={attempt}\n"); log.flush()
                proc = subprocess.run(
                    [PY,"-u","-m","src.evaluate","--config",str(cfg),"--checkpoint",str(ckpt),
                     "--n-boot",str(n_boot),"--results-dir",str(ROOT/"results"/run_id)],
                    cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
                )
                log.write(f"EXIT:{proc.returncode}\n")
            print(f"EXIT:{proc.returncode} complete={complete(run_id)} files={sorted(x.name for x in (ROOT/'results'/run_id).glob('*metrics.json'))}", flush=True)
            if proc.returncode == 0 and complete(run_id):
                ok = True
                break
            time.sleep(25)
        if not ok:
            print(f"FAIL {run_id}", flush=True)
            raise SystemExit(1)
        time.sleep(8)
    print("HYBRID_S42_EVAL_ALL_OK", flush=True)
    with LOG.open("a") as log:
        log.write("HYBRID_S42_EVAL_ALL_OK\n")

if __name__ == "__main__":
    main()
