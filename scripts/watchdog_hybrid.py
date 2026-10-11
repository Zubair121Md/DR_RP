#!/usr/bin/env python3
"""Hybrid Phase G/H watchdog (user session; Desktop-accessible). Seed-specific done markers."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
LOG = ROOT / "logs" / "hybrid_watchdog.log"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
CELLS = sys.argv[2:] or ["H1", "H2", "H3", "H4"]
PHASE = ROOT / "logs" / f"hybrid_2x2_phase_s{SEED}.log"
DONE = f"HYBRID_2X2_ALL_OK_SEEDS_{SEED}"
LOG.parent.mkdir(exist_ok=True)


def alive() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "src.train --config configs/hybrid_runs"], text=True
        )
        return bool(out.strip())
    except subprocess.CalledProcessError:
        pass
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "scripts/run_hybrid_2x2.py"], text=True
        )
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def log(msg: str) -> None:
    line = f"{msg}\n"
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(line)


log(f"PY_WATCHDOG_START seed={SEED} cells={CELLS}")
while True:
    if PHASE.exists() and DONE in PHASE.read_text(errors="ignore"):
        log(f"PY_WATCHDOG_DONE {DONE}")
        break
    if not alive():
        log(f"PY_WATCHDOG_RELAUNCH seed={SEED}")
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            }
        )
        cmd = [
            "caffeinate",
            "-is",
            sys.executable,
            "-u",
            "scripts/run_hybrid_2x2.py",
            "--seed",
            str(SEED),
            "--cells",
            *CELLS,
        ]
        with (ROOT / "logs" / f"hybrid_2x2_stdout_s{SEED}.log").open("a") as out:
            p = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (ROOT / "logs" / f"hybrid_2x2_s{SEED}.pid").write_text(str(p.pid))
        log(f"PY_WATCHDOG_STARTED pid={p.pid}")
    time.sleep(60)
