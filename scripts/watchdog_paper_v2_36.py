#!/usr/bin/env python3
"""Watchdog: keep scripts/run_paper_v2_36.py alive under caffeinate until ALL_OK.

Detects orchestrator / src.train / src.evaluate for paper_v2 configs.
Relaunches with caffeinate -is if nothing is running and DONE marker absent.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
LOG = ROOT / "logs" / "paper_v2_36_watchdog.log"
PHASE = ROOT / "logs" / "paper_v2_36.log"
DONE = "PAPER_V2_36_ALL_OK"
PID_FILE = ROOT / "logs" / "paper_v2_36.pid"
OUTER_LOG = ROOT / "logs" / "paper_v2_36_stdout.log"
POLL_SEC = int(os.environ.get("PAPER_V2_WATCHDOG_POLL", "90"))

PATTERNS = (
    "scripts/run_paper_v2_36.py",
    "src.train --config configs/paper_v2",
    "src.evaluate --config",
)


def log(msg: str) -> None:
    line = f"{msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def alive() -> bool:
    for pat in PATTERNS:
        try:
            out = subprocess.check_output(["pgrep", "-f", pat], text=True)
            if out.strip():
                return True
        except subprocess.CalledProcessError:
            continue
    return False


def done() -> bool:
    return PHASE.is_file() and DONE in PHASE.read_text(errors="ignore")


def relaunch() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    cmd = [
        "caffeinate",
        "-is",
        sys.executable,
        "-u",
        "scripts/run_paper_v2_36.py",
    ]
    OUTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUTER_LOG.open("a") as out:
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_FILE.write_text(str(p.pid))
    log(f"WATCHDOG_RELAUNCH pid={p.pid}")


def main() -> None:
    log(f"WATCHDOG_START poll={POLL_SEC}s")
    while True:
        if done():
            log(f"WATCHDOG_DONE {DONE}")
            break
        if not alive():
            log("WATCHDOG_IDLE — relaunching orchestrator")
            relaunch()
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
