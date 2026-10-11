#!/usr/bin/env python3
"""Watchdog: keep scripts/run_paper_v3.py alive under caffeinate until ALL_OK."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
LOG = ROOT / "logs" / "paper_v3_watchdog.log"
PHASE = ROOT / "logs" / "paper_v3.log"
DONE = "PAPER_V3_ALL_OK"
PID_FILE = ROOT / "logs" / "paper_v3.pid"
OUTER_LOG = ROOT / "logs" / "paper_v3_stdout.log"
POLL_SEC = int(os.environ.get("PAPER_V3_WATCHDOG_POLL", "90"))

PATTERNS = (
    "scripts/run_paper_v3.py",
    "src.train --config configs/paper_v3",
    "src.evaluate --config",
)


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


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
    # Preserve group selection across relaunches.
    if "PAPER_V3_GROUPS" not in env:
        env["PAPER_V3_GROUPS"] = "core,decomp"
    cmd = ["caffeinate", "-is", sys.executable, "-u", "scripts/run_paper_v3.py"]
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
    log(f"WATCHDOG_RELAUNCH pid={p.pid} groups={env.get('PAPER_V3_GROUPS')}")


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
