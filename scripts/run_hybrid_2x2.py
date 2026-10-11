#!/usr/bin/env python3
"""
Phase G/H — Hybrid 2×2 under frozen_protocol_v1.

Primary cells (H1–H4), fixed run_ids for interrupt/jetsam resume.
Completion markers are seed-specific so Phase H does not stop on Phase G's ALL_OK.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PY = sys.executable

CELL_FILES = {
    "H1": "hybrid_mobilevit_s_rgb_ce_s{seed}.yaml",
    "H2": "hybrid_mobilevit_s_green_clahe_ce_s{seed}.yaml",
    "H3": "hybrid_mobilevit_s_rgb_mild_coral_s{seed}.yaml",
    "H4": "hybrid_mobilevit_s_green_clahe_mild_coral_s{seed}.yaml",
}


def run_id_for(cell: str, seed: int) -> str:
    return f"hybrid_{cell.lower()}_s{seed}_v1"


def log_path_for(seeds: list[int]) -> Path:
    tag = "_".join(str(s) for s in seeds)
    return PROJECT / "logs" / f"hybrid_2x2_phase_s{tag}.log"


def done_marker(seeds: list[int]) -> str:
    tag = "_".join(str(s) for s in seeds)
    return f"HYBRID_2X2_ALL_OK_SEEDS_{tag}"


def run(cmd: list[str], log_file: Path) -> int:
    line = " ".join(cmd)
    print(f"\n>>> {line}", flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n>>> {line}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"EXIT:{proc.returncode}\n")
        log.flush()
    print(f"EXIT:{proc.returncode}", flush=True)
    return int(proc.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--cells", nargs="+", default=["H1", "H2", "H3", "H4"])
    args = p.parse_args()
    seeds = args.seeds or ([args.seed] if args.seed is not None else [42])
    log_file = log_path_for(seeds)
    marker = done_marker(seeds)

    if not log_file.exists():
        log_file.write_text(f"HYBRID_2X2_START seeds={seeds}\n", encoding="utf-8")
    else:
        with log_file.open("a") as log:
            log.write("\nHYBRID_2X2_RESUME_ORCHESTRATOR\n")

    fails: list[str] = []
    for seed in seeds:
        for cell in args.cells:
            if cell not in CELL_FILES:
                raise SystemExit(f"Unknown cell {cell}; choose {list(CELL_FILES)}")
            rel = f"configs/hybrid_runs/{CELL_FILES[cell].format(seed=seed)}"
            path = PROJECT / rel
            if not path.is_file():
                raise SystemExit(f"Missing config {path}")
            rid = run_id_for(cell, seed)
            summary = PROJECT / "results" / rid / "train_summary.json"
            if not summary.is_file():
                summary = PROJECT / "results" / rid / "train_summary.json"
            if summary.is_file():
                print(f"SKIP complete {rid}", flush=True)
                continue
            rc = run([PY, "-m", "src.train", "--config", rel, "--run-id", rid], log_file)
            if rc == 75:
                print(f"CHUNK_CONTINUE {cell} seed={seed} run_id={rid}", flush=True)
                import time as _time

                while not (PROJECT / "results" / rid / "train_summary.json").is_file():
                    _time.sleep(8)
                    rc = run([PY, "-m", "src.train", "--config", rel, "--run-id", rid], log_file)
                    if rc == 0:
                        break
                    if rc == 75:
                        print(f"CHUNK_CONTINUE {rid}", flush=True)
                        continue
                    fails.append(f"{cell}_s{seed}")
                    print(f"TRAIN_FAIL {cell} seed={seed} run_id={rid} rc={rc}", flush=True)
                    raise SystemExit(rc)
                continue
            if rc != 0:
                import time as _time

                _time.sleep(5)
                fails.append(f"{cell}_s{seed}")
                print(f"TRAIN_FAIL {cell} seed={seed} run_id={rid}", flush=True)
                raise SystemExit(rc)

    if fails:
        print(f"FAILED: {fails}", flush=True)
        raise SystemExit(1)
    print(marker, flush=True)
    with log_file.open("a") as log:
        log.write(f"{marker}\n")
        if seeds == [42]:
            log.write("HYBRID_2X2_ALL_OK\n")


if __name__ == "__main__":
    main()
