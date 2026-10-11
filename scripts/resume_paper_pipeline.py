#!/usr/bin/env python3
"""
Resume paper pipeline from MobileViT hybrid grid onward (mild-CORAL B0 already done).

Memory-safe defaults for 8GB Macs (Cursor + unified memory):
  - hybrid configs: device=cpu, batch_size=2
  - OMP/MKL threads capped at 2
  - 45s pause between jobs; retry killed runs up to 5×
  - external sweep n_boot=200

Run under caffeinate (see scripts/resume_hybrids_caffeinated.sh).
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LOG = PROJECT / "logs" / "resume_pipeline.log"
STDOUT = PROJECT / "logs" / "resume_pipeline_stdout.log"
PY = sys.executable

HYBRID_ORDER = [
    "hybrid_mobilevit_s_green_clahe_ce_s42.yaml",
    "hybrid_mobilevit_s_green_clahe_coral_s42.yaml",
    "hybrid_mobilevit_s_green_clahe_mild_coral_s42.yaml",
    "hybrid_mobilevit_s_rgb_ce_s42.yaml",
    "hybrid_mobilevit_s_rgb_coral_s42.yaml",
    "hybrid_mobilevit_s_rgb_mild_coral_s42.yaml",
]

PAUSE_SEC = 60
MAX_TRAIN_RETRIES = 40


def _log(msg: str) -> None:
    print(msg, flush=True)
    with STDOUT.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(cmd: list[str]) -> int:
    line = " ".join(cmd)
    _log(f"\n>>> {line}")
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n>>> {line}\n")
        log.flush()
        env = {
            **os.environ,
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "VECLIB_MAXIMUM_THREADS": "2",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        }
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,  # jetsam/OOM kills train without taking the orchestrator
        )
        log.write(f"EXIT:{proc.returncode}\n")
        log.flush()
    _log(f"EXIT:{proc.returncode}")
    return int(proc.returncode)


def _hybrid_done(config_stem: str) -> bool:
    import json

    # Stable resume run id used by this orchestrator
    stable = PROJECT / "results" / f"{config_stem}_stable" / "train_summary.json"
    if stable.is_file():
        try:
            s = json.loads(stable.read_text())
            if s.get("epochs") == 20 and s.get("best_epoch", -1) >= 1:
                return True
        except Exception:
            pass
    for p in (PROJECT / "results").glob(f"{config_stem}_*/train_summary.json"):
        try:
            s = json.loads(p.read_text())
            if s.get("epochs") == 20 and s.get("best_epoch", -1) >= 1:
                return True
        except Exception:
            continue
    return False


def _reclaim() -> None:
    gc.collect()
    _log(f"PAUSE {PAUSE_SEC}s for memory reclaim")
    time.sleep(PAUSE_SEC)


def _train_with_retries(cfg_rel: str, stem: str) -> bool:
    run_id = f"{stem}_stable"
    for attempt in range(1, MAX_TRAIN_RETRIES + 1):
        if _hybrid_done(stem):
            _log(f"SKIP_DONE {stem}")
            return True
        _log(f"TRAIN_ATTEMPT {attempt}/{MAX_TRAIN_RETRIES} {stem} run_id={run_id}")
        _reclaim()
        rc = run(
            [
                PY,
                "-m",
                "src.train",
                "--config",
                cfg_rel,
                "--run-id",
                run_id,
            ]
        )
        if rc == 0 and _hybrid_done(stem):
            _log(f"TRAIN_OK {stem}")
            return True
        _log(f"TRAIN_FAIL rc={rc} {stem} — will retry/resume after pause")
    return _hybrid_done(stem)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Append across relaunches so history is preserved
    with LOG.open("a", encoding="utf-8") as log:
        log.write("\nRESUME_PIPELINE_START\n")
    with STDOUT.open("a", encoding="utf-8") as f:
        f.write("\nRESUME_PIPELINE_START\n")
    _log("RESUME_PIPELINE_START (hybrids → eval sweep → efficiency)")

    hybrid_dir = PROJECT / "configs" / "hybrid_runs"
    for name in HYBRID_ORDER:
        cfg = hybrid_dir / name
        if not cfg.is_file():
            _log(f"MISSING_CONFIG {name}")
            continue
        stem = cfg.stem
        if _hybrid_done(stem):
            _log(f"SKIP_DONE {stem}")
            continue
        ok = _train_with_retries(str(cfg.relative_to(PROJECT)), stem)
        if not ok:
            _log(f"TRAIN_GAVE_UP {stem} — continuing remaining jobs")
        _reclaim()

    _log("HYBRIDS_PHASE_DONE — starting external eval sweep")
    sweep_out = PROJECT / "results" / "external_eval_sweep.json"
    if sweep_out.is_file():
        _log(f"SKIP_DONE external_eval_sweep ({sweep_out})")
    else:
        _reclaim()
        # Focus on paper models (hybrids + mild-CORAL B0). Full archive is 22 ckpts — too heavy for 8GB.
        run(
            [
                PY,
                "scripts/eval_external_sweep.py",
                "--n-boot",
                "200",
                "--only",
                "_stable",
                "mild_coral",
            ]
        )
    _reclaim()
    run([PY, "scripts/efficiency_b0.py"])
    _reclaim()
    eff_out = PROJECT / "results" / "efficiency_models.json"
    if eff_out.is_file():
        _log(f"SKIP_DONE efficiency_models ({eff_out})")
    else:
        run([PY, "scripts/efficiency_models.py"])

    with LOG.open("a", encoding="utf-8") as log:
        log.write("RESUME_PIPELINE_DONE\n")
    _log("RESUME_PIPELINE_DONE")


if __name__ == "__main__":
    main()
