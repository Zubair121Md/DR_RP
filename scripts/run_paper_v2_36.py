#!/usr/bin/env python3
"""Serial orchestrator for all 36 paper v2 trainings (+ 4-domain eval).

Skips completed jobs (train_summary + four domain metrics present).
Resumes chunked training (exit code 75). Never touches archived/non-v2 dirs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "logs" / "paper_v2_36.log"
DONE = "PAPER_V2_36_ALL_OK"
NEEDED_METRICS = [
    "aptos_test_metrics.json",
    "idrid_external_metrics.json",
    "messidor2_external_metrics.json",
    "deepdrid_external_metrics.json",
]
N_BOOT = int(os.environ.get("PAPER_V2_N_BOOT", "20"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict) -> str:
    blob = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_jobs() -> list[dict]:
    # Prefer generated manifest; regenerate if missing.
    man = ROOT / "configs" / "paper_v2" / "manifest_36.json"
    if not man.is_file():
        rc = subprocess.run(
            [PY, "-u", "scripts/generate_paper_v2_configs.py"],
            cwd=str(ROOT),
        )
        if rc.returncode != 0:
            raise SystemExit("config generation failed")
    data = json.loads(man.read_text())
    jobs = data["jobs"]
    if len(jobs) != 36:
        raise SystemExit(f"manifest has {len(jobs)} jobs, need 36")
    return jobs


def train_done(run_id: str) -> bool:
    return (ROOT / "results" / run_id / "train_summary.json").is_file()


def eval_done(run_id: str) -> bool:
    d = ROOT / "results" / run_id
    return all((d / n).is_file() for n in NEEDED_METRICS)


def job_done(run_id: str) -> bool:
    return train_done(run_id) and eval_done(run_id)


def write_run_meta(job: dict, cfg: dict, stage: str) -> None:
    run_id = job["run_id"]
    meta = {
        "run_id": run_id,
        "family": job["family"],
        "cell": job["cell"],
        "seed": job["seed"],
        "stage": stage,
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config_rel": job["config_rel"],
        "batch_size": cfg["train"]["batch_size"],
        "device": cfg["train"]["device"],
        "freeze_bn": cfg["train"]["freeze_bn"],
        "epochs": cfg["train"]["epochs"],
        "learning_rate": cfg["train"]["learning_rate"],
        "weight_decay": cfg["train"]["weight_decay"],
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for base in (
        ROOT / "logs" / "paper_v2_36_meta",
        ROOT / "experiments" / run_id,
        ROOT / "results" / run_id,
    ):
        base.mkdir(parents=True, exist_ok=True)
        (base / "run_meta.json").write_text(json.dumps(meta, indent=2))


def run_cmd(cmd: list[str], log_file: Path) -> int:
    line = " ".join(cmd)
    print(f">>> {line}", flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n>>> {line}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            },
        )
        log.write(f"EXIT:{proc.returncode}\n")
        log.flush()
    print(f"EXIT:{proc.returncode}", flush=True)
    return int(proc.returncode)


def train_until_done(job: dict) -> None:
    run_id = job["run_id"]
    cfg_rel = job["config_rel"]
    cfg = yaml.safe_load((ROOT / cfg_rel).read_text())
    write_run_meta(job, cfg, stage="train")
    summary = ROOT / "results" / run_id / "train_summary.json"
    while not summary.is_file():
        rc = run_cmd(
            [PY, "-u", "-m", "src.train", "--config", cfg_rel, "--run-id", run_id],
            LOG,
        )
        if rc == 0 and summary.is_file():
            return
        if rc == 75:
            print(f"CHUNK_CONTINUE {run_id}", flush=True)
            time.sleep(5)
            continue
        if rc == 0 and not summary.is_file():
            raise SystemExit(f"train rc=0 but missing summary for {run_id}")
        raise SystemExit(f"train failed {run_id} rc={rc}")


def eval_until_done(job: dict) -> None:
    run_id = job["run_id"]
    if eval_done(run_id):
        print(f"SKIP eval {run_id}", flush=True)
        return
    ckpt = ROOT / "checkpoints" / run_id / "best.pt"
    cfg = ROOT / "experiments" / run_id / "config.yaml"
    if not cfg.is_file():
        cfg = ROOT / job["config_rel"]
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    out = ROOT / "results" / run_id
    for attempt in range(1, 9):
        if eval_done(run_id):
            return
        print(f"EVAL {run_id} attempt={attempt}", flush=True)
        rc = run_cmd(
            [
                PY,
                "-u",
                "-m",
                "src.evaluate",
                "--config",
                str(cfg),
                "--checkpoint",
                str(ckpt),
                "--n-boot",
                str(N_BOOT),
                "--results-dir",
                str(out),
            ],
            ROOT / "logs" / f"paper_v2_eval_{run_id}.log",
        )
        if rc == 0 and eval_done(run_id):
            return
        time.sleep(20)
    raise SystemExit(f"eval failed {run_id}")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text(
            f"PAPER_V2_36_START {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    else:
        with LOG.open("a") as f:
            f.write(f"\nPAPER_V2_36_RESUME {datetime.now(timezone.utc).isoformat()}\n")

    jobs = load_jobs()
    print(f"JOBS={len(jobs)} DONE_MARKER={DONE}", flush=True)

    for i, job in enumerate(jobs, 1):
        run_id = job["run_id"]
        print(f"\n=== [{i}/36] {run_id} family={job['family']} cell={job['cell']} seed={job['seed']} ===", flush=True)
        if job_done(run_id):
            print(f"SKIP complete {run_id}", flush=True)
            continue
        if not train_done(run_id):
            train_until_done(job)
        else:
            print(f"SKIP train {run_id}", flush=True)
        eval_until_done(job)
        print(f"JOB_OK {run_id}", flush=True)
        with LOG.open("a") as f:
            f.write(f"JOB_OK {run_id}\n")

    print(DONE, flush=True)
    with LOG.open("a") as f:
        f.write(f"{DONE}\n")


if __name__ == "__main__":
    main()
