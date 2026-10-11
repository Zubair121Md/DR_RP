#!/usr/bin/env python3
"""
Paper pipeline queue (sequential, MPS-safe):

1) Smoke-check Messidor-2 + DeepDRiD loaders
2) Re-eval one existing B0 ckpt (logits + grade-1 F1 + Messidor/DeepDRiD + temp ECE)
3) Train mild-aware CORAL B0: G+CLAHE then RGB (seed 42)
4) Train MobileViT-S hybrid grid seed 42 (6 cells)
5) Full external eval sweep over all checkpoints
6) Efficiency table (B0 + MobileViT)

Logs: logs/paper_pipeline.log
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LOG = PROJECT / "logs" / "paper_pipeline.log"
PY = sys.executable


def run(cmd: list[str], cwd: Path | None = None) -> int:
    cwd = cwd or PROJECT
    line = " ".join(cmd)
    print(f"\n>>> {line}", flush=True)
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n>>> {line}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"EXIT:{proc.returncode}\n")
        log.flush()
    print(f"EXIT:{proc.returncode}", flush=True)
    return int(proc.returncode)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("PAPER_PIPELINE_START\n", encoding="utf-8")

    # 1) Smoke loaders
    smoke = r"""
from src.dataset import Messidor2Dataset, DeepDRiDDataset
from pathlib import Path
m = Messidor2Dataset(root=Path('Messidor '), split='all')
print('MESSIDOR_OK', len(m))
d = DeepDRiDDataset(root=Path('datasets/DeepDRiD'), split='all')
print('DEEPDRID_OK', len(d))
"""
    rc = run([PY, "-c", smoke])
    if rc != 0:
        raise SystemExit("Loader smoke failed")

    # 2) Quick re-eval of best Green+CLAHE+CORAL seed42
    ckpt = PROJECT / "checkpoints" / "baseline_efficientnet_b0_green_clahe_coral_20260918_203508" / "best.pt"
    cfg = PROJECT / "configs" / "baseline_efficientnet_b0_green_clahe_coral_local.yaml"
    if ckpt.is_file():
        run(
            [
                PY,
                "-m",
                "src.evaluate",
                "--config",
                str(cfg),
                "--checkpoint",
                str(ckpt),
                "--n-boot",
                "200",
            ]
        )

    # 3) Mild-aware CORAL B0 seed 42
    for cfg in (
        "configs/baseline_efficientnet_b0_green_clahe_mild_coral_local.yaml",
        "configs/baseline_efficientnet_b0_rgb_mild_coral_local.yaml",
    ):
        rc = run([PY, "-m", "src.train", "--config", cfg])
        if rc != 0:
            print(f"TRAIN_FAIL {cfg}", flush=True)

    # 4) Hybrid grid seed 42
    hybrid_cfgs = sorted((PROJECT / "configs" / "hybrid_runs").glob("*.yaml"))
    for cfg in hybrid_cfgs:
        rc = run([PY, "-m", "src.train", "--config", str(cfg.relative_to(PROJECT))])
        if rc != 0:
            print(f"TRAIN_FAIL {cfg.name}", flush=True)

    # 5) Full external sweep (lighter bootstrap for throughput)
    run([PY, "scripts/eval_external_sweep.py", "--n-boot", "500"])

    # 6) Efficiency
    run([PY, "scripts/efficiency_b0.py"])
    run([PY, "scripts/efficiency_models.py"])

    with LOG.open("a", encoding="utf-8") as log:
        log.write("PAPER_PIPELINE_DONE\n")
    print("PAPER_PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
