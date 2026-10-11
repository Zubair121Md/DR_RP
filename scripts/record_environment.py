#!/usr/bin/env python3
"""Record the software/hardware environment and run-time facts used in the Methods.

Reads: system information, installed package metadata, logs/*.log, results/*/train_summary.json,
results/*/run_meta.json. Writes paper/tables/environment.csv.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables" / "environment.csv"
PKGS = ["torch", "torchvision", "timm", "opencv-python", "numpy", "pillow", "scikit-learn", "pandas", "matplotlib"]
READ: list[str] = []


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unavailable"


def pkg_info(name: str) -> tuple[str, str]:
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return "not installed", ""
    files = dist.files or []
    meta = next((f for f in files if str(f).endswith("METADATA")), None)
    ts = ""
    if meta is not None:
        p = Path(dist.locate_file(meta))
        if p.is_file():
            ts = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d")
    return dist.version, ts


def main() -> None:
    rows = []

    def add(k: str, v) -> None:
        rows.append({"item": k, "value": v})

    add("CPU", sh(["sysctl", "-n", "machdep.cpu.brand_string"]))
    mem = sh(["sysctl", "-n", "hw.memsize"])
    add("RAM (GB)", f"{int(mem) / 2**30:.0f}" if mem.isdigit() else mem)
    add("CPU cores", sh(["sysctl", "-n", "hw.ncpu"]))
    add("OS", f"macOS {platform.mac_ver()[0]}")
    add("Python", platform.python_version())
    add("Python interpreter", sys.executable)
    for p in PKGS:
        v, ts = pkg_info(p)
        add(f"{p} version", v)
        add(f"{p} install date (metadata mtime, UTC)", ts)

    interp = set()
    for lg in ("hybrid_2x2_phase_g.log", "hybrid_2x2_phase_s43.log", "hybrid_2x2_phase_s44.log", "paper_v3.log"):
        p = ROOT / "logs" / lg
        if not p.is_file():
            raise SystemExit(f"MISSING input: logs/{lg}")
        READ.append(f"logs/{lg}")
        txt = p.read_text(errors="ignore")
        for m in re.findall(r"^>>> (\S+python3)", txt, flags=re.M):
            interp.add((lg, m))
    for lg in sorted({a for a, _ in interp}):
        add(f"interpreter in {lg}", "; ".join(sorted({b for a, b in interp if a == lg})))

    # seeds 42-44: count resumes per run in the v1 logs
    resumes = {}
    for lg in ("hybrid_2x2_phase_g.log", "hybrid_2x2_phase_s43.log", "hybrid_2x2_phase_s44.log"):
        cur = None
        for line in (ROOT / "logs" / lg).read_text(errors="ignore").splitlines():
            m = re.search(r"--run-id (hybrid_h\d_s\d+_v1)", line)
            if m:
                cur = m.group(1)
            if cur and line.startswith("Resume at"):
                resumes[cur] = resumes.get(cur, 0) + 1
    if resumes:
        add("seeds 42-44: resumes per run (min-max, from v1 logs)", f"{min(resumes.values())}-{max(resumes.values())}")
        add("seeds 42-44: runs with resume records", len(resumes))

    # v3 runs: single-launch training time from train_summary (elapsed covers the whole run when not resumed)
    v3_el = []
    dates = []
    for p in sorted((ROOT / "results").glob("*_v3/train_summary.json")):
        READ.append(str(p.relative_to(ROOT)))
        s = json.loads(p.read_text())
        v3_el.append(float(s["elapsed_sec"]) / 3600)
        mp = p.parent / "run_meta.json"
        if mp.is_file():
            READ.append(str(mp.relative_to(ROOT)))
            dates.append(json.loads(mp.read_text())["date_utc"][:10])
    hyb224 = []
    for c in (1, 2, 3, 4):
        for s in (45, 46, 47):
            p = ROOT / "results" / f"hybrid_h{c}_s{s}_v3" / "train_summary.json"
            hyb224.append(float(json.loads(p.read_text())["elapsed_sec"]) / 3600)
    add("v3 runs: number", len(v3_el))
    add("MobileViT-S 224 (seeds 45-47) training time per run, hours (median)", f"{median(hyb224):.1f}")
    add("MobileViT-S 224 (seeds 45-47) training time per run, hours (range)", f"{min(hyb224):.1f}-{max(hyb224):.1f}")
    if dates:
        add("v3 run dates (UTC)", f"{min(dates)} to {max(dates)}")
    v1_dates = []
    for c in (1, 2, 3, 4):
        for s in (42, 43, 44):
            p = ROOT / "results" / f"hybrid_h{c}_s{s}_v1" / "train_summary.json"
            READ.append(str(p.relative_to(ROOT)))
            v1_dates.append(dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d"))
    add("seeds 42-44 completion dates (train_summary mtime, UTC)", f"{min(v1_dates)} to {max(v1_dates)}")

    def day(t: float) -> str:
        return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")

    for prep in ("rgb", "green_clahe"):
        files = [p for p in (ROOT / "datasets" / "cache" / prep / "aptos").rglob("*") if p.is_file()]
        READ.append(f"datasets/cache/{prep}/aptos/** ({len(files)} files)")
        m = [p.stat().st_mtime for p in files]
        add(f"APTOS {prep} cache: files", len(files))
        add(f"APTOS {prep} cache: written (UTC)", f"{day(min(m))} to {day(max(m))}")
    for name in ("train.py", "dataset.py", "preprocessing.py", "models.py", "losses.py", "utils.py", "metrics.py"):
        p = ROOT / "src" / name
        READ.append(str(p.relative_to(ROOT)))
        add(f"src/{name} last modified (UTC)", day(p.stat().st_mtime))
    add("version control", "none (no git repository)" if not (ROOT / ".git").exists() else "git")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "value"])
        w.writeheader()
        w.writerows(rows)
    print("READ:", *sorted(set(READ)), sep="\n  ")
    for r in rows:
        print(f"{r['item']}: {r['value']}")
    print("WROTE", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
