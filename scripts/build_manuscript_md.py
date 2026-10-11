#!/usr/bin/env python3
"""
Fill paper/MANUSCRIPT_TEMPLATE.md with values computed from paper/tables/*.csv,
experiments/*/config.yaml and results/*/*.json, and write paper/MANUSCRIPT.md.

Placeholders have the form {{kind:arg:arg}}. Every resolved value is logged to
paper/tables/manuscript_values.csv with the files it came from. Qualitative claims in
the prose are asserted against the same sources; the build fails if any claim is false.
An unknown placeholder or a missing source file stops the build.
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper" / "tables"
RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"
TEMPLATE = ROOT / "paper" / "MANUSCRIPT_TEMPLATE.md"
OUT_MD = ROOT / "paper" / "MANUSCRIPT.md"
LEDGER = TABLES / "manuscript_values.csv"

CELLS = ["H1", "H2", "H3", "H4"]
DOMAINS = ["APTOS", "IDRiD", "Messidor-2", "DeepDRiD"]
EXTERNAL = DOMAINS[1:]
CELL_SETUP = {
    "H1": ("RGB", "Cross-entropy"),
    "H2": ("Green-channel CLAHE", "Cross-entropy"),
    "H3": ("RGB", "MW threshold loss"),
    "H4": ("Green-channel CLAHE", "MW threshold loss"),
}
MINUS = "\u2212"
NI = "n.i."
SEEDS_A, SEEDS_B = "42–44", "45–47"
SCHED_SHORT = {"truncated": "truncated", "full cosine": "full cosine"}

_read: set[str] = set()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def missing(p: Path) -> None:
    print("MISSING", rel(p))
    raise SystemExit(f"MISSING input: {rel(p)}")


def read_csv(name: str) -> list[dict[str, str]]:
    p = TABLES / name
    if not p.is_file():
        missing(p)
    if rel(p) not in _read:
        print("READ", rel(p))
        _read.add(rel(p))
    with p.open() as f:
        return list(csv.DictReader(f))


def read_json(p: Path) -> dict:
    if not p.is_file():
        missing(p)
    if rel(p) not in _read:
        print("READ", rel(p))
        _read.add(rel(p))
    return json.loads(p.read_text())


def read_cfg(run_dir: str) -> dict:
    p = EXPERIMENTS / Path(run_dir).name / "config.yaml"
    if not p.is_file():
        missing(p)
    if rel(p) not in _read:
        print("READ", rel(p))
        _read.add(rel(p))
    return yaml.safe_load(p.read_text())


def num(x: float, d: int = 3) -> str:
    s = f"{x:.{d}f}"
    return s.replace("-", MINUS) if s.startswith("-") else s


def signed(x: float, d: int = 3) -> str:
    return f"{x:+.{d}f}".replace("-", MINUS)


def mean_sd(vals: list[float], d: int = 3, sign: bool = False) -> str:
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{signed(m, d) if sign else num(m, d)} ± {num(s, d)}"


def pm(text: str) -> str:
    """Normalise an 'a±b' string from a CSV to the manuscript style 'a ± b'."""
    text = text.strip()
    if text in {"—", "", "MISSING"}:
        raise SystemExit(f"MISSING value in source table: {text!r}")
    a, b = text.split("±")
    return f"{num(float(a))} ± {num(float(b))}"


def pm_mean(text: str) -> float:
    return float(text.split("±")[0])


def thousands(n: int) -> str:
    return f"{n:,}"


def sci(x: float) -> str:
    if x == 0:
        return "0"
    e = int(f"{x:e}".split("e")[1])
    m = x / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.2f} × 10{sup}" if abs(m - round(m)) > 1e-9 else f"{int(round(m))} × 10{sup}"


def pval(p: float) -> str:
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ---------------------------------------------------------------------------
# Source accessors
# ---------------------------------------------------------------------------

def master() -> list[dict[str, str]]:
    rows = read_csv("master_results_per_run.csv")
    if len(rows) != 24:
        raise SystemExit(f"MISSING: master table should have 24 runs, found {len(rows)}")
    return rows


def mcol(cell: str, col: str, seeds: tuple[int, ...] | None = None) -> list[float]:
    rows = [r for r in master() if r["cell"] == cell and (seeds is None or int(r["seed"]) in seeds)]
    if len(rows) != (6 if seeds is None else len(seeds)):
        raise SystemExit(f"MISSING master rows for {cell} {seeds}")
    if any(r[col] == NI for r in rows):
        raise SystemExit(f"Requested non-interpretable column {col} for {cell}")
    return [float(r[col]) for r in rows]


def qwk_vals(cell: str, dom: str) -> list[float]:
    return mcol(cell, f"QWK_{dom}")


def aptos_metric(metric: str, cell: str) -> list[float]:
    col = {"mild_dr_precision": "APTOS_mild_precision", "mild_dr_recall": "APTOS_mild_recall",
           "mild_dr_f1": "APTOS_mild_F1", "macro_f1": "APTOS_macro_F1", "accuracy": "APTOS_accuracy"}[metric]
    return mcol(cell, col)


def ece6(cell: str, dom: str) -> list[dict[str, str]]:
    rows = [r for r in read_csv("ece_ce_cells_n6_per_seed.csv") if r["cell"] == cell and r["domain"] == dom]
    if len(rows) != 6:
        raise SystemExit(f"MISSING: expected 6 ECE rows for {cell} {dom}")
    return rows


PC_METRIC = {"qwk": "QWK", "g1": "grade-1 recall", "f1": "macro-F1", "acc": "accuracy", "ece": "ECE"}


def pc(metric: str, contrast: str, dom: str) -> dict[str, str]:
    c = contrast.replace("-", MINUS)
    rows = [r for r in read_csv("paired_contrasts_n6.csv")
            if r["metric"] == PC_METRIC[metric] and r["contrast"] == c and r["domain"] == dom]
    if len(rows) != 1:
        raise SystemExit(f"MISSING paired contrast {metric} {contrast} {dom}")
    return rows[0]


def g1(cell: str, dom: str) -> str:
    rows = [r for r in read_csv("external_grade1_recall_summary.csv") if r["cell"] == cell]
    if len(rows) != 1 or rows[0]["n_seeds"] != "6":
        raise SystemExit(f"MISSING grade-1 recall summary for {cell}")
    return rows[0][dom]


def g1run(cell: str, seed: int) -> dict[str, str]:
    rows = [r for r in read_csv("external_grade1_recall_per_run.csv") if r["cell"] == cell and int(r["seed"]) == seed]
    if len(rows) != 1:
        raise SystemExit(f"MISSING grade-1 recall row {cell} seed {seed}")
    return rows[0]


def pred_share(cell: str, seed: int, split: str, grade: int) -> tuple[int, int, str]:
    """Number of images of a test set assigned a given grade by one primary run, from its saved predictions."""
    run = f"hybrid_{cell.lower()}_s{seed}_{'v1' if seed <= 44 else 'v3'}"
    rel = f"results/{run}/{split}_predictions.json"
    y = read_json(RESULTS / run / f"{split}_predictions.json")["y_pred"]
    return sum(int(v) == grade for v in y), len(y), rel


def paired_group(metric: str, contrast: str, dom: str, group: str) -> list[float]:
    """Seed-level differences a−b within one schedule group, as in Table 8."""
    a, b = contrast.split("-")
    seeds = (42, 43, 44) if group == "a" else (45, 46, 47)
    if metric == "qwk":
        by = {(r["cell"], int(r["seed"])): r for r in master()}
        return [float(by[(a, s)][f"QWK_{dom}"]) - float(by[(b, s)][f"QWK_{dom}"]) for s in seeds]
    return [float(g1run(a, s)[f"{dom}_grade1_recall"]) - float(g1run(b, s)[f"{dom}_grade1_recall"]) for s in seeds]


def g1n(dom: str) -> int:
    vals = {r[f"{dom}_n_grade1"] for r in read_csv("external_grade1_recall_summary.csv")}
    if len(vals) != 1:
        raise SystemExit(f"grade-1 counts differ across cells for {dom}: {vals}")
    return int(vals.pop())


def smc(cell: str, dom: str, metric: str) -> str:
    rows = [r for r in read_csv("schedule_matched_comparators.csv")
            if r["cell"] == cell and r["domain"] == dom and r["metric"] == metric]
    if len(rows) != 1:
        raise SystemExit(f"MISSING schedule-matched comparator {cell} {dom} {metric}")
    return rows[0]["value"]


def sb(cell: str, dom: str, metric: str, batch: str) -> str:
    rows = [r for r in read_csv("seed_batch_summary.csv")
            if r["cell"] == cell and r["domain"] == dom and r["metric"] == metric]
    if len(rows) != 1:
        raise SystemExit(f"MISSING seed-batch row {cell} {dom} {metric}")
    return rows[0]["seeds_42_44" if batch == "a" else "seeds_45_47"]


def kv(name: str) -> dict[str, str]:
    return {r["item"]: r["value"] for r in read_csv(name)}


ENV_KEYS = {
    "cpu": "CPU", "ram": "RAM (GB)", "cores": "CPU cores", "os": "OS", "python": "Python",
    "torch": "torch version", "torchvision": "torchvision version", "timm": "timm version",
    "opencv": "opencv-python version", "numpy": "numpy version", "pillow": "pillow version",
    "sklearn": "scikit-learn version", "resumes": "seeds 42-44: resumes per run (min-max, from v1 logs)",
    "hours": "MobileViT-S 224 (seeds 45-47) training time per run, hours (median)",
    "hoursrange": "MobileViT-S 224 (seeds 45-47) training time per run, hours (range)",
    "v1dates": "seeds 42-44 completion dates (train_summary mtime, UTC)", "v3dates": "v3 run dates (UTC)",
    "cache": "APTOS green_clahe cache: written (UTC)", "cachergb": "APTOS rgb cache: written (UTC)",
    "trainmod": "src/train.py last modified (UTC)", "prepmod": "src/preprocessing.py last modified (UTC)",
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def fmt_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]} {y}"

AUD_KEYS = {
    "total": "total images", "pairs": "exact duplicate pairs, all", "pixonly": "exact duplicate pairs found only by decoded pixels",
    "trtr": "exact duplicate pairs train-train", "trva": "exact duplicate pairs train-val",
    "trte": "exact duplicate pairs train-test", "vate": "exact duplicate pairs val-test",
    "vava": "exact duplicate pairs val-val", "tete": "exact duplicate pairs test-test",
    "difflabel": "cross-split exact pairs with different labels",
    "testdup": "test images with an exact duplicate in train or validation",
    "valdup": "validation images with an exact duplicate in train",
    "idov": "id overlap train-test",
}


def dedup_rows() -> list[dict[str, str]]:
    rows = read_csv("aptos_dedup_sensitivity.csv")
    if len(rows) != 24:
        raise SystemExit("MISSING dedup sensitivity rows")
    return rows


def coral_rows() -> list[dict[str, str]]:
    rows = read_csv("coral_verification_runs.csv")
    if len(rows) != 21:
        raise SystemExit(f"MISSING: expected 21 threshold-head runs, found {len(rows)}")
    return rows


def nonmono_fracs(r: dict[str, str]) -> list[float]:
    return [float(r[f"{d}_nonmono_frac"]) for d in DOMAINS]


def audit_rows() -> list[dict[str, str]]:
    rows = read_csv("training_schedule_audit.csv")
    if len(rows) != 60:
        raise SystemExit(f"MISSING: training audit should have 60 runs, found {len(rows)}")
    return rows


def norm_layers(backbone: str) -> dict[str, str]:
    rows = [r for r in read_csv("normalisation_layers.csv") if r["backbone"] == backbone]
    if len(rows) != 1:
        raise SystemExit(f"MISSING normalisation-layer row for {backbone}")
    return rows[0]


def decomp_row(cell: str) -> dict[str, str]:
    rows = [r for r in read_csv("table_v3_decomp_clahe.csv") if r["Cell"] == cell]
    if len(rows) != 1:
        raise SystemExit(f"MISSING decomp row {cell}")
    return rows[0]


def res448_row(cell: str) -> dict[str, str]:
    rows = [r for r in read_csv("table_v3_res448_vs_224.csv") if r["Cell"] == cell]
    if len(rows) != 1:
        raise SystemExit(f"MISSING res448 row {cell}")
    return rows[0]


def mess_pred0_res(cell: str, size: str) -> str:
    rows = [r for r in read_csv("table_v3_messidor_pred0_res_summary.csv")
            if r["cell"] == cell and r["image_size"] == str(size)]
    if len(rows) != 1:
        raise SystemExit(f"MISSING mess pred0 {cell} {size}")
    return rows[0]["pred_frac_grade0"]


def mess_pred0_224(cell: str, seeds: str) -> str:
    rows = [r for r in read_csv("messidor_pred0_n6_summary.csv") if r["cell"] == cell and r["seeds"] == seeds]
    if len(rows) != 1:
        raise SystemExit(f"MISSING Messidor grade-0 share {cell} {seeds}")
    return rows[0]["pred_frac_grade0"]


def b0_row(cell: str) -> dict[str, str]:
    rows = [r for r in read_csv("table_v3_matched_b0_vs_hybrid.csv") if r["Cell"] == cell]
    if len(rows) != 1:
        raise SystemExit(f"MISSING matched B0 row {cell}")
    return rows[0]


def arch_row(model: str) -> dict[str, str]:
    rows = [r for r in read_csv("table_phase_o_arch.csv") if r["Model"] == model]
    if len(rows) != 1:
        raise SystemExit(f"MISSING architecture row {model}")
    return rows[0]


def manifest(dataset: str, split: str) -> dict[str, str]:
    rows = [r for r in read_csv("dataset_manifest.csv") if r["dataset"] == dataset and r["split"] == split]
    if len(rows) != 1:
        raise SystemExit(f"MISSING manifest row {dataset} {split}")
    return rows[0]


def messidor_runs(cell: str) -> list[dict[str, str]]:
    rows = [r for r in read_csv("messidor_pred0_n6_per_run.csv") if r["cell"] == cell]
    if len(rows) != 6:
        raise SystemExit(f"MISSING Messidor runs for {cell}")
    return rows


LODO_KEY = {"IDRiD": "idrid", "Messidor-2": "messidor2", "DeepDRiD": "deepdrid"}
LODO_PARTS = {
    "IDRiD": [("APTOS train", "APTOS", "train"), ("Messidor-2", "Messidor-2", "external_gradable"),
              ("DeepDRiD", "DeepDRiD", "external_eval")],
    "Messidor-2": [("APTOS train", "APTOS", "train"), ("IDRiD", "IDRiD", "external_all"),
                   ("DeepDRiD", "DeepDRiD", "external_eval")],
    "DeepDRiD": [("APTOS train", "APTOS", "train"), ("IDRiD", "IDRiD", "external_all"),
                 ("Messidor-2", "Messidor-2", "external_gradable")],
}


def lodo_train_sizes() -> dict[str, int]:
    out = {}
    for h, key in LODO_KEY.items():
        sizes = set()
        for s in (42, 43, 44):
            run = f"phase_p_lodo_holdout_{key}_s{s}_v1"
            sizes.add(read_json(RESULTS / run / "train_summary.json")["train_size"])
            cfg = read_cfg(run)
            if cfg["preprocessing"]["name"] != "green_clahe" or cfg["loss"]["name"] != "cross_entropy":
                raise SystemExit(f"LODO run {run} is not CLAHE × cross-entropy")
        if len(sizes) != 1:
            raise SystemExit(f"LODO train size differs across seeds for {h}: {sizes}")
        out[key] = sizes.pop()
    return out


def lodo_composition(h: str) -> str:
    parts = [(lab, int(manifest(d, s)["n_images"])) for lab, d, s in LODO_PARTS[h]]
    total = sum(n for _, n in parts)
    if total != lodo_train_sizes()[LODO_KEY[h]]:
        raise SystemExit(f"LODO composition for {h} does not add up to the logged training size")
    return " + ".join(f"{lab} ({thousands(n)})" for lab, n in parts)


# ---------------------------------------------------------------------------
# Placeholder resolution
# ---------------------------------------------------------------------------

def resolve(key: str) -> tuple[str, str]:
    """Return (value, source description) for a placeholder key."""
    parts = key.split(":")
    kind, args = parts[0], parts[1:]

    if kind == "qwk":
        return mean_sd(qwk_vals(*args)), "master_results_per_run.csv"
    if kind in ("qwkmin", "qwkmax"):
        means = [st.mean(qwk_vals(c, args[0])) for c in CELLS]
        return num(min(means) if kind == "qwkmin" else max(means)), "master_results_per_run.csv"
    if kind == "apt":
        return mean_sd(aptos_metric(args[0], args[1])), "master_results_per_run.csv"
    if kind == "mildk":
        seeds = tuple(range(42, 45)) if args[1] == "a" else tuple(range(45, 48))
        n = g1n("APTOS")
        ks = sorted({round(v * n) for v in mcol(args[0], "APTOS_mild_recall", seeds)})
        return (" or ".join(str(k) for k in ks)), "master_results_per_run.csv"
    if kind == "g1":
        return pm(g1(args[0], args[1])), "external_grade1_recall_summary.csv"
    if kind == "g1n":
        return str(g1n(args[0])), "external_grade1_recall_summary.csv"
    if kind == "pc":
        r = pc(args[0], args[1], args[2])
        field = args[3]
        if field == "d":
            return f"{signed(float(r['mean_delta']))} ± {num(float(r['sd_delta']))}", "paired_contrasts_n6.csv"
        if field == "npos":
            return r["n_pos"], "paired_contrasts_n6.csv"
        if field == "nneg":
            return r["n_neg"], "paired_contrasts_n6.csv"
        if field == "p":
            return pval(float(r["p_signflip_exact"])), "paired_contrasts_n6.csv"
        if field == "holm":
            return pval(float(r["p_holm_within_metric"])), "paired_contrasts_n6.csv"
    if kind == "pcfam":
        rows = [r for r in read_csv("paired_contrasts_n6.csv") if r["metric"] == PC_METRIC[args[0]]]
        if args[1] == "n":
            return str(len(rows)), "paired_contrasts_n6.csv"
        if args[1] == "minholm":
            return pval(min(float(r["p_holm_within_metric"]) for r in rows)), "paired_contrasts_n6.csv"
        if args[1] == "minp":
            return pval(min(float(r["p_signflip_exact"]) for r in rows)), "paired_contrasts_n6.csv"
    if kind == "smc":
        return pm(smc(args[0], args[1], args[2])), "schedule_matched_comparators.csv"
    if kind == "sb":
        return pm(sb(args[0], args[1], args[2], args[3])), "seed_batch_summary.csv"
    if kind == "env":
        v = kv("environment.csv")[ENV_KEYS[args[0]]]
        if args[0] in ("resumes", "hoursrange"):
            v = v.replace("-", "–")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}( to \d{4}-\d{2}-\d{2})?", v):
            v = " to ".join(dict.fromkeys(fmt_date(x) for x in v.split(" to ")))
        return v, "environment.csv"
    if kind == "aud":
        v = kv("aptos_split_audit.csv")[AUD_KEYS[args[0]]]
        return thousands(int(v)), "aptos_split_audit.csv"
    if kind == "dedup":
        rows = dedup_rows()
        if args[0] == "nkept":
            vals = {r["n_kept"] for r in rows}
            if len(vals) != 1:
                raise SystemExit("dedup n_kept differs")
            return vals.pop(), "aptos_dedup_sensitivity.csv"
        if args[0] == "nmild":
            vals = {r["n_mild_kept"] for r in rows}
            if len(vals) != 1:
                raise SystemExit("dedup n_mild_kept differs")
            return vals.pop(), "aptos_dedup_sensitivity.csv"
        if args[0] == "maxchange":
            ch = [abs(st.mean(float(r["qwk_dedup"]) for r in rows if r["cell"] == c)
                      - st.mean(float(r["qwk_all"]) for r in rows if r["cell"] == c)) for c in CELLS]
            return num(max(ch)), "aptos_dedup_sensitivity.csv"
        if args[0] == "grpmax":
            ch = [abs(st.mean(float(r["qwk_dedup"]) for r in rows if r["cell"] == c and int(r["seed"]) in g)
                      - st.mean(float(r["qwk_all"]) for r in rows if r["cell"] == c and int(r["seed"]) in g))
                  for c in CELLS for g in ((42, 43, 44), (45, 46, 47))]
            return num(max(ch)), "aptos_dedup_sensitivity.csv"
        if args[0] == "runmax":
            return num(max(abs(float(r["qwk_dedup"]) - float(r["qwk_all"])) for r in rows)), "aptos_dedup_sensitivity.csv"
        if args[0] == "sb":
            grp = {"a": (42, 43, 44), "b": (45, 46, 47)}[args[2]]
            cell, col = args[1], {"all": "qwk_all", "dedup": "qwk_dedup"}[args[3]]
            return mean_sd([float(r[col]) for r in rows if r["cell"] == cell and int(r["seed"]) in grp]), "aptos_dedup_sensitivity.csv"
        cell, col = args[0], {"all": "qwk_all", "dedup": "qwk_dedup"}[args[1]]
        return mean_sd([float(r[col]) for r in rows if r["cell"] == cell]), "aptos_dedup_sensitivity.csv"
    if kind == "coral":
        rows = coral_rows()
        if args[0] == "nruns":
            return str(len(rows)), "coral_verification_runs.csv"
        if args[0] == "nordered":
            return str(sum(r["biases_nonincreasing"] == "True" for r in rows)), "coral_verification_runs.csv"
        if args[0] == "minnonmono":
            v = min(min(nonmono_fracs(r)) for r in rows)
            return f"{100 * v:.1f}%", "coral_verification_runs.csv"
        if args[0] == "init":
            sys.path.insert(0, str(ROOT))
            from src.models import CoralLayer
            print("READ src/models.py (CoralLayer initialisation)")
            b = CoralLayer(8, 5).bias.detach().tolist()
            return "(" + ", ".join(num(x, 2) if abs(x - round(x)) > 1e-6 else num(x, 0) for x in b) + ")", "src/models.py"
    if kind == "ece6":
        col = {"raw": "ece", "T": "ece_T"}[args[2]]
        return mean_sd([float(r[col]) for r in ece6(args[0], args[1])]), "ece_ce_cells_n6_per_seed.csv"
    if kind == "T6":
        return mean_sd([float(r["T"]) for r in ece6(args[0], "APTOS")]), "ece_ce_cells_n6_per_seed.csv"
    if kind == "sched":
        rows = audit_rows()
        col = {"lr10": "lr_epoch10", "lrfinal": "lr_final"}[args[0]]
        want = "truncated" if args[1] == "trunc" else "full cosine"
        vals = {float(r[col]) for r in rows if r["lr_schedule"].startswith(want)}
        if len(vals) != 1:
            raise SystemExit(f"{col} differs within {want} runs: {vals}")
        return sci(vals.pop()), "training_schedule_audit.csv"
    if kind == "nruns":
        rows = audit_rows()
        if args[0] == "trunc":
            return str(sum(r["lr_schedule"].startswith("truncated") for r in rows)), "training_schedule_audit.csv"
        if args[0] == "all":
            return str(len(rows)), "training_schedule_audit.csv"
    if kind == "steps":
        rows = audit_rows()
        vals = {r["chunk_steps"] for r in rows if r["lr_schedule"].startswith(
            "truncated" if args[0] == "trunc" else "full cosine") and r["family"].startswith("MobileViT-S H")}
        if len(vals) != 1:
            raise SystemExit(f"chunk sizes differ: {vals}")
        return vals.pop(), "training_schedule_audit.csv"
    if kind == "stepsepoch":
        n = int(manifest("APTOS", "train")["n_images"])
        bs = {int(r["batch_size"]) for r in audit_rows()}
        if len(bs) != 1:
            raise SystemExit(f"batch sizes differ: {bs}")
        b = bs.pop()
        return str(-(-n // b)), "dataset_manifest.csv; training_schedule_audit.csv"
    if kind == "decomp":
        col = {"qwk": f"{args[1]} QWK", "mild": f"{args[1]} mild recall", "ece": f"{args[1]} ECE"}[args[2]]
        return pm(decomp_row(args[0])[col]), "table_v3_decomp_clahe.csv"
    if kind == "res448":
        return pm(res448_row(args[0])[f"{args[1]} {args[2]}"]), "table_v3_res448_vs_224.csv"
    if kind == "mess448":
        return pm(mess_pred0_res(args[0], args[1])), "table_v3_messidor_pred0_res_summary.csv"
    if kind == "mp0":
        return pm(mess_pred0_224(args[0], args[1].replace("-", "–"))), "messidor_pred0_n6_summary.csv"
    if kind == "b0m":
        return pm(b0_row(args[0])[f"B0 {args[1]}"]), "table_v3_matched_b0_vs_hybrid.csv"
    if kind == "b0delta":
        rows = [r for r in read_csv("table_v3_matched_b0_paired_delta.csv")
                if r["Cell"] == args[0] and r["Domain"] == args[1]]
        if len(rows) != 1:
            raise SystemExit(f"MISSING B0 paired delta {args}")
        return rows[0]["Δ QWK (B0−hybrid)"].replace("±", " ± ").replace("-", MINUS), "table_v3_matched_b0_paired_delta.csv"
    if kind == "n":
        return thousands(int(manifest(args[0], args[1])["n_images"])), "dataset_manifest.csv"
    if kind == "eff":
        rows = [r for r in read_csv("table_efficiency.csv") if r["backbone"] == args[0] and r["head"] == args[1]]
        if len(rows) != 1 or rows[0][args[2]] == "MISSING":
            raise SystemExit(f"MISSING efficiency value {key}")
        v = rows[0][args[2]]
        if args[2] == "batch1_ms_mean":
            v = f"{float(v):.1f}"
        if args[2] == "device":
            v = {"mps": "the Apple M2 integrated GPU (PyTorch MPS backend)", "cpu": "the CPU"}[v]
        return v, "table_efficiency.csv"
    if kind == "lodo" and args[0] == "ntrain":
        sizes = lodo_train_sizes()
        if args[1] in ("min", "max"):
            v = min(sizes.values()) if args[1] == "min" else max(sizes.values())
        else:
            v = sizes[args[1]]
        return thousands(v), "results/phase_p_lodo_holdout_*_v1/train_summary.json"
    if kind == "bestep":
        eps = [int(r["best_epoch"]) for r in master()]
        return str(min(eps) if args[0] == "min" else max(eps)), "master_results_per_run.csv"
    if kind == "mess":
        sub = args[0]
        runs = read_csv("messidor_pred0_n6_per_run.csv")
        if sub == "true":
            vals = {r[f"true_n{args[1]}"] for r in runs}
            if len(vals) != 1:
                raise SystemExit(f"CLAIM FAILED: Messidor-2 labels differ across runs: {vals}")
            return thousands(int(vals.pop())), "messidor_pred0_n6_per_run.csv"
        if sub == "n":
            vals = {r["n"] for r in runs}
            if len(vals) != 1:
                raise SystemExit(f"CLAIM FAILED: Messidor-2 n differs across runs: {vals}")
            return thousands(int(vals.pop())), "messidor_pred0_n6_per_run.csv"
        if sub == "predfrac0":
            return mean_sd([float(r["pred_frac_grade0"]) for r in messidor_runs(args[1])]), "messidor_pred0_n6_per_run.csv"
        if sub == "pred0":
            rows = [r for r in messidor_runs(args[1]) if r["seed"] == args[2]]
            return thousands(int(rows[0]["pred_n0"])), "messidor_pred0_n6_per_run.csv"
    if kind == "ref":
        rows = [r for r in read_csv("table_B_referable_dr.csv") if r["cell"] == args[1] and r["domain"] == args[2]]
        if len(rows) != 3:
            raise SystemExit(f"MISSING referable rows {args}")
        return mean_sd([float(r[args[0]]) for r in rows]), "table_B_referable_dr.csv"
    if kind == "arch":
        return pm(arch_row(args[0])[args[1]]), "table_phase_o_arch.csv"
    if kind == "boot":
        vals = {r["n_boot"] for r in read_csv("table_C_qwk_bootstrap_ci.csv")}
        if len(vals) != 1:
            raise SystemExit(f"bootstrap n differs across runs: {vals}")
        return thousands(int(vals.pop())), "table_C_qwk_bootstrap_ci.csv"
    if kind == "g1grp":
        seeds = (42, 43, 44) if args[2] == "a" else (45, 46, 47)
        return mean_sd([float(g1run(args[0], x)[f"{args[1]}_grade1_recall"]) for x in seeds]), "external_grade1_recall_per_run.csv"
    if kind == "g1run":
        r = g1run(args[0], int(args[1]))
        return f"{r[f'{args[2]}_grade1_correct']} of {r[f'{args[2]}_grade1_n']}", "external_grade1_recall_per_run.csv"
    if kind == "predshare":
        k, n, src = pred_share(args[0], int(args[1]), args[2], int(args[3]))
        return f"{thousands(k)} of {thousands(n)} ({100 * k / n:.1f}%)", src
    if kind == "pgrp":
        d = paired_group(args[0], args[1], args[2], args[3])
        if args[4] == "npos":
            return str(sum(x > 0 for x in d)), "master_results_per_run.csv; external_grade1_recall_per_run.csv"
        return f"{signed(st.mean(d))} ± {num(st.stdev(d))}", "master_results_per_run.csv; external_grade1_recall_per_run.csv"
    if kind == "table":
        return TABLE_BUILDERS[args[0]](), f"table:{args[0]}"
    raise SystemExit(f"UNKNOWN placeholder {{{{{key}}}}}")


# ---------------------------------------------------------------------------
# Tables (markdown pipe tables with a bold caption line above)
# ---------------------------------------------------------------------------

def md_table(caption: str, header: list[str], rows: list[list[str]], note: str | None = None) -> str:
    out = [caption, "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    if note:
        out += ["", note]
    return "\n".join(out)


def t_datasets() -> str:
    def counts_from_preds(split: str) -> list[int]:
        p = read_json(RESULTS / "hybrid_h2_s42_v1" / f"{split}_predictions.json")
        yt = [int(v) for v in p["y_true"]]
        return [yt.count(g) for g in range(5)]

    def fmt(c: list[int]) -> str:
        return " / ".join(thousands(x) for x in c)

    rows = []
    for split, role in (("train", "Training"), ("val", "Checkpoint selection and temperature fitting"),
                        ("test", "Internal held-out test")):
        m = manifest("APTOS", split)
        rows.append([f"APTOS {split if split != 'val' else 'validation'}", role, thousands(int(m["n_images"])),
                     fmt([int(m[f"n{g}"]) for g in range(5)])])
    idr = manifest("IDRiD", "external_all")
    ic = counts_from_preds("idrid_external")
    if sum(ic) != int(idr["n_images"]):
        raise SystemExit("IDRiD prediction count does not match manifest")
    rows.append(["IDRiD (training and test parts pooled)", "External test", thousands(int(idr["n_images"])), fmt(ic)])
    mes = manifest("Messidor-2", "external_gradable")
    rows.append(["Messidor-2 (gradable)", "External test", thousands(int(mes["n_images"])),
                 fmt([int(mes[f"n{g}"]) for g in range(5)])])
    dd = manifest("DeepDRiD", "external_eval")
    dc = counts_from_preds("deepdrid_external")
    if sum(dc) != int(dd["n_images"]):
        raise SystemExit("DeepDRiD prediction count does not match manifest")
    rows.append(["DeepDRiD (regular fundus)", "External test", thousands(int(dd["n_images"])), fmt(dc)])
    return md_table("**Table 1.** Datasets and splits. Grade counts are listed from grade 0 to grade 4.",
                    ["Dataset", "Role", "Images", "Grades 0 / 1 / 2 / 3 / 4"], rows)


def _family_runs(prefix: str, size: int = 224) -> list[dict[str, str]]:
    return [r for r in audit_rows() if r["family"].startswith(prefix) and int(r["image_size"]) == size]


def _one(vals: set, what: str) -> str:
    if len(vals) != 1:
        raise SystemExit(f"{what} is not constant within a model family: {vals}")
    return str(vals.pop())


def _by_seed_group(runs: list[dict[str, str]], key) -> str:
    groups: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        groups[key(r)].append(int(r["seed"]))
    if len(groups) == 1:
        return next(iter(groups))
    parts = []
    for v, seeds in sorted(groups.items(), key=lambda kv: min(kv[1])):
        seeds = sorted(set(seeds))
        parts.append(f"{v} (seeds {seeds[0]}–{seeds[-1]})")
    return "; ".join(parts)


def t_settings() -> str:
    fams = {
        "MobileViT-S (H1–H4)": (_family_runs("MobileViT-S H"), "MobileViT-S"),
        "EfficientNet-B0 (matched)": (_family_runs("EfficientNet-B0 matched"), "EfficientNet-B0"),
        "Swin-Tiny": (_family_runs("Swin-Tiny"), "Swin-Tiny"),
    }
    fams["MobileViT-S (H1–H4)"] = (
        [r for r in fams["MobileViT-S (H1–H4)"][0] if re.fullmatch(r"MobileViT-S H[1-4]", r["family"])], "MobileViT-S")
    cols = list(fams)
    rows: dict[str, list[str]] = defaultdict(list)
    order = ["Pretrained weights", "Input size", "Optimiser", "Learning rate", "Weight decay", "Epochs",
             "Configured schedule", "Learning rate at last epoch", "Batch size", "Normalisation layers",
             "Training device", "Evaluation device", "Optimiser steps per launch", "Checkpoint selection", "Seeds"]
    for col in cols:
        runs, backbone = fams[col]
        cfgs = [read_cfg(r["run_dir"]) for r in runs]
        tr = [c["train"] for c in cfgs]
        nl = norm_layers(backbone)
        rows["Pretrained weights"].append(nl["weights"])
        rows["Input size"].append(_one({f"{c['preprocessing']['image_size']}×{c['preprocessing']['image_size']}" for c in cfgs}, "size"))
        rows["Optimiser"].append(_one({t["optimizer"].replace("adamw", "AdamW") for t in tr}, "optimiser"))
        rows["Learning rate"].append(sci(float(_one({t["learning_rate"] for t in tr}, "lr"))))
        rows["Weight decay"].append(sci(float(_one({t["weight_decay"] for t in tr}, "wd"))))
        rows["Epochs"].append(_one({t["epochs"] for t in tr}, "epochs"))
        rows["Configured schedule"].append(_one({t["scheduler"] for t in tr}, "scheduler") + ", stepped per epoch")
        rows["Learning rate at last epoch"].append(_by_seed_group(runs, lambda r: sci(float(r["lr_final"]))))
        rows["Batch size"].append(_one({t["batch_size"] for t in tr}, "batch size"))
        bn, ln = int(nl["batchnorm_layers"]), int(nl["layernorm_layers"])
        frozen = _one({str(t.get("freeze_bn", False)) for t in tr}, "freeze_bn") == "True"
        parts = []
        if bn:
            parts.append(f"{bn} BatchNorm ({'frozen at ImageNet statistics' if frozen else 'trained'})")
        if ln:
            parts.append(f"{ln} LayerNorm (trained)")
        rows["Normalisation layers"].append(" + ".join(parts) if bn else parts[0] + "; no BatchNorm layers")
        rows["Training device"].append(_one({r["train_device"].upper() for r in runs}, "train device") + f" ({kv('environment.csv')['CPU']})")
        rows["Evaluation device"].append(_one({r["eval_device"] for r in runs}, "eval device").replace("mps", "Apple GPU (MPS)"))
        rows["Optimiser steps per launch"].append(_by_seed_group(runs, lambda r: r["chunk_steps"]))
        rows["Checkpoint selection"].append(
            _one({t["selection_metric"] for t in tr}, "selection").replace("quadratic_weighted_kappa", "APTOS validation QWK"))
        seeds = sorted({int(r["seed"]) for r in runs})
        rows["Seeds"].append(f"{seeds[0]}–{seeds[-1]}")
    body = [[k] + rows[k] for k in order]
    return md_table(
        "**Table 2.** Training settings, read from the configuration file of every run and from the per-epoch learning rates logged during training. The loss-decomposition (D1, D2) and 448×448 MobileViT-S runs and the leave-one-dataset-out runs used the MobileViT-S settings with a full cosine schedule and seeds 42–44 (Supplementary Table S2).",
        ["Setting"] + cols, body)


def t_qwk() -> str:
    rows = []
    for c in CELLS:
        inp, loss = CELL_SETUP[c]
        for label, seeds, sched in (("42–44", (42, 43, 44), "truncated"),
                                   ("45–47", (45, 46, 47), "full cosine")):
            rows.append([c, inp, loss, f"{label}, {sched}"]
                        + [mean_sd(mcol(c, f"QWK_{d}", seeds)) for d in DOMAINS])
    return md_table(
        "**Table 3.** Schedule-stratified QWK of the four MobileViT-S cells on the APTOS internal test split and three external datasets (mean ± sample standard deviation over three seeds per row). Seeds 42–44 received a truncated learning-rate schedule; seeds 45–47 received the intended full cosine schedule. The two groups are not pooled as exchangeable replicates. Per-seed values are in Supplementary Table S4.",
        ["Cell", "Input", "Loss", "Schedule group"] + DOMAINS, rows)


def t_mild() -> str:
    rows = []
    for c in CELLS:
        for label, seeds, sched in (("42–44", (42, 43, 44), "truncated"),
                                   ("45–47", (45, 46, 47), "full cosine")):
            rows.append([c, f"{label}, {sched}"]
                        + [mean_sd(mcol(c, col, seeds)) for col in
                           ("APTOS_mild_precision", "APTOS_mild_recall", "APTOS_mild_F1",
                            "APTOS_macro_F1", "APTOS_accuracy")])
    return md_table(
        f"**Table 4.** Schedule-stratified mild-stage and overall metrics on the APTOS internal test split (mean ± sample standard deviation over three seeds per row). Mild-stage metrics treat grade 1 as the positive class; the test split contains {g1n('APTOS')} grade-1 images.",
        ["Cell", "Schedule group", "Mild precision", "Mild recall", "Mild F1", "Macro-F1", "Accuracy"], rows)


def t_g1() -> str:
    rows = []
    g = read_csv("external_grade1_recall_per_run.csv")
    for c in CELLS:
        for label, seeds, sched in (("42–44", (42, 43, 44), "truncated"),
                                   ("45–47", (45, 46, 47), "full cosine")):
            vals = []
            for d in DOMAINS:
                ys = [float(r[f"{d}_grade1_recall"]) for r in g if r["cell"] == c and int(r["seed"]) in seeds]
                if len(ys) != 3:
                    raise SystemExit(f"MISSING grade-1 values {c} {d} {seeds}")
                vals.append(mean_sd(ys))
            rows.append([c, f"{label}, {sched}"] + vals)
    return md_table(
        "**Table 5.** Schedule-stratified grade-1 (mild) recall on every test set (mean ± sample standard deviation over three seeds per row). This is a descriptive analysis computed from the saved predictions. The H3 IDRiD value for seeds 42–44 reflects one run that assigned grade 1 to most IDRiD images (Section 4.3).",
        ["Cell", "Schedule group"] + [f"{d} (n = {g1n(d)})" for d in DOMAINS], rows)


def t_ece() -> str:
    rows = []
    for c in ("H1", "H2"):
        for label, seeds, sched in (("42–44", (42, 43, 44), "truncated"),
                                   ("45–47", (45, 46, 47), "full cosine")):
            row = [c, f"{label}, {sched}"]
            for d in DOMAINS:
                rr = [r for r in ece6(c, d) if int(r["seed"]) in seeds]
                if len(rr) != 3:
                    raise SystemExit(f"MISSING ECE {c} {d} {seeds}")
                row.append(f"{mean_sd([float(r['ece']) for r in rr])} → {mean_sd([float(r['ece_T']) for r in rr])}")
            row.append(mean_sd([float(r["T"]) for r in ece6(c, "APTOS") if int(r["seed"]) in seeds]))
            rows.append(row)
    return md_table(
        "**Table 6.** Expected calibration error of the cross-entropy cells before and after temperature scaling (before → after; mean ± sample standard deviation over three seeds per schedule group; 15 equal-width bins). T is the temperature fitted on APTOS validation data. Threshold-head configurations are excluded because their cumulative outputs were non-monotonic and derived class probabilities were not interpretable.",
        ["Cell", "Schedule group"] + DOMAINS + ["T"], rows)


def t_b0() -> str:
    rows = []
    for c in CELLS:
        b0 = b0_row(c)
        rows.append([c, "EfficientNet-B0", f"{SEEDS_A}, full cosine"] + [pm(b0[f"B0 {d}"]) for d in DOMAINS])
        rows.append([c, "MobileViT-S", f"{SEEDS_A}, truncated"] + [pm(b0[f"Hybrid {d}"]) for d in DOMAINS])
        rows.append([c, "MobileViT-S", f"{SEEDS_B}, full cosine"] + [pm(smc(c, d, "QWK")) for d in DOMAINS])
    return md_table(
        "**Table 7.** Secondary exploratory comparison: QWK of matched EfficientNet-B0 and MobileViT-S under the same inputs, losses, batch size, frozen batch normalisation and CPU training (mean ± standard deviation over three seeds each). The schedule column gives the seeds and the learning-rate schedule actually delivered; MobileViT-S seeds 45–47 are the schedule-matched comparison.",
        ["Cell", "Backbone", "Seeds, schedule"] + DOMAINS, rows)


CONTRASTS = [("H2−H1", "CLAHE under cross-entropy"), ("H4−H3", "CLAHE under MW threshold loss"),
             ("H3−H1", "MW threshold loss with RGB"), ("H4−H2", "MW threshold loss with CLAHE")]


def t_paired() -> str:
    """Descriptive paired differences within each schedule group; no confirmatory p-values."""
    rows = []
    by = {(r["cell"], int(r["seed"])): r for r in master()}
    g1rows = {(r["cell"], int(r["seed"])): r for r in read_csv("external_grade1_recall_per_run.csv")}
    for c, label in CONTRASTS:
        a, b = c.replace(MINUS, "-").split("-")
        for group, seeds, sched in (("42–44", (42, 43, 44), "truncated"),
                                   ("45–47", (45, 46, 47), "full cosine")):
            for d in DOMAINS:
                dq = [float(by[(a, s)][f"QWK_{d}"]) - float(by[(b, s)][f"QWK_{d}"]) for s in seeds]
                dg = [float(g1rows[(a, s)][f"{d}_grade1_recall"]) - float(g1rows[(b, s)][f"{d}_grade1_recall"])
                      for s in seeds]
                signs_q = f"{sum(x > 0 for x in dq)} / {sum(x < 0 for x in dq)}"
                signs_g = f"{sum(x > 0 for x in dg)} / {sum(x < 0 for x in dg)}"
                rows.append([f"{c} ({label})", d, f"{group}, {sched}",
                             f"{signed(st.mean(dq))} ± {num(st.stdev(dq))}", signs_q,
                             f"{signed(st.mean(dg))} ± {num(st.stdev(dg))}", signs_g])
    return md_table(
        "**Table 8.** Descriptive paired differences between cells within each schedule group (mean ± sample standard deviation of the three seed-level differences). Positive / negative counts seeds with a difference above or below zero. These summaries are exploratory and are not used as confirmatory hypothesis tests. The large spread of the H3 IDRiD grade-1 contrasts for seeds 42–44 is inherited from one degenerate H3 run (Section 4.3).",
        ["Contrast", "Test set", "Schedule group", "Δ QWK", "Positive / negative",
         "Δ grade-1 recall", "Positive / negative"], rows)


def t_arch() -> str:
    sw, h2 = arch_row("Swin-Tiny"), arch_row("MobileViT-S H2")
    b0 = b0_row("H2")
    rows = [
        ["Swin-Tiny", f"{SEEDS_A}, truncated"] + [pm(sw[d]) for d in DOMAINS],
        ["MobileViT-S", f"{SEEDS_A}, truncated"] + [pm(h2[d]) for d in DOMAINS],
        ["MobileViT-S", f"{SEEDS_B}, full cosine"] + [pm(smc("H2", d, "QWK")) for d in DOMAINS],
        ["EfficientNet-B0 (matched)", f"{SEEDS_A}, full cosine"] + [pm(b0[f"B0 {d}"]) for d in DOMAINS],
    ]
    return md_table(
        "**Table 9.** QWK of three backbones (four rows: MobileViT-S H2 is shown for both schedule groups) trained on APTOS only with green-channel CLAHE and cross-entropy (mean ± standard deviation over three seeds each). Swin-Tiny and MobileViT-S seeds 42–44 received the same truncated learning-rate schedule.",
        ["Backbone", "Seeds, schedule"] + DOMAINS, rows)


def t_lodo() -> str:
    rows_csv = read_csv("table_lodo_vs_h2_per_seed.csv")
    sizes = lodo_train_sizes()
    rows = []
    for h in EXTERNAL:
        rr = [r for r in rows_csv if r["holdout"] == h]
        if len(rr) != 3:
            raise SystemExit(f"MISSING LODO seeds for {h}")
        rows.append([h, lodo_composition(h), thousands(sizes[LODO_KEY[h]]),
                     mean_sd([float(r["lodo_qwk"]) for r in rr]), mean_sd([float(r["h2_qwk"]) for r in rr]),
                     mean_sd([float(r["delta"]) for r in rr], sign=True), pm(smc("H2", h, "QWK"))])
    return md_table(
        "**Table 10.** Leave-one-dataset-out (LODO) training with green-channel CLAHE and cross-entropy, compared with APTOS-only H2 on each held-out dataset (QWK, mean ± standard deviation over three seeds). LODO and H2 seeds 42–44 are paired by seed; the difference is LODO minus H2. LODO runs received a full cosine schedule, so H2 seeds 45–47 are shown as a schedule-matched reference.",
        ["Held-out dataset", "Training composition (images)", "Training images", "LODO (seeds 42–44)",
         "H2 (seeds 42–44)", "Difference", "H2 (seeds 45–47)"], rows)


def t_decomp() -> str:
    def row(label, setup, seeds, aq, am, ae, iq, dq):
        return [label, setup, seeds, aq, am, ae, iq, dq]

    rows = []
    for cell, setup in (("H2", "CLAHE × CE"), ("D1", "CLAHE × unweighted threshold loss"),
                        ("D2", "CLAHE × MW-CE"), ("H4", "CLAHE × MW threshold loss")):
        r = decomp_row(cell)
        sched = "full cosine" if cell in ("D1", "D2") else "truncated"
        ece = pm(r["APTOS ECE"]) if cell in ("H2", "D2") else NI
        rows.append(row(cell, setup, f"{SEEDS_A}, {sched}", pm(r["APTOS QWK"]), pm(r["APTOS mild recall"]), ece,
                        pm(r["IDRiD QWK"]), pm(r["DeepDRiD QWK"])))
        if cell in ("H2", "H4"):
            rows.append(row(cell, setup, f"{SEEDS_B}, full cosine", pm(smc(cell, "APTOS", "QWK")),
                            pm(smc(cell, "APTOS", "mild recall")),
                            pm(smc(cell, "APTOS", "ECE")) if cell == "H2" else NI,
                            pm(smc(cell, "IDRiD", "QWK")), pm(smc(cell, "DeepDRiD", "QWK"))))
    order = {"H2": 0, "D2": 1, "D1": 2, "H4": 3}
    rows.sort(key=lambda r: (order[r[0]], r[2]))
    return md_table(
        "**Table 11.** Decomposition of the MW threshold loss under green-channel CLAHE (mean ± standard deviation over three seeds per row). D1 uses the threshold loss without mild weighting; D2 is cross-entropy with mild sample weight 2 (MW-CE). D1 and D2 were trained with seeds 42–44 under the full cosine schedule, so H2 and H4 are shown for both seed sets. n.i., not interpretable (Section 3.5).",
        ["Cell", "Setup", "Seeds, schedule", "APTOS QWK", "APTOS mild recall", "APTOS ECE", "IDRiD QWK",
         "DeepDRiD QWK"], rows)


def t_res448() -> str:
    rows = []
    for c in ("H1", "H2"):
        r = res448_row(c)
        rows.append([c, "224×224", f"{SEEDS_A}, truncated"] + [pm(r[f"224 {d}"]) for d in DOMAINS]
                    + [pm(r["224 mild recall"]), pm(mess_pred0_224(c, SEEDS_A))])
        rows.append([c, "224×224", f"{SEEDS_B}, full cosine"] + [pm(smc(c, d, "QWK")) for d in DOMAINS]
                    + [pm(smc(c, "APTOS", "mild recall")), pm(mess_pred0_224(c, SEEDS_B))])
        rows.append([c, "448×448", f"{SEEDS_A}, full cosine"] + [pm(r[f"448 {d}"]) for d in DOMAINS]
                    + [pm(r["448 mild recall"]), pm(mess_pred0_res(c, "448"))])
    return md_table(
        "**Table 12.** MobileViT-S H1 and H2 at 224×224 and 448×448 (mean ± standard deviation over three seeds per row). QWK on each test set, APTOS mild recall, and the share of Messidor-2 images predicted as grade 0. The 448×448 runs received the full cosine schedule, so both 224×224 seed sets are shown.",
        ["Cell", "Input size", "Seeds, schedule"] + [f"{d} QWK" for d in DOMAINS]
        + ["APTOS mild recall", "Messidor-2 share predicted grade 0"], rows)


# Supplementary tables ------------------------------------------------------

def ts_split() -> str:
    a = kv("aptos_split_audit.csv")
    rows = []
    for split, lab in (("train", "Training"), ("val", "Validation"), ("test", "Test")):
        rows.append([f"{lab} images (grades 0 / 1 / 2 / 3 / 4)",
                     f"{thousands(int(a[f'{split} images']))} ({' / '.join(thousands(int(a[f'{split} grade {g}'])) for g in range(5))})"])
    rows += [
        ["Image identifiers shared between any two partitions", a["id overlap train-val"] if
         a["id overlap train-val"] == a["id overlap train-test"] == a["id overlap val-test"] else "differs"],
        ["Byte-identical or pixel-identical image pairs, all", a[AUD_KEYS["pairs"]]],
        ["  within training", a[AUD_KEYS["trtr"]]], ["  training–validation", a[AUD_KEYS["trva"]]],
        ["  training–test", a[AUD_KEYS["trte"]]], ["  validation–test", a[AUD_KEYS["vate"]]],
        ["  within validation", a[AUD_KEYS["vava"]]], ["  within test", a[AUD_KEYS["tete"]]],
        ["Pairs found only by decoded-pixel comparison", a[AUD_KEYS["pixonly"]]],
        ["Cross-partition pairs with different grades", a[AUD_KEYS["difflabel"]]],
        ["Test images with an exact copy in training or validation", a[AUD_KEYS["testdup"]]],
        ["Validation images with an exact copy in training", a[AUD_KEYS["valdup"]]],
    ]
    return md_table(
        "**Table S1.** Audit of the APTOS partitions used in this study. Exact duplicates were identified by MD5 hashes of the original files and of the decoded pixel arrays; near-duplicates were not assessed.",
        ["Item", "Value"], rows)


def ts_regime() -> str:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in audit_rows():
        k = (r["family"], r["lr_schedule"].split(" (")[0], r["train_device"].upper(), r["eval_device"].replace("mps", "MPS"),
             r["batch_size"], "frozen" if r["freeze_bn"] == "True" else "trained", r["chunk_steps"],
             sci(float(r["lr_epoch10"])), sci(float(r["lr_final"])), r["trainer"].split("/")[-1].replace(".py", ""))
        groups[k].append(int(r["seed"]))
    rows = []
    for k, seeds in sorted(groups.items(), key=lambda kv: (kv[0][0], min(kv[1]))):
        seeds = sorted(seeds)
        s = f"{seeds[0]}–{seeds[-1]}" if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[-1] + 1)) else ", ".join(map(str, seeds))
        fam = k[0].replace("matched ", "").replace("LODO holdout ", "LODO, held out: ").replace(
            "messidor2", "Messidor-2").replace("idrid", "IDRiD").replace("deepdrid", "DeepDRiD")
        rows.append([fam, s, k[1], f"{k[2]} / {k[3]}", k[4], k[5], k[6], k[7], k[8]])
    spe = resolve("stepsepoch")[0]
    return md_table(
        f"**Table S2.** Training regime actually delivered to every run, from the run configuration and the learning rate logged at each epoch. Steps per launch is the number of optimiser steps run before the process saved a checkpoint and restarted; an epoch has {spe} steps at batch size 4, so a smaller value means that every epoch spanned several launches.",
        ["Model family", "Seeds", "Schedule", "Train / eval device", "Batch", "BatchNorm", "Steps per launch",
         "LR at epoch 10", "LR at epoch 20"], rows)


def ts_coral() -> str:
    rows = []
    for r in coral_rows():
        fr = nonmono_fracs(r)
        rows.append([f"{r['arch']} {r['cell']}", r["seed"]] + [num(float(r[f"bias_b{k}"])) for k in range(4)]
                    + ["yes" if r["biases_nonincreasing"] == "True" else "no", f"{min(fr):.3f}–{max(fr):.3f}"])
    return md_table(
        "**Table S3.** Learned threshold biases of every threshold-head run and the fraction of test images whose cumulative probabilities σ(z₀) ≥ σ(z₁) ≥ σ(z₂) ≥ σ(z₃) are violated (range over the four test sets). Rank consistency requires b₀ ≥ b₁ ≥ b₂ ≥ b₃; the initial values are given in Section 3.5.",
        ["Model and cell", "Seed", "b₀", "b₁", "b₂", "b₃", "Biases non-increasing", "Non-monotonic fraction"], rows)


def ts_master() -> str:
    rows = []
    for r in master():
        ece = (f"{num(float(r['APTOS_ECE_raw']))} → {num(float(r['APTOS_ECE_T']))}" if r["APTOS_ECE_raw"] != NI else NI)
        t = num(float(r["T"])) if r["T"] != NI else NI
        rows.append([r["cell"], r["seed"], r["best_epoch"], num(float(r["val_QWK"]))]
                    + [num(float(r[f"QWK_{d}"])) for d in DOMAINS]
                    + [num(float(r["APTOS_mild_recall"])), ece, t,
                       "truncated" if r["lr_schedule"].startswith("truncated") else "full cosine"])
    return md_table(
        "**Table S4.** Per-run results of the 24 primary MobileViT-S runs (seeds 42–47). Best epoch is the epoch of the saved checkpoint with the highest APTOS validation QWK. ECE is on the APTOS internal test split before → after temperature scaling. n.i., not interpretable (Section 3.5).",
        ["Cell", "Seed", "Best epoch", "Validation QWK", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD",
         "APTOS mild recall", "APTOS ECE", "T", "Schedule"], rows)


def ts_batch() -> str:
    rows = [[r["cell"], r["domain"], r["metric"], pm(r["seeds_42_44"]), pm(r["seeds_45_47"])]
            for r in read_csv("seed_batch_summary.csv")]
    return md_table(
        "**Table S5.** Primary MobileViT-S results summarised separately for seeds 42–44 and seeds 45–47 (mean ± standard deviation over three seeds each).",
        ["Cell", "Test set", "Metric", "Seeds 42–44", "Seeds 45–47"], rows)


def ts_dedup() -> str:
    metrics = ["qwk", "mild_recall", "macro_f1", "accuracy", "referable_sensitivity",
               "referable_specificity", "referable_auc", "ece", "nll", "brier"]
    rows = []
    for r in read_csv("deduplication_sensitivity_summary.csv"):
        subset = "all" if r["evaluation_subset"].startswith("all") else "deduplicated"
        grp = r["schedule_group"].replace("42-44", "42–44,").replace("45-47", "45–47,")
        rows.append([r["cell"], grp, f"{subset} ({r['n_images']})"]
                    + [NI if r[m] == NI else pm(r[m]) for m in metrics])
    return md_table(
        "**Table S6.** Schedule-stratified APTOS test-set deduplication sensitivity (mean ± sample standard deviation over three seeds per row). The deduplicated subset removes the test images that have an exact copy in training or validation. Referable DR is grade 2 or above, scored as in Section 3.7. ECE, NLL and multiclass Brier score are computed from softmax probabilities of the cross-entropy cells only; n.i., not interpretable for threshold-head cells (Section 3.5).",
        ["Cell", "Schedule group", "Test images (n)", "QWK", "Mild recall", "Macro-F1", "Accuracy",
         "Referable sensitivity", "Referable specificity", "Referable AUC", "ECE", "NLL", "Brier"], rows)


TABLE_BUILDERS = {
    "datasets": t_datasets, "settings": t_settings, "qwk": t_qwk, "mild": t_mild, "g1": t_g1, "ece": t_ece,
    "b0": t_b0, "paired": t_paired, "arch": t_arch, "lodo": t_lodo, "decomp": t_decomp, "res448": t_res448,
    "s_split": ts_split, "s_regime": ts_regime, "s_coral": ts_coral, "s_master": ts_master, "s_batch": ts_batch,
    "s_dedup": ts_dedup,
}


# ---------------------------------------------------------------------------
# Claims stated in words in the template
# ---------------------------------------------------------------------------

def check_claims() -> list[str]:
    fails: list[str] = []

    def claim(ok: bool, text: str) -> None:
        print(("CLAIM_OK   " if ok else "CLAIM_FAIL ") + text)
        if not ok:
            fails.append(text)

    m = {(c, d): st.mean(qwk_vals(c, d)) for c in CELLS for d in DOMAINS}
    claim(max(CELLS, key=lambda c: m[(c, "APTOS")]) == "H2", "H2 has the highest mean APTOS QWK (n=6)")
    claim(max(CELLS, key=lambda c: m[(c, "DeepDRiD")]) == "H2", "H2 has the highest mean DeepDRiD QWK (n=6)")
    claim(max(CELLS, key=lambda c: m[(c, "IDRiD")]) == "H4", "H4 has the highest mean IDRiD QWK (n=6)")
    claim(all(m[(c, "APTOS")] - m[(c, d)] > 0.25 for c in CELLS for d in EXTERNAL),
          "QWK drops by more than 0.25 from APTOS to every external set for every cell (n=6)")
    claim(max(m[(c, "Messidor-2")] for c in CELLS) < 0.1, "No cell exceeds 0.1 mean QWK on Messidor-2")
    claim(m[("H2", "IDRiD")] > m[("H1", "IDRiD")], "H2 mean IDRiD QWK above H1 (n=6)")
    claim(st.stdev(qwk_vals("H1", "IDRiD")) > 0.09 and st.stdev(qwk_vals("H2", "IDRiD")) > 0.09,
          "IDRiD QWK SD is large for H1 and H2")

    rec = {c: st.mean(aptos_metric("mild_dr_recall", c)) for c in CELLS}
    prec = {c: st.mean(aptos_metric("mild_dr_precision", c)) for c in CELLS}
    claim(min(CELLS, key=rec.get) == "H2", "H2 has the lowest APTOS mild recall (n=6)")
    claim(rec["H3"] > rec["H2"] and rec["H4"] > rec["H2"], "H3 and H4 have higher mild recall than H2 (n=6)")
    claim(max(prec["H3"], prec["H4"]) < min(prec["H1"], prec["H2"]), "H3 and H4 have lower mild precision than H1 and H2")
    claim(g1n("APTOS") == 30, "APTOS test has 30 grade-1 images")
    claim(mildk_set("H2", "a") == {5}, "H2 found exactly 5 of 30 mild images in each of seeds 42–44")
    claim(min(mcol("H2", "APTOS_mild_recall", (45, 46, 47))) > max(mcol("H2", "APTOS_mild_recall", (42, 43, 44))),
          "H2 mild recall is higher in every seed of 45–47 than in any seed of 42–44")

    g = {(c, d): pm_mean(g1(c, d)) for c in CELLS for d in DOMAINS}
    claim(max(CELLS, key=lambda c: g[(c, "IDRiD")]) == "H4" and max(CELLS, key=lambda c: g[(c, "Messidor-2")]) == "H4"
          and max(CELLS, key=lambda c: g[(c, "DeepDRiD")]) == "H4", "H4 has the highest grade-1 recall on every external set")
    claim(all(g[(c, d)] < 0.05 for c in ("H1", "H2") for d in ("Messidor-2", "DeepDRiD")),
          "CE cells detect almost no grade-1 images on Messidor-2 and DeepDRiD")
    claim(all(int(pc("g1", "H4-H2", d)["n_pos"]) == 6 for d in EXTERNAL), "H4−H2 grade-1 recall positive in 6/6 seeds on every external set")

    # ECE, cross-entropy cells only (n=6)
    e = {(c, d): st.mean(float(r["ece"]) for r in ece6(c, d)) for c in ("H1", "H2") for d in DOMAINS}
    claim(all(e[(c, d)] > e[(c, "APTOS")] for c in ("H1", "H2") for d in EXTERNAL), "CE ECE higher on every external set than APTOS")
    claim(e[("H2", "IDRiD")] < e[("H1", "IDRiD")] and e[("H2", "DeepDRiD")] < e[("H1", "DeepDRiD")],
          "H2 ECE below H1 on IDRiD and DeepDRiD")
    tvals = [float(r["T"]) for c in ("H1", "H2") for r in ece6(c, "APTOS")]
    claim(min(tvals) > 0.8 and max(tvals) < 1.25, "CE fitted temperatures are close to 1")
    claim(all(abs(st.mean(float(r["ece_T"]) for r in ece6(c, d)) - e[(c, d)]) < 0.02 for c in ("H1", "H2") for d in DOMAINS),
          "Temperature scaling changes mean CE ECE by less than 0.02 everywhere")

    # Threshold head
    cr = coral_rows()
    claim(not any(r["biases_nonincreasing"] == "True" for r in cr), "No threshold-head run has non-increasing biases")
    claim(min(min(nonmono_fracs(r)) for r in cr) > 0.95, "Over 95% of test images non-monotonic in every run/test set")
    claim(all(r[f"{d}_decode_matches_saved"] == "True" and r[f"{d}_probs_match_saved"] == "True" for r in cr for d in DOMAINS),
          "Saved predictions and probabilities reproduce from stored logits")
    claim({r["score_def"] for r in read_csv("table_B_referable_dr.csv") if r["head"] == "coral"} == {"sigmoid(z_1)"},
          "Referable AUC for threshold heads uses sigmoid(z_1)")

    # Paired contrasts (n=6)
    claim(all(float(r["p_holm_within_metric"]) >= 0.5 for r in read_csv("paired_contrasts_n6.csv") if r["metric"] == "QWK"),
          "No QWK contrast has a Holm-adjusted p below 0.5")
    for c, d in (("H2-H1", "APTOS"), ("H2-H1", "DeepDRiD"), ("H4-H3", "APTOS"), ("H4-H3", "IDRiD"), ("H4-H3", "DeepDRiD"),
                 ("H4-H2", "IDRiD")):
        claim(int(pc("qwk", c, d)["n_pos"]) == 6, f"{c} QWK positive in 6/6 seeds on {d}")
    claim(int(pc("qwk", "H4-H2", "DeepDRiD")["n_neg"]) == 6, "H4−H2 DeepDRiD QWK negative in 6/6 seeds")
    claim(int(pc("qwk", "H2-H1", "IDRiD")["n_pos"]) == 4, "H2−H1 IDRiD QWK positive in 4/6 seeds")
    claim(float(pc("qwk", "H2-H1", "Messidor-2")["mean_delta"]) < 0, "H2−H1 Messidor-2 mean negative")
    claim(float(pc("qwk", "H4-H3", "Messidor-2")["mean_delta"]) < 0, "H4−H3 Messidor-2 mean negative")
    claim(0 < int(pc("qwk", "H2-H1", "IDRiD")["n_neg"]), "H2−H1 IDRiD not positive in every seed")
    claim(abs(float(pc("qwk", "H3-H1", "APTOS")["mean_delta"])) < 0.01, "H3−H1 APTOS near zero")
    claim(float(pc("qwk", "H3-H1", "DeepDRiD")["mean_delta"]) < 0, "H3−H1 DeepDRiD negative")

    # Seed batches and schedules
    for c in CELLS:
        claim(pm_mean(sb(c, "APTOS", "QWK", "a")) != pm_mean(sb(c, "APTOS", "QWK", "b")), f"{c} APTOS differs between seed sets")
    claim(pm_mean(sb("H1", "DeepDRiD", "QWK", "b")) < 0.05 < pm_mean(sb("H1", "DeepDRiD", "QWK", "a")),
          "H1 DeepDRiD near zero for 45–47 but not 42–44")
    sched = {(r["family"], r["lr_schedule"].split(" (")[0]) for r in audit_rows()}
    spe = int(resolve("stepsepoch")[0])
    claim(all(r["lr_schedule"].startswith("truncated") == (int(r["chunk_steps"]) < spe)
              for r in audit_rows() if r["trainer"] == "src/train.py"),
          "Main trainer: truncated schedule exactly in runs whose launches were shorter than one epoch")
    claim(all(r["lr_schedule"].startswith("full cosine") and int(r["chunk_steps"]) < spe
              for r in audit_rows() if r["trainer"] != "src/train.py"),
          "LODO trainer: launches shorter than an epoch yet full cosine schedule")
    claim({f for f, s in sched if s == "truncated"} == {"MobileViT-S H1", "MobileViT-S H2", "MobileViT-S H3",
                                                         "MobileViT-S H4", "Swin-Tiny"},
          "Only primary 42–44 and Swin runs have the truncated schedule")
    trunc_seeds = {int(r["seed"]) for r in audit_rows() if r["lr_schedule"].startswith("truncated")}
    claim(trunc_seeds == {42, 43, 44}, "Truncated runs are seeds 42–44 only")
    claim(int(kv("environment.csv")["seeds 42-44: runs with resume records"]) == 12, "All 12 primary 42–44 runs resumed")

    # Dedup sensitivity
    claim(float(resolve("dedup:maxchange")[0]) <= 0.01, "Removing duplicated test images changes mean QWK by at most 0.01")
    claim(float(resolve("dedup:grpmax")[0]) <= 0.01,
          "Removing duplicated test images changes schedule-group mean QWK by at most 0.01")

    # Statements added after the external review (schedule-stratified)
    for d in ("APTOS", "DeepDRiD"):
        for g in ("a", "b"):
            claim(sum(x > 0 for x in paired_group("qwk", "H2-H1", d, g)) == 3, f"H2−H1 {d} QWK positive in 3/3 seeds of group {g}")
    claim(all(float(g1run("H1", x)[f"{d}_grade1_recall"]) == 0 for x in (45, 46, 47) for d in EXTERNAL),
          "H1 full-cosine external grade-1 recall is zero in every seed")
    claim(pm_mean(mess_pred0_224("H1", "45–47")) == 1.0, "H1 full-cosine predicts grade 0 for all Messidor-2 images")
    claim(all(pm_mean(sb("H1", d, "QWK", "b")) < 0.15 for d in EXTERNAL), "H1 full-cosine external QWK below 0.15 everywhere")
    k, n, _ = pred_share("H3", 42, "idrid_external", 1)
    claim(k / n > 0.5 and g1run("H3", 42)["IDRiD_grade1_correct"] == g1run("H3", 42)["IDRiD_grade1_n"]
          and all(g1run("H3", x)["IDRiD_grade1_correct"] == "0" for x in (43, 44)),
          "H3 seed 42 recalls every IDRiD grade-1 image by predicting grade 1 for most images; seeds 43–44 recall none")
    d2, h2a, h2b = (pm_mean(decomp_row("D2")["APTOS mild recall"]), pm_mean(sb("H2", "APTOS", "mild recall", "a")),
                    pm_mean(sb("H2", "APTOS", "mild recall", "b")))
    claim(d2 - h2a > 0.2 and 0 < d2 - h2b < 0.1, "D2 mild recall well above truncated H2 but only modestly above full-cosine H2")
    r448 = res448_row("H2")
    claim(pm_mean(r448["448 IDRiD"]) - max(pm_mean(sb("H2", "IDRiD", "QWK", g)) for g in "ab") > 0.2,
          "H2 448 IDRiD QWK clearly above both 224 groups")
    claim(abs(pm_mean(r448["448 mild recall"]) - h2b) < 0.05, "H2 448 mild recall within 0.05 of full-cosine 224")

    # Matched B0 beats hybrid in both seed sets
    for c in CELLS:
        r = b0_row(c)
        for d in DOMAINS:
            b = pm_mean(r[f"B0 {d}"])
            claim(b > pm_mean(r[f"Hybrid {d}"]) and b > pm_mean(smc(c, d, "QWK")),
                  f"Matched B0 beats MobileViT-S (42–44 and 45–47) on {d} for {c}")

    # Architecture reference
    sw, h2 = arch_row("Swin-Tiny"), arch_row("MobileViT-S H2")
    for d in DOMAINS:
        claim(pm_mean(sw[d]) > pm_mean(h2[d]) and pm_mean(sw[d]) > pm_mean(smc("H2", d, "QWK")),
              f"Swin-Tiny beats MobileViT-S H2 (both seed sets) on {d}")
    claim(pm_mean(sw["Messidor-2"]) < pm_mean(b0_row("H2")["B0 Messidor-2"]), "Matched B0 beats Swin on Messidor-2")
    claim(pm_mean(sw["Messidor-2"]) > max(m[(c, "Messidor-2")] for c in CELLS),
          "Swin Messidor-2 higher than every MobileViT-S cell")
    claim(all(pm_mean(sw[d]) < pm_mean(b0_row("H2")[f"B0 {d}"]) for d in EXTERNAL), "Matched B0 H2 beats Swin on every external set")

    # Decomposition
    D = {c: decomp_row(c) for c in ("H2", "D1", "D2", "H4")}
    mr = {c: pm_mean(D[c]["APTOS mild recall"]) for c in D}
    claim(mr["D1"] > mr["H2"] and mr["D2"] > mr["H2"], "D1 and D2 mild recall above H2 (42–44)")
    claim(mr["D2"] > pm_mean(smc("H2", "APTOS", "mild recall")) and mr["D1"] > pm_mean(smc("H2", "APTOS", "mild recall")),
          "D1 and D2 mild recall above schedule-matched H2 (45–47)")
    claim(mr["D2"] - pm_mean(smc("H2", "APTOS", "mild recall")) < mr["D2"] - mr["H2"],
          "Gap D2 vs H2 smaller against schedule-matched H2")
    claim(mr["D1"] > mr["D2"], "D1 mild recall above D2")
    claim(abs(pm_mean(D["D2"]["APTOS ECE"]) - pm_mean(D["H2"]["APTOS ECE"])) < 0.01, "D2 ECE close to H2")
    iq = {c: pm_mean(D[c]["IDRiD QWK"]) for c in D}
    claim(iq["D1"] > iq["D2"] > iq["H2"] and iq["D1"] > pm_mean(smc("H2", "IDRiD", "QWK")) and
          iq["D2"] > pm_mean(smc("H2", "IDRiD", "QWK")), "IDRiD: D1 > D2 > H2 (both H2 seed sets)")
    claim(iq["D1"] > iq["H4"] and iq["D1"] > pm_mean(smc("H4", "IDRiD", "QWK")), "D1 IDRiD above H4 (both seed sets)")
    claim(mr["D1"] > pm_mean(smc("H4", "APTOS", "mild recall")) and mr["D1"] > mr["H4"], "D1 mild recall above H4 (both seed sets)")

    # 448
    r448 = {c: res448_row(c) for c in ("H1", "H2")}
    claim(pm_mean(r448["H2"]["448 IDRiD"]) > pm_mean(r448["H2"]["224 IDRiD"]) and
          pm_mean(r448["H2"]["448 IDRiD"]) > pm_mean(smc("H2", "IDRiD", "QWK")), "H2 IDRiD higher at 448 than both 224 sets")
    claim(pm_mean(r448["H2"]["448 mild recall"]) > pm_mean(r448["H2"]["224 mild recall"]) and
          pm_mean(r448["H2"]["448 mild recall"]) > pm_mean(smc("H2", "APTOS", "mild recall")), "H2 mild recall higher at 448 than both")
    claim(pm_mean(r448["H1"]["448 IDRiD"]) < pm_mean(r448["H1"]["224 IDRiD"]) and
          pm_mean(r448["H1"]["448 DeepDRiD"]) < pm_mean(r448["H1"]["224 DeepDRiD"]), "H1 declines on IDRiD and DeepDRiD vs 42–44")
    claim(pm_mean(r448["H1"]["448 IDRiD"]) > pm_mean(smc("H1", "IDRiD", "QWK")) and
          pm_mean(r448["H1"]["448 DeepDRiD"]) > pm_mean(smc("H1", "DeepDRiD", "QWK")), "H1 at 448 above schedule-matched 224 on IDRiD, DeepDRiD")
    claim(all(abs(pm_mean(r448[c]["448 Messidor-2"])) < 0.1 for c in ("H1", "H2")), "Messidor-2 QWK near zero at 448 for H1 and H2")
    p448 = pm_mean(mess_pred0_res("H2", "448"))
    claim(p448 < 0.5 * min(pm_mean(mess_pred0_224("H2", SEEDS_A)), pm_mean(mess_pred0_224("H2", SEEDS_B))),
          "H2 Messidor grade-0 share at 448 less than half of either 224 seed set")
    claim(pm_mean(r448["H2"]["448 DeepDRiD"]) < pm_mean(r448["H2"]["224 DeepDRiD"]), "H2 DeepDRiD at 448 below 224 (42–44)")

    # Messidor
    claim(min(float(r["pred_frac_grade0"]) for c in ("H1", "H2", "H3") for r in messidor_runs(c)) > 0.85,
          "H1–H3 predict grade 0 for more than 85% of Messidor-2 in every run")
    claim(max(float(r["pred_frac_grade0"]) for r in messidor_runs("H4")) < 0.85, "H4 below 85% grade-0 share in every run")

    # LODO
    lodo = read_csv("table_lodo_vs_h2_per_seed.csv")
    dl = {h: [float(r["delta"]) for r in lodo if r["holdout"] == h] for h in EXTERNAL}
    claim(all(st.mean(v) > 0 for v in dl.values()), "LODO beats APTOS-only H2 on every held-out set")
    claim(st.mean(dl["DeepDRiD"]) < st.stdev(dl["DeepDRiD"]), "LODO gain on DeepDRiD is smaller than its seed spread")
    claim(min(st.mean(dl["IDRiD"]), st.mean(dl["Messidor-2"])) > st.mean(dl["DeepDRiD"]), "LODO gains largest for IDRiD and Messidor-2")
    lq = {h: st.mean(float(r["lodo_qwk"]) for r in lodo if r["holdout"] == h) for h in EXTERNAL}
    claim(all(lq[h] > pm_mean(smc("H2", h, "QWK")) for h in EXTERNAL), "LODO above schedule-matched H2 on every held-out set")
    sizes = lodo_train_sizes()
    n_aptos = int(manifest("APTOS", "train")["n_images"])
    claim(min(sizes.values()) > 1.5 * n_aptos, "LODO training sets are much larger than APTOS train")
    return fails


def mildk_set(cell: str, batch: str) -> set[int]:
    seeds = (42, 43, 44) if batch == "a" else (45, 46, 47)
    n = g1n("APTOS")
    return {round(v * n) for v in mcol(cell, "APTOS_mild_recall", seeds)}


def main() -> None:
    if not TEMPLATE.is_file():
        missing(TEMPLATE)
    print("READ", rel(TEMPLATE))
    text = TEMPLATE.read_text()
    ledger: list[tuple[str, str, str]] = []

    def sub(mo: re.Match) -> str:
        key = mo.group(1).strip()
        val, src = resolve(key)
        if not key.startswith("table:"):
            ledger.append((key, val, src))
        else:
            for line in val.splitlines():
                if line.startswith("|") and "---" not in line:
                    for cell in line.strip("|").split("|"):
                        for tok in re.findall(r"[−-]?\d[\d,]*(?:\.\d+)?(?: ± \d+\.\d+)?", cell):
                            ledger.append((key, tok.strip(), src))
        return val

    filled = re.sub(r"\{\{([^}]+)\}\}", sub, text)
    fails = check_claims()
    if fails:
        print("BUILD_MANUSCRIPT_MD_FAIL")
        for f in fails:
            print("  ", f)
        raise SystemExit(1)

    header = ("<!-- Generated by scripts/build_manuscript_md.py from paper/MANUSCRIPT_TEMPLATE.md. "
              "Edit the template, not this file. -->\n\n")
    OUT_MD.write_text(header + filled)
    print("WROTE", rel(OUT_MD))
    with LEDGER.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["placeholder", "value", "source"])
        w.writerows(ledger)
    print("WROTE", rel(LEDGER), "values", len(ledger))
    print("SOURCES", len(_read))
    print("BUILD_MANUSCRIPT_MD_OK")


if __name__ == "__main__":
    sys.exit(main())
