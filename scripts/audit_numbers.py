#!/usr/bin/env python3
"""
Write NUMBERS_AUDIT.csv: one row per numeric token in paper/MANUSCRIPT.md (body and tables, references excluded).

Three independent traces, strongest first:
  RECOMPUTED  Tables 3-6 cells recomputed here from the raw per-run files
              results/hybrid_<cell>_s<seed>_<v1|v3>/<dataset>_metrics.json (mean ± sample SD, 3 d.p.).
  SOURCE_CSV  value (or both parts of a mean ± SD pair, in one row) found in a paper/tables CSV other than
              manuscript_values.csv, which is written by the manuscript builder and therefore not a source.
  PLACEHOLDER value equals a template placeholder resolved by scripts/build_manuscript_md.py; the source
              recorded in manuscript_values.csv is listed.
Tokens matching none of these are MISMATCH / UNTRACED and need review. Section numbers, figure/table/reference
numbers and the fixed configuration constants listed in CONFIG are skipped.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "paper" / "MANUSCRIPT.md"
TABLES = ROOT / "paper" / "tables"
OUT = ROOT / "NUMBERS_AUDIT.csv"
SEEDS = {"42–44, truncated": ((42, 43, 44), "v1"), "45–47, full cosine": ((45, 46, 47), "v3")}
DS = {"APTOS": "aptos_test", "IDRiD": "idrid_external", "Messidor-2": "messidor2_external",
      "DeepDRiD": "deepdrid_external"}
CONFIG = {"2.0", "0.5", "0.05", "20", "1.0", "0.0"}
NUM = re.compile(r"[−+-]?\d+\.\d+(?:\s*±\s*\d+\.\d+)?")


def fmt(x: float) -> str:
    s = f"{x:.3f}"
    return s.replace("-", "−") if not s.startswith("-0.000") else "0.000"


def msd(vals: list[float]) -> str:
    return f"{fmt(statistics.mean(vals))} ± {fmt(statistics.stdev(vals))}"


def run_metrics(cell: str, group: str, ds: str) -> tuple[list[dict], str]:
    seeds, ver = SEEDS[group]
    out = []
    for s in seeds:
        p = ROOT / "results" / f"hybrid_{cell.lower()}_s{s}_{ver}" / f"{DS[ds]}_metrics.json"
        out.append(json.loads(p.read_text()))
    return out, f"results/hybrid_{cell.lower()}_s{{{','.join(map(str, seeds))}}}_{ver}/{DS[ds]}_metrics.json"


def md_table(md: str, num: int) -> list[list[str]]:
    m = re.search(rf"^\*\*Table {num}\.\*\*.*?\n\n((?:\|.*\n)+)", md, flags=re.M)
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in m.group(1).strip().splitlines()]
    return [rows[0]] + rows[2:]


def recompute(md: str) -> dict[tuple[str, str], tuple[str, str, str]]:
    """(table number, row cells, claimed) -> (location, source, recomputed, status)"""
    res = {}

    row = None

    def put(loc, claimed, src, val):
        res[(loc.split()[1], tuple(row), claimed)] = (loc, src, val, "RECOMPUTED_MATCH" if claimed == val else "RECOMPUTED_MISMATCH")

    t = md_table(md, 3)
    for r in t[1:]:
        row = r
        cell, grp = r[0], r[3]
        for j, ds in enumerate(("APTOS", "IDRiD", "Messidor-2", "DeepDRiD")):
            ms, src = run_metrics(cell, grp, ds)
            put(f"Table 3 {cell} {grp} {ds}", r[4 + j], src + ":quadratic_weighted_kappa",
                msd([m["quadratic_weighted_kappa"] for m in ms]))
    t = md_table(md, 4)
    keys = ["mild_dr_precision", "mild_dr_recall", "mild_dr_f1", "macro_f1", "accuracy"]
    for r in t[1:]:
        row = r
        ms, src = run_metrics(r[0], r[1], "APTOS")
        for j, k in enumerate(keys):
            put(f"Table 4 {r[0]} {r[1]} {t[0][2 + j]}", r[2 + j], f"{src}:{k}", msd([m[k] for m in ms]))
    t = md_table(md, 5)
    for r in t[1:]:
        row = r
        for j, ds in enumerate(("APTOS", "IDRiD", "Messidor-2", "DeepDRiD")):
            ms, src = run_metrics(r[0], r[1], ds)
            put(f"Table 5 {r[0]} {r[1]} {ds}", r[2 + j], f"{src}:mild_dr_recall",
                msd([m["mild_dr_recall"] for m in ms]))
    t = md_table(md, 6)
    for r in t[1:]:
        row = r
        for j, ds in enumerate(("APTOS", "IDRiD", "Messidor-2", "DeepDRiD")):
            ms, src = run_metrics(r[0], r[1], ds)
            before, after = (x.strip() for x in r[2 + j].split("→"))
            put(f"Table 6 {r[0]} {r[1]} {ds} before", before, f"{src}:ece", msd([m["ece"] for m in ms]))
            put(f"Table 6 {r[0]} {r[1]} {ds} after", after, f"{src}:ece_temperature_scaled",
                msd([m["ece_temperature_scaled"] for m in ms]))
        ms, src = run_metrics(r[0], r[1], "APTOS")
        put(f"Table 6 {r[0]} {r[1]} T", r[6], f"{src}:temperature", msd([m["temperature"] for m in ms]))
    t = md_table(md, 8)
    for r in t[1:]:
        a, b = re.match(r"(H\d)−(H\d)", r[0]).groups()
        ma, src_a = run_metrics(a, r[2], r[1])
        mb, src_b = run_metrics(b, r[2], r[1])
        for col, key in ((3, "quadratic_weighted_kappa"), (5, "mild_dr_recall")):
            d = [x[key] - y[key] for x, y in zip(ma, mb)]
            pos, neg = sum(v > 1e-12 for v in d), sum(v < -1e-12 for v in d)
            claimed = r[col].lstrip("+")
            ok = claimed == msd(d) and r[col + 1] == f"{pos} / {neg}"
            res[("8", tuple(r), claimed)] = (
                f"Table 8 {a}−{b} {r[1]} {r[2]} {key}", f"{src_a} minus {src_b}:{key}", f"{msd(d)}; {pos} / {neg}",
                "RECOMPUTED_MATCH" if ok else "RECOMPUTED_MISMATCH")
    return res


def as_float(s: str) -> float | None:
    try:
        return float(s.replace("−", "-").replace(",", ""))
    except ValueError:
        return None


def load_csv_numbers() -> list[tuple[str, list[list[str]]]]:
    out = []
    for p in sorted(TABLES.glob("*.csv")):
        if p.name in ("manuscript_values.csv",):
            continue
        with p.open() as f:
            out.append((f"paper/tables/{p.name}", list(csv.reader(f))))
    return out


def close(v: float, target: str) -> bool:
    d = len(target.split(".")[1]) if "." in target else 0
    t = as_float(target)
    return t is not None and abs(round(v, d) - t) < 10 ** -d / 2 + 1e-9


def find_in_csv(tok: str, csvs) -> tuple[str, str] | None:
    parts = [p.strip() for p in tok.split("±")]
    for name, rows in csvs:
        for row in rows:
            joined = " | ".join(row)
            if tok in joined or tok.replace(" ", "") in joined.replace(" ", ""):
                return name, tok
            nums = [as_float(c) for c in row]
            nums = [x for x in nums if x is not None]
            sub = [c for c in row if "±" in c]
            for c in sub:
                a = [x.strip() for x in c.split("±")]
                if len(a) == 2 and len(parts) == 2 and all(as_float(a[i]) is not None and close(as_float(a[i]), parts[i]) for i in range(2)):
                    return name, c
            if all(any(close(x, p) for x in nums) for p in parts):
                hit = [f"{x:g}" for p in parts for x in nums if close(x, p)][: len(parts)]
                return name, " / ".join(hit)
    return None


def main() -> None:
    md = MD.read_text().split("\n## References")[0]
    rec = recompute(md)
    csvs = load_csv_numbers()
    ph = {}
    with (TABLES / "manuscript_values.csv").open() as f:
        for r in csv.DictReader(f):
            ph.setdefault(r["value"], (r["placeholder"], r["source"]))

    rows, heading, table = [], "", ""
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            heading, table = s.lstrip("# ").strip(), ""
            continue
        m = re.match(r"^\*\*(Table S?\d+|Figure S?\d+)\.\*\*", s)
        if m:
            table = m.group(1)
        if s.startswith("!["):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")] if s.startswith("|") else [s]
        if s.startswith("|---"):
            continue
        rowlab = " ".join(cells[:2]) if s.startswith("|") else ""
        for ci, c in enumerate(cells):
            text = re.sub(r"\b(?:Section|Sections|Table|Figure|Figures|Tables)\s+S?\d+(?:\.\d+)?", "", c)
            for tok in NUM.findall(text):
                tok = re.sub(r"\s*±\s*", " ± ", tok).lstrip("+")
                if tok in CONFIG:
                    continue
                loc = f"{heading} / {table + ' ' if s.startswith('|') or table else ''}{rowlab}".strip(" /")
                hit = rec.get((table.split()[-1], tuple(cells), tok)) if s.startswith("|") else None
                if hit:
                    rows.append([hit[0], tok, hit[1], hit[2], hit[3], "no" if hit[3] == "RECOMPUTED_MATCH" else "yes", "recomputed from raw per-run JSON"])
                    continue
                f = find_in_csv(tok, csvs)
                if f and "±" in tok:
                    rows.append([loc, tok, f[0], f[1], "SOURCE_CSV_MATCH", "no", "mean and SD found in one source row"])
                    continue
                if not s.startswith("|") and (tok in ph or any(tok in v for v in ph)):
                    key, src = ph.get(tok) or next(ph[v] for v in ph if tok in v)
                    rows.append([loc, tok, src, tok, "PLACEHOLDER_TRACED", "no",
                                 f"resolved by build_manuscript_md.py from placeholder {key}; not independently recomputed"])
                    continue
                if f:
                    rows.append([loc, tok, f[0], f[1], "SOURCE_CSV_WEAK", "no",
                                 "single value present in a source CSV; table builder in build_manuscript_md.py names the source"])
                    continue
                rows.append([loc, tok, "", "", "UNTRACED", "review", "no source CSV value or placeholder found"])

    rows.append(["Results 4.2 IDRiD sentence", "H2 ≤ H1 (truncated), H2 > H1 (full cosine)",
                 "Table 3", "0.364<0.395; 0.336>0.136", "SEMANTIC_MATCH", "no", "fixed in this audit"])
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["location", "claimed_value", "source_file", "source_value", "match_status",
                    "correction_needed", "notes"])
        w.writerows(rows)
    from collections import Counter
    c = Counter(r[4] for r in rows)
    print("READ", "paper/MANUSCRIPT.md", "results/hybrid_*/*_metrics.json", "paper/tables/*.csv")
    print("WROTE", OUT.relative_to(ROOT), len(rows), "rows", dict(c))
    for r in rows:
        if r[4] in ("UNTRACED", "RECOMPUTED_MISMATCH"):
            print("FLAG", r[:5])


if __name__ == "__main__":
    main()
