#!/usr/bin/env python3
"""Submission-stage tables built only from saved run outputs (no training, no estimation).

Outputs (paper/tables/):
  master_results_per_run.csv          one row per MobileViT-S H1-H4 run, seeds 42-47
  master_crosscheck.txt               agreement of the master table with every per-seed table used in the paper
  training_schedule_audit.csv         chunking / learning-rate schedule / device for every run family in the paper
  seed_batch_summary.csv              QWK per cell and test set for seeds 42-44 versus 45-47
  paired_contrasts_n6.csv             H2-H1, H4-H3, H3-H1, H4-H2 per test set and metric, exact sign-flip p, Holm
  external_grade1_recall_per_run.csv  grade-1 recall on every test set from saved predictions
  external_grade1_recall_summary.csv  mean +- SD by cell (six seeds) with grade-1 counts
  schedule_matched_comparators.csv    H cells at seeds 45-47 (full cosine) next to v3 secondary runs
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
CELLS = ["H1", "H2", "H3", "H4"]
SEEDS = [42, 43, 44, 45, 46, 47]
DOMS = {"APTOS": "aptos_test", "IDRiD": "idrid_external", "Messidor-2": "messidor2_external", "DeepDRiD": "deepdrid_external"}
INPUT = {"H1": "RGB", "H2": "green-channel CLAHE", "H3": "RGB", "H4": "green-channel CLAHE"}
LOSS = {"H1": "cross-entropy", "H2": "cross-entropy", "H3": "mild-weighted threshold", "H4": "mild-weighted threshold"}
ORDINAL = {"H3", "H4"}
READ: set[str] = set()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def load(p: Path):
    if not p.is_file():
        raise SystemExit(f"MISSING input: {rel(p)}")
    READ.add(rel(p))
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(p.read_text())
    with p.open() as f:
        return list(csv.DictReader(f))


def hyb(cell: str, seed: int) -> str:
    return f"hybrid_{cell.lower()}_s{seed}_{'v1' if seed <= 44 else 'v3'}"


def write(name: str, rows: list[dict]) -> None:
    p = OUT / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", rel(p), len(rows), "rows")


def run_info(run: str) -> dict:
    cfg = load(ROOT / "experiments" / run / "config.yaml")
    summ = load(ROOT / "results" / run / "train_summary.json")
    hist = load(ROOT / "results" / run / "history.json")
    ev = load(ROOT / "results" / run / "eval_report.json")
    t = cfg["train"]
    ckp = ROOT / "checkpoints" / run / "best.pt"
    if not ckp.is_file():
        raise SystemExit(f"MISSING input: {rel(ckp)}")
    READ.add(rel(ckp))
    best = int(torch.load(ckp, map_location="cpu", weights_only=False)["epoch"])
    argmax_ep = max(hist, key=lambda r: r["val_qwk"])["epoch"]
    return {
        "cfg": cfg, "summ": summ, "hist": hist, "ev": ev,
        "best_epoch": best, "history_argmax_epoch": argmax_ep, "summary_best_epoch": int(summ["best_epoch"]),
        "val_qwk": float(summ["best_selection_score"]),
        "train_device": t.get("device") or summ.get("device"),
        "eval_device": ev.get("device"),
        "batch_size": t["batch_size"],
        "bn_mode": "frozen (ImageNet statistics)" if t.get("freeze_bn") else "trained",
        "chunk_steps": t.get("max_steps_per_launch"),
        "lr_epoch1": hist[0]["lr"], "lr_epoch10": hist[9]["lr"] if len(hist) >= 10 else None,
        "lr_final": hist[-1]["lr"], "epochs_logged": len(hist),
        "image_size": cfg["preprocessing"]["image_size"],
    }


def schedule_label(lr_final: float) -> str:
    return "full cosine (final LR 0)" if lr_final < 1e-7 else f"truncated (final LR {lr_final:.2e})"


def signflip_p(d: list[float]) -> float:
    obs = abs(mean(d))
    hits = 0
    tot = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        tot += 1
        if abs(mean(s * x for s, x in zip(signs, d))) >= obs - 1e-12:
            hits += 1
    return hits / tot


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    running = 0.0
    m = len(ps)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def grade1_recall(pred: dict) -> tuple[int, int]:
    yt = pred["y_true"]
    yp = pred["y_pred"]
    n1 = sum(1 for y in yt if y == 1)
    c1 = sum(1 for y, p in zip(yt, yp) if y == 1 and p == 1)
    return c1, n1


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master = []
    info = {}
    for c in CELLS:
        for s in SEEDS:
            run = hyb(c, s)
            ri = run_info(run)
            info[(c, s)] = ri
            ev = ri["ev"]
            a = ev["aptos_test"]["metrics"]
            row = {
                "cell": c, "seed": s, "input": INPUT[c], "loss": LOSS[c],
                "best_epoch": ri["best_epoch"], "val_QWK": round(ri["val_qwk"], 6),
            }
            for d, k in DOMS.items():
                row[f"QWK_{d}"] = round(ev[k]["metrics"]["quadratic_weighted_kappa"], 6)
            row.update({
                "APTOS_mild_precision": round(a["mild_dr_precision"], 6),
                "APTOS_mild_recall": round(a["mild_dr_recall"], 6),
                "APTOS_mild_F1": round(a["mild_dr_f1"], 6),
                "APTOS_macro_F1": round(a["macro_f1"], 6),
                "APTOS_accuracy": round(a["accuracy"], 6),
                "APTOS_ECE_raw": round(a["ece"], 6) if c not in ORDINAL else "n.i.",
                "APTOS_ECE_T": round(a["ece_temperature_scaled"], 6) if c not in ORDINAL else "n.i.",
                "T": round(ev["temperature"]["value"], 6) if c not in ORDINAL else "n.i.",
                "train_device": ri["train_device"], "eval_device": ri["eval_device"],
                "batch_size": ri["batch_size"], "BN_mode": ri["bn_mode"],
                "chunk_steps": ri["chunk_steps"], "lr_schedule": schedule_label(ri["lr_final"]),
                "run_dir": f"results/{run}",
            })
            master.append(row)
    write("master_results_per_run.csv", master)

    # ---- cross-check master against every per-seed table used by the manuscript
    msgs = []
    bad = 0

    def chk(label: str, a: float, b: float, tol: float = 1e-5) -> None:
        nonlocal bad
        if abs(float(a) - float(b)) > tol:
            bad += 1
            msgs.append(f"MISMATCH {label}: master={a} table={b}")

    mrow = {(r["cell"], r["seed"]): r for r in master}
    for r in load(OUT / "table_v3_hybrid_qwk_n6_per_seed.csv"):
        chk(f"qwk {r['cell']} {r['seed']} {r['domain']}", mrow[(r["cell"], int(r["seed"]))][f"QWK_{r['domain']}"], r["qwk"])
    mk = {"mild_dr_recall": "APTOS_mild_recall", "mild_dr_precision": "APTOS_mild_precision", "mild_dr_f1": "APTOS_mild_F1",
          "macro_f1": "APTOS_macro_F1", "accuracy": "APTOS_accuracy"}
    for r in load(OUT / "table_v3_hybrid_mild_n6_long.csv"):
        if r["metric"] in mk:
            chk(f"{r['metric']} {r['cell']} {r['seed']}", mrow[(r["cell"], int(r["seed"]))][mk[r["metric"]]], r["value"])
    for r in load(OUT / "table_D_ece_all_domains.csv"):
        c, s = r["cell"], int(r["seed"])
        ev = info[(c, s)]["ev"][DOMS[r["domain"]]]["metrics"]
        chk(f"ece {c} {s} {r['domain']}", ev["ece"], r["ece"])
        chk(f"eceT {c} {s} {r['domain']}", ev["ece_temperature_scaled"], r["ece_at_T"])
        chk(f"T {c} {s}", info[(c, s)]["ev"]["temperature"]["value"], r["temperature_T"])
    for r in load(OUT / "table_v3_decomp_clahe_per_seed.csv"):
        if r["cell"] in CELLS:
            chk(f"decomp qwk {r['cell']} {r['seed']} {r['domain']}", mrow[(r["cell"], int(r["seed"]))][f"QWK_{r['domain']}"], r["qwk"])
    for r in load(OUT / "table_v3_res448_per_seed.csv"):
        if r["image_size"] == "224":
            chk(f"res224 {r['cell']} {r['seed']} {r['domain']}", mrow[(r["cell"], int(r["seed"]))][f"QWK_{r['domain']}"], r["qwk"])
    for r in load(OUT / "table_v3_matched_b0_vs_hybrid_per_seed.csv"):
        if r["model"] == "hybrid":
            chk(f"b0cmp {r['cell']} {r['seed']} {r['domain']}", mrow[(r["cell"], int(r["seed"]))][f"QWK_{r['domain']}"], r["qwk"])
    for r in load(OUT / "table_lodo_vs_h2_per_seed.csv"):
        chk(f"lodo h2 {r['seed']} {r['holdout']}", mrow[("H2", int(r["seed"]))][f"QWK_{r['holdout']}"], r["h2_qwk"])
    for (c, s), ri in info.items():
        if ri["best_epoch"] != ri["history_argmax_epoch"]:
            bad += 1
            msgs.append(f"MISMATCH best_epoch {c} {s}: best.pt={ri['best_epoch']} history_argmax={ri['history_argmax_epoch']}")
        if ri["summary_best_epoch"] != ri["best_epoch"]:
            msgs.append(f"NOTE train_summary.json best_epoch={ri['summary_best_epoch']} for {c} s{s}, "
                        f"but best.pt (the evaluated checkpoint) is from epoch {ri['best_epoch']}")
    msgs.append(f"checks finished, mismatches={bad}")
    (OUT / "master_crosscheck.txt").write_text("\n".join(msgs) + "\n")
    print("WROTE paper/tables/master_crosscheck.txt;", msgs[-1])
    if bad:
        print("\n".join(msgs[:20]))
        raise SystemExit("MASTER_CROSSCHECK_FAIL")

    # ---- schedule audit across all run families in the paper
    fam = []
    for c in CELLS:
        for s in SEEDS:
            fam.append((f"MobileViT-S {c}", s, hyb(c, s)))
    for c in CELLS:
        for s in (42, 43, 44):
            fam.append((f"EfficientNet-B0 matched {c}", s, f"matched_b0_{c.lower()}_s{s}_v3"))
    for s in (42, 43, 44):
        fam.append(("MobileViT-S D1", s, f"hybrid_d1_coral_clahe_s{s}_v3"))
        fam.append(("MobileViT-S D2", s, f"hybrid_d2_mildce_clahe_s{s}_v3"))
        fam.append(("MobileViT-S H1 448", s, f"hybrid_h1_448_s{s}_v3"))
        fam.append(("MobileViT-S H2 448", s, f"hybrid_h2_448_s{s}_v3"))
        fam.append(("Swin-Tiny", s, f"phase_o_swin_tiny_s{s}_v1"))
        for h in ("idrid", "messidor2", "deepdrid"):
            fam.append((f"LODO holdout {h}", s, f"phase_p_lodo_holdout_{h}_s{s}_v1"))
    audit = []
    for label, s, run in fam:
        cfg = load(ROOT / "experiments" / run / "config.yaml")
        hist = load(ROOT / "results" / run / "history.json")
        t = cfg["train"]
        evp = ROOT / "results" / run / "eval_report.json"
        evdev = load(evp).get("device") if evp.is_file() else ""
        audit.append({
            "family": label, "seed": s, "model": cfg["model"]["name"], "loss": cfg["loss"]["name"],
            "input": cfg["preprocessing"]["name"], "image_size": cfg["preprocessing"]["image_size"],
            "train_device": t.get("device"), "eval_device": evdev, "batch_size": t["batch_size"],
            "freeze_bn": t.get("freeze_bn"), "chunk_steps": t.get("max_steps_per_launch"),
            "trainer": "src/train_lodo.py" if run.startswith("phase_p_lodo") else "src/train.py",
            "lr_epoch1": f"{hist[0]['lr']:.3e}", "lr_epoch10": f"{hist[9]['lr']:.3e}", "lr_final": f"{hist[-1]['lr']:.3e}",
            "lr_schedule": schedule_label(hist[-1]["lr"]), "epochs": len(hist), "run_dir": f"results/{run}",
        })
    write("training_schedule_audit.csv", audit)

    # ---- seed batch summary
    sb = []
    for c in CELLS:
        for d in DOMS:
            a = [mrow[(c, s)][f"QWK_{d}"] for s in (42, 43, 44)]
            b = [mrow[(c, s)][f"QWK_{d}"] for s in (45, 46, 47)]
            sb.append({"cell": c, "domain": d, "metric": "QWK",
                       "seeds_42_44": f"{mean(a):.3f}±{stdev(a):.3f}", "seeds_45_47": f"{mean(b):.3f}±{stdev(b):.3f}",
                       "mean_42_44": round(mean(a), 6), "mean_45_47": round(mean(b), 6)})
        a = [mrow[(c, s)]["APTOS_mild_recall"] for s in (42, 43, 44)]
        b = [mrow[(c, s)]["APTOS_mild_recall"] for s in (45, 46, 47)]
        sb.append({"cell": c, "domain": "APTOS", "metric": "mild recall",
                   "seeds_42_44": f"{mean(a):.3f}±{stdev(a):.3f}", "seeds_45_47": f"{mean(b):.3f}±{stdev(b):.3f}",
                   "mean_42_44": round(mean(a), 6), "mean_45_47": round(mean(b), 6)})
    write("seed_batch_summary.csv", sb)

    # ---- external grade-1 recall
    g1 = []
    for c in CELLS:
        for s in SEEDS:
            run = hyb(c, s)
            row = {"cell": c, "seed": s}
            for d, k in DOMS.items():
                pr = load(ROOT / "results" / run / f"{k}_predictions.json")
                c1, n1 = grade1_recall(pr)
                row[f"{d}_grade1_correct"] = c1
                row[f"{d}_grade1_n"] = n1
                row[f"{d}_grade1_recall"] = round(c1 / n1, 6) if n1 else ""
            g1.append(row)
    write("external_grade1_recall_per_run.csv", g1)
    g1s = []
    for c in CELLS:
        rows = [r for r in g1 if r["cell"] == c]
        out = {"cell": c, "n_seeds": len(rows)}
        for d in DOMS:
            v = [r[f"{d}_grade1_recall"] for r in rows]
            out[f"{d}_n_grade1"] = rows[0][f"{d}_grade1_n"]
            out[d] = f"{mean(v):.3f}±{stdev(v):.3f}"
        g1s.append(out)
    write("external_grade1_recall_summary.csv", g1s)

    # ---- paired contrasts, six seeds
    contrasts = [("H2", "H1", "CLAHE under cross-entropy"), ("H4", "H3", "CLAHE under threshold loss"),
                 ("H3", "H1", "threshold loss under RGB"), ("H4", "H2", "threshold loss under CLAHE")]
    metrics = [("QWK", lambda c, s, d: mrow[(c, s)][f"QWK_{d}"]),
               ("grade-1 recall", lambda c, s, d: next(r for r in g1 if r["cell"] == c and r["seed"] == s)[f"{d}_grade1_recall"]),
               ("macro-F1", lambda c, s, d: info[(c, s)]["ev"][DOMS[d]]["metrics"]["macro_f1"]),
               ("accuracy", lambda c, s, d: info[(c, s)]["ev"][DOMS[d]]["metrics"]["accuracy"])]
    pc = []
    for m, fn in metrics:
        for a, b, lab in contrasts:
            for d in DOMS:
                diffs = [fn(a, s, d) - fn(b, s, d) for s in SEEDS]
                pc.append({"metric": m, "contrast": f"{a}−{b}", "meaning": lab, "domain": d,
                           "mean_delta": round(mean(diffs), 6), "sd_delta": round(stdev(diffs), 6),
                           "n_pos": sum(x > 0 for x in diffs), "n_neg": sum(x < 0 for x in diffs), "n_seeds": len(diffs),
                           "p_signflip_exact": round(signflip_p(diffs), 6),
                           **{f"delta_s{s}": round(x, 6) for s, x in zip(SEEDS, diffs)}})
    for a, b, lab in [("H2", "H1", "CLAHE under cross-entropy")]:
        for d, k in DOMS.items():
            diffs = [info[(a, s)]["ev"][k]["metrics"]["ece"] - info[(b, s)]["ev"][k]["metrics"]["ece"] for s in SEEDS]
            pc.append({"metric": "ECE", "contrast": f"{a}−{b}", "meaning": lab, "domain": d,
                       "mean_delta": round(mean(diffs), 6), "sd_delta": round(stdev(diffs), 6),
                       "n_pos": sum(x > 0 for x in diffs), "n_neg": sum(x < 0 for x in diffs), "n_seeds": len(diffs),
                       "p_signflip_exact": round(signflip_p(diffs), 6),
                       **{f"delta_s{s}": round(x, 6) for s, x in zip(SEEDS, diffs)}})
    for m in {r["metric"] for r in pc}:
        idx = [i for i, r in enumerate(pc) if r["metric"] == m]
        adj = holm([pc[i]["p_signflip_exact"] for i in idx])
        for i, p in zip(idx, adj):
            pc[i]["p_holm_within_metric"] = round(p, 6)
            pc[i]["n_tests_in_family"] = len(idx)
    write("paired_contrasts_n6.csv", pc)

    # ---- schedule-matched comparators (H cells at seeds 45-47 share the full cosine schedule of v3 runs)
    smc = []
    for c in CELLS:
        for d in DOMS:
            v = [mrow[(c, s)][f"QWK_{d}"] for s in (45, 46, 47)]
            smc.append({"cell": c, "seeds": "45–47", "domain": d, "metric": "QWK", "value": f"{mean(v):.3f}±{stdev(v):.3f}"})
        v = [mrow[(c, s)]["APTOS_mild_recall"] for s in (45, 46, 47)]
        smc.append({"cell": c, "seeds": "45–47", "domain": "APTOS", "metric": "mild recall", "value": f"{mean(v):.3f}±{stdev(v):.3f}"})
        if c not in ORDINAL:
            v = [info[(c, s)]["ev"]["aptos_test"]["metrics"]["ece"] for s in (45, 46, 47)]
            smc.append({"cell": c, "seeds": "45–47", "domain": "APTOS", "metric": "ECE", "value": f"{mean(v):.3f}±{stdev(v):.3f}"})
    write("schedule_matched_comparators.csv", smc)

    # ---- ECE / temperature for the cross-entropy cells, six seeds
    ece_rows = []
    for c in ("H1", "H2"):
        for s in SEEDS:
            ev = info[(c, s)]["ev"]
            for d, k in DOMS.items():
                m = ev[k]["metrics"]
                ece_rows.append({"cell": c, "seed": s, "domain": d, "ece": round(m["ece"], 6),
                                 "ece_T": round(m["ece_temperature_scaled"], 6),
                                 "T": round(ev["temperature"]["value"], 6)})
    write("ece_ce_cells_n6_per_seed.csv", ece_rows)

    # ---- Messidor-2 share of grade-0 predictions, all 24 runs
    mp = []
    for c in CELLS:
        for s in SEEDS:
            cm = info[(c, s)]["ev"]["messidor2_external"]["metrics"]["confusion_matrix"]
            n = sum(sum(r) for r in cm)
            pred0 = sum(r[0] for r in cm)
            mp.append({"cell": c, "seed": s, "n": n, "pred_n0": pred0, "pred_frac_grade0": round(pred0 / n, 6),
                       **{f"true_n{g}": sum(cm[g]) for g in range(5)}})
    write("messidor_pred0_n6_per_run.csv", mp)
    mps = []
    for c in CELLS:
        for label, seeds in (("42–47", SEEDS), ("42–44", [42, 43, 44]), ("45–47", [45, 46, 47])):
            v = [r["pred_frac_grade0"] for r in mp if r["cell"] == c and r["seed"] in seeds]
            mps.append({"cell": c, "seeds": label, "pred_frac_grade0": f"{mean(v):.3f}±{stdev(v):.3f}"})
    write("messidor_pred0_n6_summary.csv", mps)

    print("READ (%d files):" % len(READ))
    for p in sorted(READ):
        print("  ", p)
    print("SUBMISSION_TABLES_OK")


if __name__ == "__main__":
    main()
