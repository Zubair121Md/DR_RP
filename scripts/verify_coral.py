#!/usr/bin/env python3
"""Verify that the ordinal head used by H3/H4 (and matched B0 H3/H4, D1) is CORAL.

Checks, from code + checkpoints + saved predictions (no training):
  1. Head structure: one shared weight vector (1 x F) + K-1 = 4 independent biases.
  2. Loss targets 1{y > k} for k = 0..3; threshold weights and mild sample weight from config.
  3. Decoding = number of sigmoid(z_k) > 0.5 (recomputed from stored logits).
  4. Calibration probabilities = successive differences of cumulative sigmoids (recomputed).
  5. Learned bias ordering (rank consistency) and frequency of non-monotonic cumulative
     probabilities on every test set; how non-monotonic rows are handled in code.

Writes paper/tables/coral_verification_runs.csv and paper/tables/coral_verification.md.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import losses as L  # noqa: E402
from src import models as M  # noqa: E402

OUT = ROOT / "paper" / "tables"
SPLITS = {
    "APTOS": "aptos_test_predictions.json",
    "IDRiD": "idrid_external_predictions.json",
    "Messidor-2": "messidor2_external_predictions.json",
    "DeepDRiD": "deepdrid_external_predictions.json",
}
READ: list[str] = []


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def runs() -> list[tuple[str, str, int, str]]:
    out = []
    for c in ("h3", "h4"):
        for s in (42, 43, 44):
            out.append((f"hybrid_{c}_s{s}_v1", c.upper(), s, "MobileViT-S"))
        for s in (45, 46, 47):
            out.append((f"hybrid_{c}_s{s}_v3", c.upper(), s, "MobileViT-S"))
        for s in (42, 43, 44):
            out.append((f"matched_b0_{c}_s{s}_v3", c.upper(), s, "EfficientNet-B0"))
    for s in (42, 43, 44):
        out.append((f"hybrid_d1_coral_clahe_s{s}_v3", "D1", s, "MobileViT-S"))
    return out


def structural_checks() -> list[tuple[str, bool, str]]:
    res = []
    layer = M.CoralLayer(640, 5)
    shapes = {n: tuple(p.shape) for n, p in layer.named_parameters()}
    res.append(("CoralLayer parameters", shapes == {"weight": (1, 640), "bias": (4,)},
                f"{shapes} (shared weight vector, 4 threshold biases)"))
    x = torch.randn(3, 640)
    z = layer(x)
    shared = torch.allclose(z - layer.bias, (x @ layer.weight.t()).expand(-1, 4))
    res.append(("All thresholds share w^T x", bool(shared), "z_k - b_k identical across k"))
    init = layer.bias.detach().numpy().round(3).tolist()
    res.append(("Bias initialisation", True, f"{init} (increasing; ordering is learned)"))

    y = torch.tensor([0, 1, 2, 3, 4])
    levels = (y.unsqueeze(1) > torch.arange(4).unsqueeze(0)).float()
    exp = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]).float()
    src = inspect.getsource(L.MildAwareCORALLoss) + inspect.getsource(L.CORALLoss)
    res.append(("Targets 1{y>k}", bool(torch.equal(levels, exp)) and "torch.arange(self.num_classes - 1" in src,
                "levels = y.unsqueeze(1) > arange(K-1) in CORALLoss and MildAwareCORALLoss"))

    lossf = L.MildAwareCORALLoss(num_classes=5, threshold_weights=[2, 2, 1, 1], mild_sample_weight=2.0)
    logits = torch.zeros(2, 4)
    tgt = torch.tensor([1, 0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, (tgt.unsqueeze(1) > torch.arange(4)).float(), reduction="none")
    manual = (bce * torch.tensor([2, 2, 1, 1.]) * torch.tensor([[2.], [1.]])).mean()
    got = lossf(logits, tgt)
    res.append(("Mild-aware weighting", bool(torch.allclose(manual, got)),
                f"loss={float(got):.6f} manual={float(manual):.6f}; w_thr=(2,2,1,1), w_samp=2 for y=1; mean over B*(K-1)"))

    zz = torch.tensor([[2.0, 1.0, -1.0, -2.0], [-1.0, 1.0, 2.0, -3.0]])
    dec = M.decode_predictions(zz, "coral").tolist()
    res.append(("Decoding = count(sigmoid>0.5)", dec == [2, 2], f"{dec}"))
    p = M.logits_to_probs(zz, "coral")
    c = torch.sigmoid(zz)
    raw = torch.stack([1 - c[:, 0], c[:, 0] - c[:, 1], c[:, 1] - c[:, 2], c[:, 2] - c[:, 3], c[:, 3]], 1)
    manual_p = raw.clamp(min=0)
    manual_p = manual_p / manual_p.sum(1, keepdim=True)
    res.append(("Probabilities = cumulative differences", bool(torch.allclose(p, manual_p)),
                "p0=1-s0, pk=s(k-1)-s(k), p4=s3; negatives clipped to 0, row renormalised"))
    return res


def bias_key(sd: dict) -> str:
    for k in ("classifier.bias", "backbone.classifier.1.bias"):
        if k in sd:
            return k
    raise KeyError("no CORAL bias")


def main() -> None:
    rows = []
    for run, cell, seed, arch in runs():
        cfgp = ROOT / "experiments" / run / "config.yaml"
        ck = ROOT / "checkpoints" / run / "best.pt"
        if not cfgp.is_file() or not ck.is_file():
            raise SystemExit(f"MISSING input for {run}: {cfgp if not cfgp.is_file() else ck}")
        READ.extend([rel(cfgp), rel(ck)])
        cfg = yaml.safe_load(cfgp.read_text())
        state = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
        bk = bias_key(state)
        b = state[bk].numpy().astype(float)
        wk = bk.replace("bias", "weight")
        wshape = tuple(state[wk].shape)
        row = {
            "run": run, "arch": arch, "cell": cell, "seed": seed,
            "loss": cfg["loss"]["name"],
            "threshold_weights": json.dumps(cfg["loss"].get("threshold_weights")),
            "mild_sample_weight": cfg["loss"].get("mild_sample_weight"),
            "weight_shape": "x".join(map(str, wshape)),
            "bias_b0": b[0], "bias_b1": b[1], "bias_b2": b[2], "bias_b3": b[3],
            "biases_nonincreasing": bool(np.all(np.diff(b) <= 0)),
        }
        for dom, fn in SPLITS.items():
            pp = ROOT / "results" / run / fn
            if not pp.is_file():
                raise SystemExit(f"MISSING input: {rel(pp)}")
            READ.append(rel(pp))
            d = json.loads(pp.read_text())
            z = torch.tensor(np.asarray(d["logits"], dtype=np.float32))
            s = torch.sigmoid(z).numpy()
            nonmono = np.any(np.diff(s, axis=1) > 1e-7, axis=1)
            dec = M.decode_predictions(z, "coral").numpy()
            probs = M.logits_to_probs(z, "coral").numpy()
            row[f"{dom}_n"] = len(dec)
            row[f"{dom}_nonmono_frac"] = float(nonmono.mean())
            row[f"{dom}_decode_matches_saved"] = bool(np.array_equal(dec, np.asarray(d["y_pred"])))
            row[f"{dom}_probs_match_saved"] = bool(np.allclose(probs, np.asarray(d["probs"]), atol=1e-5))
        rows.append(row)

    import csv
    OUT.mkdir(parents=True, exist_ok=True)
    csvp = OUT / "coral_verification_runs.csv"
    with csvp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    st = structural_checks()
    all_dec = all(r[f"{d}_decode_matches_saved"] for r in rows for d in SPLITS)
    all_p = all(r[f"{d}_probs_match_saved"] for r in rows for d in SPLITS)
    n_mono = sum(r["biases_nonincreasing"] for r in rows)
    max_nm = max(r[f"{d}_nonmono_frac"] for r in rows for d in SPLITS)
    lines = ["# CORAL implementation verification", "",
             "Generated by `scripts/verify_coral.py` from `src/models.py`, `src/losses.py`, run configs, "
             "`best.pt` checkpoints and saved test logits.", "", "## Code checks", "",
             "| Check | Pass | Detail |", "|---|---|---|"]
    for name, ok, det in st:
        lines.append(f"| {name} | {'yes' if ok else 'NO'} | {det} |")
    lines += ["", "## Learned checkpoints and saved predictions", "",
              f"- Runs checked: {len(rows)} (MobileViT-S H3/H4 seeds 42–47, matched EfficientNet-B0 H3/H4 seeds 42–44, D1 seeds 42–44).",
              f"- Learned threshold biases non-increasing (b0 >= b1 >= b2 >= b3): {n_mono}/{len(rows)} runs.",
              f"- Maximum fraction of test images with non-monotonic cumulative probabilities, any run and test set: {max_nm:.4f}.",
              f"- Decoded grades recomputed from stored logits match saved predictions: {all_dec}.",
              f"- Probabilities recomputed from stored logits match saved probabilities: {all_p}.",
              "- Non-monotonic handling in `logits_to_probs`: negative differences are clipped to 0 and each row is renormalised to sum to 1. Decoding counts thresholds above 0.5 and does not require monotonicity.",
              "", "Conclusion: the ordinal head is a CORAL layer (shared weight vector, four free threshold biases) "
              "trained with binary cross-entropy on targets 1{y>k}; the mild-aware variant multiplies the per-threshold "
              "terms by fixed task weights (2,2,1,1) and by a per-sample weight of 2 for grade-1 images."]
    mdp = OUT / "coral_verification.md"
    mdp.write_text("\n".join(lines) + "\n")
    print("READ:", "src/models.py", "src/losses.py", *sorted(set(READ)), sep="\n  ")
    print("WROTE:", rel(csvp), rel(mdp))
    print("\n".join(lines))
    if not all(ok for _, ok, _ in st) or not all_dec or not all_p:
        raise SystemExit("CORAL_VERIFY_FAIL")
    print("CORAL_VERIFY_OK")


if __name__ == "__main__":
    main()
