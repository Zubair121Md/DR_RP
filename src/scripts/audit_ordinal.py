#!/usr/bin/env python3
"""
PHASE C — Audit CE / CORAL / MildAwareCORAL target generation and decoding.

No training. Writes experiments/results/ordinal_loss_audit.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.losses import CORALLoss, MildAwareCORALLoss, build_loss  # noqa: E402
from src.models import CoralLayer, decode_predictions, logits_to_probs  # noqa: E402


def coral_levels(y: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    return (
        y.unsqueeze(1)
        > torch.arange(num_classes - 1, device=y.device, dtype=y.dtype)
    ).float()


def main() -> None:
    issues = []
    warnings = []

    # --- Target generation ---
    expected = {
        0: [0, 0, 0, 0],
        1: [1, 0, 0, 0],
        2: [1, 1, 0, 0],
        3: [1, 1, 1, 0],
        4: [1, 1, 1, 1],
    }
    target_rows = {}
    for g, exp in expected.items():
        got = coral_levels(torch.tensor([g])).squeeze(0).tolist()
        target_rows[str(g)] = {"expected": exp, "got": got, "ok": got == exp}
        if got != exp:
            issues.append(f"CORAL target mismatch for grade {g}: {got} vs {exp}")

    # --- Decode monotonic cumulative logits (large positive/negative) ---
    decode_cases = []
    for bits, grade in [
        ([0, 0, 0, 0], 0),
        ([1, 0, 0, 0], 1),
        ([1, 1, 0, 0], 2),
        ([1, 1, 1, 0], 3),
        ([1, 1, 1, 1], 4),
    ]:
        # Map 0/1 to logits ±20 so sigmoid ≈ 0/1
        logits = torch.tensor([[20.0 if b else -20.0 for b in bits]])
        pred = int(decode_predictions(logits, head="coral").item())
        decode_cases.append({"bits": bits, "expected_grade": grade, "pred": pred, "ok": pred == grade})
        if pred != grade:
            issues.append(f"Decode failed for {bits}: got {pred} expected {grade}")

    # --- Non-monotonic ---
    nonmono = torch.tensor([[20.0, -20.0, 20.0, -20.0]])  # sigmoid ≈ [1,0,1,0]
    nm_pred = int(decode_predictions(nonmono, head="coral").item())
    nm_probs = logits_to_probs(nonmono, head="coral")
    nm_argmax = int(nm_probs.argmax(dim=1).item())
    nonmono_report = {
        "logits_pattern": [1, 0, 1, 0],
        "decode_sum_gt_0.5": nm_pred,
        "argmax_probs": nm_argmax,
        "probs": nm_probs.squeeze(0).tolist(),
        "decode_vs_argmax_agree": nm_pred == nm_argmax,
    }
    if nm_pred != nm_argmax:
        warnings.append(
            "Non-monotonic CORAL: decode_predictions (threshold-sum) disagrees with "
            "argmax(class probs). Inference uses threshold-sum; calibration uses probs. "
            "Document this; do not silently change without re-eval."
        )

    # --- MildAware vs CORAL ---
    y = torch.tensor([0, 1, 2, 3, 4])
    logits = torch.zeros(5, 4)
    coral = CORALLoss(num_classes=5)
    mild = MildAwareCORALLoss(
        num_classes=5, mild_sample_weight=2.0, threshold_weights=[2.0, 2.0, 1.0, 1.0]
    )
    loss_c = float(coral(logits, y).item())
    loss_m = float(mild(logits, y).item())
    # Element-wise unpack of mild weighting for documentation
    levels = coral_levels(y)
    bce = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")
    tw = torch.tensor([2.0, 2.0, 1.0, 1.0])
    sample_w = torch.ones(5) + (y == 1).float() * 1.0  # mild_sample_weight 2 → +1
    manual = float((bce * tw.unsqueeze(0) * sample_w.unsqueeze(1)).mean().item())

    mild_math = {
        "description": (
            "MildAwareCORAL = mean( BCE_none(logits, 1{y>k}) * threshold_weights[k] * sample_w[i] ). "
            "Default threshold_weights boost k=0 (Y>0) and k=1 (Y>1). "
            "sample_w = mild_sample_weight when y==mild_class else 1."
        ),
        "default_threshold_weights": [2.0, 2.0, 1.0, 1.0],
        "default_mild_sample_weight": 2.0,
        "mild_class": 1,
        "coral_loss_zero_logits": loss_c,
        "mild_loss_zero_logits": loss_m,
        "manual_mild_matches_impl": abs(manual - loss_m) < 1e-5,
        "differs_from_coral": abs(loss_m - loss_c) > 1e-6,
        "train_infer_consistent": (
            "Same CoralLayer head and decode_predictions as CORAL; "
            "only the training loss weighting differs."
        ),
    }
    if not mild_math["manual_mild_matches_impl"]:
        issues.append("MildAwareCORAL manual expansion does not match implementation")

    # --- Dtype / shape ---
    layer = CoralLayer(16, 5)
    z = torch.randn(3, 16)
    out = layer(z)
    shape_ok = list(out.shape) == [3, 4]

    # CE sanity
    ce = build_loss("cross_entropy")
    ce_loss = float(ce(torch.randn(4, 5), torch.tensor([0, 1, 2, 3])).item())

    report = {
        "note": "PHASE C ordinal / CE audit",
        "tasks": "P(y>=1), P(y>=2), P(y>=3), P(y>=4) via targets 1{y>k} for k=0..3",
        "target_generation": target_rows,
        "decode_monotonic": decode_cases,
        "non_monotonic": nonmono_report,
        "mild_aware_coral": mild_math,
        "coral_layer_out_shape_ok": shape_ok,
        "ce_loss_finite": ce_loss == ce_loss,
        "issues": issues,
        "warnings": warnings,
        "status": "PASS" if not issues else "BLOCKED",
    }

    out = PROJECT / "experiments" / "results" / "ordinal_loss_audit.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("SAVED", out)


if __name__ == "__main__":
    main()
