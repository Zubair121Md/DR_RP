#!/usr/bin/env python3
"""
PHASE B — Synthetic audit of MobileViT-S / EfficientNet-B0 heads.

Does not train. Does not overwrite checkpoints.
Writes experiments/results/mv_hybrid_audit.json (+ human md written separately).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.losses import build_loss  # noqa: E402
from src.models import build_model, decode_predictions, logits_to_probs  # noqa: E402


def _count_modules(model: nn.Module) -> dict[str, int]:
    return {
        "BatchNorm2d": sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d)),
        "BatchNorm1d": sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d)),
        "LayerNorm": sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm)),
        "GroupNorm": sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm)),
        "Dropout": sum(1 for m in model.modules() if isinstance(m, nn.Dropout)),
    }


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item())
    return total


def audit_forward_backward(name: str, head: str, batch: int) -> dict:
    device = torch.device("cpu")
    model = build_model(name, num_classes=5, pretrained=False, head=head).to(device)
    model.train()
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, 5, (batch,), device=device)
    loss_name = "coral" if head == "coral" else "cross_entropy"
    criterion = build_loss(loss_name, num_classes=5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    gnorm = _grad_norm(model)
    opt.step()

    changed = 0
    total_p = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total_p += 1
        if not torch.allclose(p.detach(), before[n]):
            changed += 1

    preds = decode_predictions(logits.detach(), head=head)
    probs = logits_to_probs(logits.detach(), head=head)
    expected_out = 4 if head == "coral" else 5

    # BN running-stat probe: evaluate vs train mode with same weights after 1 step
    model.eval()
    with torch.no_grad():
        logits_eval = model(x)
    model.train()
    with torch.no_grad():
        logits_train = model(x)
    train_eval_diff = float((logits_train - logits_eval).abs().mean().item())

    return {
        "model": name,
        "head": head,
        "batch": batch,
        "logits_shape": list(logits.shape),
        "expected_logits_dim": expected_out,
        "shape_ok": logits.shape == (batch, expected_out),
        "loss": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "grad_norm": gnorm,
        "grad_nonzero": gnorm > 0,
        "params_updated": changed,
        "params_trainable": total_p,
        "preds_shape": list(preds.shape),
        "probs_shape": list(probs.shape),
        "probs_sum_mean": float(probs.sum(dim=1).mean().item()),
        "module_counts": _count_modules(model),
        "num_features": int(getattr(model, "backbone", model).num_features)
        if hasattr(getattr(model, "backbone", model), "num_features")
        else None,
        "train_eval_logit_absdiff_mean": train_eval_diff,
        "nan_in_logits": bool(torch.isnan(logits).any().item()),
        "inf_in_logits": bool(torch.isinf(logits).any().item()),
    }


def main() -> None:
    rows = []
    for name in ("mobilevit_s", "efficientnet_b0"):
        for head in ("ce", "coral"):
            for batch in (1, 8, 32):
                rows.append(audit_forward_backward(name, head, batch))

    # Parameter freeze check: all params require_grad by default
    m = build_model("mobilevit_s", num_classes=5, pretrained=False, head="ce")
    frozen = sum(1 for p in m.parameters() if not p.requires_grad)
    trainable = sum(1 for p in m.parameters() if p.requires_grad)

    report = {
        "note": "PHASE B synthetic MV/hybrid audit — no training, no checkpoint overwrite",
        "definition": {
            "MV": "timm mobilevit_s + Linear/CoralLayer",
            "cnn_transformer_hybrid_alias": "MobileViTS",
            "swin_tiny": "NotImplementedError",
        },
        "params": {"trainable": trainable, "frozen": frozen},
        "cases": rows,
        "critical_findings": [],
        "warnings": [],
    }

    # Flag batch-1 BN risk if BN present and train/eval diverge strongly on tiny batches
    mv_bn = next(r for r in rows if r["model"] == "mobilevit_s" and r["head"] == "ce" and r["batch"] == 1)
    if mv_bn["module_counts"]["BatchNorm2d"] > 0:
        report["critical_findings"].append(
            {
                "id": "MV_BN_BATCH1",
                "file": "src/models.py :: MobileViTS",
                "problem": (
                    f"MobileViT-S contains {mv_bn['module_counts']['BatchNorm2d']} BatchNorm2d layers. "
                    "Historical hybrid *_stable runs used batch_size=1 on CPU. "
                    "BatchNorm with batch=1 yields ill-defined batch statistics and corrupts running stats."
                ),
                "scientific_consequence": (
                    "All 6 MobileViT seed-42 stable external results are protocol-confounded "
                    "and MUST NOT be used for architecture claims. Mark REQUIRES RERUN with batch>=16."
                ),
                "proposed_correction": (
                    "Retrain hybrid grid with batch_size>=16 (prefer 32 to match B0), GPU/MPS if available; "
                    "or document SyncBatchNorm/GroupNorm alternative — do not silently change architecture."
                ),
                "experiments_affected": [
                    "hybrid_mobilevit_s_*_s42_stable (all 6 cells)",
                    "results/external_eval_sweep.json rows for those runs",
                ],
            }
        )

    shape_fail = [r for r in rows if not r["shape_ok"] or not r["loss_finite"] or not r["grad_nonzero"]]
    if shape_fail:
        report["critical_findings"].append(
            {
                "id": "FORWARD_FAIL",
                "problem": "One or more forward/backward checks failed",
                "cases": shape_fail,
            }
        )
    else:
        report["warnings"].append("All synthetic CE/CORAL forward-backward checks passed for batch in {1,8,32}.")

    # train/eval gap larger at batch 1 is expected with BN
    b1 = mv_bn["train_eval_logit_absdiff_mean"]
    b32 = next(
        r for r in rows if r["model"] == "mobilevit_s" and r["head"] == "ce" and r["batch"] == 32
    )["train_eval_logit_absdiff_mean"]
    report["bn_train_eval_gap"] = {"batch1": b1, "batch32": b32}

    out = PROJECT / "experiments" / "results" / "mv_hybrid_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("SAVED", out)


if __name__ == "__main__":
    main()
