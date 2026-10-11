#!/usr/bin/env python3
"""
Deployable-compute snapshot: params, approx FLOPs, MPS latency for EfficientNet-B0.
Supports CE and CORAL heads (research question efficiency block).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT))

from src.models import build_model  # noqa: E402


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_flops(model: torch.nn.Module, device: torch.device) -> float | None:
    try:
        from torchvision.models.feature_extraction import get_graph_node_names  # noqa: F401
    except Exception:
        pass
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        x = torch.randn(1, 3, 224, 224, device=device)
        with FlopCounterMode(display=False) as fc:
            model(x)
        return float(fc.get_total_flops())
    except Exception:
        try:
            from thop import profile  # type: ignore

            x = torch.randn(1, 3, 224, 224, device=device)
            flops, _ = profile(model, inputs=(x,), verbose=False)
            return float(flops)
        except Exception:
            return None


@torch.no_grad()
def latency_ms(model: torch.nn.Module, device: torch.device, warmup: int = 20, reps: int = 50) -> dict:
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    return {"batch1_ms_mean": 1000.0 * elapsed / reps, "reps": reps}


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for head in ("ce", "coral"):
        model = build_model("efficientnet_b0", num_classes=5, pretrained=False, head=head).to(device)
        flops = try_flops(model, device)
        rows.append(
            {
                "backbone": "efficientnet_b0",
                "head": head,
                "params": count_params(model),
                "params_m": round(count_params(model) / 1e6, 3),
                "flops": flops,
                "flops_g": None if flops is None else round(flops / 1e9, 3),
                "latency": latency_ms(model, device),
                "device": str(device),
            }
        )
        print(rows[-1])

    out = PROJECT / "results" / "efficiency_efficientnet_b0.json"
    out.write_text(
        json.dumps(
            {
                "note": "Deployable compute snapshot for B0 CE/CORAL heads",
                "models": rows,
            },
            indent=2,
        )
    )
    print("SAVED", out)


if __name__ == "__main__":
    main()
