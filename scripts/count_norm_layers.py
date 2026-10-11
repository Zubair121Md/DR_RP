#!/usr/bin/env python3
"""Count normalisation layers in each backbone exactly as built by src/models.py.

`freeze_bn: true` in src/train.py puts BatchNorm1d/2d modules in eval mode; LayerNorm and
GroupNorm modules are unaffected. Writes paper/tables/normalisation_layers.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model  # noqa: E402

OUT = ROOT / "paper" / "tables" / "normalisation_layers.csv"


def main() -> None:
    rows = []
    for name, label in (("mobilevit_s", "MobileViT-S"), ("efficientnet_b0", "EfficientNet-B0"), ("swin_tiny", "Swin-Tiny")):
        m = build_model(name, num_classes=5, pretrained=False, head="ce")
        bn = sum(isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d)) for x in m.modules())
        ln = sum(isinstance(x, nn.LayerNorm) for x in m.modules())
        gn = sum(isinstance(x, nn.GroupNorm) for x in m.modules())
        cfg = getattr(getattr(m, "backbone", None), "pretrained_cfg", None)
        if cfg:
            weights = f"timm {cfg.get('architecture')}.{cfg.get('tag')}"
        else:
            from torchvision.models import EfficientNet_B0_Weights
            weights = f"torchvision EfficientNet_B0_Weights.{EfficientNet_B0_Weights.IMAGENET1K_V1.name}"
        rows.append({"backbone": label, "weights": weights, "batchnorm_layers": bn, "layernorm_layers": ln,
                     "groupnorm_layers": gn, "affected_by_freeze_bn": bn > 0})
        print(label, weights, "BN", bn, "LN", ln, "GN", gn)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("READ src/models.py (timm / torchvision definitions, no weights)")
    print("WROTE", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
