"""
Model definitions for DR grading (5-class ICDR 0–4).

Supports:
- EfficientNet-B0 + CE / CORAL head
- MobileViT-S (timm) CNN–Transformer hybrid + CE / CORAL head
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

MODEL_REGISTRY = {
    "efficientnet_b0": "EfficientNetB0",
    "mobilevit_s": "MobileViTS",
    "cnn_transformer_hybrid": "MobileViTS",
    "swin_tiny": "SwinTiny",
}


class CoralLayer(nn.Module):
    """
    CORAL ordinal head (Cao et al.): shared weight + (K-1) biases.

    Output logits_k ≈ P(Y > k) in logit space for k = 0..K-2.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORAL requires num_classes >= 2")
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(1, in_features))
        self.bias = nn.Parameter(torch.empty(num_classes - 1))
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.bias.copy_(
                torch.linspace(-(num_classes - 1) / 2.0, (num_classes - 1) / 2.0, num_classes - 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.weight.t()) + self.bias


def decode_predictions(logits: torch.Tensor, head: str = "ce") -> torch.Tensor:
    """Map model logits to integer DR grades."""
    key = head.lower().strip()
    if key in {"coral", "ordinal", "mild_aware_coral"}:
        return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()
    return logits.argmax(dim=1)


def logits_to_probs(logits: torch.Tensor, head: str = "ce") -> torch.Tensor:
    """
    Convert logits to class probabilities.

    CE: softmax.
    CORAL / mild-aware CORAL: differences of cumulative P(Y > k).
    """
    key = head.lower().strip()
    if key in {"coral", "ordinal", "mild_aware_coral"}:
        cum = torch.sigmoid(logits)  # (B, K-1) ≈ P(Y > k)
        bsz, km1 = cum.shape
        probs = torch.zeros(bsz, km1 + 1, device=logits.device, dtype=logits.dtype)
        probs[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, km1):
            probs[:, k] = cum[:, k - 1] - cum[:, k]
        probs[:, -1] = cum[:, -1]
        probs = probs.clamp(min=0.0)
        denom = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return probs / denom
    return F.softmax(logits, dim=1)


def _is_coral_head(head: str) -> bool:
    return head.lower().strip() in {"coral", "ordinal", "mild_aware_coral"}


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 with CE (K) or CORAL (K-1) head."""

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        head: str = "ce",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.num_classes = num_classes
        self.head = head.lower().strip()
        if _is_coral_head(self.head):
            self.head = "coral"
            self.backbone.classifier[1] = CoralLayer(in_features, num_classes)
        else:
            self.head = "ce"
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileViTS(nn.Module):
    """
    MobileViT-S CNN–Transformer hybrid via timm.

    Lightweight hybrid backbone for the paper's hybrid grid.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        head: str = "ce",
        timm_name: str = "mobilevit_s",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "MobileViT-S requires timm. Install with: pip install timm"
            ) from exc

        self.num_classes = num_classes
        self.head = head.lower().strip()
        coral = _is_coral_head(self.head)
        # Create without classifier; attach CE Linear or CoralLayer
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = int(self.backbone.num_features)
        if coral:
            self.head = "coral"
            self.classifier = CoralLayer(in_features, num_classes)
        else:
            self.head = "ce"
            self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)


class SwinTiny(nn.Module):
    """
    Swin-Tiny transformer baseline via timm (Phase O).

    Pure transformer counterpart to EfficientNet-B0 (CNN) and MobileViT-S (hybrid).
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        head: str = "ce",
        timm_name: str = "swin_tiny_patch4_window7_224",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "Swin-Tiny requires timm. Install with: pip install timm"
            ) from exc

        self.num_classes = num_classes
        self.head = head.lower().strip()
        coral = _is_coral_head(self.head)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = int(self.backbone.num_features)
        if coral:
            self.head = "coral"
            self.classifier = CoralLayer(in_features, num_classes)
        else:
            self.head = "ce"
            self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Factory: build a model by config name."""
    key = name.lower().strip()
    if key in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
        return EfficientNetB0(**kwargs)
    if key in {
        "mobilevit_s",
        "mobilevit-s",
        "mobilevits",
        "cnn_transformer_hybrid",
        "hybrid",
    }:
        return MobileViTS(**kwargs)
    if key in {"swin_tiny", "swin-tiny", "swintiny"}:
        return SwinTiny(**kwargs)
    raise ValueError(f"Unknown model {name!r}. Known: {list(MODEL_REGISTRY)}")
