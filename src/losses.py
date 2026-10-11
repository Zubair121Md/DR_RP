"""
Loss functions for ordinal DR grading.

Supported: Cross Entropy, Focal (stub), CORAL, mild-aware CORAL.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_REGISTRY = {
    "cross_entropy": "CrossEntropyLoss",
    "mild_weighted_ce": "MildWeightedCrossEntropyLoss",
    "focal": "FocalLoss",
    "coral": "CORALLoss",
    "mild_aware_coral": "MildAwareCORALLoss",
}


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy over DR grades 0–4."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


class MildWeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with an extra sample weight on mild (grade 1) images.

    Used to decompose mild-aware CORAL: isolate mild-sample weighting without
    an ordinal head or threshold weights.
    """

    def __init__(
        self,
        mild_class: int = 1,
        mild_sample_weight: float = 2.0,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.mild_class = int(mild_class)
        self.mild_sample_weight = float(mild_sample_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        w = torch.ones_like(per)
        w = w + (targets == self.mild_class).to(per.dtype) * (self.mild_sample_weight - 1.0)
        return (per * w).mean()


class FocalLoss(nn.Module):
    """Reserved for later experiments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        raise NotImplementedError("Focal Loss is deferred.")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CORALLoss(nn.Module):
    """
    CORAL ordinal loss (Cao et al.).

    For label y in {0..K-1}, binary targets are 1{y > k} for k=0..K-2.
    Logits must have shape (B, K-1).
    """

    def __init__(self, num_classes: int = 5, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.size(1) != self.num_classes - 1:
            raise ValueError(
                f"CORAL logits must be (B, {self.num_classes - 1}), got {tuple(logits.shape)}"
            )
        levels = (
            targets.unsqueeze(1)
            > torch.arange(self.num_classes - 1, device=targets.device, dtype=targets.dtype)
        ).float()
        return F.binary_cross_entropy_with_logits(logits, levels)


class MildAwareCORALLoss(nn.Module):
    """
    CORAL with elevated weights on mild-boundary thresholds and mild samples.

    Default: boost k=0 (Y>0) and k=1 (Y>1) — the cuts that separate No-DR / Mild /
    Moderate — and upweight rows where y==1 (Mild NPDR). Keeps the ordinal CORAL
    head; improves mild F1 without switching to a retitled CE story.
    """

    def __init__(
        self,
        num_classes: int = 5,
        mild_class: int = 1,
        mild_sample_weight: float = 2.0,
        threshold_weights: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.mild_class = int(mild_class)
        self.mild_sample_weight = float(mild_sample_weight)
        km1 = self.num_classes - 1
        if threshold_weights is None:
            # Boost first two cumulative tasks (around mild)
            tw = [2.0 if k < 2 else 1.0 for k in range(km1)]
        else:
            tw = list(threshold_weights)
            if len(tw) != km1:
                raise ValueError(f"threshold_weights length must be {km1}, got {len(tw)}")
        self.register_buffer(
            "threshold_weights",
            torch.tensor(tw, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.size(1) != self.num_classes - 1:
            raise ValueError(
                f"CORAL logits must be (B, {self.num_classes - 1}), got {tuple(logits.shape)}"
            )
        levels = (
            targets.unsqueeze(1)
            > torch.arange(self.num_classes - 1, device=targets.device, dtype=targets.dtype)
        ).float()
        # per-element BCE
        loss = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")
        tw = self.threshold_weights.to(device=loss.device, dtype=loss.dtype)
        loss = loss * tw.unsqueeze(0)
        sample_w = torch.ones(targets.size(0), device=targets.device, dtype=loss.dtype)
        sample_w = sample_w + (targets == self.mild_class).to(loss.dtype) * (
            self.mild_sample_weight - 1.0
        )
        loss = loss * sample_w.unsqueeze(1)
        return loss.mean()


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Factory: build a loss by config name."""
    key = name.lower().strip()
    if key in {"cross_entropy", "ce", "crossentropy"}:
        return CrossEntropyLoss(**kwargs)
    if key in {"mild_weighted_ce", "mild_weighted_cross_entropy", "ce_mild_weight"}:
        return MildWeightedCrossEntropyLoss(**kwargs)
    if key == "focal":
        return FocalLoss(**kwargs)
    if key == "coral":
        return CORALLoss(**kwargs)
    if key in {"mild_aware_coral", "mild-aware-coral", "mild_coral"}:
        return MildAwareCORALLoss(**kwargs)
    raise ValueError(f"Unknown loss {name!r}. Known: {list(LOSS_REGISTRY)}")
