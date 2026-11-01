"""Loss functions with float32 numerical stability under mixed precision."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels: Sequence[int]) -> torch.Tensor:
    """Compute balanced class weights: w_c = N / (2 * n_c).

    MUST be computed strictly on inner training labels of each individual fit.
    """
    labels_arr = np.array(labels)
    n_samples = len(labels_arr)
    n_classes = 2

    counts = np.bincount(labels_arr, minlength=n_classes)
    if n_samples == 0 or len(counts) != n_classes or np.any(counts == 0):
        raise ValueError("Balanced cross-entropy requires non-empty inner-train labels from both classes")
    weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        weights[c] = n_samples / (n_classes * counts[c])

    return torch.tensor(weights, dtype=torch.float32)


class WeightedCrossEntropyLoss(nn.Module):
    """Numerically stable weighted cross entropy in float32."""

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=logits.device.type, enabled=False):
            return F.cross_entropy(logits.float(), targets, weight=self.weight)


class FocalLoss(nn.Module):
    """Numerically stable binary focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f32 = logits.float()
        ce_loss = F.cross_entropy(logits_f32, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            at = self.alpha[targets]
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
