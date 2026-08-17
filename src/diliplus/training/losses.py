"""Loss functions for the single-task AHI-proxy prediction problem."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnweightedFocalLoss(nn.Module):
    """Multiclass focal modulation without a class-weighting ``alpha``.

    For each sample, ``CE = -log(p_t)`` and
    ``loss = (1 - p_t) ** gamma * CE``.  The implementation deliberately has
    no scalar ``alpha``: multiplying every class by the same scalar is not
    class balancing.  Any future class weighting must be introduced as a
    separate, pre-specified method change.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        target_probability = torch.exp(-cross_entropy)
        loss = (1.0 - target_probability).pow(self.gamma) * cross_entropy
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
