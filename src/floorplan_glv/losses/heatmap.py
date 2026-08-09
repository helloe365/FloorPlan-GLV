"""CenterNet-style modified focal loss for point heatmaps."""

from __future__ import annotations

import torch


def _heatmap_valid_mask(
    valid_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.float32)
    if valid_mask.shape != reference.shape:
        raise ValueError("heatmap valid mask must match logits and targets")
    return valid_mask.to(dtype=torch.float32)


def centernet_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Return CenterNet modified focal loss accumulated in float32.

    A target value of exactly one is a positive center. Values below one are
    negatives weighted by their distance from the Gaussian center.
    """
    if logits.shape != targets.shape:
        raise ValueError("heatmap logits and targets must have the same shape")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = logits.float().sigmoid().clamp(1e-6, 1.0 - 1e-6)
        targets_float = targets.float()
        mask = _heatmap_valid_mask(valid_mask, probabilities)
        positive_mask = targets_float.eq(1.0).to(dtype=torch.float32) * mask
        negative_mask = targets_float.lt(1.0).to(dtype=torch.float32) * mask
        negative_weights = (1.0 - targets_float).pow(beta)
        positive_loss = (
            probabilities.log() * (1.0 - probabilities).pow(alpha) * positive_mask
        )
        negative_loss = (
            (1.0 - probabilities).log()
            * probabilities.pow(alpha)
            * negative_weights
            * negative_mask
        )
        positive_count = positive_mask.sum()
        normalized_loss = -(
            positive_loss.sum() + negative_loss.sum()
        ) / positive_count.clamp_min(1.0)
        negative_only_loss = -negative_loss.sum()
        return torch.where(
            positive_count > 0,
            normalized_loss,
            negative_only_loss,
        )
