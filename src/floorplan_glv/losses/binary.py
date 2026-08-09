"""Binary mask, Dice, and topology-aware centerline losses."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional


def _valid_mask(
    valid_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.float32)
    if valid_mask.shape == reference.shape:
        expanded = valid_mask
    elif (
        valid_mask.ndim == reference.ndim
        and valid_mask.shape[0] == reference.shape[0]
        and valid_mask.shape[1] == 1
        and valid_mask.shape[2:] == reference.shape[2:]
    ):
        expanded = valid_mask.expand_as(reference)
    else:
        raise ValueError(
            "valid_mask must match the prediction shape or have one channel"
        )
    return expanded.to(dtype=torch.float32)


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return alpha-balanced focal BCE accumulated in float32.

    Args:
        logits: Raw binary logits ``[N, C, H, W]``.
        targets: Binary or soft targets with the same shape.
        valid_mask: Optional boolean mask with one or ``C`` channels.
        alpha: Positive-class balancing factor.
        gamma: Focusing exponent.
    """
    if logits.shape != targets.shape:
        raise ValueError("focal BCE logits and targets must have the same shape")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_float = logits.float()
        targets_float = targets.float()
        mask = _valid_mask(valid_mask, logits_float)
        binary_cross_entropy = functional.binary_cross_entropy_with_logits(
            logits_float,
            targets_float,
            reduction="none",
        )
        probabilities = logits_float.sigmoid()
        probability_target = probabilities * targets_float + (1.0 - probabilities) * (
            1.0 - targets_float
        )
        alpha_target = alpha * targets_float + (1.0 - alpha) * (1.0 - targets_float)
        values = (
            alpha_target * (1.0 - probability_target).pow(gamma) * binary_cross_entropy
        )
        return cast(
            torch.Tensor,
            (values * mask).sum() / mask.sum().clamp_min(1.0),
        )


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Return soft Dice loss for raw logits in float32."""
    if logits.shape != targets.shape:
        raise ValueError("Dice logits and targets must have the same shape")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = logits.float().sigmoid()
        targets_float = targets.float()
        mask = _valid_mask(valid_mask, probabilities)
        probabilities = probabilities * mask
        targets_float = targets_float * mask
        intersection = (probabilities * targets_float).sum()
        denominator = probabilities.sum() + targets_float.sum()
        return cast(
            torch.Tensor,
            1.0 - (2.0 * intersection + smooth) / (denominator + smooth),
        )


def _soft_erode(tensor: torch.Tensor) -> torch.Tensor:
    return -functional.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)


def _soft_open(tensor: torch.Tensor) -> torch.Tensor:
    eroded = _soft_erode(tensor)
    return functional.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)


def _soft_skeleton(tensor: torch.Tensor, iterations: int) -> torch.Tensor:
    opened = _soft_open(tensor)
    skeleton = functional.relu(tensor - opened)
    eroded = tensor
    for _ in range(iterations):
        eroded = _soft_erode(eroded)
        opened = _soft_open(eroded)
        delta = functional.relu(eroded - opened)
        skeleton = skeleton + functional.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    smooth: float = 1.0,
    iterations: int = 20,
) -> torch.Tensor:
    """Return differentiable centerline Dice loss for raw logits.

    Inputs have shape ``[N, 1, H, W]``. Soft skeletonization is restricted to
    valid pixels and accumulated in float32.
    """
    if logits.shape != targets.shape:
        raise ValueError("clDice logits and targets must have the same shape")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("clDice expects tensors shaped [N, 1, H, W]")
    if iterations <= 0:
        raise ValueError("clDice iterations must be positive")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = logits.float().sigmoid()
        targets_float = targets.float()
        mask = _valid_mask(valid_mask, probabilities)
        probabilities = probabilities * mask
        targets_float = targets_float * mask
        prediction_skeleton = _soft_skeleton(probabilities, iterations)
        target_skeleton = _soft_skeleton(targets_float, iterations)
        topology_precision = ((prediction_skeleton * targets_float).sum() + smooth) / (
            prediction_skeleton.sum() + smooth
        )
        topology_sensitivity = ((target_skeleton * probabilities).sum() + smooth) / (
            target_skeleton.sum() + smooth
        )
        return cast(
            torch.Tensor,
            1.0
            - (
                2.0
                * topology_precision
                * topology_sensitivity
                / (topology_precision + topology_sensitivity).clamp_min(1e-12)
            ),
        )


def binary_mask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
    dice_smooth: float,
) -> torch.Tensor:
    """Return the specified equal Focal BCE and Soft Dice mixture."""
    return 0.5 * focal_bce_loss(
        logits,
        targets,
        valid_mask,
        alpha=focal_alpha,
        gamma=focal_gamma,
    ) + 0.5 * soft_dice_loss(
        logits,
        targets,
        valid_mask,
        smooth=dice_smooth,
    )


def centerline_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
    dice_smooth: float,
    cldice_iterations: int,
) -> torch.Tensor:
    """Return the specified Focal BCE, Soft Dice, and clDice mixture."""
    focal = focal_bce_loss(
        logits,
        targets,
        valid_mask,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )
    dice = soft_dice_loss(
        logits,
        targets,
        valid_mask,
        smooth=dice_smooth,
    )
    topology = cldice_loss(
        logits,
        targets,
        valid_mask,
        smooth=dice_smooth,
        iterations=cldice_iterations,
    )
    return 0.25 * focal + 0.25 * dice + 0.50 * topology
