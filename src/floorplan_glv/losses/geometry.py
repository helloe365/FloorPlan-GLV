"""Masked geometry regression, orientation, and opening-type losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional


def _expanded_mask(
    valid_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
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
            "geometry valid mask must match the prediction or have one channel"
        )
    return expanded.to(dtype=torch.float32)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Return masked SmoothL1 in float32, including differentiable empty zero."""
    if prediction.shape != target.shape:
        raise ValueError("SmoothL1 prediction and target must have the same shape")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_float = prediction.float()
        target_float = target.float()
        mask = _expanded_mask(valid_mask, prediction_float)
        values = functional.smooth_l1_loss(
            prediction_float,
            target_float,
            reduction="none",
            beta=beta,
        )
        return _masked_mean(values, mask)


def orientation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    unit_penalty_weight: float = 0.1,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    """Return normalized orientation SmoothL1 plus raw unit-norm penalty.

    Prediction and target shapes are ``[N, 2, H, W]``. The target represents
    ``sin(2 theta), cos(2 theta)`` and validity has one or two channels.
    """
    if prediction.shape != target.shape:
        raise ValueError("orientation prediction and target shapes must match")
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("orientation tensors must have shape [N, 2, H, W]")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_float = prediction.float()
        target_float = target.float()
        normalized = functional.normalize(
            prediction_float,
            dim=1,
            eps=1e-6,
        )
        component_mask = _expanded_mask(valid_mask, normalized)
        regression = _masked_mean(
            functional.smooth_l1_loss(
                normalized,
                target_float,
                reduction="none",
                beta=smooth_l1_beta,
            ),
            component_mask,
        )
        pixel_mask = valid_mask
        if pixel_mask.shape[1] == 2:
            pixel_mask = pixel_mask[:, :1]
        pixel_mask_float = pixel_mask.to(dtype=torch.float32)
        norm_penalty = _masked_mean(
            (torch.linalg.vector_norm(prediction_float, dim=1, keepdim=True) - 1.0).pow(
                2
            ),
            pixel_mask_float,
        )
        return regression + unit_penalty_weight * norm_penalty


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    class_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return masked opening-type cross entropy in float32."""
    if logits.ndim != 4:
        raise ValueError("type logits must have shape [N, C, H, W]")
    expected_target_shape = (logits.shape[0], *logits.shape[2:])
    if tuple(target.shape) != expected_target_shape:
        raise ValueError(
            f"type target must have shape {expected_target_shape}, "
            f"got {tuple(target.shape)}"
        )
    if valid_mask.shape == (logits.shape[0], 1, *logits.shape[2:]):
        valid_pixels = valid_mask[:, 0]
    elif tuple(valid_mask.shape) == expected_target_shape:
        valid_pixels = valid_mask
    else:
        raise ValueError("type valid mask must have shape [N, 1, H, W] or [N, H, W]")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_float = logits.float()
        valid_pixels = valid_pixels.bool()
        safe_target = torch.where(
            valid_pixels,
            target.to(dtype=torch.int64),
            torch.zeros_like(target, dtype=torch.int64),
        )
        weight_tensor = (
            torch.tensor(
                tuple(class_weights),
                dtype=torch.float32,
                device=logits.device,
            )
            if class_weights is not None
            else None
        )
        values = functional.cross_entropy(
            logits_float,
            safe_target,
            weight=weight_tensor,
            reduction="none",
        )
        valid_float = valid_pixels.to(dtype=torch.float32)
        denominator = (
            (weight_tensor[safe_target] * valid_float).sum()
            if weight_tensor is not None
            else valid_float.sum()
        )
        return (values * valid_float).sum() / denominator.clamp_min(1.0)
