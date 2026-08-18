"""Validated multi-task loss aggregation and reporting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, cast

import torch
from pydantic import Field

from floorplan_glv.config.models import StrictConfigModel
from floorplan_glv.geometry.rasterize import TARGET_KEYS, TargetMaps
from floorplan_glv.losses.binary import binary_mask_loss, centerline_loss
from floorplan_glv.losses.geometry import (
    masked_cross_entropy_loss,
    masked_smooth_l1_loss,
    orientation_loss,
)
from floorplan_glv.losses.heatmap import centernet_focal_loss
from floorplan_glv.models.types import FloorPlanModelOutput

NonnegativeFloat = Annotated[
    float,
    Field(ge=0.0, allow_inf_nan=False),
]
PositiveFloat = Annotated[
    float,
    Field(gt=0.0, allow_inf_nan=False),
]
PositiveInt = Annotated[int, Field(gt=0)]
Probability = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]

TASK_NAMES: tuple[str, ...] = (
    "wall_mask",
    "wall_centerline",
    "wall_junction",
    "wall_orientation",
    "wall_thickness",
    "opening_mask",
    "opening_center",
    "opening_endpoint",
    "opening_type",
    "opening_orientation",
    "opening_length",
)

OUTPUT_KEYS: tuple[str, ...] = tuple(FloorPlanModelOutput.__annotations__)


class LossWeights(StrictConfigModel):
    """Validated Stage 2 task-loss weights from the data specification."""

    wall_mask: NonnegativeFloat = 2.0
    wall_centerline: NonnegativeFloat = 2.0
    wall_junction: NonnegativeFloat = 1.0
    wall_orientation: NonnegativeFloat = 0.5
    wall_thickness: NonnegativeFloat = 0.5
    opening_mask: NonnegativeFloat = 2.0
    opening_center: NonnegativeFloat = 3.0
    opening_endpoint: NonnegativeFloat = 1.0
    opening_type: NonnegativeFloat = 1.5
    opening_orientation: NonnegativeFloat = 0.75
    opening_length: NonnegativeFloat = 0.75


class LossConfig(StrictConfigModel):
    """Validated weights and numeric settings for multi-task loss computation."""

    weights: LossWeights = Field(default_factory=LossWeights)
    focal_alpha: Probability = 0.25
    focal_gamma: NonnegativeFloat = 2.0
    dice_smooth: PositiveFloat = 1.0
    cldice_iterations: PositiveInt = 20
    centernet_alpha: NonnegativeFloat = 2.0
    centernet_beta: NonnegativeFloat = 4.0
    smooth_l1_beta: PositiveFloat = 1.0
    orientation_unit_penalty_weight: NonnegativeFloat = 0.1
    opening_type_class_weights: tuple[PositiveFloat, PositiveFloat] | None = None


@dataclass(frozen=True, slots=True)
class LossReport:
    """One differentiable aggregate plus detached-friendly task breakdowns."""

    total: torch.Tensor
    raw_losses: dict[str, torch.Tensor]
    weighted_losses: dict[str, torch.Tensor]
    positive_counts: dict[str, torch.Tensor]


def _combined_validity(
    task_validity: torch.Tensor,
    valid_pixels: torch.Tensor,
) -> torch.Tensor:
    return task_validity.bool() & valid_pixels.bool()


def _positive_count(
    target: torch.Tensor,
    validity: torch.Tensor,
    *,
    exact_peak: bool = False,
) -> torch.Tensor:
    positive = target.eq(1.0) if exact_peak else target.gt(0.5)
    return (positive & validity.bool()).sum()


def _validate_boundary_keys(
    output: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> None:
    if set(output) != set(OUTPUT_KEYS):
        missing = sorted(set(OUTPUT_KEYS) - set(output))
        extra = sorted(set(output) - set(OUTPUT_KEYS))
        raise ValueError(
            f"output keys do not match contract; missing={missing}, extra={extra}"
        )
    if set(targets) != set(TARGET_KEYS):
        missing = sorted(set(TARGET_KEYS) - set(targets))
        extra = sorted(set(targets) - set(TARGET_KEYS))
        raise ValueError(
            f"target keys do not match contract; missing={missing}, extra={extra}"
        )


def _validate_finite_tensors(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    floating_tensors = tuple(
        (name, tensor) for name, tensor in named_tensors if tensor.is_floating_point()
    )
    finite_flags = torch.stack(
        tuple(torch.isfinite(tensor).all() for _, tensor in floating_tensors)
    )
    if bool(finite_flags.all()):
        return
    first_bad_index = int(torch.nonzero(~finite_flags, as_tuple=False)[0, 0].item())
    name = floating_tensors[first_bad_index][0]
    raise ValueError(f"{name} contains NaN or Infinity")


def compute_losses(
    output: FloorPlanModelOutput,
    targets: TargetMaps,
    config: LossConfig,
) -> LossReport:
    """Compute and validate all 11 V1 task losses.

    Args:
        output: Raw model tensors shaped ``[N, C, H, W]``.
        targets: Dense target tensors at the same spatial resolution.
        config: Validated task weights and loss numeric settings.

    Returns:
        Differentiable float32 total, raw/weighted task scalars, and positive
        pixel counts.
    """
    output_mapping = cast(Mapping[str, torch.Tensor], output)
    target_mapping = cast(Mapping[str, torch.Tensor], targets)
    _validate_boundary_keys(output_mapping, target_mapping)
    _validate_finite_tensors(
        tuple(
            (f"output tensor {key}", tensor) for key, tensor in output_mapping.items()
        )
        + tuple(
            (f"target tensor {key}", tensor) for key, tensor in target_mapping.items()
        )
    )

    valid_pixels = targets["valid_pixels"]
    wall_orientation_valid = _combined_validity(
        targets["wall_orientation_valid"],
        valid_pixels,
    )
    wall_thickness_valid = _combined_validity(
        targets["wall_thickness_valid"],
        valid_pixels,
    )
    opening_type_valid = _combined_validity(
        targets["opening_type_valid"],
        valid_pixels,
    )
    opening_orientation_valid = _combined_validity(
        targets["opening_orientation_valid"],
        valid_pixels,
    )
    opening_length_valid = _combined_validity(
        targets["opening_length_valid"],
        valid_pixels,
    )

    loss_functions: dict[str, Callable[[], torch.Tensor]] = {
        "wall_mask": lambda: binary_mask_loss(
            output["wall_mask_logits"],
            targets["wall_mask"],
            valid_pixels,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            dice_smooth=config.dice_smooth,
        ),
        "wall_centerline": lambda: centerline_loss(
            output["wall_centerline_logits"],
            targets["wall_centerline"],
            valid_pixels,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            dice_smooth=config.dice_smooth,
            cldice_iterations=config.cldice_iterations,
        ),
        "wall_junction": lambda: centernet_focal_loss(
            output["wall_junction_logits"],
            targets["wall_junction"],
            valid_pixels,
            alpha=config.centernet_alpha,
            beta=config.centernet_beta,
        ),
        "wall_orientation": lambda: orientation_loss(
            output["wall_orientation_raw"],
            targets["wall_orientation"],
            wall_orientation_valid,
            unit_penalty_weight=config.orientation_unit_penalty_weight,
            smooth_l1_beta=config.smooth_l1_beta,
        ),
        "wall_thickness": lambda: masked_smooth_l1_loss(
            output["wall_log_half_thickness"],
            targets["wall_log_half_thickness"],
            wall_thickness_valid,
            beta=config.smooth_l1_beta,
        ),
        "opening_mask": lambda: binary_mask_loss(
            output["opening_mask_logits"],
            targets["opening_mask"],
            valid_pixels,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            dice_smooth=config.dice_smooth,
        ),
        "opening_center": lambda: centernet_focal_loss(
            output["opening_center_logits"],
            targets["opening_center"],
            valid_pixels,
            alpha=config.centernet_alpha,
            beta=config.centernet_beta,
        ),
        "opening_endpoint": lambda: centernet_focal_loss(
            output["opening_endpoint_logits"],
            targets["opening_endpoint"],
            valid_pixels,
            alpha=config.centernet_alpha,
            beta=config.centernet_beta,
        ),
        "opening_type": lambda: masked_cross_entropy_loss(
            output["opening_type_logits"],
            targets["opening_type"],
            opening_type_valid,
            class_weights=config.opening_type_class_weights,
        ),
        "opening_orientation": lambda: orientation_loss(
            output["opening_orientation_raw"],
            targets["opening_orientation"],
            opening_orientation_valid,
            unit_penalty_weight=config.orientation_unit_penalty_weight,
            smooth_l1_beta=config.smooth_l1_beta,
        ),
        "opening_length": lambda: masked_smooth_l1_loss(
            output["opening_log_half_length"],
            targets["opening_log_half_length"],
            opening_length_valid,
            beta=config.smooth_l1_beta,
        ),
    }
    enabled_tasks = tuple(
        task_name
        for task_name in TASK_NAMES
        if float(getattr(config.weights, task_name)) > 0.0
    )
    if not enabled_tasks:
        raise ValueError("at least one task loss weight must be positive")
    zero_loss = next(iter(output_mapping.values())).float().new_zeros(())
    raw_losses = {
        task_name: loss_functions[task_name]()
        if task_name in enabled_tasks
        else zero_loss.clone()
        for task_name in TASK_NAMES
    }
    for task_name, loss in raw_losses.items():
        if loss.dtype != torch.float32:
            raise ValueError(f"loss {task_name} was not accumulated in float32")
    _validate_finite_tensors(
        tuple((f"loss {task_name}", loss) for task_name, loss in raw_losses.items())
    )

    weighted_losses = {
        task_name: raw_losses[task_name] * float(getattr(config.weights, task_name))
        for task_name in TASK_NAMES
    }
    total = torch.stack(tuple(weighted_losses.values())).sum()
    positive_counts = {
        "wall_mask": _positive_count(
            targets["wall_mask"],
            valid_pixels,
        ),
        "wall_centerline": _positive_count(
            targets["wall_centerline"],
            valid_pixels,
        ),
        "wall_junction": _positive_count(
            targets["wall_junction"],
            valid_pixels,
            exact_peak=True,
        ),
        "wall_orientation": wall_orientation_valid.sum(),
        "wall_thickness": wall_thickness_valid.sum(),
        "opening_mask": _positive_count(
            targets["opening_mask"],
            valid_pixels,
        ),
        "opening_center": _positive_count(
            targets["opening_center"],
            valid_pixels,
            exact_peak=True,
        ),
        "opening_endpoint": _positive_count(
            targets["opening_endpoint"],
            valid_pixels,
            exact_peak=True,
        ),
        "opening_type": opening_type_valid.sum(),
        "opening_orientation": opening_orientation_valid.sum(),
        "opening_length": opening_length_valid.sum(),
    }
    return LossReport(
        total=total,
        raw_losses=raw_losses,
        weighted_losses=weighted_losses,
        positive_counts=positive_counts,
    )
