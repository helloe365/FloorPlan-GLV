"""Typed tensor contracts shared by FloorPlan-GLV model components."""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import torch


class FeaturePyramid(NamedTuple):
    """Four local SegFormer stages for a 512 x 512 patch.

    Shapes:
        stage1: ``[M, 64, 128, 128]``.
        stage2: ``[M, 128, 64, 64]``.
        stage3: ``[M, 320, 32, 32]``.
        stage4: ``[M, 512, 16, 16]``.
    """

    stage1: torch.Tensor
    stage2: torch.Tensor
    stage3: torch.Tensor
    stage4: torch.Tensor


class FloorPlanModelOutput(TypedDict):
    """Raw stride-2 predictions for one flattened patch batch.

    Every tensor has shape ``[M, C, 256, 256]``. Binary logits and scalar
    regressions use ``C=1``; orientation and opening-type outputs use ``C=2``.
    Task-specific activations and transformations are intentionally excluded.
    """

    wall_mask_logits: torch.Tensor
    wall_centerline_logits: torch.Tensor
    wall_junction_logits: torch.Tensor
    wall_orientation_raw: torch.Tensor
    wall_log_half_thickness: torch.Tensor
    opening_mask_logits: torch.Tensor
    opening_center_logits: torch.Tensor
    opening_endpoint_logits: torch.Tensor
    opening_type_logits: torch.Tensor
    opening_orientation_raw: torch.Tensor
    opening_log_half_length: torch.Tensor


LOCAL_FEATURE_CONTRACT: tuple[tuple[int, int], ...] = (
    (64, 128),
    (128, 64),
    (320, 32),
    (512, 16),
)
