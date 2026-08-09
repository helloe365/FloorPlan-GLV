"""Independent raw prediction heads for all FloorPlan-GLV V1 targets."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from floorplan_glv.models.types import FloorPlanModelOutput

INPUT_CHANNELS = 256
HIDDEN_CHANNELS = 64

OUTPUT_CHANNELS: dict[str, int] = {
    "wall_mask_logits": 1,
    "wall_centerline_logits": 1,
    "wall_junction_logits": 1,
    "wall_orientation_raw": 2,
    "wall_log_half_thickness": 1,
    "opening_mask_logits": 1,
    "opening_center_logits": 1,
    "opening_endpoint_logits": 1,
    "opening_type_logits": 2,
    "opening_orientation_raw": 2,
    "opening_log_half_length": 1,
}


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class PredictionHead(nn.Sequential):
    """Map shared decoder features to one unprocessed task tensor."""

    def __init__(self, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                INPUT_CHANNELS,
                HIDDEN_CHANNELS,
                kernel_size=3,
                padding=1,
            ),
            _group_norm(HIDDEN_CHANNELS),
            nn.GELU(),
            nn.Conv2d(HIDDEN_CHANNELS, output_channels, kernel_size=1),
        )


class PredictionHeads(nn.Module):
    """Produce the 11 raw V1 predictions from shared decoder features.

    Input shape: ``[M, 256, H, W]``.
    Output shapes: ``[M, C, H, W]``, where ``C`` is defined by
    :data:`OUTPUT_CHANNELS`. For the V1 decoder, ``H=W=256``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                key: PredictionHead(output_channels)
                for key, output_channels in OUTPUT_CHANNELS.items()
            }
        )

    def forward(self, shared_feature: torch.Tensor) -> FloorPlanModelOutput:
        """Return raw logits and regression fields without postprocessing."""
        if shared_feature.ndim != 4 or shared_feature.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                "shared_feature must have shape [M, 256, H, W], "
                f"got {tuple(shared_feature.shape)}"
            )
        output = {key: head(shared_feature) for key, head in self.heads.items()}
        return cast(FloorPlanModelOutput, output)
