"""Acceptance-only deterministic components injected into :class:`FloorPlanGLV`."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from floorplan_glv.config.models import ModelConfig
from floorplan_glv.models.model import FloorPlanGLV
from floorplan_glv.models.types import FeaturePyramid, FloorPlanModelOutput

ACCEPTANCE_IMPLEMENTATION = "deterministic_floorplan_glv_v2"
_LOGIT_MAGNITUDE = 8.0


class _AcceptanceGlobalEncoder(nn.Module):
    forward_count: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.int64))

    def forward(self, images: torch.Tensor) -> FeaturePyramid:
        if self.forward_count.item() != 0:
            raise RuntimeError("acceptance global encoder called more than once")
        self.forward_count.add_(1)
        gate = (images.amin(dim=(1, 2, 3), keepdim=True) < 0.0).to(images.dtype)
        return FeaturePyramid(gate, gate, gate, gate)


class _AcceptanceLocalEncoder(nn.Module):
    def forward(self, patches: torch.Tensor) -> FeaturePyramid:
        pixels = functional.avg_pool2d(patches, kernel_size=2, stride=2)
        stage = torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=1)
        return FeaturePyramid(stage, stage, stage, stage)


class _AcceptanceFusion(nn.Module):
    def forward(
        self,
        local_feature: torch.Tensor,
        global_feature: torch.Tensor,
        patch_to_image: torch.Tensor,
        patch_boxes_global_xyxy: torch.Tensor,
    ) -> torch.Tensor:
        if patch_boxes_global_xyxy.shape != (local_feature.shape[0], 4):
            raise ValueError("acceptance patch boxes must have shape [M, 4]")
        gate = global_feature[patch_to_image]
        return local_feature * gate


class _AcceptanceDecoder(nn.Module):
    def forward(
        self,
        features: FeaturePyramid,
        rgb_patch: torch.Tensor,
    ) -> torch.Tensor:
        if features.stage1.shape[0] != rgb_patch.shape[0]:
            raise ValueError("acceptance features and RGB patches must share a batch")
        return sum(
            (feature[:, :3] for feature in features[1:]),
            start=features[0][:, :3],
        ) / float(len(features))


class _AcceptanceHeads(nn.Module):
    wall_geometry: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("wall_geometry", torch.empty(4))

    def forward(self, shared: torch.Tensor) -> FloorPlanModelOutput:
        if shared.ndim != 4 or shared.shape[1] != 3:
            raise ValueError("acceptance heads require a three-channel decoder output")
        wall = shared.amin(dim=1, keepdim=True) < 0.0
        red, green, blue = shared[:, :1], shared[:, 1:2], shared[:, 2:]
        door = (red > 0.0) & (green < 0.0) & (blue < 0.0)
        window = (red < 0.0) & (green < 0.0) & (blue > 0.0)
        shape = (shared.shape[0], 1, *shared.shape[-2:])
        center = torch.zeros(shape, dtype=torch.bool, device=shared.device)
        endpoint = torch.zeros_like(center)
        log_half_length = torch.zeros(shape, dtype=shared.dtype, device=shared.device)
        for index, masks in enumerate(zip(door, window, strict=True)):
            for mask in masks:
                y_coordinates, x_coordinates = torch.where(mask[0])
                if x_coordinates.numel() == 0:
                    continue
                y = (y_coordinates.min() + y_coordinates.max()) // 2
                x0, x1 = x_coordinates.min(), x_coordinates.max()
                center[index, 0, y, (x0 + x1) // 2] = True
                endpoint[index, 0, y, x0] = endpoint[index, 0, y, x1] = True
                log_half_length[index, 0][mask[0]] = torch.log1p((x1 - x0).float())
        orientation = torch.zeros(
            (shape[0], 2, *shape[2:]), dtype=shared.dtype, device=shared.device
        )
        orientation[:, 1].fill_(1.0)
        door_type = torch.where(
            door,
            _LOGIT_MAGNITUDE,
            torch.where(window, -_LOGIT_MAGNITUDE, 0.0),
        )

        def logits(mask: torch.Tensor) -> torch.Tensor:
            return torch.where(mask, _LOGIT_MAGNITUDE, -_LOGIT_MAGNITUDE)

        return cast(
            FloorPlanModelOutput,
            {
                "wall_mask_logits": logits(wall),
                "wall_centerline_logits": logits(wall),
                "wall_junction_logits": torch.full(
                    shape,
                    -_LOGIT_MAGNITUDE,
                    dtype=shared.dtype,
                    device=shared.device,
                ),
                "wall_orientation_raw": orientation,
                "wall_log_half_thickness": torch.full(
                    shape,
                    math.log1p(float(self.wall_geometry[3])),
                    dtype=shared.dtype,
                    device=shared.device,
                ),
                "opening_mask_logits": logits(door | window),
                "opening_center_logits": logits(center),
                "opening_endpoint_logits": logits(endpoint),
                "opening_type_logits": torch.cat((door_type, -door_type), dim=1),
                "opening_orientation_raw": orientation.clone(),
                "opening_log_half_length": log_half_length,
            },
        )


def build_acceptance_model(config: ModelConfig) -> FloorPlanGLV:
    """Build the offline deterministic fixture through the real model assembly."""
    return FloorPlanGLV(
        config,
        local_encoder=_AcceptanceLocalEncoder(),
        global_encoder=_AcceptanceGlobalEncoder(),
        fusion_blocks=[_AcceptanceFusion() for _ in range(4)],
        decoder=_AcceptanceDecoder(),
        heads=_AcceptanceHeads(),
        debug_shapes=False,
    )
