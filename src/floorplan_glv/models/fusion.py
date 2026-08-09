"""Stage-wise gated fusion of local features with global ROI context."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchvision.ops import roi_align  # type: ignore[import-untyped]

GATE_INITIAL_BIAS = -2.0
ROI_SAMPLING_RATIO = 2


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class GatedRoiFusion(nn.Module):
    """Fuse one local stage with aligned context from one global stage.

    Inputs:
        local: Local features shaped ``[M, C_local, H_local, W_local]``.
        global_feature: Global features shaped
            ``[B, C_global, H_global, W_global]``.
        patch_to_image: Source-image indices shaped ``[M]``.
        boxes_global: Patch boxes in global-canvas xyxy coordinates,
            shaped ``[M, 4]``.

    Returns:
        Fused local features shaped ``[M, C_local, H_local, W_local]``.
    """

    def __init__(
        self,
        channels: int,
        *,
        global_channels: int | None = None,
        global_canvas_size: int = 1024,
        debug_shapes: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if global_channels is not None and global_channels <= 0:
            raise ValueError("global_channels must be positive")
        if global_canvas_size <= 0:
            raise ValueError("global_canvas_size must be positive")
        self.channels = channels
        self.global_channels = global_channels or channels
        self.global_canvas_size = global_canvas_size
        self.debug_shapes = debug_shapes
        self.global_projection = nn.Conv2d(
            self.global_channels,
            channels,
            kernel_size=1,
        )
        self.gate_projection = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=1,
        )
        gate_bias = self.gate_projection.bias
        if gate_bias is None:
            raise RuntimeError("gate projection must include a bias")
        nn.init.constant_(gate_bias, GATE_INITIAL_BIAS)
        self.norm = _group_norm(channels)

    def forward(
        self,
        local: torch.Tensor,
        global_feature: torch.Tensor,
        patch_to_image: torch.Tensor,
        boxes_global: torch.Tensor,
    ) -> torch.Tensor:
        """ROI-align global context and apply gated residual fusion."""
        if self.debug_shapes:
            self._validate_inputs(
                local,
                global_feature,
                patch_to_image,
                boxes_global,
            )
        feature_height, feature_width = global_feature.shape[-2:]
        box_scale = boxes_global.new_tensor(
            (
                feature_width / self.global_canvas_size,
                feature_height / self.global_canvas_size,
                feature_width / self.global_canvas_size,
                feature_height / self.global_canvas_size,
            )
        )
        scaled_boxes = boxes_global * box_scale
        rois = torch.cat(
            (
                patch_to_image.to(dtype=scaled_boxes.dtype).unsqueeze(1),
                scaled_boxes,
            ),
            dim=1,
        ).to(device=global_feature.device, dtype=global_feature.dtype)
        global_roi = roi_align(
            global_feature,
            rois,
            output_size=local.shape[-2:],
            spatial_scale=1.0,
            sampling_ratio=ROI_SAMPLING_RATIO,
            aligned=True,
        )
        projected_global = self.global_projection(global_roi)
        gate = torch.sigmoid(
            self.gate_projection(torch.cat((local, projected_global), dim=1))
        )
        return cast(torch.Tensor, self.norm(local + gate * projected_global))

    def _validate_inputs(
        self,
        local: torch.Tensor,
        global_feature: torch.Tensor,
        patch_to_image: torch.Tensor,
        boxes_global: torch.Tensor,
    ) -> None:
        patch_count = local.shape[0]
        if local.ndim != 4 or local.shape[1] != self.channels:
            raise ValueError(
                f"local must have shape [M, {self.channels}, H, W], "
                f"got {tuple(local.shape)}"
            )
        if global_feature.ndim != 4 or global_feature.shape[1] != self.global_channels:
            raise ValueError(
                "global_feature must have shape "
                f"[B, {self.global_channels}, H, W], "
                f"got {tuple(global_feature.shape)}"
            )
        if tuple(patch_to_image.shape) != (patch_count,):
            raise ValueError(
                f"patch_to_image must have shape ({patch_count},), "
                f"got {tuple(patch_to_image.shape)}"
            )
        if patch_to_image.dtype != torch.int64:
            raise TypeError("patch_to_image must have dtype torch.int64")
        if tuple(boxes_global.shape) != (patch_count, 4):
            raise ValueError(
                f"boxes_global must have shape ({patch_count}, 4), "
                f"got {tuple(boxes_global.shape)}"
            )
        if not boxes_global.is_floating_point():
            raise TypeError("boxes_global must have a floating-point dtype")
        if not torch.isfinite(boxes_global).all():
            raise ValueError("boxes_global must contain only finite coordinates")
        if (boxes_global < 0).any() or (boxes_global > self.global_canvas_size).any():
            raise ValueError("boxes_global coordinates are outside the global canvas")
        if (boxes_global[:, 2] <= boxes_global[:, 0]).any() or (
            boxes_global[:, 3] <= boxes_global[:, 1]
        ).any():
            raise ValueError("boxes_global must contain positive-area xyxy boxes")
        if patch_to_image.numel() and (
            (patch_to_image < 0).any()
            or (patch_to_image >= global_feature.shape[0]).any()
        ):
            raise ValueError("patch_to_image contains an out-of-range source index")
        if local.device != global_feature.device:
            raise ValueError("local and global_feature must be on the same device")
        if local.dtype != global_feature.dtype:
            raise TypeError("local and global_feature must have the same dtype")
