"""Exact stride-2 multi-scale decoder for local floor-plan features."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn

from floorplan_glv.models.types import (
    LOCAL_FEATURE_CONTRACT,
    FeaturePyramid,
)

PYRAMID_WIDTH = 128
DECODER_WIDTH = 256
DETAIL_WIDTH = 64
PYRAMID_SPATIAL_SIZE = (128, 128)
OUTPUT_SPATIAL_SIZE = (256, 256)
RGB_PATCH_SIZE = 512


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormGelu(nn.Sequential):
    """Convolution followed by GroupNorm and GELU."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            _group_norm(output_channels),
            nn.GELU(),
        )


class ResidualConvBlock(nn.Module):
    """Two 3 x 3 convolutions with a normalized residual connection."""

    def __init__(self, channels: int = DECODER_WIDTH) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            ConvNormGelu(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            _group_norm(channels),
        )
        self.activation = nn.GELU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return one residual refinement of ``tensor``."""
        residual = cast(torch.Tensor, self.residual(tensor))
        return cast(torch.Tensor, self.activation(tensor + residual))


class MultiScaleDecoder(nn.Module):
    """Fuse four local stages with stride-2 RGB detail.

    Inputs:
        features: MiT-B4 tensors described by :class:`FeaturePyramid`.
        rgb_patch: Normalized local RGB tensor ``[M, 3, 512, 512]``.

    Returns:
        Shared decoder feature ``[M, 256, 256, 256]``.
    """

    def __init__(self, *, debug_shapes: bool = True) -> None:
        super().__init__()
        self.debug_shapes = debug_shapes
        self.detail_stem = nn.Sequential(
            ConvNormGelu(
                3,
                DETAIL_WIDTH,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvNormGelu(
                DETAIL_WIDTH,
                DETAIL_WIDTH,
                kernel_size=3,
                padding=1,
            ),
        )
        self.pyramid_projections = nn.ModuleList(
            [
                ConvNormGelu(
                    channels,
                    PYRAMID_WIDTH,
                    kernel_size=1,
                )
                for channels, _ in LOCAL_FEATURE_CONTRACT
            ]
        )
        self.pyramid_fusion = nn.Sequential(
            ConvNormGelu(
                4 * PYRAMID_WIDTH,
                DECODER_WIDTH,
                kernel_size=3,
                padding=1,
            ),
            ConvNormGelu(
                DECODER_WIDTH,
                DECODER_WIDTH,
                kernel_size=3,
                padding=1,
            ),
        )
        self.detail_fusion = ConvNormGelu(
            DECODER_WIDTH + DETAIL_WIDTH,
            DECODER_WIDTH,
            kernel_size=3,
            padding=1,
        )
        self.residual_blocks = nn.Sequential(
            ResidualConvBlock(),
            ResidualConvBlock(),
        )

    def forward(
        self,
        features: FeaturePyramid,
        rgb_patch: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one flattened local-patch batch at stride 2."""
        batch_size = rgb_patch.shape[0]
        if self.debug_shapes:
            _validate_decoder_inputs(features, rgb_patch)

        projected = [
            functional.interpolate(
                projection(feature),
                size=PYRAMID_SPATIAL_SIZE,
                mode="bilinear",
                align_corners=False,
            )
            for projection, feature in zip(
                self.pyramid_projections,
                features,
                strict=True,
            )
        ]
        pyramid = self.pyramid_fusion(torch.cat(projected, dim=1))
        pyramid = functional.interpolate(
            pyramid,
            size=OUTPUT_SPATIAL_SIZE,
            mode="bilinear",
            align_corners=False,
        )
        detail = self.detail_stem(rgb_patch)
        output = cast(
            torch.Tensor,
            self.residual_blocks(
                self.detail_fusion(torch.cat((pyramid, detail), dim=1))
            ),
        )
        if self.debug_shapes:
            _require_shape(
                output,
                (batch_size, DECODER_WIDTH, *OUTPUT_SPATIAL_SIZE),
                "decoder output",
            )
        return output


def _validate_decoder_inputs(
    features: FeaturePyramid,
    rgb_patch: torch.Tensor,
) -> None:
    batch_size = rgb_patch.shape[0]
    _require_shape(
        rgb_patch,
        (batch_size, 3, RGB_PATCH_SIZE, RGB_PATCH_SIZE),
        "rgb_patch",
    )
    for index, (feature, (channels, spatial_size)) in enumerate(
        zip(features, LOCAL_FEATURE_CONTRACT, strict=True),
        start=1,
    ):
        _require_shape(
            feature,
            (batch_size, channels, spatial_size, spatial_size),
            f"decoder stage {index}",
        )


def _require_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    name: str,
) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
        )
