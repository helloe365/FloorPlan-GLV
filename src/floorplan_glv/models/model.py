"""FloorPlan-GLV model assembly for local-only and global-local modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from floorplan_glv.config.models import ConfigurationError, ModelConfig
from floorplan_glv.data.collate import ModelBatch
from floorplan_glv.models.decoder import DECODER_WIDTH, MultiScaleDecoder
from floorplan_glv.models.encoders import (
    GLOBAL_IMAGE_SIZE,
    GlobalSegformerEncoder,
    LocalSegformerEncoder,
)
from floorplan_glv.models.fusion import GatedRoiFusion
from floorplan_glv.models.heads import OUTPUT_CHANNELS, PredictionHeads
from floorplan_glv.models.types import (
    LOCAL_FEATURE_CONTRACT,
    FeaturePyramid,
    FloorPlanModelOutput,
)


class FloorPlanGLV(nn.Module):
    """Run the approved encoders, optional ROI fusion, decoder, and heads."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        local_encoder: nn.Module | None = None,
        global_encoder: nn.Module | None = None,
        fusion_blocks: Sequence[nn.Module] | None = None,
        decoder: nn.Module | None = None,
        heads: nn.Module | None = None,
        debug_shapes: bool = True,
    ) -> None:
        super().__init__()
        if config.decoder_width != DECODER_WIDTH:
            raise ConfigurationError(
                f"decoder_width must be {DECODER_WIDTH} for the V1 architecture"
            )
        if config.global_enabled and config.global_size != GLOBAL_IMAGE_SIZE:
            raise ConfigurationError(
                f"global_size must be {GLOBAL_IMAGE_SIZE} when global mode is enabled"
            )
        self.config = config
        self.debug_shapes = debug_shapes
        self.local_encoder = (
            local_encoder
            if local_encoder is not None
            else LocalSegformerEncoder(
                checkpoint=config.local_checkpoint,
                debug_shapes=debug_shapes,
            )
        )
        self.global_encoder: nn.Module | None
        if config.global_enabled:
            self.global_encoder = (
                global_encoder
                if global_encoder is not None
                else GlobalSegformerEncoder(
                    checkpoint=config.global_checkpoint,
                    debug_shapes=debug_shapes,
                )
            )
            configured_fusion = (
                list(fusion_blocks)
                if fusion_blocks is not None
                else [
                    GatedRoiFusion(
                        channels,
                        global_canvas_size=GLOBAL_IMAGE_SIZE,
                        debug_shapes=debug_shapes,
                    )
                    for channels, _ in LOCAL_FEATURE_CONTRACT
                ]
            )
            if len(configured_fusion) != len(LOCAL_FEATURE_CONTRACT):
                raise ConfigurationError("global mode requires four fusion blocks")
            self.fusion_blocks = nn.ModuleList(configured_fusion)
        else:
            self.global_encoder = None
            self.fusion_blocks = nn.ModuleList()
        self.decoder = (
            decoder
            if decoder is not None
            else MultiScaleDecoder(debug_shapes=debug_shapes)
        )
        self.heads = heads if heads is not None else PredictionHeads()

    def forward(self, batch: ModelBatch) -> FloorPlanModelOutput:
        """Predict all task maps for a flattened source-image batch.

        Args:
            batch: Global-local batch whose local patches have shape
                ``[M, 3, 512, 512]``.

        Returns:
            Eleven raw tensors shaped ``[M, C, 256, 256]``.
        """
        features = cast(FeaturePyramid, self.local_encoder(batch.local_patches))
        if self.config.global_enabled:
            if self.global_encoder is None:
                raise RuntimeError("global encoder is not configured")
            global_features = cast(
                FeaturePyramid,
                self.global_encoder(batch.global_images),
            )
            fused_stages = [
                cast(
                    torch.Tensor,
                    fusion(
                        local_stage,
                        global_stage,
                        batch.patch_to_image,
                        batch.patch_boxes_global_xyxy,
                    ),
                )
                for fusion, local_stage, global_stage in zip(
                    self.fusion_blocks,
                    features,
                    global_features,
                    strict=True,
                )
            ]
            features = FeaturePyramid(*fused_stages)
        shared_feature = cast(
            torch.Tensor,
            self.decoder(features, batch.local_patches),
        )
        output = cast(FloorPlanModelOutput, self.heads(shared_feature))
        if self.debug_shapes:
            _validate_output_contract(output, batch.local_patches.shape[0])
        return output


def _validate_output_contract(
    output: FloorPlanModelOutput,
    batch_size: int,
) -> None:
    if set(output) != set(OUTPUT_CHANNELS):
        raise ValueError("model output keys do not match the V1 contract")
    for key, channels in OUTPUT_CHANNELS.items():
        value = output[key]  # type: ignore[literal-required]
        expected = (batch_size, channels, 256, 256)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"model output {key} must have shape {expected}, "
                f"got {tuple(value.shape)}"
            )
