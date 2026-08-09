"""Hugging Face SegFormer wrappers with explicit feature contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn
from transformers import SegformerModel

from floorplan_glv.config.models import ConfigurationError
from floorplan_glv.models.types import (
    LOCAL_FEATURE_CONTRACT,
    FeaturePyramid,
)

LOCAL_SEGFORMER_CHECKPOINT = "nvidia/mit-b4"
LOCAL_PATCH_SIZE = 512
GLOBAL_SEGFORMER_CHECKPOINT = "nvidia/mit-b2"
GLOBAL_IMAGE_SIZE = 1024
GLOBAL_FEATURE_CONTRACT: tuple[tuple[int, int], ...] = (
    (64, 256),
    (128, 128),
    (320, 64),
    (512, 32),
)


class CheckpointError(RuntimeError):
    """Raised when a configured model checkpoint cannot be loaded."""


class LocalSegformerEncoder(nn.Module):
    """Encode normalized local patches into the four MiT-B4 feature stages.

    Input shape: ``[M, 3, 512, 512]``.
    Output shapes are documented by :class:`FeaturePyramid`.
    """

    def __init__(
        self,
        checkpoint: str = LOCAL_SEGFORMER_CHECKPOINT,
        *,
        backbone: nn.Module | None = None,
        gradient_checkpointing: bool = False,
        debug_shapes: bool = True,
    ) -> None:
        super().__init__()
        if checkpoint != LOCAL_SEGFORMER_CHECKPOINT:
            raise ConfigurationError(
                f"local encoder checkpoint must be {LOCAL_SEGFORMER_CHECKPOINT}"
            )
        self.backbone = (
            backbone if backbone is not None else _load_local_backbone(checkpoint)
        )
        self.debug_shapes = debug_shapes
        self._gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            self.set_gradient_checkpointing(True)

    @property
    def gradient_checkpointing_enabled(self) -> bool:
        """Return whether checkpointing has been enabled on the backbone."""
        return self._gradient_checkpointing_enabled

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable or disable Hugging Face gradient checkpointing."""
        method_name = (
            "gradient_checkpointing_enable"
            if enabled
            else "gradient_checkpointing_disable"
        )
        method = getattr(self.backbone, method_name, None)
        if not callable(method):
            raise ConfigurationError(f"local backbone does not support {method_name}")
        method()
        self._gradient_checkpointing_enabled = enabled

    def forward(self, pixel_values: torch.Tensor) -> FeaturePyramid:
        """Return MiT-B4 stages for ``pixel_values``.

        Args:
            pixel_values: Normalized float tensor shaped ``[M, 3, 512, 512]``.

        Returns:
            Four tensors matching the local feature-pyramid contract.
        """
        if self.debug_shapes:
            _require_shape(
                pixel_values,
                (pixel_values.shape[0], 3, LOCAL_PATCH_SIZE, LOCAL_PATCH_SIZE),
                "pixel_values",
            )
        output: Any = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        raw_hidden_states = getattr(output, "hidden_states", None)
        if raw_hidden_states is None:
            raise ValueError("local backbone did not return hidden states")
        hidden_states = cast(Sequence[object], raw_hidden_states)
        if len(hidden_states) != 4:
            raise ValueError("local backbone must return exactly four hidden states")
        if not all(isinstance(stage, torch.Tensor) for stage in hidden_states):
            raise TypeError("local backbone hidden states must be tensors")
        stages = cast(Sequence[torch.Tensor], hidden_states)
        features = FeaturePyramid(
            stage1=stages[0],
            stage2=stages[1],
            stage3=stages[2],
            stage4=stages[3],
        )
        if self.debug_shapes:
            _validate_feature_pyramid(features, pixel_values.shape[0])
        return features


class GlobalSegformerEncoder(nn.Module):
    """Encode normalized global canvases into four MiT-B2 feature stages.

    Input shape: ``[B, 3, 1024, 1024]``.
    Output shapes: ``[B, 64, 256, 256]``, ``[B, 128, 128, 128]``,
    ``[B, 320, 64, 64]``, and ``[B, 512, 32, 32]``.
    """

    def __init__(
        self,
        checkpoint: str = GLOBAL_SEGFORMER_CHECKPOINT,
        *,
        backbone: nn.Module | None = None,
        gradient_checkpointing: bool = False,
        debug_shapes: bool = True,
    ) -> None:
        super().__init__()
        if checkpoint != GLOBAL_SEGFORMER_CHECKPOINT:
            raise ConfigurationError(
                f"global encoder checkpoint must be {GLOBAL_SEGFORMER_CHECKPOINT}"
            )
        self.backbone = (
            backbone if backbone is not None else _load_global_backbone(checkpoint)
        )
        self.debug_shapes = debug_shapes
        self._gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            self.set_gradient_checkpointing(True)

    @property
    def gradient_checkpointing_enabled(self) -> bool:
        """Return whether checkpointing has been enabled on the backbone."""
        return self._gradient_checkpointing_enabled

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable or disable Hugging Face gradient checkpointing."""
        method_name = (
            "gradient_checkpointing_enable"
            if enabled
            else "gradient_checkpointing_disable"
        )
        method = getattr(self.backbone, method_name, None)
        if not callable(method):
            raise ConfigurationError(f"global backbone does not support {method_name}")
        method()
        self._gradient_checkpointing_enabled = enabled

    def forward(self, pixel_values: torch.Tensor) -> FeaturePyramid:
        """Return MiT-B2 stages for a global image batch.

        Args:
            pixel_values: Normalized float tensor shaped ``[B, 3, 1024, 1024]``.

        Returns:
            Four tensors matching the global feature-pyramid contract.
        """
        if self.debug_shapes:
            _require_shape(
                pixel_values,
                (pixel_values.shape[0], 3, GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE),
                "pixel_values",
            )
        output: Any = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        raw_hidden_states = getattr(output, "hidden_states", None)
        if raw_hidden_states is None:
            raise ValueError("global backbone did not return hidden states")
        hidden_states = cast(Sequence[object], raw_hidden_states)
        if len(hidden_states) != 4:
            raise ValueError("global backbone must return exactly four hidden states")
        if not all(isinstance(stage, torch.Tensor) for stage in hidden_states):
            raise TypeError("global backbone hidden states must be tensors")
        stages = cast(Sequence[torch.Tensor], hidden_states)
        features = FeaturePyramid(
            stage1=stages[0],
            stage2=stages[1],
            stage3=stages[2],
            stage4=stages[3],
        )
        if self.debug_shapes:
            _validate_global_feature_pyramid(features, pixel_values.shape[0])
        return features


def _load_local_backbone(checkpoint: str) -> nn.Module:
    try:
        return cast(
            nn.Module,
            SegformerModel.from_pretrained(
                checkpoint,
                output_hidden_states=True,
            ),
        )
    except (OSError, ValueError) as exc:
        raise CheckpointError(
            f"failed to load local encoder checkpoint {checkpoint}: {exc}"
        ) from exc


def _load_global_backbone(checkpoint: str) -> nn.Module:
    try:
        return cast(
            nn.Module,
            SegformerModel.from_pretrained(
                checkpoint,
                output_hidden_states=True,
            ),
        )
    except (OSError, ValueError) as exc:
        raise CheckpointError(
            f"failed to load global encoder checkpoint {checkpoint}: {exc}"
        ) from exc


def _validate_feature_pyramid(
    features: FeaturePyramid,
    batch_size: int,
) -> None:
    for index, (feature, (channels, spatial_size)) in enumerate(
        zip(features, LOCAL_FEATURE_CONTRACT, strict=True),
        start=1,
    ):
        _require_shape(
            feature,
            (batch_size, channels, spatial_size, spatial_size),
            f"local backbone stage {index}",
        )


def _validate_global_feature_pyramid(
    features: FeaturePyramid,
    batch_size: int,
) -> None:
    for index, (feature, (channels, spatial_size)) in enumerate(
        zip(features, GLOBAL_FEATURE_CONTRACT, strict=True),
        start=1,
    ):
        _require_shape(
            feature,
            (batch_size, channels, spatial_size, spatial_size),
            f"global backbone stage {index}",
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
