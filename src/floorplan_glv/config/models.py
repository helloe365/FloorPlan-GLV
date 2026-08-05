"""Validated application configuration models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0)]


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be loaded or validated."""


class StrictConfigModel(BaseModel):
    """Immutable configuration base that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictConfigModel):
    """Neural-network and patch geometry settings for the approved V1 model."""

    global_enabled: bool = True
    global_checkpoint: str = "nvidia/mit-b2"
    local_checkpoint: str = "nvidia/mit-b4"
    global_size: int = 1024
    patch_size: int = 512
    stride: int = 384
    padding_value: Annotated[int, Field(ge=0, le=255)] = 255
    output_stride: Literal[2] = 2
    decoder_width: PositiveInt = 256

    @field_validator("global_size", "patch_size")
    @classmethod
    def validate_encoder_size(cls, value: int, info: object) -> int:
        """Require positive encoder sizes compatible with four MiT stages."""
        field_name = getattr(info, "field_name", "size")
        if value <= 0 or value % 32 != 0:
            raise ValueError(f"{field_name} must be positive and divisible by 32")
        return value

    @model_validator(mode="after")
    def validate_model_contract(self) -> Self:
        """Enforce the fixed V1 backbones and valid patch stride."""
        if not 0 < self.stride <= self.patch_size:
            raise ValueError("stride must be in (0, patch_size]")
        if self.local_checkpoint != "nvidia/mit-b4":
            raise ValueError("local_checkpoint must be nvidia/mit-b4")
        if self.global_enabled and self.global_checkpoint != "nvidia/mit-b2":
            raise ValueError(
                "global_checkpoint must be nvidia/mit-b2 when global_enabled"
            )
        return self


class DataConfig(StrictConfigModel):
    """Dataset boundary defaults shared by conversion and loading."""

    minimum_image_size: PositiveInt = 64
    train_fraction: Probability = 0.80
    validation_fraction: Probability = 0.10
    test_fraction: Probability = 0.10

    @model_validator(mode="after")
    def validate_split_fractions(self) -> Self:
        """Require source-image split fractions to form one complete dataset."""
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("data split fractions must sum to 1.0")
        return self


class SchedulerConfig(StrictConfigModel):
    """Learning-rate scheduler defaults from the training specification."""

    name: Literal["cosine"] = "cosine"
    warmup_steps: Annotated[int, Field(ge=0)] = 1500
    min_lr_ratio: Probability = 0.01


class LearningRateConfig(StrictConfigModel):
    """Discriminative learning rates for the five trainable model sections."""

    global_encoder: PositiveFloat = 0.00005
    local_encoder: PositiveFloat = 0.00001
    fusion: PositiveFloat = 0.00020
    decoder: PositiveFloat = 0.00010
    heads: PositiveFloat = 0.00015


class TrainingLossWeights(StrictConfigModel):
    """Validated per-task weights consumed by the Task 8 loss aggregator."""

    wall_mask: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 2.0
    wall_centerline: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 2.0
    wall_junction: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 1.0
    wall_orientation: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.5
    wall_thickness: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.5
    opening_mask: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 2.0
    opening_center: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 3.0
    opening_endpoint: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 1.0
    opening_type: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 1.5
    opening_orientation: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.75
    opening_length: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.75


class TrainConfig(StrictConfigModel):
    """Validated optimizer, batching, precision, and run settings."""

    dataset_index: Path = Path("data/processed/index.jsonl")
    validation_index: Path | None = None
    output_dir: Path = Path("runs/train")
    initial_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    epochs: PositiveInt = 60
    seed: NonnegativeInt = 1337
    patches_per_image: PositiveInt = 4
    source_images_per_gpu: PositiveInt = 1
    num_workers: NonnegativeInt = 0
    gradient_accumulation_steps: PositiveInt = 8
    weight_decay: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.01
    betas: tuple[Probability, Probability] = (0.9, 0.999)
    gradient_clip_norm: PositiveFloat = 1.0
    ema_decay: Probability = 0.9998
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    freeze_local_encoder_epochs: NonnegativeInt = 0
    gradient_checkpointing: bool = True
    deterministic_algorithms: bool = True
    hard_negative_mining: bool = False
    learning_rates: LearningRateConfig = Field(default_factory=LearningRateConfig)
    loss_weights: TrainingLossWeights = Field(default_factory=TrainingLossWeights)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Require conventional AdamW beta values below one."""
        if any(beta >= 1.0 for beta in value):
            raise ValueError("optimizer betas must be less than 1.0")
        return value

    @model_validator(mode="after")
    def validate_stage_schedule(self) -> Self:
        """Keep the local freeze window within the configured run."""
        if self.freeze_local_encoder_epochs > self.epochs:
            raise ValueError("freeze_local_encoder_epochs cannot exceed epochs")
        if self.initial_checkpoint is not None and self.resume_checkpoint is not None:
            raise ValueError(
                "initial_checkpoint and resume_checkpoint are mutually exclusive"
            )
        return self


class WallPostprocessConfig(StrictConfigModel):
    """Wall vectorization defaults from the architecture specification."""

    mask_threshold: Probability = 0.45
    centerline_threshold: Probability = 0.35
    junction_threshold: Probability = 0.30
    junction_nms_radius_px: PositiveInt = 6
    junction_cluster_radius_px: PositiveInt = 8
    min_component_area_px: PositiveInt = 20
    min_wall_length_px: PositiveFloat = 12.0
    max_trace_gap_px: PositiveFloat = 5.0
    rdp_epsilon_px: PositiveFloat = 2.0
    split_angle_deg: PositiveFloat = 12.0
    merge_angle_deg: PositiveFloat = 5.0
    merge_gap_px: PositiveFloat = 8.0
    endpoint_snap_px: PositiveFloat = 6.0
    dominant_angle_snap_deg: PositiveFloat = 4.0
    min_raster_support: Probability = 0.55


class OpeningPostprocessConfig(StrictConfigModel):
    """Opening candidate defaults from the architecture specification."""

    center_threshold: Probability = 0.30
    center_nms_radius_px: PositiveInt = 6
    max_candidates: PositiveInt = 512
    final_confidence_threshold: Probability = 0.40


class AttachmentConfig(StrictConfigModel):
    """Opening-to-wall attachment gates and score weights."""

    max_distance_px: PositiveFloat = 12.0
    max_distance_wall_thickness_factor: PositiveFloat = 0.75
    max_angle_difference_deg: PositiveFloat = 12.0
    min_projected_overlap_ratio: Probability = 0.65
    distance_weight: Probability = 0.35
    angle_weight: Probability = 0.25
    segment_overlap_weight: Probability = 0.25
    wall_confidence_weight: Probability = 0.15

    @model_validator(mode="after")
    def validate_score_weights(self) -> Self:
        """Require attachment score weights to form a convex combination."""
        total = (
            self.distance_weight
            + self.angle_weight
            + self.segment_overlap_weight
            + self.wall_confidence_weight
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("attachment score weights must sum to 1.0")
        return self


class OpeningCleanupConfig(StrictConfigModel):
    """Opening duplicate-removal thresholds."""

    center_distance_mean_length_factor: PositiveFloat = 0.25
    max_angle_difference_deg: PositiveFloat = 8.0
    min_projected_overlap_ratio: Probability = 0.70


class PostprocessConfig(StrictConfigModel):
    """Complete deterministic postprocessing configuration."""

    hann_weight_floor: PositiveFloat = 0.05
    wall: WallPostprocessConfig = Field(default_factory=WallPostprocessConfig)
    opening: OpeningPostprocessConfig = Field(default_factory=OpeningPostprocessConfig)
    attachment: AttachmentConfig = Field(default_factory=AttachmentConfig)
    opening_cleanup: OpeningCleanupConfig = Field(default_factory=OpeningCleanupConfig)


class AppConfig(StrictConfigModel):
    """Root configuration used by FloorPlan-GLV commands."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
