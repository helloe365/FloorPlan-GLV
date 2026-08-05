"""Source-image dataset with synchronized augmentation and local patches."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import Dataset

from floorplan_glv.config.models import DataConfig, ModelConfig
from floorplan_glv.data.annotation_schema import (
    AnnotationError,
    TrainingAnnotation,
)
from floorplan_glv.data.index import IndexRecord
from floorplan_glv.data.patch_sampler import (
    PatchRequest,
    PatchSampler,
    PatchSamplingConfig,
)
from floorplan_glv.geometry.rasterize import (
    PatchSample,
    TargetGenerationConfig,
    TargetMaps,
    generate_patch_targets,
)

Augmentation: TypeAlias = Callable[
    [Image.Image, TrainingAnnotation, np.random.Generator],
    tuple[Image.Image, TrainingAnnotation],
]

_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STANDARD_DEVIATION = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class ImageTransform:
    """Mapping from source image coordinates to the global canvas."""

    source_width: int
    source_height: int
    scale: float
    pad_x: int
    pad_y: int

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source dimensions must be positive")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("global image scale must be finite and positive")
        if self.pad_x < 0 or self.pad_y < 0:
            raise ValueError("global image padding must be non-negative")

    def map_box(
        self,
        source_box: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float]:
        """Map one source-coordinate box into global-canvas coordinates."""
        x0, y0, x1, y1 = source_box
        return (
            x0 * self.scale + self.pad_x,
            y0 * self.scale + self.pad_y,
            x1 * self.scale + self.pad_x,
            y1 * self.scale + self.pad_y,
        )


@dataclass(frozen=True, slots=True)
class LocalPatchItem:
    """One normalized local patch and its complete target set."""

    image: torch.Tensor
    request: PatchRequest
    box_global_xyxy: tuple[float, float, float, float]
    targets: TargetMaps


@dataclass(frozen=True, slots=True)
class SourceImageItem:
    """One global source tensor and its sampled local training patches."""

    sample_id: str
    global_image: torch.Tensor
    image_transform: ImageTransform
    patches: tuple[LocalPatchItem, ...]


class FloorPlanDataset(Dataset[SourceImageItem]):
    """Load normalized samples and deterministically prepare K local patches."""

    def __init__(
        self,
        index_path: Path,
        *,
        patches_per_image: int = 4,
        patch_sampler: PatchSampler | None = None,
        augmentation: Augmentation | None = None,
        seed: int = 0,
        model_config: ModelConfig | None = None,
        data_config: DataConfig | None = None,
        target_config: TargetGenerationConfig | None = None,
    ) -> None:
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive")
        if seed < 0:
            raise ValueError("dataset seed must be non-negative")
        self.index_path = Path(index_path)
        self.records = _load_index(self.index_path)
        self.patches_per_image = patches_per_image
        self.model_config = model_config or ModelConfig()
        self.data_config = data_config or DataConfig()
        self.target_config = target_config or TargetGenerationConfig()
        self.patch_sampler = patch_sampler or PatchSampler(
            PatchSamplingConfig(patch_size=self.model_config.patch_size)
        )
        if self.patch_sampler.config.patch_size != self.model_config.patch_size:
            raise ValueError(
                "patch sampler size must match the validated model patch size"
            )
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic sampling stream for one training epoch."""
        if epoch < 0:
            raise ValueError("dataset epoch must be non-negative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> SourceImageItem:
        record = self.records[index]
        image, annotation = self._load_sample(record)
        rng = np.random.default_rng(
            np.random.SeedSequence((self.seed, self.epoch, index))
        )
        if self.augmentation is not None:
            image, annotation = self.augmentation(image, annotation, rng)
            _validate_image_annotation_pair(image, annotation, record.sample_id)

        global_image, image_transform = _prepare_global_image(
            image,
            global_size=self.model_config.global_size,
            padding_value=self.model_config.padding_value,
        )
        output_extent = self.model_config.patch_size // self.model_config.output_stride
        patches: list[LocalPatchItem] = []
        for _ in range(self.patches_per_image):
            request = self.patch_sampler.sample(annotation, rng)
            local_image = _crop_with_padding(
                image,
                request.source_box,
                padding_value=self.model_config.padding_value,
            )
            targets = generate_patch_targets(
                PatchSample(
                    annotation=annotation,
                    source_box=request.source_box,
                ),
                output_size=(output_extent, output_extent),
                config=self.target_config,
            )
            patches.append(
                LocalPatchItem(
                    image=_normalize_image(local_image),
                    request=request,
                    box_global_xyxy=image_transform.map_box(request.source_box),
                    targets=targets,
                )
            )
        return SourceImageItem(
            sample_id=record.sample_id,
            global_image=global_image,
            image_transform=image_transform,
            patches=tuple(patches),
        )

    def _load_sample(
        self,
        record: IndexRecord,
    ) -> tuple[Image.Image, TrainingAnnotation]:
        root = self.index_path.parent
        image_path = root / record.image
        annotation_path = root / record.annotation
        try:
            with Image.open(image_path) as encoded:
                image = _decode_rgb(encoded)
        except OSError as exc:
            raise AnnotationError(
                f"failed to decode sample {record.sample_id}: {image_path}: {exc}"
            ) from exc
        if min(image.size) < self.data_config.minimum_image_size:
            raise AnnotationError(
                f"sample {record.sample_id} image must be at least "
                f"{self.data_config.minimum_image_size} x "
                f"{self.data_config.minimum_image_size}"
            )
        try:
            annotation = TrainingAnnotation.model_validate_json(
                annotation_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise AnnotationError(
                f"failed to load sample {record.sample_id} annotation "
                f"{annotation_path}: {exc}"
            ) from exc
        if annotation.sample_id != record.sample_id:
            raise AnnotationError(
                f"index sample ID {record.sample_id} does not match "
                f"annotation sample ID {annotation.sample_id}"
            )
        _validate_image_annotation_pair(image, annotation, record.sample_id)
        return image, annotation


def _load_index(path: Path) -> tuple[IndexRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnnotationError(f"failed to read dataset index {path}: {exc}") from exc
    records: list[IndexRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(IndexRecord.model_validate_json(line))
        except ValidationError as exc:
            raise AnnotationError(
                f"invalid dataset index record {path}:{line_number}: {exc}"
            ) from exc
    if not records:
        raise AnnotationError(f"dataset index contains no samples: {path}")
    sample_ids = tuple(record.sample_id for record in records)
    if len(sample_ids) != len(set(sample_ids)):
        raise AnnotationError(f"dataset index has duplicate sample IDs: {path}")
    return tuple(records)


def _decode_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        foreground = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, foreground).convert("RGB")
    return image.convert("RGB")


def _validate_image_annotation_pair(
    image: Image.Image,
    annotation: TrainingAnnotation,
    sample_id: str,
) -> None:
    expected = (annotation.image.width, annotation.image.height)
    if image.size != expected:
        raise AnnotationError(
            f"sample {sample_id} image size {image.size} does not match "
            f"annotation size {expected}"
        )


def _prepare_global_image(
    image: Image.Image,
    *,
    global_size: int,
    padding_value: int,
) -> tuple[torch.Tensor, ImageTransform]:
    width, height = image.size
    scale = min(global_size / width, global_size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_x = (global_size - resized_width) // 2
    pad_y = (global_size - resized_height) // 2
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    fill = (padding_value, padding_value, padding_value)
    canvas = Image.new("RGB", (global_size, global_size), color=fill)
    canvas.paste(resized, (pad_x, pad_y))
    return (
        _normalize_image(canvas),
        ImageTransform(
            source_width=width,
            source_height=height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        ),
    )


def _crop_with_padding(
    image: Image.Image,
    source_box: tuple[int, int, int, int],
    *,
    padding_value: int,
) -> Image.Image:
    x0, y0, x1, y1 = source_box
    patch_width = x1 - x0
    patch_height = y1 - y0
    fill = (padding_value, padding_value, padding_value)
    patch = Image.new("RGB", (patch_width, patch_height), color=fill)
    crop_box = (
        max(0, x0),
        max(0, y0),
        min(image.width, x1),
        min(image.height, y1),
    )
    if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
        crop = image.crop(crop_box)
        patch.paste(crop, (crop_box[0] - x0, crop_box[1] - y0))
    return patch


def _normalize_image(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    channels_first = np.ascontiguousarray(pixels.transpose(2, 0, 1))
    tensor = torch.from_numpy(channels_first)
    mean = torch.tensor(_IMAGE_MEAN, dtype=torch.float32)[:, None, None]
    standard_deviation = torch.tensor(
        _IMAGE_STANDARD_DEVIATION,
        dtype=torch.float32,
    )[:, None, None]
    return (tensor - mean) / standard_deviation


__all__ = [
    "Augmentation",
    "FloorPlanDataset",
    "ImageTransform",
    "LocalPatchItem",
    "PatchSample",
    "SourceImageItem",
]
