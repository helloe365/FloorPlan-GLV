"""Shape-checked collation for source-image global-local batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch

from floorplan_glv.data.annotation_schema import AnnotationError
from floorplan_glv.data.dataset import LocalPatchItem, SourceImageItem
from floorplan_glv.geometry.rasterize import TARGET_KEYS

_FLOAT_SCALAR_KEYS = {
    "wall_mask",
    "wall_centerline",
    "wall_junction",
    "wall_log_half_thickness",
    "opening_mask",
    "opening_center",
    "opening_endpoint",
    "opening_log_half_length",
}
_ORIENTATION_KEYS = {"wall_orientation", "opening_orientation"}
_VALIDITY_KEYS = {
    "wall_orientation_valid",
    "wall_thickness_valid",
    "opening_type_valid",
    "opening_orientation_valid",
    "opening_length_valid",
    "valid_pixels",
}


@dataclass(frozen=True, slots=True)
class ModelBatch:
    """Global-local tensors consumed by the FloorPlan-GLV model.

    Shapes:
        global_images: ``[B, 3, 1024, 1024]`` float32.
        local_patches: ``[M, 3, 512, 512]`` float32.
        patch_to_image: ``[M]`` int64.
        patch_boxes_global_xyxy: ``[M, 4]`` float32.
        patch_valid_masks: ``[M, 1, 256, 256]`` bool.
        targets: Batched scalar, orientation, type, and validity target maps.
    """

    global_images: torch.Tensor
    local_patches: torch.Tensor
    patch_to_image: torch.Tensor
    patch_boxes_global_xyxy: torch.Tensor
    patch_valid_masks: torch.Tensor
    targets: dict[str, torch.Tensor]


def collate_source_images(items: Sequence[SourceImageItem]) -> ModelBatch:
    """Stack B source tensors and flatten their local patches into M."""
    if not items:
        raise AnnotationError("collate requires at least one source image")
    for item in items:
        _require_tensor(
            item.global_image,
            expected_shape=(3, 1024, 1024),
            expected_dtype=torch.float32,
            name=f"{item.sample_id} global image",
        )
        if not item.patches:
            raise AnnotationError(
                f"source image {item.sample_id} contains no local patches"
            )

    flat_patches: list[LocalPatchItem] = []
    patch_to_image: list[int] = []
    for image_index, item in enumerate(items):
        for patch in item.patches:
            _validate_patch(patch, item.sample_id)
            flat_patches.append(patch)
            patch_to_image.append(image_index)

    target_mappings: list[Mapping[str, torch.Tensor]] = [
        cast(Mapping[str, torch.Tensor], patch.targets) for patch in flat_patches
    ]
    targets = {
        key: torch.stack([target[key] for target in target_mappings])
        for key in TARGET_KEYS
    }
    patch_valid_masks = targets["valid_pixels"]
    return ModelBatch(
        global_images=torch.stack([item.global_image for item in items]),
        local_patches=torch.stack([patch.image for patch in flat_patches]),
        patch_to_image=torch.tensor(patch_to_image, dtype=torch.int64),
        patch_boxes_global_xyxy=torch.tensor(
            [patch.box_global_xyxy for patch in flat_patches],
            dtype=torch.float32,
        ),
        patch_valid_masks=patch_valid_masks,
        targets=targets,
    )


def _validate_patch(patch: LocalPatchItem, sample_id: str) -> None:
    _require_tensor(
        patch.image,
        expected_shape=(3, 512, 512),
        expected_dtype=torch.float32,
        name=f"{sample_id} local patch",
    )
    if (
        len(patch.box_global_xyxy) != 4
        or not torch.isfinite(
            torch.tensor(patch.box_global_xyxy, dtype=torch.float64)
        ).all()
    ):
        raise AnnotationError(f"{sample_id} patch global box must be finite xyxy")
    target_mapping = cast(Mapping[str, torch.Tensor], patch.targets)
    if set(target_mapping) != set(TARGET_KEYS):
        raise AnnotationError(f"{sample_id} patch target keys do not match contract")
    for key, value in target_mapping.items():
        shape: tuple[int, ...]
        if key in _FLOAT_SCALAR_KEYS:
            shape = (1, 256, 256)
            dtype = torch.float32
        elif key in _ORIENTATION_KEYS:
            shape = (2, 256, 256)
            dtype = torch.float32
        elif key in _VALIDITY_KEYS:
            shape = (1, 256, 256)
            dtype = torch.bool
        elif key == "opening_type":
            shape = (256, 256)
            dtype = torch.int64
        else:
            raise AnnotationError(f"unexpected patch target key: {key}")
        _require_tensor(
            value,
            expected_shape=shape,
            expected_dtype=dtype,
            name=f"{sample_id} target {key}",
        )


def _require_tensor(
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    name: str,
) -> None:
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise AnnotationError(
            f"{name} must have shape {expected_shape} and dtype {expected_dtype}"
        )
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise AnnotationError(f"{name} contains non-finite values")
