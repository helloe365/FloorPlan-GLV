"""Seam-resistant deterministic merging of overlapping patch predictions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypedDict, cast

import torch
import torch.nn.functional as functional

from floorplan_glv.config.models import PostprocessConfig
from floorplan_glv.geometry.primitives import GeometryError
from floorplan_glv.models.types import FloorPlanModelOutput

_NORMALIZATION_EPSILON = 1e-12

_BINARY_MAPS = {
    "wall_mask_logits": "wall_mask",
    "wall_centerline_logits": "wall_centerline",
    "wall_junction_logits": "wall_junction",
    "opening_mask_logits": "opening_mask",
    "opening_center_logits": "opening_center",
    "opening_endpoint_logits": "opening_endpoint",
}
_ORIENTATION_MAPS = {
    "wall_orientation_raw": "wall_orientation",
    "opening_orientation_raw": "opening_orientation",
}
_REGRESSION_MAPS = {
    "wall_log_half_thickness": "wall_log_half_thickness",
    "opening_log_half_length": "opening_log_half_length",
}
_TYPE_MAP = ("opening_type_logits", "opening_type")
_RAW_CHANNELS = {
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
_FULL_CHANNELS = {
    **{name: 1 for name in _BINARY_MAPS.values()},
    **{name: 2 for name in _ORIENTATION_MAPS.values()},
    **{name: 1 for name in _REGRESSION_MAPS.values()},
    _TYPE_MAP[1]: 2,
}


class FullResolutionMaps(TypedDict):
    """Merged source-resolution maps shaped ``[C, height, width]``."""

    wall_mask: torch.Tensor
    wall_centerline: torch.Tensor
    wall_junction: torch.Tensor
    wall_orientation: torch.Tensor
    wall_log_half_thickness: torch.Tensor
    opening_mask: torch.Tensor
    opening_center: torch.Tensor
    opening_endpoint: torch.Tensor
    opening_type: torch.Tensor
    opening_orientation: torch.Tensor
    opening_log_half_length: torch.Tensor


@dataclass(frozen=True, slots=True)
class PatchOutput:
    """One patch's raw stride-2 predictions and stride-2 validity mask.

    ``predictions`` contains tensors shaped ``[C, patch_size / 2,
    patch_size / 2]``. ``valid_mask`` is boolean with shape
    ``[1, patch_size / 2, patch_size / 2]``.
    """

    predictions: FloorPlanModelOutput
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class PatchPlacement:
    """One row-major patch position on the source-image canvas."""

    index: int
    x: int
    y: int
    valid_width: int
    valid_height: int


@dataclass(frozen=True, slots=True)
class PatchGrid:
    """Deterministic row-major inference grid for a ``(width, height)`` image."""

    image_size: tuple[int, int]
    patch_size: int = 512
    stride: int = 384
    x_starts: tuple[int, ...] = field(init=False)
    y_starts: tuple[int, ...] = field(init=False)
    placements: tuple[PatchPlacement, ...] = field(init=False)

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise GeometryError("patch-grid image dimensions must be positive")
        if self.patch_size <= 0 or self.patch_size % 2 != 0:
            raise GeometryError("patch size must be positive and divisible by 2")
        if not 0 < self.stride <= self.patch_size:
            raise GeometryError("patch stride must be in (0, patch_size]")
        x_starts = _dimension_starts(width, self.patch_size, self.stride)
        y_starts = _dimension_starts(height, self.patch_size, self.stride)
        placements: list[PatchPlacement] = []
        for y in y_starts:
            for x in x_starts:
                placements.append(
                    PatchPlacement(
                        index=len(placements),
                        x=x,
                        y=y,
                        valid_width=min(self.patch_size, width - x),
                        valid_height=min(self.patch_size, height - y),
                    )
                )
        object.__setattr__(self, "x_starts", x_starts)
        object.__setattr__(self, "y_starts", y_starts)
        object.__setattr__(self, "placements", tuple(placements))

    def __len__(self) -> int:
        return len(self.placements)


class TiledAccumulator:
    """Accumulate activated patch maps with a validity-masked Hann window."""

    def __init__(
        self,
        grid: PatchGrid,
        *,
        image_size: tuple[int, int],
        hann_weight_floor: float | None = None,
    ) -> None:
        if image_size != grid.image_size:
            raise GeometryError("merge image_size must match the patch grid")
        self.grid = grid
        self.image_size = image_size
        self.hann_weight_floor = _resolve_hann_weight_floor(hann_weight_floor)
        self._sum_maps: dict[str, torch.Tensor] | None = None
        self._sum_weight: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._added = 0

    def add(self, placement: PatchPlacement, output: PatchOutput) -> None:
        """Add the next row-major patch after activation and stride-2 resize."""
        if self._added >= len(self.grid):
            raise GeometryError("more patch outputs than grid placements")
        expected = self.grid.placements[self._added]
        if placement != expected:
            raise GeometryError("patch outputs must follow row-major grid order")
        prepared, valid_mask = _prepare_patch(
            output,
            patch_size=self.grid.patch_size,
        )
        device = next(iter(prepared.values())).device
        if self._device is None:
            self._initialize(device)
        elif device != self._device:
            raise GeometryError("all patch outputs must use the same device")
        assert self._sum_maps is not None
        assert self._sum_weight is not None
        window = _hann_window(
            self.grid.patch_size,
            device=device,
            floor=self.hann_weight_floor,
        )
        geometry_mask = torch.zeros_like(valid_mask)
        geometry_mask[
            :,
            : placement.valid_height,
            : placement.valid_width,
        ] = 1.0
        local_weight = window * valid_mask * geometry_mask
        y_slice = slice(placement.y, placement.y + placement.valid_height)
        x_slice = slice(placement.x, placement.x + placement.valid_width)
        local_y = slice(0, placement.valid_height)
        local_x = slice(0, placement.valid_width)
        cropped_weight = local_weight[:, local_y, local_x]
        self._sum_weight[:, y_slice, x_slice].add_(cropped_weight)
        for name, value in prepared.items():
            self._sum_maps[name][:, y_slice, x_slice].add_(
                value[:, local_y, local_x] * cropped_weight
            )
        self._added += 1

    def finalize(self) -> FullResolutionMaps:
        """Return finite maps after exact coverage and orientation checks."""
        if self._added != len(self.grid):
            raise GeometryError(
                "merge requires one output per grid placement "
                f"(received {self._added}, expected {len(self.grid)})"
            )
        assert self._sum_maps is not None
        assert self._sum_weight is not None
        if torch.any(self._sum_weight <= 0.0):
            raise GeometryError("valid patch masks leave source pixels uncovered")
        denominator = self._sum_weight.clamp_min(1e-6)
        merged = {name: value / denominator for name, value in self._sum_maps.items()}
        for name in _ORIENTATION_MAPS.values():
            merged[name] = _normalize_orientation(merged[name])
        if any(not torch.isfinite(value).all() for value in merged.values()):
            raise GeometryError("merged prediction maps contain non-finite values")
        return cast(FullResolutionMaps, merged)

    def _initialize(self, device: torch.device) -> None:
        width, height = self.image_size
        self._device = device
        self._sum_maps = {
            name: torch.zeros(
                (channels, height, width),
                dtype=torch.float32,
                device=device,
            )
            for name, channels in _FULL_CHANNELS.items()
        }
        self._sum_weight = torch.zeros(
            (1, height, width),
            dtype=torch.float32,
            device=device,
        )


def merge_predictions(
    grid: PatchGrid,
    outputs: Iterable[PatchOutput],
    image_size: tuple[int, int],
    *,
    hann_weight_floor: float | None = None,
) -> FullResolutionMaps:
    """Merge one raw output per row-major grid patch into source-pixel maps.

    Args:
        grid: Deterministic placements over a ``(width, height)`` image.
        outputs: Per-patch raw maps shaped ``[C, patch_size/2, patch_size/2]``.
        image_size: Source image ``(width, height)`` in pixels.
        hann_weight_floor: Minimum Hann-window weight. ``None`` uses the
            validated postprocess default.

    Returns:
        Activated and Hann-merged maps shaped ``[C, height, width]``.
    """
    accumulator = TiledAccumulator(
        grid,
        image_size=image_size,
        hann_weight_floor=hann_weight_floor,
    )
    iterator = iter(outputs)
    for placement in grid.placements:
        try:
            output = next(iterator)
        except StopIteration as exc:
            raise GeometryError("merge requires one output per grid placement") from exc
        accumulator.add(placement, output)
    try:
        next(iterator)
    except StopIteration:
        return accumulator.finalize()
    raise GeometryError("merge requires one output per grid placement")


def _dimension_starts(length: int, patch_size: int, stride: int) -> tuple[int, ...]:
    if length <= patch_size:
        return (0,)
    final_start = length - patch_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _prepare_patch(
    output: PatchOutput,
    *,
    patch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not isinstance(output.predictions, Mapping):
        raise GeometryError("patch predictions must be a tensor mapping")
    raw = cast(Mapping[str, torch.Tensor], output.predictions)
    if set(raw) != set(_RAW_CHANNELS):
        raise GeometryError("patch prediction keys do not match the model contract")
    output_size = patch_size // 2
    device: torch.device | None = None
    resized: dict[str, torch.Tensor] = {}
    for name, channels in _RAW_CHANNELS.items():
        value = raw[name]
        expected_shape = (channels, output_size, output_size)
        if not isinstance(value, torch.Tensor):
            raise GeometryError(f"{name} must be a floating tensor")
        if value.shape != expected_shape or not value.is_floating_point():
            raise GeometryError(
                f"{name} must be a floating tensor shaped {expected_shape}"
            )
        if not torch.isfinite(value).all():
            raise GeometryError(f"{name} contains non-finite values")
        if device is None:
            device = value.device
        elif value.device != device:
            raise GeometryError("one patch output cannot span multiple devices")
        resized[name] = functional.interpolate(
            value.to(dtype=torch.float32).unsqueeze(0),
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    assert device is not None
    expected_mask_shape = (1, output_size, output_size)
    if not isinstance(output.valid_mask, torch.Tensor):
        raise GeometryError("patch valid_mask must be a boolean tensor")
    if (
        output.valid_mask.shape != expected_mask_shape
        or output.valid_mask.dtype != torch.bool
    ):
        raise GeometryError(
            f"patch valid_mask must be boolean and shaped {expected_mask_shape}"
        )
    if output.valid_mask.device != device:
        raise GeometryError("patch valid_mask must share the prediction device")
    valid_mask = functional.interpolate(
        output.valid_mask.to(dtype=torch.float32).unsqueeze(0),
        size=(patch_size, patch_size),
        mode="nearest",
    ).squeeze(0)
    prepared: dict[str, torch.Tensor] = {}
    for raw_name, output_name in _BINARY_MAPS.items():
        prepared[output_name] = torch.sigmoid(resized[raw_name])
    for raw_name, output_name in _ORIENTATION_MAPS.items():
        prepared[output_name] = _normalize_orientation(resized[raw_name])
    for raw_name, output_name in _REGRESSION_MAPS.items():
        prepared[output_name] = resized[raw_name]
    prepared[_TYPE_MAP[1]] = torch.softmax(resized[_TYPE_MAP[0]], dim=0)
    return prepared, valid_mask


def _resolve_hann_weight_floor(value: float | None) -> float:
    """Resolve and validate the configured Hann-window floor."""
    resolved = PostprocessConfig().hann_weight_floor if value is None else value
    if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise GeometryError("hann_weight_floor must be finite and in (0, 1]")
    return resolved


def _hann_window(
    size: int,
    *,
    device: torch.device,
    floor: float,
) -> torch.Tensor:
    one_dimensional = torch.hann_window(
        size,
        periodic=False,
        dtype=torch.float32,
        device=device,
    )
    return torch.outer(one_dimensional, one_dimensional).clamp_min(floor).unsqueeze(0)


def _normalize_orientation(value: torch.Tensor) -> torch.Tensor:
    return functional.normalize(
        value,
        dim=0,
        eps=_NORMALIZATION_EPSILON,
    )
