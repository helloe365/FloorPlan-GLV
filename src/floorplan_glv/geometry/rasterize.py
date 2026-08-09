"""Pure deterministic rasterization of one local training patch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Self, TypedDict

import cv2
import numpy as np
import torch
from pydantic import Field, model_validator
from shapely.geometry import Polygon as ShapelyPolygon  # type: ignore[import-untyped]

from floorplan_glv.config.models import StrictConfigModel
from floorplan_glv.data.annotation_schema import (
    Polygon,
    TrainingAnnotation,
)
from floorplan_glv.geometry.primitives import Segment2D

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
PositiveProbability = Annotated[
    float,
    Field(gt=0.0, le=1.0, allow_inf_nan=False),
]
Point = tuple[float, float]


class TargetMaps(TypedDict):
    """All dense targets for one patch at the model output resolution."""

    wall_mask: torch.Tensor
    wall_centerline: torch.Tensor
    wall_junction: torch.Tensor
    wall_orientation: torch.Tensor
    wall_orientation_valid: torch.Tensor
    wall_log_half_thickness: torch.Tensor
    wall_thickness_valid: torch.Tensor
    opening_mask: torch.Tensor
    opening_center: torch.Tensor
    opening_endpoint: torch.Tensor
    opening_type: torch.Tensor
    opening_type_valid: torch.Tensor
    opening_orientation: torch.Tensor
    opening_orientation_valid: torch.Tensor
    opening_log_half_length: torch.Tensor
    opening_length_valid: torch.Tensor
    valid_pixels: torch.Tensor


TARGET_KEYS: tuple[str, ...] = tuple(TargetMaps.__annotations__)


class TargetGenerationConfig(StrictConfigModel):
    """Validated rasterization constants from the target specification."""

    wall_area_positive_fraction: PositiveProbability = 0.25
    area_supersample: PositiveInt = 4
    wall_centerline_radius_factor: PositiveFloat = 0.15
    wall_centerline_min_radius: PositiveFloat = 1.0
    wall_centerline_max_radius: PositiveFloat = 3.0
    junction_cluster_distance_source_px: PositiveFloat = 2.0
    heatmap_sigma_target_px: PositiveFloat = 2.0
    opening_geometry_radius_target_px: PositiveInt = 3
    segment_mask_half_width_source_px: PositiveFloat = 2.0

    @model_validator(mode="after")
    def validate_radius_range(self) -> Self:
        """Require a non-empty configured wall-centerline radius range."""
        if self.wall_centerline_min_radius > self.wall_centerline_max_radius:
            raise ValueError("wall centerline minimum radius exceeds maximum")
        return self


@dataclass(frozen=True, slots=True)
class PatchSample:
    """Annotation and source box needed to rasterize one local patch.

    The source box uses source-image boundary coordinates and may extend beyond
    the image. Padding outside the annotated image is represented by
    ``valid_pixels=False``.
    """

    annotation: TrainingAnnotation
    source_box: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.source_box
        if x1 <= x0 or y1 <= y0:
            raise ValueError("patch source box must have positive area")


def generate_patch_targets(
    patch: PatchSample,
    output_size: tuple[int, int],
    *,
    config: TargetGenerationConfig | None = None,
) -> TargetMaps:
    """Generate all model targets directly at ``output_size``.

    Args:
        patch: Source-coordinate annotation and local source box.
        output_size: ``(height, width)`` of every generated spatial target.
        config: Validated rasterization constants.

    Returns:
        Float targets with shape ``[1, H, W]`` or ``[2, H, W]``, an opening
        type target with shape ``[H, W]``, and boolean validity targets with
        shape ``[1, H, W]``.
    """
    height, width = output_size
    if height <= 0 or width <= 0:
        raise ValueError("target output dimensions must be positive")
    settings = config or TargetGenerationConfig()
    x0, y0, x1, y1 = patch.source_box
    scale_x = width / float(x1 - x0)
    scale_y = height / float(y1 - y0)
    scale = math.sqrt(scale_x * scale_y)
    valid = _valid_pixel_mask(patch, output_size)

    wall_mask = _rasterize_polygons(
        tuple(wall.polygon for wall in patch.annotation.walls),
        patch.source_box,
        output_size,
        settings,
    )
    wall_centerline = np.zeros((height, width), dtype=np.float32)
    wall_junction = np.zeros((height, width), dtype=np.float32)
    wall_orientation = np.zeros((2, height, width), dtype=np.float32)
    wall_orientation_valid = np.zeros((height, width), dtype=np.bool_)
    wall_log_half_thickness = np.zeros((height, width), dtype=np.float32)
    wall_thickness_valid = np.zeros((height, width), dtype=np.bool_)

    for wall in patch.annotation.walls:
        segment_mask = np.zeros((height, width), dtype=np.uint8)
        radius = max(
            settings.wall_centerline_min_radius,
            min(
                settings.wall_centerline_max_radius,
                settings.wall_centerline_radius_factor * wall.thickness_px * scale,
            ),
        )
        _draw_segment(
            segment_mask,
            wall.segment,
            patch.source_box,
            output_size,
            thickness=2 * math.ceil(radius) + 1,
        )
        band = segment_mask.astype(np.bool_) & valid
        wall_centerline[band] = 1.0
        sin_2theta, cos_2theta = _orientation(wall.segment)
        wall_orientation[0, band] = sin_2theta
        wall_orientation[1, band] = cos_2theta
        wall_orientation_valid[band] = True
        wall_log_half_thickness[band] = math.log1p(0.5 * wall.thickness_px)
        wall_thickness_valid[band] = True

    for junction in _wall_junctions(
        patch.annotation,
        settings.junction_cluster_distance_source_px,
    ):
        _compose_gaussian(
            wall_junction,
            _map_point(junction, patch.source_box, output_size),
            settings.heatmap_sigma_target_px,
        )

    opening_mask = _rasterize_polygons(
        tuple(
            opening.polygon
            for opening in patch.annotation.openings
            if opening.polygon is not None
        ),
        patch.source_box,
        output_size,
        settings,
    )
    opening_center = np.zeros((height, width), dtype=np.float32)
    opening_endpoint = np.zeros((height, width), dtype=np.float32)
    opening_type = np.zeros((height, width), dtype=np.int64)
    opening_type_valid = np.zeros((height, width), dtype=np.bool_)
    opening_orientation = np.zeros((2, height, width), dtype=np.float32)
    opening_orientation_valid = np.zeros((height, width), dtype=np.bool_)
    opening_log_half_length = np.zeros((height, width), dtype=np.float32)
    opening_length_valid = np.zeros((height, width), dtype=np.bool_)

    for opening in patch.annotation.openings:
        if opening.polygon is None:
            segment_mask = np.zeros(output_size, dtype=np.uint8)
            assert opening.segment is not None
            _draw_segment(
                segment_mask,
                opening.segment,
                patch.source_box,
                output_size,
                thickness=max(
                    1,
                    2 * math.ceil(settings.segment_mask_half_width_source_px * scale)
                    + 1,
                ),
            )
            opening_mask = np.maximum(opening_mask, segment_mask)
        center = (
            _segment_midpoint(opening.segment)
            if opening.segment is not None
            else _polygon_representative_point(opening.polygon)
        )
        assert center is not None
        if not _point_is_in_patch(center, patch):
            continue
        center_target = _map_point(center, patch.source_box, output_size)
        _compose_gaussian(
            opening_center,
            center_target,
            settings.heatmap_sigma_target_px,
        )
        geometry_disk = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            geometry_disk,
            _rounded_pixel(center_target),
            settings.opening_geometry_radius_target_px,
            color=1,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        geometry_valid = geometry_disk.astype(np.bool_) & valid
        opening_type[geometry_valid] = 0 if opening.type == "door" else 1
        opening_type_valid[geometry_valid] = True
        if opening.segment is None:
            continue
        for endpoint in opening.segment:
            _compose_gaussian(
                opening_endpoint,
                _map_point(endpoint, patch.source_box, output_size),
                settings.heatmap_sigma_target_px,
            )
        sin_2theta, cos_2theta = _orientation(opening.segment)
        opening_orientation[0, geometry_valid] = sin_2theta
        opening_orientation[1, geometry_valid] = cos_2theta
        opening_orientation_valid[geometry_valid] = True
        start, end = opening.segment
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        opening_log_half_length[geometry_valid] = math.log1p(0.5 * length)
        opening_length_valid[geometry_valid] = True

    valid_float = valid.astype(np.float32)
    wall_mask *= valid_float
    wall_centerline *= valid_float
    wall_junction *= valid_float
    opening_mask *= valid_float
    opening_center *= valid_float
    opening_endpoint *= valid_float

    return TargetMaps(
        wall_mask=_float_scalar(wall_mask),
        wall_centerline=_float_scalar(wall_centerline),
        wall_junction=_float_scalar(wall_junction),
        wall_orientation=torch.from_numpy(wall_orientation),
        wall_orientation_valid=_bool_scalar(wall_orientation_valid),
        wall_log_half_thickness=_float_scalar(wall_log_half_thickness),
        wall_thickness_valid=_bool_scalar(wall_thickness_valid),
        opening_mask=_float_scalar(opening_mask),
        opening_center=_float_scalar(opening_center),
        opening_endpoint=_float_scalar(opening_endpoint),
        opening_type=torch.from_numpy(opening_type),
        opening_type_valid=_bool_scalar(opening_type_valid),
        opening_orientation=torch.from_numpy(opening_orientation),
        opening_orientation_valid=_bool_scalar(opening_orientation_valid),
        opening_log_half_length=_float_scalar(opening_log_half_length),
        opening_length_valid=_bool_scalar(opening_length_valid),
        valid_pixels=_bool_scalar(valid),
    )


def _valid_pixel_mask(
    patch: PatchSample,
    output_size: tuple[int, int],
) -> np.ndarray:
    height, width = output_size
    x0, y0, x1, y1 = patch.source_box
    source_x = x0 + (np.arange(width, dtype=np.float64) + 0.5) * ((x1 - x0) / width)
    source_y = y0 + (np.arange(height, dtype=np.float64) + 0.5) * ((y1 - y0) / height)
    valid_x = (source_x >= 0.0) & (source_x < patch.annotation.image.width)
    valid_y = (source_y >= 0.0) & (source_y < patch.annotation.image.height)
    return valid_y[:, None] & valid_x[None, :]


def _rasterize_polygons(
    polygons: tuple[Polygon, ...],
    source_box: tuple[int, int, int, int],
    output_size: tuple[int, int],
    config: TargetGenerationConfig,
) -> np.ndarray:
    height, width = output_size
    supersample = config.area_supersample
    canvas = np.zeros(
        (height * supersample, width * supersample),
        dtype=np.uint8,
    )
    target_size = (height * supersample, width * supersample)
    contours = [
        np.rint(
            np.asarray(
                [_map_point(point, source_box, target_size) for point in polygon],
                dtype=np.float64,
            )
        ).astype(np.int32)
        for polygon in polygons
    ]
    for contour in contours:
        cv2.fillPoly(canvas, [contour], color=1, lineType=cv2.LINE_8)
    coverage = canvas.reshape(
        height,
        supersample,
        width,
        supersample,
    ).mean(axis=(1, 3))
    return np.asarray(
        coverage >= config.wall_area_positive_fraction,
        dtype=np.float32,
    )


def _draw_segment(
    canvas: np.ndarray,
    segment: tuple[Point, Point],
    source_box: tuple[int, int, int, int],
    output_size: tuple[int, int],
    *,
    thickness: int,
) -> None:
    cv2.line(
        canvas,
        _rounded_pixel(_map_point(segment[0], source_box, output_size)),
        _rounded_pixel(_map_point(segment[1], source_box, output_size)),
        color=1,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )


def _map_point(
    point: Point,
    source_box: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> Point:
    height, width = output_size
    x0, y0, x1, y1 = source_box
    return (
        (point[0] - x0) * width / (x1 - x0),
        (point[1] - y0) * height / (y1 - y0),
    )


def _rounded_pixel(point: Point) -> tuple[int, int]:
    return (round(point[0]), round(point[1]))


def _orientation(segment: tuple[Point, Point]) -> tuple[float, float]:
    start, end = segment
    theta = math.atan2(end[1] - start[1], end[0] - start[0])
    return (math.sin(2.0 * theta), math.cos(2.0 * theta))


def _segment_midpoint(segment: tuple[Point, Point]) -> Point:
    return (
        0.5 * (segment[0][0] + segment[1][0]),
        0.5 * (segment[0][1] + segment[1][1]),
    )


def _polygon_representative_point(polygon: Polygon | None) -> Point | None:
    if polygon is None:
        return None
    point = ShapelyPolygon(polygon).representative_point()
    return (float(point.x), float(point.y))


def _compose_gaussian(
    heatmap: np.ndarray,
    center: Point,
    sigma: float,
) -> None:
    height, width = heatmap.shape
    center_x, center_y = _rounded_pixel(center)
    radius = math.ceil(3.0 * sigma)
    left = max(0, center_x - radius)
    right = min(width - 1, center_x + radius)
    top = max(0, center_y - radius)
    bottom = min(height - 1, center_y + radius)
    if left > right or top > bottom:
        return
    x = np.arange(left, right + 1, dtype=np.float32) - center_x
    y = np.arange(top, bottom + 1, dtype=np.float32) - center_y
    gaussian = np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2.0 * sigma**2))
    target = heatmap[top : bottom + 1, left : right + 1]
    np.maximum(target, gaussian, out=target)


def _wall_junctions(
    annotation: TrainingAnnotation,
    cluster_distance: float,
) -> tuple[Point, ...]:
    segments = tuple(Segment2D(*wall.segment) for wall in annotation.walls)
    candidates = [point for wall in annotation.walls for point in wall.segment]
    for index, first in enumerate(segments):
        for second in segments[index + 1 :]:
            intersection = first.intersection(second)
            if intersection is not None:
                candidates.append(intersection.as_tuple())
    candidates.sort(key=lambda point: (point[1], point[0]))
    clusters: list[list[Point]] = []
    for point in candidates:
        for cluster in clusters:
            center = _mean_point(cluster)
            if math.dist(point, center) <= cluster_distance:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    return tuple(_mean_point(cluster) for cluster in clusters)


def _mean_point(points: list[Point]) -> Point:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _point_is_in_patch(point: Point, patch: PatchSample) -> bool:
    x0, y0, x1, y1 = patch.source_box
    return (
        0.0 <= point[0] < patch.annotation.image.width
        and 0.0 <= point[1] < patch.annotation.image.height
        and x0 <= point[0] < x1
        and y0 <= point[1] < y1
    )


def _float_scalar(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value[np.newaxis, ...].astype(np.float32, copy=False))


def _bool_scalar(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value[np.newaxis, ...].astype(np.bool_, copy=False))
