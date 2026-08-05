"""Synchronized affine transforms for images and normalized annotations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

from PIL import Image
from shapely.geometry import LineString, Polygon, box  # type: ignore[import-untyped]

from floorplan_glv.data.annotation_schema import (
    HardNegativeAnnotation,
    ImageAnnotation,
    OpeningAnnotation,
    TrainingAnnotation,
    WallAnnotation,
)
from floorplan_glv.geometry.primitives import GeometryError

Point: TypeAlias = tuple[float, float]
Segment: TypeAlias = tuple[Point, Point]
Matrix: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
ImageSize: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GeometricTransform:
    """A finite similarity transform between two image canvases."""

    matrix: Matrix
    input_size: ImageSize
    output_size: ImageSize
    minimum_retained_area_ratio: float = 0.20

    def __post_init__(self) -> None:
        if len(self.matrix) != 3 or any(len(row) != 3 for row in self.matrix):
            raise GeometryError("transform matrix must be 3 by 3")
        values = tuple(value for row in self.matrix for value in row)
        if not all(math.isfinite(value) for value in values):
            raise GeometryError("transform matrix values must be finite")
        if self.matrix[2] != (0.0, 0.0, 1.0):
            raise GeometryError("transform must be a two-dimensional affine matrix")
        if any(dimension <= 0 for dimension in (*self.input_size, *self.output_size)):
            raise GeometryError("transform image dimensions must be positive")
        if not 0.0 <= self.minimum_retained_area_ratio <= 1.0:
            raise GeometryError("minimum retained area ratio must be within [0, 1]")
        first_norm = math.hypot(self.matrix[0][0], self.matrix[1][0])
        second_norm = math.hypot(self.matrix[0][1], self.matrix[1][1])
        dot_product = (
            self.matrix[0][0] * self.matrix[0][1]
            + self.matrix[1][0] * self.matrix[1][1]
        )
        if (
            first_norm <= 0.0
            or second_norm <= 0.0
            or not math.isclose(first_norm, second_norm, abs_tol=1e-9)
            or not math.isclose(dot_product, 0.0, abs_tol=1e-9)
        ):
            raise GeometryError("transform must preserve angles with uniform scale")

    @property
    def scale_factor(self) -> float:
        """Return the uniform length and thickness scale."""
        determinant = (
            self.matrix[0][0] * self.matrix[1][1]
            - self.matrix[0][1] * self.matrix[1][0]
        )
        return math.sqrt(abs(determinant))

    def transform_point(self, point: Point) -> Point:
        """Map one source boundary-coordinate point to the output canvas."""
        x, y = point
        return (
            self.matrix[0][0] * x + self.matrix[0][1] * y + self.matrix[0][2],
            self.matrix[1][0] * x + self.matrix[1][1] * y + self.matrix[1][2],
        )

    @classmethod
    def horizontal_flip(cls, image_size: ImageSize) -> GeometricTransform:
        """Reflect around the vertical centerline of an image canvas."""
        width, height = image_size
        return cls(
            matrix=((-1.0, 0.0, float(width)), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            input_size=image_size,
            output_size=(width, height),
        )

    @classmethod
    def vertical_flip(cls, image_size: ImageSize) -> GeometricTransform:
        """Reflect around the horizontal centerline of an image canvas."""
        width, height = image_size
        return cls(
            matrix=((1.0, 0.0, 0.0), (0.0, -1.0, float(height)), (0.0, 0.0, 1.0)),
            input_size=image_size,
            output_size=(width, height),
        )

    @classmethod
    def rotate90_clockwise(cls, image_size: ImageSize) -> GeometricTransform:
        """Rotate an image canvas clockwise by exactly 90 degrees."""
        width, height = image_size
        return cls(
            matrix=((0.0, -1.0, float(height)), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            input_size=image_size,
            output_size=(height, width),
        )

    @classmethod
    def uniform_scale(
        cls,
        image_size: ImageSize,
        factor: float,
    ) -> GeometricTransform:
        """Scale an image canvas and all geometry uniformly from the origin."""
        if not math.isfinite(factor) or factor <= 0.0:
            raise GeometryError("uniform scale factor must be finite and positive")
        width, height = image_size
        output_size = (
            max(1, round(width * factor)),
            max(1, round(height * factor)),
        )
        return cls(
            matrix=((factor, 0.0, 0.0), (0.0, factor, 0.0), (0.0, 0.0, 1.0)),
            input_size=image_size,
            output_size=output_size,
        )

    @classmethod
    def translation(
        cls,
        image_size: ImageSize,
        delta_x: float,
        delta_y: float,
    ) -> GeometricTransform:
        """Translate within an unchanged image canvas."""
        return cls(
            matrix=((1.0, 0.0, delta_x), (0.0, 1.0, delta_y), (0.0, 0.0, 1.0)),
            input_size=image_size,
            output_size=image_size,
        )


def apply_transform(
    image: Image.Image,
    annotation: TrainingAnnotation,
    transform: GeometricTransform,
) -> tuple[Image.Image, TrainingAnnotation]:
    """Transform image pixels and vector annotations together.

    Polygon and segment coordinates remain in source-image pixel boundary
    coordinates. Objects retaining less than the transform's configured area
    ratio are dropped; retained partial polygons and segments are clipped.
    """
    annotation_size = (annotation.image.width, annotation.image.height)
    if image.size != annotation_size or image.size != transform.input_size:
        raise GeometryError(
            "image dimensions, annotation dimensions, and transform input "
            "dimensions must match"
        )

    transformed_image = image.transform(
        transform.output_size,
        Image.Transform.AFFINE,
        _inverse_affine_coefficients(transform.matrix),
        resample=Image.Resampling.NEAREST,
        fillcolor=_white_fill(image.mode),
    )
    walls = tuple(
        wall
        for source_wall in annotation.walls
        if (wall := _transform_wall(source_wall, transform)) is not None
    )
    retained_wall_ids = {wall.source_id for wall in walls}
    openings = tuple(
        opening
        for source_opening in annotation.openings
        if (
            opening := _transform_opening(
                source_opening,
                transform,
                retained_wall_ids,
            )
        )
        is not None
    )
    hard_negatives = tuple(
        negative
        for source_negative in annotation.hard_negatives
        if (negative := _transform_hard_negative(source_negative, transform))
        is not None
    )
    transformed_annotation = TrainingAnnotation(
        schema_version=annotation.schema_version,
        sample_id=annotation.sample_id,
        image=ImageAnnotation(
            file_name=annotation.image.file_name,
            width=transform.output_size[0],
            height=transform.output_size[1],
            source=annotation.image.source,
        ),
        walls=walls,
        openings=openings,
        hard_negatives=hard_negatives,
    )
    return transformed_image, transformed_annotation


def _transform_wall(
    wall: WallAnnotation,
    transform: GeometricTransform,
) -> WallAnnotation | None:
    polygon = _transform_and_clip_polygon(wall.polygon, transform)
    if polygon is None:
        return None
    segment = _transform_and_clip_segment(wall.segment, transform)
    if segment is None:
        return None
    return WallAnnotation(
        source_id=wall.source_id,
        polygon=polygon,
        segment=segment,
        thickness_px=wall.thickness_px * transform.scale_factor,
        quality=wall.quality,
    )


def _transform_opening(
    opening: OpeningAnnotation,
    transform: GeometricTransform,
    retained_wall_ids: set[str],
) -> OpeningAnnotation | None:
    polygon = (
        _transform_and_clip_polygon(opening.polygon, transform)
        if opening.polygon is not None
        else None
    )
    if opening.polygon is not None and polygon is None:
        return None
    segment = (
        _transform_and_clip_segment(opening.segment, transform)
        if opening.segment is not None
        else None
    )
    if polygon is None and segment is None:
        return None
    host_id = opening.host_wall_source_id
    if host_id not in retained_wall_ids or segment is None:
        host_id = None
    return OpeningAnnotation(
        source_id=opening.source_id,
        type=opening.type,
        polygon=polygon,
        segment=segment,
        host_wall_source_id=host_id,
        quality=opening.quality,
    )


def _transform_hard_negative(
    negative: HardNegativeAnnotation,
    transform: GeometricTransform,
) -> HardNegativeAnnotation | None:
    polygon = _transform_and_clip_polygon(negative.polygon, transform)
    if polygon is None:
        return None
    return HardNegativeAnnotation(
        source_id=negative.source_id,
        polygon=polygon,
        category=negative.category,
    )


def _transform_and_clip_polygon(
    points: tuple[Point, ...],
    transform: GeometricTransform,
) -> tuple[Point, ...] | None:
    transformed = tuple(transform.transform_point(point) for point in points)
    original = Polygon(transformed)
    if original.is_empty or original.area <= 0.0:
        return None
    width, height = transform.output_size
    if all(0.0 <= x <= width and 0.0 <= y <= height for x, y in transformed):
        return transformed
    clipped = original.intersection(box(0.0, 0.0, float(width), float(height)))
    polygon = _largest_polygon(clipped)
    if polygon is None:
        return None
    retained_ratio = float(polygon.area / original.area)
    if retained_ratio < transform.minimum_retained_area_ratio:
        return None
    return tuple((float(x), float(y)) for x, y in polygon.exterior.coords[:-1])


def _largest_polygon(geometry: Any) -> Any | None:
    if getattr(geometry, "geom_type", None) == "Polygon":
        return geometry
    geometries = tuple(
        item
        for item in getattr(geometry, "geoms", ())
        if getattr(item, "geom_type", None) == "Polygon"
        and float(getattr(item, "area", 0.0)) > 0.0
    )
    return max(geometries, key=lambda item: float(item.area), default=None)


def _transform_and_clip_segment(
    segment: Segment,
    transform: GeometricTransform,
) -> Segment | None:
    transformed = tuple(transform.transform_point(point) for point in segment)
    width, height = transform.output_size
    if all(0.0 <= x <= width and 0.0 <= y <= height for x, y in transformed):
        return transformed  # type: ignore[return-value]
    clipped = LineString(transformed).intersection(
        box(0.0, 0.0, float(width), float(height))
    )
    lines = (
        (clipped,)
        if getattr(clipped, "geom_type", None) == "LineString"
        else tuple(
            item
            for item in getattr(clipped, "geoms", ())
            if getattr(item, "geom_type", None) == "LineString"
        )
    )
    line = max(lines, key=lambda item: float(item.length), default=None)
    if line is None or float(line.length) <= 0.0:
        return None
    coordinates = tuple((float(x), float(y)) for x, y in line.coords)
    return (coordinates[0], coordinates[-1])


def _inverse_affine_coefficients(
    matrix: Matrix,
) -> tuple[float, float, float, float, float, float]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    determinant = a * e - b * d
    if math.isclose(determinant, 0.0, abs_tol=1e-12):
        raise GeometryError("transform matrix is singular")
    return (
        e / determinant,
        -b / determinant,
        (b * f - e * c) / determinant,
        -d / determinant,
        a / determinant,
        (d * c - a * f) / determinant,
    )


def _white_fill(mode: str) -> int | tuple[int, ...]:
    if mode == "RGBA":
        return (255, 255, 255, 255)
    if mode == "RGB":
        return (255, 255, 255)
    return 255
