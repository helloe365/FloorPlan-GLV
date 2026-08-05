"""Convert one CubiCasa SVG/image pair into normalized training annotations."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Annotated, Protocol, cast

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import Field

from floorplan_glv.config.models import StrictConfigModel
from floorplan_glv.data.annotation_schema import (
    AnnotationError,
    HardNegativeAnnotation,
    ImageAnnotation,
    OpeningAnnotation,
    TrainingAnnotation,
    WallAnnotation,
)
from floorplan_glv.geometry.primitives import (
    GeometryError,
    Segment2D,
    undirected_angle_difference_deg,
)
from floorplan_glv.geometry.svg import SvgCategory, parse_svg_geometry

Point = tuple[float, float]
Segment = tuple[Point, Point]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveFinite = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
IndexArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]


class _Exterior(Protocol):
    coords: Iterable[tuple[float, float]]


class _Centroid(Protocol):
    x: float
    y: float


class _PolygonGeometry(Protocol):
    area: float
    is_valid: bool
    bounds: tuple[float, float, float, float]
    exterior: _Exterior
    centroid: _Centroid

    @property
    def minimum_rotated_rectangle(self) -> _PolygonGeometry: ...


class _RasterizePolygon(Protocol):
    def __call__(
        self,
        rows: npt.ArrayLike,
        columns: npt.ArrayLike,
        *,
        shape: tuple[int, int],
    ) -> tuple[IndexArray, IndexArray]: ...


class _Skeletonize(Protocol):
    def __call__(self, image: BoolArray) -> BoolArray: ...


_shapely_polygon = cast(
    "Callable[[tuple[Point, ...]], _PolygonGeometry]",
    import_module("shapely.geometry").Polygon,
)
_rasterize_polygon = cast(
    _RasterizePolygon,
    import_module("skimage.draw").polygon,
)
_skeletonize = cast(
    _Skeletonize,
    import_module("skimage.morphology").skeletonize,
)


class ConversionConfig(StrictConfigModel):
    """Configurable SVG geometry extraction and host-assignment gates."""

    rectangularity_threshold: Probability = 0.80
    irregular_wall_quality: Probability = 0.70
    irregular_opening_quality: Probability = 0.70
    irregular_wall_min_segment_length_px: PositiveFinite = 2.0
    host_max_distance_px: PositiveFinite = 12.0
    host_max_distance_wall_thickness_factor: PositiveFinite = 0.75
    host_max_angle_difference_deg: PositiveFinite = 12.0
    host_min_projected_overlap_ratio: Probability = 0.65


@dataclass(frozen=True, slots=True)
class ConversionRejection:
    """One rejected source object with enough context for dataset auditing."""

    source_id: str
    category: SvgCategory
    reason: str


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Normalized annotation plus non-fatal parser/converter audit records."""

    annotation: TrainingAnnotation
    rejections: tuple[ConversionRejection, ...]
    encountered_group_ids: tuple[str, ...]
    encountered_class_prefixes: tuple[str, ...]
    unknown_group_counts: dict[str, int]


def convert_cubicasa_sample(
    sample_dir: Path,
    config: ConversionConfig | None = None,
    *,
    sample_id: str | None = None,
    svg_filename: str = "model.svg",
    image_filename: str = "F1_scaled.png",
) -> ConversionResult:
    """Convert a CubiCasa sample in source-image pixel coordinates.

    Wall and opening segments are ``((x1, y1), (x2, y2))`` tuples. Irregular
    openings retain only mask geometry; irregular walls use raster skeletons.
    """
    resolved_config = config or ConversionConfig()
    svg_path = sample_dir / svg_filename
    image_path = sample_dir / image_filename
    if not svg_path.is_file():
        raise AnnotationError(f"missing CubiCasa SVG: {svg_path}")
    if not image_path.is_file():
        raise AnnotationError(f"missing CubiCasa image: {image_path}")

    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError as exc:
        raise AnnotationError(
            f"failed to read CubiCasa image {image_path}: {exc}"
        ) from exc

    try:
        parsed = parse_svg_geometry(svg_path)
    except GeometryError as exc:
        raise AnnotationError(
            f"failed to parse CubiCasa SVG {svg_path}: {exc}"
        ) from exc
    x_scale = width / parsed.width
    y_scale = height / parsed.height

    rejections = [
        ConversionRejection(item.source_id, item.category, item.reason)
        for item in parsed.rejections
    ]
    walls: list[WallAnnotation] = []
    pending_openings: list[OpeningAnnotation] = []
    hard_negatives: list[HardNegativeAnnotation] = []
    source_id_counts: Counter[str] = Counter()

    for item in parsed.objects:
        try:
            polygon_points = tuple((x * x_scale, y * y_scale) for x, y in item.polygon)
            polygon = _validated_polygon(
                polygon_points,
                width=width,
                height=height,
            )
            source_id_counts[item.source_id] += 1
            source_id = _unique_source_id(
                item.source_id,
                source_id_counts[item.source_id],
            )
            rectangularity, segment, thickness = _rectangle_geometry(polygon)
            if item.category == "wall":
                if rectangularity >= resolved_config.rectangularity_threshold:
                    walls.append(
                        WallAnnotation(
                            source_id=source_id,
                            polygon=polygon_points,
                            segment=segment,
                            thickness_px=thickness,
                            quality=rectangularity,
                        )
                    )
                else:
                    walls.extend(
                        _skeletonized_walls(
                            source_id,
                            polygon_points,
                            polygon,
                            resolved_config,
                        )
                    )
            elif item.category in {"door", "window"}:
                stable_segment = (
                    segment
                    if rectangularity >= resolved_config.rectangularity_threshold
                    else None
                )
                quality = (
                    rectangularity
                    if stable_segment is not None
                    else min(
                        rectangularity,
                        resolved_config.irregular_opening_quality,
                    )
                )
                pending_openings.append(
                    OpeningAnnotation(
                        source_id=source_id,
                        type=item.category,
                        polygon=polygon_points,
                        segment=stable_segment,
                        host_wall_source_id=None,
                        quality=quality,
                    )
                )
            else:
                hard_negatives.append(
                    HardNegativeAnnotation(
                        source_id=source_id,
                        polygon=polygon_points,
                        category="fixed_furniture",
                    )
                )
        except (GeometryError, ValueError) as exc:
            rejections.append(
                ConversionRejection(item.source_id, item.category, str(exc))
            )

    openings = tuple(
        _attach_opening(opening, walls, resolved_config) for opening in pending_openings
    )
    resolved_sample_id = sample_id or sample_dir.name
    try:
        annotation = TrainingAnnotation(
            sample_id=resolved_sample_id,
            image=ImageAnnotation(
                file_name="image.png",
                width=width,
                height=height,
                source="cubicasa5k",
            ),
            walls=tuple(walls),
            openings=openings,
            hard_negatives=tuple(hard_negatives),
        )
    except ValueError as exc:
        raise AnnotationError(
            f"normalized annotation is invalid for {sample_dir}: {exc}"
        ) from exc
    return ConversionResult(
        annotation=annotation,
        rejections=tuple(rejections),
        encountered_group_ids=parsed.encountered_group_ids,
        encountered_class_prefixes=parsed.encountered_class_prefixes,
        unknown_group_counts=parsed.unknown_group_counts,
    )


def _validated_polygon(
    points: tuple[Point, ...],
    *,
    width: int,
    height: int,
) -> _PolygonGeometry:
    if any(not (0.0 <= x <= width and 0.0 <= y <= height) for x, y in points):
        raise GeometryError("polygon is outside source-image bounds")
    polygon = _shapely_polygon(points)
    if not polygon.is_valid or polygon.area <= 0.0:
        raise GeometryError("polygon is invalid or degenerate")
    return polygon


def _rectangle_geometry(
    polygon: _PolygonGeometry,
) -> tuple[float, Segment, float]:
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.area <= 0.0:
        raise GeometryError("minimum rotated rectangle is degenerate")
    rectangle_coordinates = tuple(rectangle.exterior.coords)
    corners = tuple((float(x), float(y)) for x, y in rectangle_coordinates[:-1])
    edges = tuple(
        (
            corners[index],
            corners[(index + 1) % len(corners)],
            math.dist(corners[index], corners[(index + 1) % len(corners)]),
        )
        for index in range(len(corners))
    )
    start, end, length = max(edges, key=lambda edge: edge[2])
    thickness = min(edge[2] for edge in edges)
    direction = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
    center = (float(polygon.centroid.x), float(polygon.centroid.y))
    half = length * 0.5
    segment = _canonical_segment(
        (
            (center[0] - direction[0] * half, center[1] - direction[1] * half),
            (center[0] + direction[0] * half, center[1] + direction[1] * half),
        )
    )
    rectangularity = min(1.0, float(polygon.area / rectangle.area))
    return rectangularity, segment, thickness


def _skeletonized_walls(
    source_id: str,
    polygon_points: tuple[Point, ...],
    polygon: _PolygonGeometry,
    config: ConversionConfig,
) -> tuple[WallAnnotation, ...]:
    min_x, min_y, max_x, max_y = polygon.bounds
    origin_x = math.floor(min_x) - 1
    origin_y = math.floor(min_y) - 1
    mask_width = math.ceil(max_x) - origin_x + 2
    mask_height = math.ceil(max_y) - origin_y + 2
    x_values = np.asarray([point[0] - origin_x for point in polygon_points])
    y_values = np.asarray([point[1] - origin_y for point in polygon_points])
    rows, columns = _rasterize_polygon(
        y_values,
        x_values,
        shape=(mask_height, mask_width),
    )
    mask = np.zeros((mask_height, mask_width), dtype=bool)
    mask[rows, columns] = True
    skeleton = _skeletonize(mask)
    segments = _trace_skeleton(
        skeleton,
        origin_x=origin_x,
        origin_y=origin_y,
        minimum_length=config.irregular_wall_min_segment_length_px,
    )
    if not segments:
        raise GeometryError("irregular wall skeleton produced no stable segment")
    total_length = sum(math.dist(*segment) for segment in segments)
    thickness = max(float(polygon.area / total_length), 1e-6)
    _, _, rectangularity_thickness = _rectangle_geometry(polygon)
    thickness = min(thickness, rectangularity_thickness)
    quality = min(
        float(polygon.area / polygon.minimum_rotated_rectangle.area),
        config.irregular_wall_quality,
    )
    multiple = len(segments) > 1
    return tuple(
        WallAnnotation(
            source_id=f"{source_id}_{index:03d}" if multiple else source_id,
            polygon=polygon_points,
            segment=segment,
            thickness_px=thickness,
            quality=quality,
        )
        for index, segment in enumerate(segments, start=1)
    )


def _trace_skeleton(
    skeleton: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    minimum_length: float,
) -> tuple[Segment, ...]:
    pixels: set[tuple[int, int]] = {
        (int(point[0]), int(point[1])) for point in np.argwhere(skeleton)
    }
    if len(pixels) < 2:
        return ()

    def neighbors(pixel: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        row, column = pixel
        return tuple(
            sorted(
                (row + row_offset, column + column_offset)
                for row_offset in (-1, 0, 1)
                for column_offset in (-1, 0, 1)
                if (row_offset or column_offset)
                and (row + row_offset, column + column_offset) in pixels
            )
        )

    adjacency = {pixel: neighbors(pixel) for pixel in pixels}
    critical = {pixel for pixel, linked in adjacency.items() if len(linked) != 2}
    if not critical:
        critical = {min(pixels), max(pixels)}
    visited: set[frozenset[tuple[int, int]]] = set()
    segments: list[Segment] = []
    for start in sorted(critical):
        for next_pixel in adjacency[start]:
            edge = frozenset((start, next_pixel))
            if edge in visited:
                continue
            visited.add(edge)
            previous, current = start, next_pixel
            while current not in critical:
                candidates = tuple(
                    pixel for pixel in adjacency[current] if pixel != previous
                )
                if not candidates:
                    break
                following = candidates[0]
                visited.add(frozenset((current, following)))
                previous, current = current, following
            segment = _pixel_segment(start, current, origin_x, origin_y)
            if math.dist(*segment) >= minimum_length:
                segments.append(_canonical_segment(segment))
    if not segments:
        endpoints = sorted(
            pixels,
            key=lambda pixel: (
                pixel[0] + pixel[1],
                pixel[0],
                pixel[1],
            ),
        )
        fallback = _canonical_segment(
            _pixel_segment(endpoints[0], endpoints[-1], origin_x, origin_y)
        )
        if math.dist(*fallback) >= minimum_length:
            segments.append(fallback)
    return tuple(sorted(set(segments)))


def _pixel_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    origin_x: int,
    origin_y: int,
) -> Segment:
    return (
        (float(start[1] + origin_x), float(start[0] + origin_y)),
        (float(end[1] + origin_x), float(end[0] + origin_y)),
    )


def _attach_opening(
    opening: OpeningAnnotation,
    walls: list[WallAnnotation],
    config: ConversionConfig,
) -> OpeningAnnotation:
    if opening.segment is None:
        return opening
    candidate = Segment2D(*opening.segment)
    compatible: list[tuple[float, float, float, str]] = []
    for wall in walls:
        host = Segment2D(*wall.segment)
        distance = host.point_to_line_distance(candidate.midpoint)
        maximum_distance = max(
            config.host_max_distance_px,
            config.host_max_distance_wall_thickness_factor * wall.thickness_px,
        )
        angle = undirected_angle_difference_deg(
            host.undirected_angle_deg,
            candidate.undirected_angle_deg,
        )
        overlap = host.clipped_projected_overlap_ratio(candidate)
        if (
            distance <= maximum_distance
            and angle <= config.host_max_angle_difference_deg
            and overlap >= config.host_min_projected_overlap_ratio
        ):
            compatible.append((distance, angle, -overlap, wall.source_id))
    host_id = min(compatible)[3] if compatible else None
    return opening.model_copy(update={"host_wall_source_id": host_id})


def _canonical_segment(segment: Segment) -> Segment:
    return tuple(sorted(segment, key=lambda point: (point[1], point[0])))  # type: ignore[return-value]


def _unique_source_id(source_id: str, occurrence: int) -> str:
    return source_id if occurrence == 1 else f"{source_id}_{occurrence:03d}"
