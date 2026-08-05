"""Immutable two-dimensional geometry primitives in source-image pixels."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

EPSILON = 1e-8


class GeometryError(ValueError):
    """Raised when geometric input is non-finite or degenerate."""


@dataclass(frozen=True, slots=True)
class Point2D:
    """A finite point in source-image pixel coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        try:
            x = float(self.x)
            y = float(self.y)
        except (TypeError, ValueError) as exc:
            raise GeometryError("point coordinates must be numeric") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise GeometryError("point coordinates must be finite")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def as_tuple(self) -> tuple[float, float]:
        """Return the point as an ``(x, y)`` tuple."""
        return (self.x, self.y)


PointLike = Point2D | tuple[float, float]


@dataclass(frozen=True, slots=True, init=False)
class Segment2D:
    """A finite, non-degenerate directed segment."""

    start: Point2D
    end: Point2D

    def __init__(self, start: PointLike, end: PointLike) -> None:
        start_point = _as_point(start)
        end_point = _as_point(end)
        if distance(start_point, end_point) <= EPSILON:
            raise GeometryError("segment endpoints must be distinct")
        object.__setattr__(self, "start", start_point)
        object.__setattr__(self, "end", end_point)

    @property
    def length(self) -> float:
        """Return Euclidean segment length in pixels."""
        return distance(self.start, self.end)

    @property
    def midpoint(self) -> Point2D:
        """Return the point halfway between both endpoints."""
        return Point2D(
            0.5 * (self.start.x + self.end.x),
            0.5 * (self.start.y + self.end.y),
        )

    @property
    def normalized_direction(self) -> tuple[float, float]:
        """Return the unit vector from ``start`` to ``end``."""
        return (
            (self.end.x - self.start.x) / self.length,
            (self.end.y - self.start.y) / self.length,
        )

    @property
    def undirected_angle_deg(self) -> float:
        """Return orientation in degrees modulo 180."""
        dx, dy = self.normalized_direction
        return math.degrees(math.atan2(dy, dx)) % 180.0

    def project_parameter(self, point: PointLike) -> float:
        """Project a point onto the infinite line as a segment parameter."""
        return projection_parameter(point, self)

    def point_to_line_distance(self, point: PointLike) -> float:
        """Return perpendicular distance to the infinite supporting line."""
        return point_to_line_distance(point, self)

    def point_to_segment_distance(self, point: PointLike) -> float:
        """Return shortest distance to the finite segment."""
        return point_to_segment_distance(point, self)

    def clipped_projected_overlap_ratio(self, candidate: Segment2D) -> float:
        """Return how much of a projected candidate lies within this segment."""
        return clipped_projected_overlap_ratio(self, candidate)

    def intersection(self, other: Segment2D) -> Point2D | None:
        """Return the unique finite intersection point, when one exists."""
        return segment_intersection(self, other)


@dataclass(frozen=True, slots=True, init=False)
class Polygon2D:
    """A finite polygon with at least three distinct, non-collinear points."""

    points: tuple[Point2D, ...]

    def __init__(self, points: Iterable[PointLike]) -> None:
        point_tuple = tuple(_as_point(point) for point in points)
        distinct = {(point.x, point.y) for point in point_tuple}
        if len(point_tuple) < 3 or len(distinct) < 3:
            raise GeometryError("polygon must contain at least three distinct points")
        object.__setattr__(self, "points", point_tuple)
        if self.area <= EPSILON:
            raise GeometryError("polygon area must be greater than zero")

    @property
    def area(self) -> float:
        """Return absolute polygon area in square pixels."""
        return abs(_signed_double_area(self.points)) * 0.5

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``(min_x, min_y, max_x, max_y)``."""
        x_values = tuple(point.x for point in self.points)
        y_values = tuple(point.y for point in self.points)
        return (
            min(x_values),
            min(y_values),
            max(x_values),
            max(y_values),
        )


def distance(first: PointLike, second: PointLike) -> float:
    """Return Euclidean distance between two points."""
    first_point = _as_point(first)
    second_point = _as_point(second)
    return math.hypot(
        second_point.x - first_point.x,
        second_point.y - first_point.y,
    )


def midpoint(segment: Segment2D) -> Point2D:
    """Return a segment midpoint."""
    return segment.midpoint


def normalized_direction(segment: Segment2D) -> tuple[float, float]:
    """Return a segment unit direction vector."""
    return segment.normalized_direction


def undirected_angle_deg(segment: Segment2D) -> float:
    """Return a segment orientation modulo 180 degrees."""
    return segment.undirected_angle_deg


def undirected_angle_difference_deg(first: float, second: float) -> float:
    """Return the smallest difference between undirected angles in degrees."""
    try:
        first_angle = float(first)
        second_angle = float(second)
    except (TypeError, ValueError) as exc:
        raise GeometryError("angles must be numeric") from exc
    if not (math.isfinite(first_angle) and math.isfinite(second_angle)):
        raise GeometryError("angles must be finite")
    difference = abs((first_angle - second_angle) % 180.0)
    return min(difference, 180.0 - difference)


def projection_parameter(point: PointLike, segment: Segment2D) -> float:
    """Return projection parameter on an infinite supporting line."""
    projected_point = _as_point(point)
    vx = segment.end.x - segment.start.x
    vy = segment.end.y - segment.start.y
    denominator = vx * vx + vy * vy
    return (
        (projected_point.x - segment.start.x) * vx
        + (projected_point.y - segment.start.y) * vy
    ) / denominator


def point_to_line_distance(point: PointLike, segment: Segment2D) -> float:
    """Return perpendicular distance to a segment's infinite line."""
    projected_point = _point_at(segment, projection_parameter(point, segment))
    return distance(point, projected_point)


def point_to_segment_distance(point: PointLike, segment: Segment2D) -> float:
    """Return shortest distance from a point to a finite segment."""
    parameter = min(1.0, max(0.0, projection_parameter(point, segment)))
    return distance(point, _point_at(segment, parameter))


def clipped_projected_overlap_ratio(
    host: Segment2D,
    candidate: Segment2D,
) -> float:
    """Return candidate projection overlap with a finite host segment.

    Both candidate endpoints are projected onto the host line. The clipped
    overlap is divided by the candidate's projected length, so a candidate
    fully contained along the host direction returns one.
    """
    first = projection_parameter(candidate.start, host)
    second = projection_parameter(candidate.end, host)
    low, high = sorted((first, second))
    projected_length = high - low
    if projected_length <= EPSILON:
        return 0.0
    overlap = max(0.0, min(high, 1.0) - max(low, 0.0))
    return min(1.0, overlap / projected_length)


def segment_intersection(
    first: Segment2D,
    second: Segment2D,
) -> Point2D | None:
    """Return the unique intersection of two finite segments.

    Parallel disjoint segments and collinear segments with non-zero overlap do
    not have a unique intersection and return ``None``. Collinear segments
    touching at exactly one endpoint return that endpoint.
    """
    p = first.start
    q = second.start
    r = (first.end.x - first.start.x, first.end.y - first.start.y)
    s = (second.end.x - second.start.x, second.end.y - second.start.y)
    r_cross_s = _cross(r, s)
    q_minus_p = (q.x - p.x, q.y - p.y)

    if abs(r_cross_s) <= EPSILON:
        if abs(_cross(q_minus_p, r)) > EPSILON:
            return None
        shared = _shared_endpoints(first, second)
        return shared[0] if len(shared) == 1 else None

    first_parameter = _cross(q_minus_p, s) / r_cross_s
    second_parameter = _cross(q_minus_p, r) / r_cross_s
    if (
        -EPSILON <= first_parameter <= 1.0 + EPSILON
        and -EPSILON <= second_parameter <= 1.0 + EPSILON
    ):
        return _point_at(first, min(1.0, max(0.0, first_parameter)))
    return None


def polygon_area(polygon: Polygon2D) -> float:
    """Return polygon area in square pixels."""
    return polygon.area


def polygon_bounds(polygon: Polygon2D) -> tuple[float, float, float, float]:
    """Return polygon axis-aligned bounds."""
    return polygon.bounds


def _as_point(point: PointLike) -> Point2D:
    if isinstance(point, Point2D):
        return point
    try:
        x, y = point
    except (TypeError, ValueError) as exc:
        raise GeometryError("point must contain exactly two coordinates") from exc
    return Point2D(x, y)


def _point_at(segment: Segment2D, parameter: float) -> Point2D:
    return Point2D(
        segment.start.x + parameter * (segment.end.x - segment.start.x),
        segment.start.y + parameter * (segment.end.y - segment.start.y),
    )


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _shared_endpoints(
    first: Segment2D,
    second: Segment2D,
) -> tuple[Point2D, ...]:
    shared: list[Point2D] = []
    for candidate in (first.start, first.end):
        if point_to_segment_distance(candidate, second) <= EPSILON and all(
            distance(candidate, existing) > EPSILON for existing in shared
        ):
            shared.append(candidate)
    for candidate in (second.start, second.end):
        if point_to_segment_distance(candidate, first) <= EPSILON and all(
            distance(candidate, existing) > EPSILON for existing in shared
        ):
            shared.append(candidate)
    return tuple(shared)


def _signed_double_area(points: tuple[Point2D, ...]) -> float:
    return sum(
        current.x * following.y - following.x * current.y
        for current, following in zip(points, points[1:] + points[:1], strict=True)
    )
