"""Deterministic geometric cleanup for wall-segment candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from shapely.geometry import LineString  # type: ignore[import-untyped]

from floorplan_glv.geometry.primitives import (
    Point2D,
    Segment2D,
    distance,
    undirected_angle_difference_deg,
)


@dataclass(frozen=True, slots=True)
class RawWallSegment:
    """A fitted wall centerline before graph nodes and final IDs exist."""

    start: Point2D
    end: Point2D

    @property
    def segment(self) -> Segment2D:
        """Return the segment as a validated geometry primitive."""
        return Segment2D(self.start, self.end)


def merge_collinear_segments(
    segments: list[RawWallSegment],
    *,
    angle_tolerance_deg: float,
    gap_tolerance_px: float,
) -> list[RawWallSegment]:
    """Merge aligned degree-one segment ends separated by a configured gap."""
    merged = list(segments)
    while True:
        endpoint_degrees = _endpoint_degrees(merged)
        endpoints = tuple(
            point for segment in merged for point in (segment.start, segment.end)
        )
        match: tuple[int, int] | None = None
        for first_index, first in enumerate(merged):
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                if _can_merge(
                    first,
                    second,
                    endpoint_degrees=endpoint_degrees,
                    endpoints=endpoints,
                    angle_tolerance_deg=angle_tolerance_deg,
                    gap_tolerance_px=gap_tolerance_px,
                ):
                    match = (first_index, second_index)
                    break
            if match is not None:
                break
        if match is None:
            return merged
        first_index, second_index = match
        combined = _fit_outer_segment(
            (
                merged[first_index].start,
                merged[first_index].end,
                merged[second_index].start,
                merged[second_index].end,
            )
        )
        merged = [segment for index, segment in enumerate(merged) if index not in match]
        merged.append(combined)


def snap_segment_endpoints(
    segments: list[RawWallSegment],
    *,
    tolerance_px: float,
) -> list[RawWallSegment]:
    """Snap endpoint clusters to deterministic mean source-pixel positions."""
    if not segments:
        return []
    endpoints = [
        point for segment in segments for point in (segment.start, segment.end)
    ]
    coordinates = np.asarray(
        [(point.x, point.y) for point in endpoints],
        dtype=np.float64,
    )
    labels = _cluster_labels(coordinates, tolerance_px)
    centers: dict[int, Point2D] = {}
    for label in sorted(set(labels)):
        members = coordinates[np.asarray(labels) == label]
        centers[label] = Point2D(
            float(np.mean(members[:, 0])),
            float(np.mean(members[:, 1])),
        )

    snapped: list[RawWallSegment] = []
    for index, _segment in enumerate(segments):
        start = centers[labels[2 * index]]
        end = centers[labels[2 * index + 1]]
        if distance(start, end) > 0.0:
            snapped.append(RawWallSegment(start=start, end=end))
    return snapped


def cluster_points(
    points: np.ndarray,
    *,
    radius_px: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Cluster ``[N, 2]`` xy points and return sorted mean centers and labels."""
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64), ()
    coordinates = np.asarray(points, dtype=np.float64)
    labels = _cluster_labels(coordinates, radius_px)
    unsorted_centers = {
        label: np.mean(coordinates[np.asarray(labels) == label], axis=0)
        for label in set(labels)
    }
    ordered_labels = sorted(
        unsorted_centers,
        key=lambda label: (
            round(float(unsorted_centers[label][1]), 4),
            round(float(unsorted_centers[label][0]), 4),
        ),
    )
    remap = {old: new for new, old in enumerate(ordered_labels)}
    centers = np.asarray(
        [unsorted_centers[label] for label in ordered_labels],
        dtype=np.float64,
    )
    return centers, tuple(remap[label] for label in labels)


def sample_segment_coordinates(
    segment: Segment2D,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Return deterministic nearest-pixel samples along a segment."""
    count = max(2, math.ceil(segment.length) + 1)
    x_values = np.linspace(segment.start.x, segment.end.x, count)
    y_values = np.linspace(segment.start.y, segment.end.y, count)
    return np.column_stack(
        (
            np.clip(np.rint(x_values), 0, width - 1).astype(np.intp),
            np.clip(np.rint(y_values), 0, height - 1).astype(np.intp),
        )
    )


def sample_segment_mean(field: np.ndarray, segment: Segment2D) -> float:
    """Return the mean nearest-pixel field value along a segment."""
    coordinates = sample_segment_coordinates(
        segment,
        width=field.shape[1],
        height=field.shape[0],
    )
    return float(np.mean(field[coordinates[:, 1], coordinates[:, 0]]))


def bilinear_sample(field: np.ndarray, point: Point2D) -> float:
    """Sample one scalar ``[height, width]`` field at a source-pixel point."""
    height, width = field.shape
    if not (0.0 <= point.x <= width - 1 and 0.0 <= point.y <= height - 1):
        return 0.0
    x0 = math.floor(point.x)
    y0 = math.floor(point.y)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    x_weight = point.x - x0
    y_weight = point.y - y0
    top = (1.0 - x_weight) * field[y0, x0] + x_weight * field[y0, x1]
    bottom = (1.0 - x_weight) * field[y1, x0] + x_weight * field[y1, x1]
    return float((1.0 - y_weight) * top + y_weight * bottom)


def sample_segment_bilinear_mean(
    field: np.ndarray,
    segment: Segment2D,
) -> float:
    """Return mean bilinear field support along a source-pixel segment."""
    count = max(2, math.ceil(segment.length) + 1)
    return float(
        np.mean(
            [
                bilinear_sample(
                    field,
                    Point2D(
                        segment.start.x + parameter * (segment.end.x - segment.start.x),
                        segment.start.y + parameter * (segment.end.y - segment.start.y),
                    ),
                )
                for parameter in np.linspace(0.0, 1.0, count)
            ]
        )
    )


def sample_buffered_segment_bilinear_mean(
    field: np.ndarray,
    segment: Segment2D,
    *,
    half_width_px: float,
) -> float:
    """Return mean support across a segment's full symmetric pixel buffer."""
    direction_x, direction_y = segment.normalized_direction
    normal_x, normal_y = -direction_y, direction_x
    offset_count = max(1, math.ceil(2.0 * half_width_px) + 1)
    offsets = np.linspace(-half_width_px, half_width_px, offset_count)
    return float(
        np.mean(
            [
                sample_segment_bilinear_mean(
                    field,
                    Segment2D(
                        (
                            segment.start.x + offset * normal_x,
                            segment.start.y + offset * normal_y,
                        ),
                        (
                            segment.end.x + offset * normal_x,
                            segment.end.y + offset * normal_y,
                        ),
                    ),
                )
                for offset in offsets
            ]
        )
    )


def refine_endpoint_from_heatmap(
    endpoint: Point2D,
    endpoint_heatmap: np.ndarray,
    *,
    radius_px: int,
    minimum_probability: float,
) -> Point2D:
    """Move an endpoint to the nearest qualifying local heatmap maximum."""
    center_x = round(endpoint.x)
    center_y = round(endpoint.y)
    x_start = max(0, center_x - radius_px)
    x_end = min(endpoint_heatmap.shape[1], center_x + radius_px + 1)
    y_start = max(0, center_y - radius_px)
    y_end = min(endpoint_heatmap.shape[0], center_y + radius_px + 1)
    if x_start >= x_end or y_start >= y_end:
        return endpoint
    window = endpoint_heatmap[y_start:y_end, x_start:x_end]
    maximum = float(np.max(window))
    if maximum < minimum_probability:
        return endpoint
    maxima = np.argwhere(window == maximum)
    best_y, best_x = min(
        (
            (int(local_y + y_start), int(local_x + x_start))
            for local_y, local_x in maxima
        ),
        key=lambda point: (
            math.hypot(point[1] - endpoint.x, point[0] - endpoint.y),
            point[0],
            point[1],
        ),
    )
    return Point2D(float(best_x), float(best_y))


def opening_segments_are_duplicates(
    first: Segment2D,
    second: Segment2D,
    *,
    center_distance_mean_length_factor: float,
    angle_tolerance_deg: float,
    min_overlap_ratio: float,
) -> bool:
    """Return whether two same-host, same-type opening segments duplicate."""
    mean_length = 0.5 * (first.length + second.length)
    if (
        distance(first.midpoint, second.midpoint)
        >= center_distance_mean_length_factor * mean_length
    ):
        return False
    if (
        undirected_angle_difference_deg(
            first.undirected_angle_deg,
            second.undirected_angle_deg,
        )
        >= angle_tolerance_deg
    ):
        return False
    overlap_ratio = max(
        first.clipped_projected_overlap_ratio(second),
        second.clipped_projected_overlap_ratio(first),
    )
    return overlap_ratio > min_overlap_ratio


def wall_node_confidence(
    point: Point2D,
    *,
    junction_field: np.ndarray,
    centerline_field: np.ndarray,
    junction_threshold: float,
    neighborhood_radius_px: int,
) -> float:
    """Score a heatmap junction above lower-confidence injected endpoints."""
    center_x = min(junction_field.shape[1] - 1, max(0, round(point.x)))
    center_y = min(junction_field.shape[0] - 1, max(0, round(point.y)))
    x_start = max(0, center_x - neighborhood_radius_px)
    x_end = min(
        junction_field.shape[1],
        center_x + neighborhood_radius_px + 1,
    )
    y_start = max(0, center_y - neighborhood_radius_px)
    y_end = min(
        junction_field.shape[0],
        center_y + neighborhood_radius_px + 1,
    )
    junction_confidence = float(np.max(junction_field[y_start:y_end, x_start:x_end]))
    if junction_confidence >= junction_threshold:
        return junction_confidence
    return min(float(centerline_field[center_y, center_x]), junction_threshold)


def _cluster_labels(points: np.ndarray, radius_px: float) -> list[int]:
    parents = list(range(len(points)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    tree = cKDTree(points)
    for first, second in sorted(tree.query_pairs(radius_px)):
        union(first, second)
    roots = [root(index) for index in range(len(points))]
    unique_roots = {value: label for label, value in enumerate(sorted(set(roots)))}
    return [unique_roots[value] for value in roots]


def _can_merge(
    first: RawWallSegment,
    second: RawWallSegment,
    *,
    endpoint_degrees: dict[tuple[float, float], int],
    endpoints: tuple[Point2D, ...],
    angle_tolerance_deg: float,
    gap_tolerance_px: float,
) -> bool:
    if (
        undirected_angle_difference_deg(
            first.segment.undirected_angle_deg,
            second.segment.undirected_angle_deg,
        )
        > angle_tolerance_deg
    ):
        return False
    endpoint_pairs = [
        (distance(first_point, second_point), first_point, second_point)
        for first_point in (first.start, first.end)
        for second_point in (second.start, second.end)
    ]
    gap, first_point, second_point = min(
        endpoint_pairs,
        key=lambda item: (
            item[0],
            item[1].y,
            item[1].x,
            item[2].y,
            item[2].x,
        ),
    )
    if gap > gap_tolerance_px:
        return False
    if endpoint_degrees[_point_key(first_point)] != 1:
        return False
    if endpoint_degrees[_point_key(second_point)] != 1:
        return False
    nearby_endpoints = sum(
        min(distance(point, first_point), distance(point, second_point))
        <= gap_tolerance_px
        for point in endpoints
    )
    if nearby_endpoints > 2:
        return False
    perpendicular_tolerance = gap_tolerance_px * math.sin(
        math.radians(angle_tolerance_deg)
    )
    if (
        max(
            first.segment.point_to_line_distance(second_point),
            second.segment.point_to_line_distance(first_point),
        )
        > perpendicular_tolerance
    ):
        return False
    first_line = LineString((first.start.as_tuple(), first.end.as_tuple()))
    second_line = LineString((second.start.as_tuple(), second.end.as_tuple()))
    return float(first_line.distance(second_line)) <= gap_tolerance_px


def _fit_outer_segment(points: tuple[Point2D, ...]) -> RawWallSegment:
    coordinates = np.asarray(
        [(point.x, point.y) for point in points],
        dtype=np.float32,
    )
    direction_x, direction_y, origin_x, origin_y = (
        float(value)
        for value in cv2.fitLine(
            coordinates,
            cv2.DIST_L2,
            0.0,
            0.0,
            0.0,
        ).reshape(-1)
    )
    projections = (coordinates[:, 0] - origin_x) * direction_x + (
        coordinates[:, 1] - origin_y
    ) * direction_y
    low = float(np.min(projections))
    high = float(np.max(projections))
    start = Point2D(origin_x + low * direction_x, origin_y + low * direction_y)
    end = Point2D(origin_x + high * direction_x, origin_y + high * direction_y)
    if (round(end.y, 4), round(end.x, 4)) < (
        round(start.y, 4),
        round(start.x, 4),
    ):
        start, end = end, start
    return RawWallSegment(start=start, end=end)


def _endpoint_degrees(
    segments: list[RawWallSegment],
) -> dict[tuple[float, float], int]:
    degrees: dict[tuple[float, float], int] = {}
    for segment in segments:
        for point in (segment.start, segment.end):
            key = _point_key(point)
            degrees[key] = degrees.get(key, 0) + 1
    return degrees


def _point_key(point: Point2D) -> tuple[float, float]:
    return (round(point.x, 4), round(point.y, 4))
