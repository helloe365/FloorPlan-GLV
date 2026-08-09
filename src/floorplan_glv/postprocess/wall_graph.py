"""Deterministic vectorization of merged wall prediction maps."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from typing import Literal, cast

import cv2
import numpy as np
import torch

from floorplan_glv.config.models import WallPostprocessConfig
from floorplan_glv.geometry.primitives import (
    GeometryError,
    Point2D,
    Segment2D,
    distance,
    undirected_angle_difference_deg,
)
from floorplan_glv.postprocess.cleanup import (
    RawWallSegment,
    cluster_points,
    merge_collinear_segments,
    sample_segment_coordinates,
    sample_segment_mean,
    snap_segment_endpoints,
    wall_node_confidence,
)
from floorplan_glv.postprocess.tiled_merge import FullResolutionMaps

_NEIGHBOR_OFFSETS = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
)
_skeletonize = cast(
    Callable[[np.ndarray], np.ndarray],
    import_module("skimage.morphology").skeletonize,
)


@dataclass(frozen=True, slots=True)
class WallNodeCandidate:
    """A stable source-pixel graph node without a final serialized ID."""

    point: Point2D
    kind: Literal["endpoint", "junction"]
    confidence: float


@dataclass(frozen=True, slots=True)
class WallEdgeCandidate:
    """A stable source-pixel wall edge without a final serialized ID."""

    start_node_index: int
    end_node_index: int
    segment: Segment2D
    thickness_px: float
    confidence: float
    raster_support: float


@dataclass(frozen=True, slots=True)
class WallGraph:
    """A deterministic candidate graph ready for later ID assignment."""

    nodes: tuple[WallNodeCandidate, ...]
    edges: tuple[WallEdgeCandidate, ...]


def build_wall_graph(
    maps: FullResolutionMaps,
    config: WallPostprocessConfig,
) -> WallGraph:
    """Vectorize source-resolution dense maps into a stable wall graph.

    Required scalar maps are shaped ``[1, height, width]`` and wall orientation
    is shaped ``[2, height, width]``. Coordinates and thicknesses in the result
    remain in source-image pixels.
    """
    arrays = _validated_wall_arrays(maps)
    centerline = _clean_centerline(
        arrays["wall_mask"],
        arrays["wall_centerline"],
        config,
    )
    skeleton = _skeletonize(centerline).astype(np.uint8)
    if not np.any(skeleton):
        return WallGraph(nodes=(), edges=())

    node_pixels, centers, pixel_labels = _find_node_pixels(
        skeleton,
        arrays["wall_junction"],
        config,
    )
    traced = _trace_skeleton(skeleton, node_pixels, centers, pixel_labels)
    fitted = [
        segment for path in traced for segment in _fit_path_segments(path, config)
    ]
    merged = merge_collinear_segments(
        fitted,
        angle_tolerance_deg=config.merge_angle_deg,
        gap_tolerance_px=config.merge_gap_px,
    )
    oriented = [
        _snap_to_orientation(
            segment,
            arrays["wall_orientation"],
            tolerance_deg=config.dominant_angle_snap_deg,
        )
        for segment in merged
    ]
    snapped = snap_segment_endpoints(
        oriented,
        tolerance_px=config.endpoint_snap_px,
    )
    return _finalize_graph(snapped, arrays, config)


def _validated_wall_arrays(
    maps: FullResolutionMaps,
) -> dict[str, np.ndarray]:
    if not isinstance(maps, Mapping):
        raise GeometryError("wall maps must be a tensor mapping")
    source = cast(Mapping[str, object], maps)
    expected_channels = {
        "wall_mask": 1,
        "wall_centerline": 1,
        "wall_junction": 1,
        "wall_orientation": 2,
        "wall_log_half_thickness": 1,
    }
    arrays: dict[str, np.ndarray] = {}
    spatial_shape: tuple[int, int] | None = None
    for name, channels in expected_channels.items():
        value = source.get(name)
        if not isinstance(value, torch.Tensor):
            raise GeometryError(f"{name} must be a floating tensor")
        if value.ndim != 3 or value.shape[0] != channels:
            raise GeometryError(f"{name} must be shaped [{channels}, height, width]")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise GeometryError(f"{name} must contain finite floating values")
        shape = (int(value.shape[1]), int(value.shape[2]))
        if spatial_shape is None:
            spatial_shape = shape
        elif shape != spatial_shape:
            raise GeometryError("wall maps must share one spatial shape")
        arrays[name] = value.detach().to(dtype=torch.float32, device="cpu").numpy()
    return arrays


def _clean_centerline(
    wall_mask: np.ndarray,
    wall_centerline: np.ndarray,
    config: WallPostprocessConfig,
) -> np.ndarray:
    mask = wall_mask[0] >= config.mask_threshold
    centerline = wall_centerline[0] >= config.centerline_threshold
    radius = math.ceil(config.max_trace_gap_px)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    supported = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    candidate = (centerline & supported).astype(np.uint8)
    count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )
    cleaned = np.zeros_like(candidate)
    for label in range(1, count):
        if statistics[label, cv2.CC_STAT_AREA] >= config.min_component_area_px:
            cleaned[labels == label] = 1
    return cast(np.ndarray, cleaned.astype(bool))


def _find_node_pixels(
    skeleton: np.ndarray,
    junction_heatmap: np.ndarray,
    config: WallPostprocessConfig,
) -> tuple[
    tuple[tuple[int, int], ...],
    np.ndarray,
    dict[tuple[int, int], int],
]:
    skeleton_pixels = {(int(y), int(x)) for y, x in np.argwhere(skeleton > 0)}
    candidates = {
        pixel
        for pixel in skeleton_pixels
        if len(_skeleton_neighbors(pixel, skeleton_pixels)) != 2
    }
    heat = junction_heatmap[0]
    radius = config.junction_nms_radius_px
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    local_maximum = heat == cv2.dilate(heat, kernel)
    peaks = np.argwhere((heat >= config.junction_threshold) & local_maximum)
    for peak_y, peak_x in peaks:
        nearest = min(
            skeleton_pixels,
            key=lambda pixel: (
                math.hypot(pixel[0] - peak_y, pixel[1] - peak_x),
                pixel,
            ),
        )
        if math.hypot(nearest[0] - peak_y, nearest[1] - peak_x) <= (
            config.max_trace_gap_px
        ):
            candidates.add(nearest)
    ordered_pixels = tuple(sorted(candidates))
    _count, component_labels = cv2.connectedComponents(skeleton, connectivity=8)
    centers_by_component: list[np.ndarray] = []
    pixel_labels: dict[tuple[int, int], int] = {}
    label_offset = 0
    component_ids = sorted({int(component_labels[y, x]) for y, x in ordered_pixels})
    for component_id in component_ids:
        component_pixels = tuple(
            pixel
            for pixel in ordered_pixels
            if component_labels[pixel[0], pixel[1]] == component_id
        )
        xy_points = np.asarray(
            [(float(x), float(y)) for y, x in component_pixels],
            dtype=np.float64,
        )
        component_centers, component_cluster_labels = cluster_points(
            xy_points,
            radius_px=config.junction_cluster_radius_px,
        )
        centers_by_component.append(component_centers)
        for index, pixel in enumerate(component_pixels):
            pixel_labels[pixel] = label_offset + component_cluster_labels[index]
        label_offset += len(component_centers)
    centers = np.concatenate(centers_by_component, axis=0)
    return ordered_pixels, centers, pixel_labels


def _trace_skeleton(
    skeleton: np.ndarray,
    node_pixels: tuple[tuple[int, int], ...],
    centers: np.ndarray,
    pixel_labels: dict[tuple[int, int], int],
) -> list[tuple[Point2D, ...]]:
    skeleton_pixels = {(int(y), int(x)) for y, x in np.argwhere(skeleton > 0)}
    candidate_set = set(node_pixels)
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[tuple[Point2D, ...]] = []
    for start_pixel in node_pixels:
        for neighbor in _skeleton_neighbors(start_pixel, skeleton_pixels):
            edge_key = _pixel_edge(start_pixel, neighbor)
            if edge_key in visited:
                continue
            visited.add(edge_key)
            pixels = [start_pixel, neighbor]
            previous = start_pixel
            current = neighbor
            while current not in candidate_set:
                choices = [
                    pixel
                    for pixel in _skeleton_neighbors(current, skeleton_pixels)
                    if pixel != previous
                ]
                if len(choices) != 1:
                    break
                following = choices[0]
                visited.add(_pixel_edge(current, following))
                pixels.append(following)
                previous, current = current, following
            if current not in candidate_set:
                continue
            start_label = pixel_labels[start_pixel]
            end_label = pixel_labels[current]
            if start_label == end_label:
                continue
            start = Point2D(
                float(centers[start_label, 0]),
                float(centers[start_label, 1]),
            )
            end = Point2D(
                float(centers[end_label, 0]),
                float(centers[end_label, 1]),
            )
            interior = tuple(Point2D(float(x), float(y)) for y, x in pixels[1:-1])
            paths.append((start, *interior, end))
    return paths


def _fit_path_segments(
    path: tuple[Point2D, ...],
    config: WallPostprocessConfig,
) -> list[RawWallSegment]:
    coordinates = np.asarray(
        [(point.x, point.y) for point in path],
        dtype=np.float32,
    )
    simplified = cv2.approxPolyDP(
        coordinates.reshape(-1, 1, 2),
        config.rdp_epsilon_px,
        False,
    ).reshape(-1, 2)
    split_indices = [0]
    for index in range(1, len(simplified) - 1):
        incoming = simplified[index] - simplified[index - 1]
        outgoing = simplified[index + 1] - simplified[index]
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator == 0.0:
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
        turn_deg = math.degrees(math.acos(cosine))
        if turn_deg >= config.split_angle_deg:
            split_indices.append(index)
    split_indices.append(len(simplified) - 1)
    segments: list[RawWallSegment] = []
    for start_index, end_index in pairwise(split_indices):
        points = simplified[start_index : end_index + 1]
        segment = _fit_points(points)
        if segment is not None:
            segments.append(segment)
    return segments


def _fit_points(points: np.ndarray) -> RawWallSegment | None:
    if len(points) < 2:
        return None
    direction_x, direction_y, origin_x, origin_y = (
        float(value)
        for value in cv2.fitLine(
            points,
            cv2.DIST_L2,
            0.0,
            0.0,
            0.0,
        ).reshape(-1)
    )
    relative = points - np.asarray((origin_x, origin_y), dtype=np.float32)
    projections = relative[:, 0] * direction_x + relative[:, 1] * direction_y
    low = float(np.min(projections))
    high = float(np.max(projections))
    start = Point2D(origin_x + low * direction_x, origin_y + low * direction_y)
    end = Point2D(origin_x + high * direction_x, origin_y + high * direction_y)
    if distance(start, end) == 0.0:
        return None
    return RawWallSegment(start=start, end=end)


def _snap_to_orientation(
    raw: RawWallSegment,
    orientation: np.ndarray,
    *,
    tolerance_deg: float,
) -> RawWallSegment:
    segment = raw.segment
    samples = sample_segment_coordinates(
        segment,
        width=orientation.shape[2],
        height=orientation.shape[1],
    )
    vectors = orientation[:, samples[:, 1], samples[:, 0]]
    mean = np.mean(vectors, axis=1)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return raw
    predicted_angle = 0.5 * math.degrees(math.atan2(mean[0], mean[1])) % 180.0
    if (
        undirected_angle_difference_deg(
            segment.undirected_angle_deg,
            predicted_angle,
        )
        > tolerance_deg
    ):
        return raw
    radians = math.radians(predicted_angle)
    dx = 0.5 * segment.length * math.cos(radians)
    dy = 0.5 * segment.length * math.sin(radians)
    midpoint = segment.midpoint
    return RawWallSegment(
        start=Point2D(midpoint.x - dx, midpoint.y - dy),
        end=Point2D(midpoint.x + dx, midpoint.y + dy),
    )


def _finalize_graph(
    raw_segments: list[RawWallSegment],
    arrays: dict[str, np.ndarray],
    config: WallPostprocessConfig,
) -> WallGraph:
    measured: list[tuple[RawWallSegment, float, float, float]] = []
    for raw in raw_segments:
        segment = raw.segment
        if segment.length < config.min_wall_length_px:
            continue
        support = sample_segment_mean(arrays["wall_mask"][0], segment)
        if support < config.min_raster_support:
            continue
        thickness = _estimate_thickness(raw, arrays, config)
        confidence = sample_segment_mean(arrays["wall_centerline"][0], segment)
        measured.append((raw, thickness, confidence, support))

    points = {
        _point_key(point): point
        for raw, _thickness, _confidence, _support in measured
        for point in (raw.start, raw.end)
    }
    ordered_keys = sorted(points, key=lambda key: (key[1], key[0]))
    point_indices = {key: index for index, key in enumerate(ordered_keys)}
    edge_records: list[tuple[int, int, float, float, float]] = []
    for raw, thickness, confidence, support in measured:
        first = point_indices[_point_key(raw.start)]
        second = point_indices[_point_key(raw.end)]
        start, end = sorted((first, second))
        edge_records.append((start, end, thickness, confidence, support))
    edge_records.sort(key=lambda record: (record[0], record[1]))

    degrees = [0] * len(ordered_keys)
    for start, end, _thickness, _confidence, _support in edge_records:
        degrees[start] += 1
        degrees[end] += 1
    nodes = tuple(
        WallNodeCandidate(
            point=points[key],
            kind="junction" if degrees[index] > 1 else "endpoint",
            confidence=wall_node_confidence(
                points[key],
                junction_field=arrays["wall_junction"][0],
                centerline_field=arrays["wall_centerline"][0],
                junction_threshold=config.junction_threshold,
                neighborhood_radius_px=config.junction_cluster_radius_px,
            ),
        )
        for index, key in enumerate(ordered_keys)
    )
    edges = tuple(
        WallEdgeCandidate(
            start_node_index=start,
            end_node_index=end,
            segment=Segment2D(nodes[start].point, nodes[end].point),
            thickness_px=thickness,
            confidence=confidence,
            raster_support=support,
        )
        for start, end, thickness, confidence, support in edge_records
    )
    return WallGraph(nodes=nodes, edges=edges)


def _estimate_thickness(
    raw: RawWallSegment,
    arrays: dict[str, np.ndarray],
    config: WallPostprocessConfig,
) -> float:
    segment = raw.segment
    coordinates = sample_segment_coordinates(
        segment,
        width=arrays["wall_mask"].shape[2],
        height=arrays["wall_mask"].shape[1],
    )
    log_values = arrays["wall_log_half_thickness"][
        0,
        coordinates[:, 1],
        coordinates[:, 0],
    ]
    positive = log_values[log_values > 0.0]
    if positive.size:
        return float(2.0 * np.expm1(np.median(positive)))
    binary_mask = (arrays["wall_mask"][0] >= config.mask_threshold).astype(np.uint8)
    distance_field = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    half_widths = distance_field[coordinates[:, 1], coordinates[:, 0]]
    return float(2.0 * np.median(half_widths))


def _skeleton_neighbors(
    pixel: tuple[int, int],
    skeleton_pixels: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    y, x = pixel
    return sorted(
        (y + dy, x + dx)
        for dy, dx in _NEIGHBOR_OFFSETS
        if (y + dy, x + dx) in skeleton_pixels
    )


def _pixel_edge(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _point_key(point: Point2D) -> tuple[float, float]:
    return (round(point.x, 4), round(point.y, 4))
