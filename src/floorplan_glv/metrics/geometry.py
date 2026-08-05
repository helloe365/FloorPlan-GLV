"""Hungarian vector matching and model-selection metrics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Literal, cast

import numpy as np

from floorplan_glv.config.models import OpeningCleanupConfig
from floorplan_glv.data.output_schema import FloorPlanResult, Opening, Segment
from floorplan_glv.geometry.primitives import (
    GeometryError,
    Segment2D,
    distance,
    point_to_line_distance,
    undirected_angle_difference_deg,
)

_GEOMETRY_EQUALITY_TOLERANCE_PX = 1e-3
_linear_sum_assignment = cast(
    Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    import_module("scipy.optimize").linear_sum_assignment,
)


@dataclass(frozen=True, slots=True)
class ObjectMetrics:
    """Object-level count metrics and matched source indices."""

    matches: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    matched_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class GeometryMetricReport:
    """All V1 vector-geometry metrics for one result pair."""

    wall_edges: ObjectMetrics
    opening_objects: ObjectMetrics
    doors: ObjectMetrics
    windows: ObjectMetrics
    mean_endpoint_distance_normalized: float | None
    wall_angle_mae_deg: float | None
    wall_thickness_relative_error: float | None
    opening_length_relative_error: float | None
    host_wall_attachment_accuracy: float
    duplicate_opening_rate: float
    dangling_wall_count: int
    invalid_geometry_rate: float

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready metrics mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelSelectionMetrics:
    """Inputs to the documented checkpoint-selection score."""

    wall_edge_f1: float
    opening_object_f1: float
    junction_f1: float
    host_wall_accuracy: float
    centerline_cldice: float
    invalid_geometry_rate: float
    wall_pixel_f1: float
    json_success_rate: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise GeometryError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class ModelSelectionScore:
    """Eligibility and score for one checkpoint."""

    eligible: bool
    score: float | None


@dataclass(frozen=True, slots=True)
class _MatchDetails:
    metrics: ObjectMetrics
    endpoint_errors_px: tuple[float, ...]
    angle_errors_deg: tuple[float, ...]


def evaluate_geometry(
    predicted: FloorPlanResult,
    target: FloorPlanResult,
) -> GeometryMetricReport:
    """Evaluate two validated source-pixel results with explicit gates."""
    if (
        predicted.image.width_px,
        predicted.image.height_px,
    ) != (
        target.image.width_px,
        target.image.height_px,
    ):
        raise GeometryError("geometry results must use the same image dimensions")
    image_diagonal = math.hypot(target.image.width_px, target.image.height_px)
    wall_details = _match_segments(
        tuple(wall.segment for wall in predicted.walls),
        tuple(wall.segment for wall in target.walls),
        image_diagonal=image_diagonal,
    )
    opening_details = _match_segments(
        tuple(opening.segment for opening in predicted.openings),
        tuple(opening.segment for opening in target.openings),
        image_diagonal=image_diagonal,
        predicted_types=tuple(opening.type for opening in predicted.openings),
        target_types=tuple(opening.type for opening in target.openings),
    )
    door_metrics = _opening_type_metrics(predicted, target, "door", image_diagonal)
    window_metrics = _opening_type_metrics(predicted, target, "window", image_diagonal)
    wall_pairs = wall_details.metrics.matched_pairs
    opening_pairs = opening_details.metrics.matched_pairs
    wall_id_mapping = {
        predicted.walls[predicted_index].id: target.walls[target_index].id
        for predicted_index, target_index in wall_pairs
    }
    host_accuracy = _host_wall_accuracy(
        predicted,
        target,
        opening_pairs,
        wall_id_mapping,
    )
    return GeometryMetricReport(
        wall_edges=wall_details.metrics,
        opening_objects=opening_details.metrics,
        doors=door_metrics,
        windows=window_metrics,
        mean_endpoint_distance_normalized=(
            float(np.mean(wall_details.endpoint_errors_px)) / image_diagonal
            if wall_details.endpoint_errors_px
            else None
        ),
        wall_angle_mae_deg=(
            float(np.mean(wall_details.angle_errors_deg))
            if wall_details.angle_errors_deg
            else None
        ),
        wall_thickness_relative_error=_mean_relative_error(
            tuple(
                (
                    predicted.walls[predicted_index].thickness_px,
                    target.walls[target_index].thickness_px,
                )
                for predicted_index, target_index in wall_pairs
            )
        ),
        opening_length_relative_error=_mean_relative_error(
            tuple(
                (
                    predicted.openings[predicted_index].length_px,
                    target.openings[target_index].length_px,
                )
                for predicted_index, target_index in opening_pairs
            )
        ),
        host_wall_attachment_accuracy=host_accuracy,
        duplicate_opening_rate=_duplicate_opening_rate(predicted.openings),
        dangling_wall_count=_dangling_wall_count(predicted),
        invalid_geometry_rate=_invalid_geometry_rate(predicted),
    )


def model_selection_score(metrics: ModelSelectionMetrics) -> ModelSelectionScore:
    """Apply the documented composite weights and strict JSON gate."""
    if metrics.json_success_rate < 1.0:
        return ModelSelectionScore(eligible=False, score=None)
    score = (
        0.25 * metrics.wall_edge_f1
        + 0.20 * metrics.opening_object_f1
        + 0.15 * metrics.junction_f1
        + 0.15 * metrics.host_wall_accuracy
        + 0.10 * metrics.centerline_cldice
        + 0.10 * (1.0 - metrics.invalid_geometry_rate)
        + 0.05 * metrics.wall_pixel_f1
    )
    return ModelSelectionScore(eligible=True, score=score)


def _match_segments(
    predicted: tuple[Segment, ...],
    target: tuple[Segment, ...],
    *,
    image_diagonal: float,
    predicted_types: tuple[str, ...] | None = None,
    target_types: tuple[str, ...] | None = None,
) -> _MatchDetails:
    if not predicted or not target:
        metrics = _object_metrics(0, len(predicted), len(target), ())
        return _MatchDetails(metrics, (), ())
    endpoint_errors = np.zeros((len(predicted), len(target)), dtype=np.float64)
    angle_errors = np.zeros_like(endpoint_errors)
    eligible = np.zeros_like(endpoint_errors, dtype=bool)
    for predicted_index, predicted_segment in enumerate(predicted):
        for target_index, target_segment in enumerate(target):
            endpoint_error = _symmetric_endpoint_distance(
                predicted_segment,
                target_segment,
            )
            angle_error = undirected_angle_difference_deg(
                _segment(predicted_segment).undirected_angle_deg,
                _segment(target_segment).undirected_angle_deg,
            )
            overlap = _symmetric_overlap_ratio(predicted_segment, target_segment)
            same_type = (
                predicted_types is None
                or target_types is None
                or predicted_types[predicted_index] == target_types[target_index]
            )
            endpoint_errors[predicted_index, target_index] = endpoint_error
            angle_errors[predicted_index, target_index] = angle_error
            eligible[predicted_index, target_index] = (
                same_type
                and endpoint_error / image_diagonal < 0.01
                and angle_error < 10.0
                and overlap > 0.50
            )
    costs = endpoint_errors / image_diagonal
    gated_costs = costs.copy()
    gated_costs[~eligible] = 1e12
    rows, columns = _linear_sum_assignment(gated_costs)
    pairs = tuple(
        sorted(
            (
                (int(row), int(column))
                for row, column in zip(rows, columns, strict=True)
                if eligible[row, column]
            )
        )
    )
    metrics = _object_metrics(
        len(pairs),
        len(predicted) - len(pairs),
        len(target) - len(pairs),
        pairs,
    )
    return _MatchDetails(
        metrics=metrics,
        endpoint_errors_px=tuple(
            float(endpoint_errors[predicted_index, target_index])
            for predicted_index, target_index in pairs
        ),
        angle_errors_deg=tuple(
            float(angle_errors[predicted_index, target_index])
            for predicted_index, target_index in pairs
        ),
    )


def _opening_type_metrics(
    predicted: FloorPlanResult,
    target: FloorPlanResult,
    opening_type: Literal["door", "window"],
    image_diagonal: float,
) -> ObjectMetrics:
    predicted_openings = tuple(
        opening for opening in predicted.openings if opening.type == opening_type
    )
    target_openings = tuple(
        opening for opening in target.openings if opening.type == opening_type
    )
    return _match_segments(
        tuple(opening.segment for opening in predicted_openings),
        tuple(opening.segment for opening in target_openings),
        image_diagonal=image_diagonal,
    ).metrics


def _object_metrics(
    matches: int,
    false_positive: int,
    false_negative: int,
    pairs: tuple[tuple[int, int], ...],
) -> ObjectMetrics:
    predicted_count = matches + false_positive
    target_count = matches + false_negative
    precision = (
        matches / predicted_count
        if predicted_count
        else (1.0 if target_count == 0 else 0.0)
    )
    recall = (
        matches / target_count
        if target_count
        else (1.0 if predicted_count == 0 else 0.0)
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ObjectMetrics(
        matches=matches,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        matched_pairs=pairs,
    )


def _symmetric_endpoint_distance(first: Segment, second: Segment) -> float:
    direct = 0.5 * (distance(first[0], second[0]) + distance(first[1], second[1]))
    reverse = 0.5 * (distance(first[0], second[1]) + distance(first[1], second[0]))
    return min(direct, reverse)


def _symmetric_overlap_ratio(first: Segment, second: Segment) -> float:
    first_segment = _segment(first)
    second_segment = _segment(second)
    return min(
        first_segment.clipped_projected_overlap_ratio(second_segment),
        second_segment.clipped_projected_overlap_ratio(first_segment),
    )


def _segment(value: Segment) -> Segment2D:
    return Segment2D(value[0], value[1])


def _mean_relative_error(pairs: tuple[tuple[float, float], ...]) -> float | None:
    if not pairs:
        return None
    return float(
        np.mean([abs(predicted - target) / target for predicted, target in pairs])
    )


def _host_wall_accuracy(
    predicted: FloorPlanResult,
    target: FloorPlanResult,
    opening_pairs: tuple[tuple[int, int], ...],
    wall_id_mapping: dict[str, str],
) -> float:
    if not opening_pairs:
        return 1.0 if not predicted.openings and not target.openings else 0.0
    correct = sum(
        wall_id_mapping.get(predicted.openings[predicted_index].host_wall_id)
        == target.openings[target_index].host_wall_id
        for predicted_index, target_index in opening_pairs
    )
    return correct / len(opening_pairs)


def _duplicate_opening_rate(openings: tuple[Opening, ...]) -> float:
    if not openings:
        return 0.0
    config = OpeningCleanupConfig()
    duplicate_count = 0
    for index, candidate in enumerate(openings):
        candidate_segment = _segment(candidate.segment)
        for previous in openings[:index]:
            previous_segment = _segment(previous.segment)
            mean_length = 0.5 * (candidate.length_px + previous.length_px)
            if (
                candidate.host_wall_id == previous.host_wall_id
                and candidate.type == previous.type
                and distance(candidate.center, previous.center)
                < config.center_distance_mean_length_factor * mean_length
                and undirected_angle_difference_deg(
                    candidate_segment.undirected_angle_deg,
                    previous_segment.undirected_angle_deg,
                )
                < config.max_angle_difference_deg
                and _symmetric_overlap_ratio(candidate.segment, previous.segment)
                > config.min_projected_overlap_ratio
            ):
                duplicate_count += 1
                break
    return duplicate_count / len(openings)


def _dangling_wall_count(result: FloorPlanResult) -> int:
    degree = {node.id: 0 for node in result.nodes}
    for wall in result.walls:
        degree[wall.start_node_id] += 1
        degree[wall.end_node_id] += 1
    return sum(
        degree[wall.start_node_id] == 1 or degree[wall.end_node_id] == 1
        for wall in result.walls
    )


def _invalid_geometry_rate(result: FloorPlanResult) -> float:
    """Return the fraction of walls involved in forbidden duplicate overlap."""
    if not result.walls:
        return 0.0
    invalid_indices: set[int] = set()
    for first_index, first in enumerate(result.walls):
        first_segment = _segment(first.segment)
        for second_index in range(first_index + 1, len(result.walls)):
            second_segment = _segment(result.walls[second_index].segment)
            if _walls_overlap_collinearly(first_segment, second_segment):
                invalid_indices.update((first_index, second_index))
    return len(invalid_indices) / len(result.walls)


def _walls_overlap_collinearly(first: Segment2D, second: Segment2D) -> bool:
    if (
        point_to_line_distance(second.start, first) > _GEOMETRY_EQUALITY_TOLERANCE_PX
        or point_to_line_distance(second.end, first) > _GEOMETRY_EQUALITY_TOLERANCE_PX
    ):
        return False
    return (
        first.clipped_projected_overlap_ratio(second) > 0.0
        and second.clipped_projected_overlap_ratio(first) > 0.0
    )
