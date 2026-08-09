"""Deterministic dense-mask and point-detection metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import cast

import numpy as np
import torch

from floorplan_glv.geometry.primitives import GeometryError, Point2D, distance

_DENSE_MASK_KEYS = frozenset({"wall", "opening", "door", "window", "centerline"})
_linear_sum_assignment = cast(
    Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    import_module("scipy.optimize").linear_sum_assignment,
)
_skeletonize = cast(
    Callable[[np.ndarray], np.ndarray],
    import_module("skimage.morphology").skeletonize,
)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Confusion counts and derived metrics for one boolean mask."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    iou: float


@dataclass(frozen=True, slots=True)
class PointMetrics:
    """One-to-one point-detection metrics under a pixel tolerance."""

    matches: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    mean_distance_px: float | None
    matched_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class DenseMetricReport:
    """All dense and heatmap metrics required by the V1 specification."""

    wall: BinaryMetrics
    opening: BinaryMetrics
    door: BinaryMetrics
    window: BinaryMetrics
    centerline_cldice: float
    junction: PointMetrics
    opening_center: PointMetrics


def binary_metrics(predicted: torch.Tensor, target: torch.Tensor) -> BinaryMetrics:
    """Measure two same-shaped finite boolean masks.

    Both tensors may have any shape. Precision, recall, F1, and IoU are one
    when both masks are empty; a missed non-empty target scores zero.
    """
    _validate_mask_pair(predicted, target)
    true_positive = int(torch.count_nonzero(predicted & target).item())
    true_negative = int(torch.count_nonzero(~predicted & ~target).item())
    false_positive = int(torch.count_nonzero(predicted & ~target).item())
    false_negative = int(torch.count_nonzero(~predicted & target).item())
    precision, recall, f1 = _count_metrics(
        true_positive,
        false_positive,
        false_negative,
    )
    union = true_positive + false_positive + false_negative
    iou = 1.0 if union == 0 else true_positive / union
    return BinaryMetrics(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
    )


def centerline_cldice(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute topology-aware clDice for two finite boolean centerline masks."""
    _validate_mask_pair(predicted, target)
    if not torch.any(predicted) and not torch.any(target):
        return 1.0
    if not torch.any(predicted) or not torch.any(target):
        return 0.0
    predicted_cpu = predicted.detach().to(device="cpu")
    target_cpu = target.detach().to(device="cpu")
    predicted_skeleton = torch.from_numpy(_skeletonize(predicted_cpu.numpy())).to(
        dtype=torch.bool
    )
    target_skeleton = torch.from_numpy(_skeletonize(target_cpu.numpy())).to(
        dtype=torch.bool
    )
    topology_precision = _safe_overlap_fraction(predicted_skeleton, target_cpu)
    topology_sensitivity = _safe_overlap_fraction(target_skeleton, predicted_cpu)
    denominator = topology_precision + topology_sensitivity
    if denominator == 0.0:
        return 0.0
    return 2.0 * topology_precision * topology_sensitivity / denominator


def point_metrics(
    predicted: Sequence[Point2D],
    target: Sequence[Point2D],
    *,
    tolerance_px: float,
) -> PointMetrics:
    """Match points one-to-one with Hungarian assignment under a pixel gate."""
    if not math.isfinite(tolerance_px) or tolerance_px <= 0.0:
        raise GeometryError("point tolerance_px must be finite and positive")
    predicted_points = tuple(predicted)
    target_points = tuple(target)
    matched_pairs: tuple[tuple[int, int], ...] = ()
    matched_distances: tuple[float, ...] = ()
    if predicted_points and target_points:
        costs = np.asarray(
            [
                [
                    distance(predicted_point, target_point)
                    for target_point in target_points
                ]
                for predicted_point in predicted_points
            ],
            dtype=np.float64,
        )
        gated_costs = costs.copy()
        gated_costs[gated_costs > tolerance_px] = 1e12
        rows, columns = _linear_sum_assignment(gated_costs)
        eligible = [
            (int(row), int(column))
            for row, column in zip(rows, columns, strict=True)
            if costs[row, column] <= tolerance_px
        ]
        eligible.sort()
        matched_pairs = tuple(eligible)
        matched_distances = tuple(costs[row, column] for row, column in eligible)
    matches = len(matched_pairs)
    false_positive = len(predicted_points) - matches
    false_negative = len(target_points) - matches
    precision, recall, f1 = _count_metrics(
        matches,
        false_positive,
        false_negative,
    )
    return PointMetrics(
        matches=matches,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_distance_px=(
            float(np.mean(matched_distances)) if matched_distances else None
        ),
        matched_pairs=matched_pairs,
    )


def evaluate_dense_metrics(
    *,
    predicted_masks: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    predicted_junctions: Sequence[Point2D],
    target_junctions: Sequence[Point2D],
    predicted_opening_centers: Sequence[Point2D],
    target_opening_centers: Sequence[Point2D],
    junction_tolerance_px: float = 6.0,
    opening_center_tolerance_px: float = 8.0,
) -> DenseMetricReport:
    """Compute all documented dense metrics from same-shaped source images.

    Every mask must be a two-dimensional boolean tensor shaped
    ``[source_height, source_width]``. All predicted and target masks in one
    report must share that source-resolution shape.
    """
    if set(predicted_masks) != _DENSE_MASK_KEYS or set(target_masks) != (
        _DENSE_MASK_KEYS
    ):
        raise GeometryError(
            f"dense mask keys must be exactly {sorted(_DENSE_MASK_KEYS)}"
        )
    _validate_dense_mask_shapes(predicted_masks, target_masks)
    return DenseMetricReport(
        wall=binary_metrics(predicted_masks["wall"], target_masks["wall"]),
        opening=binary_metrics(
            predicted_masks["opening"],
            target_masks["opening"],
        ),
        door=binary_metrics(predicted_masks["door"], target_masks["door"]),
        window=binary_metrics(predicted_masks["window"], target_masks["window"]),
        centerline_cldice=centerline_cldice(
            predicted_masks["centerline"],
            target_masks["centerline"],
        ),
        junction=point_metrics(
            predicted_junctions,
            target_junctions,
            tolerance_px=junction_tolerance_px,
        ),
        opening_center=point_metrics(
            predicted_opening_centers,
            target_opening_centers,
            tolerance_px=opening_center_tolerance_px,
        ),
    )


def _validate_mask_pair(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise GeometryError("metric masks must be torch tensors")
    if predicted.shape != target.shape:
        raise GeometryError("metric masks must have the same shape")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise GeometryError("metric masks must contain finite values")
    if predicted.dtype is not torch.bool or target.dtype is not torch.bool:
        raise GeometryError("metric masks must use boolean dtype")


def _validate_dense_mask_shapes(
    predicted_masks: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
) -> None:
    values = tuple(predicted_masks.values()) + tuple(target_masks.values())
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in values):
        raise GeometryError("dense masks must share one two-dimensional source shape")
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise GeometryError("dense masks must share one two-dimensional source shape")


def _count_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    predicted_count = true_positive + false_positive
    target_count = true_positive + false_negative
    precision = (
        true_positive / predicted_count
        if predicted_count
        else (1.0 if target_count == 0 else 0.0)
    )
    recall = (
        true_positive / target_count
        if target_count
        else (1.0 if predicted_count == 0 else 0.0)
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _safe_overlap_fraction(skeleton: torch.Tensor, mask: torch.Tensor) -> float:
    denominator = int(torch.count_nonzero(skeleton).item())
    if denominator == 0:
        return 1.0 if not torch.any(mask) else 0.0
    return float(torch.count_nonzero(skeleton & mask).item() / denominator)
