"""Deterministic opening decoding, host attachment, and duplicate cleanup."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import cv2
import numpy as np
import torch

from floorplan_glv.config.models import (
    AttachmentConfig,
    OpeningCleanupConfig,
    OpeningPostprocessConfig,
    PostprocessConfig,
)
from floorplan_glv.geometry.primitives import (
    GeometryError,
    Point2D,
    Segment2D,
    distance,
    undirected_angle_difference_deg,
)
from floorplan_glv.postprocess.cleanup import (
    bilinear_sample,
    opening_segments_are_duplicates,
    refine_endpoint_from_heatmap,
    sample_buffered_segment_bilinear_mean,
    sample_segment_bilinear_mean,
)
from floorplan_glv.postprocess.tiled_merge import FullResolutionMaps
from floorplan_glv.postprocess.wall_graph import WallEdgeCandidate, WallGraph

OpeningType = Literal["door", "window"]
OpeningDropReason = Literal[
    "invalid_orientation",
    "invalid_length",
    "no_host_wall",
    "distance_gate",
    "angle_gate",
    "overlap_gate",
    "below_confidence",
    "duplicate",
]


@dataclass(frozen=True, slots=True)
class OpeningCandidate:
    """One source-pixel opening candidate before final JSON ID assignment."""

    opening_type: OpeningType
    segment: Segment2D | None
    center: Point2D
    center_probability: float
    type_probability: float
    mask_support: float
    host_wall_index: int | None = None
    attachment_score: float = 0.0
    confidence: float = 0.0
    drop_reason: OpeningDropReason | None = None

    def __post_init__(self) -> None:
        if self.opening_type not in ("door", "window"):
            raise GeometryError("opening type must be door or window")
        for name in (
            "center_probability",
            "type_probability",
            "mask_support",
            "attachment_score",
            "confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise GeometryError(f"{name} must be finite and within [0, 1]")
            object.__setattr__(self, name, value)
        if self.host_wall_index is not None and self.host_wall_index < 0:
            raise GeometryError("host wall index cannot be negative")
        if self.segment is None and self.drop_reason is None:
            raise GeometryError("an active opening candidate requires a segment")
        if (
            self.segment is not None
            and distance(
                self.center,
                self.segment.midpoint,
            )
            > 1e-6
        ):
            raise GeometryError("opening center must equal its segment midpoint")

    @property
    def length_px(self) -> float:
        """Return segment length, or zero for an invalid decoded candidate."""
        return 0.0 if self.segment is None else self.segment.length


@dataclass(frozen=True, slots=True)
class _AttachmentMatch:
    wall_index: int
    wall: WallEdgeCandidate
    score: float


def decode_openings(
    maps: FullResolutionMaps,
    wall_graph: WallGraph,
    config: PostprocessConfig,
) -> list[OpeningCandidate]:
    """Decode, attach, threshold, and deduplicate source-resolution openings.

    Opening scalar maps are shaped ``[1, height, width]``; type and
    doubled-angle orientation maps are shaped ``[2, height, width]``.
    Rejected candidates remain in the returned list with ``drop_reason`` set
    so every postprocessing removal is measurable.
    """
    arrays = _validated_opening_arrays(maps)
    decoded = _decode_candidates(arrays, config.opening)
    attached = attach_openings(decoded, wall_graph, config.attachment)
    supported = [
        _remeasure_mask_support(
            candidate,
            opening_mask=arrays["opening_mask"][0],
            wall_graph=wall_graph,
        )
        for candidate in attached
    ]
    thresholded = [
        (
            replace(candidate, drop_reason="below_confidence")
            if candidate.drop_reason is None
            and candidate.confidence < config.opening.final_confidence_threshold
            else candidate
        )
        for candidate in supported
    ]
    return deduplicate_openings(thresholded, config.opening_cleanup)


def _remeasure_mask_support(
    candidate: OpeningCandidate,
    *,
    opening_mask: np.ndarray,
    wall_graph: WallGraph,
) -> OpeningCandidate:
    if (
        candidate.drop_reason is not None
        or candidate.segment is None
        or candidate.host_wall_index is None
    ):
        return candidate
    wall = wall_graph.edges[candidate.host_wall_index]
    mask_support = sample_buffered_segment_bilinear_mean(
        opening_mask,
        candidate.segment,
        half_width_px=0.5 * wall.thickness_px,
    )
    confidence = (
        candidate.center_probability
        * candidate.type_probability
        * mask_support
        * candidate.attachment_score
    ) ** 0.25
    return replace(
        candidate,
        mask_support=mask_support,
        confidence=confidence,
    )


def attach_openings(
    candidates: Sequence[OpeningCandidate],
    wall_graph: WallGraph,
    config: AttachmentConfig,
) -> list[OpeningCandidate]:
    """Attach candidates to their best eligible wall without mutating the graph."""
    attached: list[OpeningCandidate] = []
    for candidate in candidates:
        if candidate.drop_reason is not None or candidate.segment is None:
            attached.append(candidate)
            continue
        if not wall_graph.edges:
            attached.append(replace(candidate, drop_reason="no_host_wall"))
            continue

        best: _AttachmentMatch | None = None
        passed_distance = False
        passed_angle = False
        passed_overlap = False
        for wall_index, wall in enumerate(wall_graph.edges):
            distance_limit = max(
                config.max_distance_px,
                config.max_distance_wall_thickness_factor * wall.thickness_px,
            )
            center_distance = wall.segment.point_to_segment_distance(candidate.center)
            if center_distance > distance_limit:
                continue
            passed_distance = True

            angle_difference = undirected_angle_difference_deg(
                candidate.segment.undirected_angle_deg,
                wall.segment.undirected_angle_deg,
            )
            if angle_difference > config.max_angle_difference_deg:
                continue
            passed_angle = True

            overlap_ratio = wall.segment.clipped_projected_overlap_ratio(
                candidate.segment
            )
            if overlap_ratio < config.min_projected_overlap_ratio:
                continue
            passed_overlap = True

            score = (
                config.distance_weight
                * max(0.0, 1.0 - center_distance / distance_limit)
                + config.angle_weight
                * max(
                    0.0,
                    1.0 - angle_difference / config.max_angle_difference_deg,
                )
                + config.segment_overlap_weight * overlap_ratio
                + config.wall_confidence_weight * wall.confidence
            )
            match = _AttachmentMatch(
                wall_index=wall_index,
                wall=wall,
                score=score,
            )
            if best is None or (match.score, -match.wall_index) > (
                best.score,
                -best.wall_index,
            ):
                best = match

        if best is None:
            reason: OpeningDropReason
            if not passed_distance:
                reason = "distance_gate"
            elif not passed_angle:
                reason = "angle_gate"
            elif not passed_overlap:
                reason = "overlap_gate"
            else:
                reason = "no_host_wall"
            attached.append(replace(candidate, drop_reason=reason))
            continue

        projected = Segment2D(
            _project_to_line(candidate.segment.start, best.wall.segment),
            _project_to_line(candidate.segment.end, best.wall.segment),
        )
        confidence = (
            candidate.center_probability
            * candidate.type_probability
            * candidate.mask_support
            * best.score
        ) ** 0.25
        attached.append(
            replace(
                candidate,
                segment=projected,
                center=projected.midpoint,
                host_wall_index=best.wall_index,
                attachment_score=best.score,
                confidence=confidence,
            )
        )
    return attached


def deduplicate_openings(
    candidates: Sequence[OpeningCandidate],
    config: OpeningCleanupConfig,
) -> list[OpeningCandidate]:
    """Mark lower-confidence same-host duplicates with a diagnostic reason."""
    cleaned = list(candidates)
    active_indices = sorted(
        (
            index
            for index, candidate in enumerate(cleaned)
            if candidate.drop_reason is None
            and candidate.segment is not None
            and candidate.host_wall_index is not None
        ),
        key=lambda index: (-cleaned[index].confidence, index),
    )
    kept_indices: list[int] = []
    for index in active_indices:
        candidate = cleaned[index]
        assert candidate.segment is not None
        duplicate = False
        for kept_index in kept_indices:
            kept = cleaned[kept_index]
            assert kept.segment is not None
            if (
                candidate.host_wall_index == kept.host_wall_index
                and candidate.opening_type == kept.opening_type
                and opening_segments_are_duplicates(
                    candidate.segment,
                    kept.segment,
                    center_distance_mean_length_factor=(
                        config.center_distance_mean_length_factor
                    ),
                    angle_tolerance_deg=config.max_angle_difference_deg,
                    min_overlap_ratio=config.min_projected_overlap_ratio,
                )
            ):
                duplicate = True
                break
        if duplicate:
            cleaned[index] = replace(candidate, drop_reason="duplicate")
        else:
            kept_indices.append(index)
    return cleaned


def _validated_opening_arrays(
    maps: FullResolutionMaps,
) -> dict[str, np.ndarray]:
    if not isinstance(maps, Mapping):
        raise GeometryError("opening maps must be a tensor mapping")
    source = cast(Mapping[str, object], maps)
    expected_channels = {
        "opening_mask": 1,
        "opening_center": 1,
        "opening_endpoint": 1,
        "opening_type": 2,
        "opening_orientation": 2,
        "opening_log_half_length": 1,
    }
    probability_maps = {
        "opening_mask",
        "opening_center",
        "opening_endpoint",
        "opening_type",
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
            raise GeometryError("opening maps must share one spatial shape")
        array = value.detach().to(dtype=torch.float32, device="cpu").numpy()
        if name in probability_maps and (np.any(array < 0.0) or np.any(array > 1.0)):
            raise GeometryError(f"{name} probabilities must be within [0, 1]")
        arrays[name] = array
    return arrays


def _decode_candidates(
    arrays: dict[str, np.ndarray],
    config: OpeningPostprocessConfig,
) -> list[OpeningCandidate]:
    center_heatmap = arrays["opening_center"][0]
    radius = config.center_nms_radius_px
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    local_maximum = center_heatmap == cv2.dilate(center_heatmap, kernel)
    peaks = [
        (float(center_heatmap[y, x]), int(y), int(x))
        for y, x in np.argwhere(
            (center_heatmap >= config.center_threshold) & local_maximum
        )
    ]
    peaks.sort(key=lambda peak: (-peak[0], peak[1], peak[2]))
    return [
        _decode_one(arrays, probability, x=x, y=y, config=config)
        for probability, y, x in peaks[: config.max_candidates]
    ]


def _decode_one(
    arrays: dict[str, np.ndarray],
    center_probability: float,
    *,
    x: int,
    y: int,
    config: OpeningPostprocessConfig,
) -> OpeningCandidate:
    source_center = Point2D(float(x), float(y))
    type_probabilities = (
        bilinear_sample(arrays["opening_type"][0], source_center),
        bilinear_sample(arrays["opening_type"][1], source_center),
    )
    type_index = max(
        range(len(type_probabilities)),
        key=lambda index: (type_probabilities[index], -index),
    )
    opening_type: OpeningType = "door" if type_index == 0 else "window"
    type_probability = type_probabilities[type_index]
    sin_twice_angle = bilinear_sample(
        arrays["opening_orientation"][0],
        source_center,
    )
    cos_twice_angle = bilinear_sample(
        arrays["opening_orientation"][1],
        source_center,
    )
    orientation_norm = math.hypot(sin_twice_angle, cos_twice_angle)
    if orientation_norm == 0.0:
        return OpeningCandidate(
            opening_type=opening_type,
            segment=None,
            center=source_center,
            center_probability=center_probability,
            type_probability=type_probability,
            mask_support=bilinear_sample(arrays["opening_mask"][0], source_center),
            drop_reason="invalid_orientation",
        )

    log_half_length = bilinear_sample(
        arrays["opening_log_half_length"][0],
        source_center,
    )
    try:
        half_length = math.expm1(log_half_length)
    except OverflowError:
        half_length = math.inf
    height, width = arrays["opening_center"].shape[1:]
    maximum_in_bounds_half_length = math.hypot(width - 1, height - 1)
    if (
        not math.isfinite(half_length)
        or half_length <= 0.0
        or half_length > maximum_in_bounds_half_length
    ):
        return OpeningCandidate(
            opening_type=opening_type,
            segment=None,
            center=source_center,
            center_probability=center_probability,
            type_probability=type_probability,
            mask_support=bilinear_sample(arrays["opening_mask"][0], source_center),
            drop_reason="invalid_length",
        )

    angle = 0.5 * math.atan2(sin_twice_angle, cos_twice_angle)
    direction = (math.cos(angle), math.sin(angle))
    first = Point2D(
        source_center.x - half_length * direction[0],
        source_center.y - half_length * direction[1],
    )
    second = Point2D(
        source_center.x + half_length * direction[0],
        source_center.y + half_length * direction[1],
    )
    first = refine_endpoint_from_heatmap(
        first,
        arrays["opening_endpoint"][0],
        radius_px=config.center_nms_radius_px,
        minimum_probability=config.center_threshold,
    )
    second = refine_endpoint_from_heatmap(
        second,
        arrays["opening_endpoint"][0],
        radius_px=config.center_nms_radius_px,
        minimum_probability=config.center_threshold,
    )
    if distance(first, second) <= 0.0:
        return OpeningCandidate(
            opening_type=opening_type,
            segment=None,
            center=source_center,
            center_probability=center_probability,
            type_probability=type_probability,
            mask_support=bilinear_sample(arrays["opening_mask"][0], source_center),
            drop_reason="invalid_length",
        )
    segment = Segment2D(first, second)
    return OpeningCandidate(
        opening_type=opening_type,
        segment=segment,
        center=segment.midpoint,
        center_probability=center_probability,
        type_probability=type_probability,
        mask_support=sample_segment_bilinear_mean(
            arrays["opening_mask"][0],
            segment,
        ),
    )


def _project_to_line(point: Point2D, wall: Segment2D) -> Point2D:
    parameter = wall.project_parameter(point)
    return Point2D(
        wall.start.x + parameter * (wall.end.x - wall.start.x),
        wall.start.y + parameter * (wall.end.y - wall.start.y),
    )
