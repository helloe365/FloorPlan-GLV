"""Deterministic annotation-aware training patch sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator
from shapely.geometry import LineString, Polygon, box  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]

from floorplan_glv.config.models import StrictConfigModel
from floorplan_glv.data.annotation_schema import TrainingAnnotation
from floorplan_glv.geometry.primitives import Segment2D

SamplingGroup = Literal[
    "opening_centered",
    "wall_junction_centered",
    "random_structural",
    "fixed_furniture",
    "random_background",
]
Point = tuple[float, float]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]


class PatchSamplingConfig(StrictConfigModel):
    """Validated patch geometry and sampling mixture."""

    patch_size: PositiveInt = 512
    opening_centered_probability: Probability = 0.35
    wall_junction_centered_probability: Probability = 0.25
    random_structural_probability: Probability = 0.20
    fixed_furniture_probability: Probability = 0.10
    random_background_probability: Probability = 0.10
    opening_central_fraction: Annotated[
        float,
        Field(gt=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.70
    background_candidate_attempts: PositiveInt = 64
    background_max_annotated_area_ratio: Probability = 0.0

    @model_validator(mode="after")
    def validate_sampling_mixture(self) -> Self:
        """Require one complete probability distribution."""
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("sampling probabilities must sum to 1")
        return self

    @property
    def probabilities(self) -> tuple[float, ...]:
        """Return mixture probabilities in deterministic selection order."""
        return (
            self.opening_centered_probability,
            self.wall_junction_centered_probability,
            self.random_structural_probability,
            self.fixed_furniture_probability,
            self.random_background_probability,
        )


@dataclass(frozen=True, slots=True)
class PatchRequest:
    """One square source-image patch request and its sampling provenance."""

    source_box: tuple[int, int, int, int]
    requested_group: SamplingGroup
    sampling_group: SamplingGroup
    anchor: Point | None

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.source_box
        if x1 <= x0 or y1 <= y0 or x1 - x0 != y1 - y0:
            raise ValueError("patch source box must be a non-empty square")
        if self.anchor is not None and not all(
            math.isfinite(value) for value in self.anchor
        ):
            raise ValueError("patch anchor must be finite")

    def contains(self, point: Point) -> bool:
        """Return whether a source point lies inside the closed patch box."""
        x0, y0, x1, y1 = self.source_box
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


class PatchSampler:
    """Sample annotation-aware patches using an explicit NumPy RNG."""

    _GROUPS: tuple[SamplingGroup, ...] = (
        "opening_centered",
        "wall_junction_centered",
        "random_structural",
        "fixed_furniture",
        "random_background",
    )

    def __init__(self, config: PatchSamplingConfig | None = None) -> None:
        self.config = config or PatchSamplingConfig()

    def sample(
        self,
        annotation: TrainingAnnotation,
        rng: np.random.Generator,
    ) -> PatchRequest:
        """Return one deterministic request for the supplied RNG state."""
        requested_group = self._select_group(rng)
        sampling_group, anchor = self._select_anchor(
            requested_group,
            annotation,
            rng,
        )
        if sampling_group == "random_background":
            source_box = self._background_box(annotation, rng)
            if source_box is None:
                sampling_group, anchor = self._occupied_sample_fallback(
                    annotation,
                    rng,
                )
                source_box = self._anchored_box(annotation, anchor, rng)
        else:
            assert anchor is not None
            source_box = self._anchored_box(annotation, anchor, rng)
        return PatchRequest(
            source_box=source_box,
            requested_group=requested_group,
            sampling_group=sampling_group,
            anchor=anchor,
        )

    def _occupied_sample_fallback(
        self,
        annotation: TrainingAnnotation,
        rng: np.random.Generator,
    ) -> tuple[SamplingGroup, Point]:
        for group in (
            "random_structural",
            "opening_centered",
            "fixed_furniture",
        ):
            anchors = self._anchors_for_group(group, annotation, rng)
            if anchors:
                return group, _choose(anchors, rng)
        raise ValueError("occupied sample has no annotation anchor")

    def _select_group(self, rng: np.random.Generator) -> SamplingGroup:
        draw = float(rng.random())
        cumulative = 0.0
        for group, probability in zip(
            self._GROUPS,
            self.config.probabilities,
            strict=True,
        ):
            cumulative += probability
            if draw < cumulative:
                return group
        return self._GROUPS[-1]

    def _select_anchor(
        self,
        requested_group: SamplingGroup,
        annotation: TrainingAnnotation,
        rng: np.random.Generator,
    ) -> tuple[SamplingGroup, Point | None]:
        if requested_group == "random_background":
            return "random_background", None
        anchors = self._anchors_for_group(requested_group, annotation, rng)
        if anchors:
            return requested_group, _choose(anchors, rng)
        structural = self._structural_anchors(annotation, rng)
        if structural:
            return "random_structural", _choose(structural, rng)
        return "random_background", None

    def _anchors_for_group(
        self,
        group: SamplingGroup,
        annotation: TrainingAnnotation,
        rng: np.random.Generator,
    ) -> tuple[Point, ...]:
        if group == "opening_centered":
            return tuple(
                point
                for opening in annotation.openings
                if (point := _opening_center(opening.segment, opening.polygon))
                is not None
            )
        if group == "wall_junction_centered":
            return _wall_junctions(annotation)
        if group == "random_structural":
            return self._structural_anchors(annotation, rng)
        if group == "fixed_furniture":
            return tuple(
                _polygon_anchor(negative.polygon)
                for negative in annotation.hard_negatives
            )
        return ()

    def _structural_anchors(
        self,
        annotation: TrainingAnnotation,
        rng: np.random.Generator,
    ) -> tuple[Point, ...]:
        if not annotation.walls:
            return ()
        wall = annotation.walls[int(rng.integers(0, len(annotation.walls)))]
        parameter = float(rng.random())
        start, end = wall.segment
        return (
            (
                start[0] + parameter * (end[0] - start[0]),
                start[1] + parameter * (end[1] - start[1]),
            ),
        )

    def _anchored_box(
        self,
        annotation: TrainingAnnotation,
        anchor: Point,
        rng: np.random.Generator,
    ) -> tuple[int, int, int, int]:
        patch_size = self.config.patch_size
        margin = 0.5 * (1.0 - self.config.opening_central_fraction) * patch_size
        x0 = _anchored_origin(
            anchor[0],
            image_extent=annotation.image.width,
            patch_size=patch_size,
            margin=margin,
            rng=rng,
        )
        y0 = _anchored_origin(
            anchor[1],
            image_extent=annotation.image.height,
            patch_size=patch_size,
            margin=margin,
            rng=rng,
        )
        return (x0, y0, x0 + patch_size, y0 + patch_size)

    def _background_box(
        self,
        annotation: TrainingAnnotation,
        rng: np.random.Generator,
    ) -> tuple[int, int, int, int] | None:
        patch_size = self.config.patch_size
        maximum_x = max(0, annotation.image.width - patch_size)
        maximum_y = max(0, annotation.image.height - patch_size)
        for _ in range(self.config.background_candidate_attempts):
            x0 = int(rng.integers(0, maximum_x + 1)) if maximum_x else 0
            y0 = int(rng.integers(0, maximum_y + 1)) if maximum_y else 0
            source_box = (x0, y0, x0 + patch_size, y0 + patch_size)
            if self._is_background(source_box, annotation):
                return source_box
        return None

    def _is_background(
        self,
        source_box: tuple[int, int, int, int],
        annotation: TrainingAnnotation,
    ) -> bool:
        patch_geometry = box(*source_box)
        polygons = [Polygon(wall.polygon) for wall in annotation.walls]
        polygons.extend(
            Polygon(opening.polygon)
            for opening in annotation.openings
            if opening.polygon is not None
        )
        polygons.extend(
            Polygon(negative.polygon) for negative in annotation.hard_negatives
        )
        if polygons:
            annotated_area = float(
                unary_union(polygons).intersection(patch_geometry).area
            )
            area_ratio = annotated_area / float(self.config.patch_size**2)
            if area_ratio > self.config.background_max_annotated_area_ratio:
                return False
        return all(
            opening.polygon is not None
            or opening.segment is None
            or not LineString(opening.segment).intersects(patch_geometry)
            for opening in annotation.openings
        )


def _choose(points: tuple[Point, ...], rng: np.random.Generator) -> Point:
    return points[int(rng.integers(0, len(points)))]


def _opening_center(
    segment: tuple[Point, Point] | None,
    polygon: tuple[Point, ...] | None,
) -> Point | None:
    if segment is not None:
        return (
            0.5 * (segment[0][0] + segment[1][0]),
            0.5 * (segment[0][1] + segment[1][1]),
        )
    return _polygon_anchor(polygon) if polygon is not None else None


def _polygon_anchor(polygon: tuple[Point, ...]) -> Point:
    point = Polygon(polygon).representative_point()
    return (float(point.x), float(point.y))


def _wall_junctions(annotation: TrainingAnnotation) -> tuple[Point, ...]:
    segments = tuple(Segment2D(*wall.segment) for wall in annotation.walls)
    points = {point for wall in annotation.walls for point in wall.segment}
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            intersection = first.intersection(second)
            if intersection is not None:
                points.add(intersection.as_tuple())
    return tuple(sorted(points, key=lambda point: (point[1], point[0])))


def _anchored_origin(
    coordinate: float,
    *,
    image_extent: int,
    patch_size: int,
    margin: float,
    rng: np.random.Generator,
) -> int:
    if not 0.0 <= coordinate <= image_extent:
        raise ValueError("patch anchor is outside annotation image bounds")
    lowest = math.ceil(coordinate - (patch_size - margin))
    highest = math.floor(coordinate - margin)
    if lowest <= highest:
        return int(rng.integers(lowest, highest + 1))
    return round(coordinate - patch_size * 0.5)
