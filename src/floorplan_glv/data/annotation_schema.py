"""Normalized source-image annotation schema for training data."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from floorplan_glv.geometry.primitives import (
    GeometryError,
    Polygon2D,
    Segment2D,
)

Point = tuple[float, float]
Segment = tuple[Point, Point]
Polygon = tuple[Point, ...]
NonEmptyString = Annotated[str, Field(min_length=1)]
Quality = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class AnnotationError(ValueError):
    """Raised when normalized annotation processing fails."""


class StrictAnnotationModel(BaseModel):
    """Immutable annotation base that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageAnnotation(StrictAnnotationModel):
    """Source image metadata stored with each normalized annotation."""

    file_name: NonEmptyString
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    source: NonEmptyString


class WallAnnotation(StrictAnnotationModel):
    """One straight wall record and its raster mask polygon."""

    source_id: NonEmptyString
    polygon: Polygon
    segment: Segment
    thickness_px: PositiveFiniteFloat
    quality: Quality

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: Polygon) -> Polygon:
        """Reject non-finite or degenerate wall polygons."""
        _require_polygon(value, "wall polygon")
        return value

    @field_validator("segment")
    @classmethod
    def validate_segment(cls, value: Segment) -> Segment:
        """Reject non-finite or degenerate wall segments."""
        _require_segment(value, "wall segment")
        return value


class OpeningAnnotation(StrictAnnotationModel):
    """A door or window mask with optional stable segment geometry."""

    source_id: NonEmptyString
    type: Literal["door", "window"]
    polygon: Polygon | None
    segment: Segment | None
    host_wall_source_id: NonEmptyString | None
    quality: Quality

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: Polygon | None) -> Polygon | None:
        """Reject a provided opening polygon when it is malformed."""
        if value is not None:
            _require_polygon(value, "opening polygon")
        return value

    @field_validator("segment")
    @classmethod
    def validate_segment(cls, value: Segment | None) -> Segment | None:
        """Reject a provided opening segment when it is malformed."""
        if value is not None:
            _require_segment(value, "opening segment")
        return value

    @model_validator(mode="after")
    def validate_available_geometry(self) -> Self:
        """Require mask or segment geometry for every opening."""
        if self.polygon is None and self.segment is None:
            raise ValueError("opening requires a polygon or segment")
        return self


class HardNegativeAnnotation(StrictAnnotationModel):
    """A polygon that must remain negative for opening objectness."""

    source_id: NonEmptyString
    polygon: Polygon
    category: NonEmptyString

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: Polygon) -> Polygon:
        """Reject non-finite or degenerate hard-negative polygons."""
        _require_polygon(value, "hard-negative polygon")
        return value


class TrainingAnnotation(StrictAnnotationModel):
    """Complete normalized annotation in source-image pixel coordinates."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    sample_id: NonEmptyString
    image: ImageAnnotation
    walls: tuple[WallAnnotation, ...] = ()
    openings: tuple[OpeningAnnotation, ...] = ()
    hard_negatives: tuple[HardNegativeAnnotation, ...] = ()

    @model_validator(mode="after")
    def validate_references_and_bounds(self) -> Self:
        """Validate IDs, host-wall references, and every coordinate."""
        _require_unique_ids(
            (wall.source_id for wall in self.walls),
            "wall source_id",
        )
        _require_unique_ids(
            (opening.source_id for opening in self.openings),
            "opening source_id",
        )
        _require_unique_ids(
            (negative.source_id for negative in self.hard_negatives),
            "hard-negative source_id",
        )

        wall_ids = {wall.source_id for wall in self.walls}
        for opening in self.openings:
            host_id = opening.host_wall_source_id
            if host_id is not None and host_id not in wall_ids:
                raise ValueError(
                    f"opening {opening.source_id} references missing wall {host_id}"
                )

        for name, point in self._named_points():
            _require_point_in_bounds(
                point,
                width=self.image.width,
                height=self.image.height,
                name=name,
            )
        return self

    def _named_points(self) -> tuple[tuple[str, Point], ...]:
        named: list[tuple[str, Point]] = []
        for wall in self.walls:
            named.extend(
                (f"{wall.source_id}.polygon[{index}]", point)
                for index, point in enumerate(wall.polygon)
            )
            named.extend(
                (f"{wall.source_id}.segment[{index}]", point)
                for index, point in enumerate(wall.segment)
            )
        for opening in self.openings:
            if opening.polygon is not None:
                named.extend(
                    (f"{opening.source_id}.polygon[{index}]", point)
                    for index, point in enumerate(opening.polygon)
                )
            if opening.segment is not None:
                named.extend(
                    (f"{opening.source_id}.segment[{index}]", point)
                    for index, point in enumerate(opening.segment)
                )
        for negative in self.hard_negatives:
            named.extend(
                (f"{negative.source_id}.polygon[{index}]", point)
                for index, point in enumerate(negative.polygon)
            )
        return tuple(named)


def _require_polygon(value: Polygon, name: str) -> None:
    try:
        Polygon2D(value)
    except GeometryError as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc


def _require_segment(value: Segment, name: str) -> None:
    try:
        Segment2D(*value)
    except GeometryError as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc


def _require_unique_ids(values: Iterable[str], name: str) -> None:
    identifiers: tuple[str, ...] = tuple(values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} values must be unique")


def _require_point_in_bounds(
    point: Point,
    *,
    width: int,
    height: int,
    name: str,
) -> None:
    x, y = point
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{name} contains a non-finite coordinate")
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        raise ValueError(f"{name} is outside image bounds")
