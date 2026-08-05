"""Validated deterministic JSON contract for FloorPlan-GLV results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    ValidationError,
    model_validator,
)

Point = tuple[FiniteFloat, FiniteFloat]
Segment = tuple[Point, Point]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]

_GEOMETRY_TOLERANCE_PX = 1e-3
_ZERO_LENGTH_TOLERANCE_PX = 1e-8
_MINIMUM_HOST_DISTANCE_TOLERANCE_PX = 2.0


class JsonContractError(ValueError):
    """Raised when result JSON cannot be read, validated, or exported."""


class StrictModel(BaseModel):
    """Frozen contract base that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageInfo(StrictModel):
    """Source raster identity and dimensions."""

    file_name: str = Field(min_length=1)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CoordinateSystem(StrictModel):
    """Fixed V1 source-image pixel coordinate system."""

    origin: Literal["top_left"] = "top_left"
    x_axis: Literal["right"] = "right"
    y_axis: Literal["down"] = "down"
    unit: Literal["pixel"] = "pixel"


class Node(StrictModel):
    """One stable wall endpoint or junction."""

    id: str = Field(pattern=r"^node_[0-9]{6}$")
    point: Point
    kind: Literal["endpoint", "junction"]
    confidence: Probability


class Wall(StrictModel):
    """One straight wall edge between two graph nodes."""

    id: str = Field(pattern=r"^wall_[0-9]{6}$")
    start_node_id: str = Field(pattern=r"^node_[0-9]{6}$")
    end_node_id: str = Field(pattern=r"^node_[0-9]{6}$")
    segment: Segment
    thickness_px: PositiveFiniteFloat
    confidence: Probability
    raster_support: Probability


class Opening(StrictModel):
    """One door or window projected onto a host wall."""

    id: str = Field(pattern=r"^opening_[0-9]{6}$")
    type: Literal["door", "window"]
    segment: Segment
    center: Point
    length_px: PositiveFiniteFloat
    host_wall_id: str = Field(pattern=r"^wall_[0-9]{6}$")
    confidence: Probability
    attachment_score: Probability


class RunMetadata(StrictModel):
    """Hashes needed to identify the inference inputs."""

    model_name: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    postprocess_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FloorPlanResult(StrictModel):
    """Complete V1 floor-plan result with cross-object validation."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    image: ImageInfo
    coordinate_system: CoordinateSystem
    nodes: tuple[Node, ...]
    walls: tuple[Wall, ...]
    openings: tuple[Opening, ...]
    metadata: RunMetadata

    @model_validator(mode="after")
    def validate_references_and_geometry(self) -> Self:
        """Validate bounds, graph references, geometry, and stable ordering."""
        _validate_unique_ids(self.nodes, self.walls, self.openings)
        _validate_sorted_nodes(self.nodes)
        node_by_id = {node.id: node for node in self.nodes}
        wall_by_id = {wall.id: wall for wall in self.walls}

        for node in self.nodes:
            self._check_point(node.point, node.id)
        for wall in self.walls:
            self._validate_wall(wall, node_by_id)
        _validate_sorted_walls(self.walls)
        for opening in self.openings:
            self._validate_opening(opening, wall_by_id)
        _validate_sorted_openings(self.openings, wall_by_id)
        return self

    def _check_point(self, point: Point, name: str) -> None:
        x, y = point
        if not (-0.5 <= x <= self.image.width_px - 0.5):
            raise ValueError(f"{name}.x is outside image bounds")
        if not (-0.5 <= y <= self.image.height_px - 0.5):
            raise ValueError(f"{name}.y is outside image bounds")

    def _validate_wall(self, wall: Wall, node_by_id: dict[str, Node]) -> None:
        if wall.start_node_id == wall.end_node_id:
            raise ValueError(f"{wall.id} references the same node twice")
        if wall.start_node_id not in node_by_id or wall.end_node_id not in node_by_id:
            raise ValueError(f"{wall.id} references a missing node")
        self._check_point(wall.segment[0], f"{wall.id}.segment[0]")
        self._check_point(wall.segment[1], f"{wall.id}.segment[1]")
        if _distance(*wall.segment) <= _ZERO_LENGTH_TOLERANCE_PX:
            raise ValueError(f"{wall.id} has zero length")
        if (
            _distance(wall.segment[0], node_by_id[wall.start_node_id].point)
            > _GEOMETRY_TOLERANCE_PX
        ):
            raise ValueError(f"{wall.id} start endpoint differs from its node")
        if (
            _distance(wall.segment[1], node_by_id[wall.end_node_id].point)
            > _GEOMETRY_TOLERANCE_PX
        ):
            raise ValueError(f"{wall.id} end endpoint differs from its node")

    def _validate_opening(
        self,
        opening: Opening,
        wall_by_id: dict[str, Wall],
    ) -> None:
        if opening.host_wall_id not in wall_by_id:
            raise ValueError(f"{opening.id} references a missing host wall")
        self._check_point(opening.segment[0], f"{opening.id}.segment[0]")
        self._check_point(opening.segment[1], f"{opening.id}.segment[1]")
        self._check_point(opening.center, f"{opening.id}.center")
        if _distance(_midpoint(opening.segment), opening.center) > (
            _GEOMETRY_TOLERANCE_PX
        ):
            raise ValueError(f"{opening.id} center is not its segment midpoint")
        actual_length = _distance(*opening.segment)
        if abs(actual_length - opening.length_px) > _GEOMETRY_TOLERANCE_PX:
            raise ValueError(f"{opening.id} length_px is inconsistent")
        host = wall_by_id[opening.host_wall_id]
        allowed_distance = max(
            _MINIMUM_HOST_DISTANCE_TOLERANCE_PX,
            0.5 * host.thickness_px,
        )
        if _point_line_distance(opening.center, host.segment) > allowed_distance:
            raise ValueError(f"{opening.id} is too far from its host wall")
        first = _projection_parameter(opening.segment[0], host.segment)
        second = _projection_parameter(opening.segment[1], host.segment)
        overlap = max(
            0.0,
            min(max(first, second), 1.0) - max(min(first, second), 0.0),
        )
        if overlap <= 0.0:
            raise ValueError(f"{opening.id} does not overlap its host wall")


def validate_result_json(path: Path) -> FloorPlanResult:
    """Read and validate one UTF-8 result file as the exact V1 contract."""
    try:
        payload = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise JsonContractError(f"{path}: file is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise JsonContractError(f"{path}: failed to read JSON: {exc}") from exc
    try:
        return FloorPlanResult.model_validate_json(payload)
    except ValidationError as exc:
        raise JsonContractError(f"{path}: invalid FloorPlanResult: {exc}") from exc


def _validate_unique_ids(
    nodes: tuple[Node, ...],
    walls: tuple[Wall, ...],
    openings: tuple[Opening, ...],
) -> None:
    for name, values in (
        ("node", nodes),
        ("wall", walls),
        ("opening", openings),
    ):
        identifiers = [value.id for value in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{name} IDs are not unique")


def _validate_sorted_nodes(nodes: tuple[Node, ...]) -> None:
    expected = sorted(
        nodes,
        key=lambda node: (
            round(node.point[1], 4),
            round(node.point[0], 4),
            node.kind,
        ),
    )
    if list(nodes) != expected:
        raise ValueError("nodes are not deterministically sorted")


def _validate_sorted_walls(walls: tuple[Wall, ...]) -> None:
    expected = sorted(
        walls,
        key=lambda wall: tuple(sorted((wall.start_node_id, wall.end_node_id))),
    )
    if list(walls) != expected:
        raise ValueError("walls are not deterministically sorted")


def _validate_sorted_openings(
    openings: tuple[Opening, ...],
    wall_by_id: dict[str, Wall],
) -> None:
    expected = sorted(
        openings,
        key=lambda opening: (
            opening.host_wall_id,
            _projection_parameter(
                opening.center,
                wall_by_id[opening.host_wall_id].segment,
            ),
            opening.type,
        ),
    )
    if list(openings) != expected:
        raise ValueError("openings are not deterministically sorted")


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _midpoint(segment: Segment) -> Point:
    return (
        0.5 * (segment[0][0] + segment[1][0]),
        0.5 * (segment[0][1] + segment[1][1]),
    )


def _projection_parameter(point: Point, segment: Segment) -> float:
    first_x, first_y = segment[0]
    delta_x = segment[1][0] - first_x
    delta_y = segment[1][1] - first_y
    denominator = delta_x * delta_x + delta_y * delta_y
    return (
        (point[0] - first_x) * delta_x + (point[1] - first_y) * delta_y
    ) / denominator


def _point_line_distance(point: Point, segment: Segment) -> float:
    parameter = _projection_parameter(point, segment)
    projected = (
        segment[0][0] + parameter * (segment[1][0] - segment[0][0]),
        segment[0][1] + parameter * (segment[1][1] - segment[0][1]),
    )
    return _distance(point, projected)
