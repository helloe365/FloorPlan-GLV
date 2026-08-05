"""Final deterministic ID assignment and atomic result export."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from floorplan_glv.data.output_schema import (
    CoordinateSystem,
    FloorPlanResult,
    ImageInfo,
    JsonContractError,
    Node,
    Opening,
    RunMetadata,
    Wall,
)
from floorplan_glv.geometry.primitives import Segment2D
from floorplan_glv.postprocess.openings import OpeningCandidate
from floorplan_glv.postprocess.wall_graph import WallEdgeCandidate, WallGraph


@dataclass(frozen=True, slots=True)
class _WallRecord:
    original_index: int
    start_node_id: str
    end_node_id: str
    segment: Segment2D
    candidate: WallEdgeCandidate


@dataclass(frozen=True, slots=True)
class _OpeningRecord:
    candidate: OpeningCandidate
    host_wall_id: str
    center_projection: float


def build_floorplan_result(
    wall_graph: WallGraph,
    openings: Sequence[OpeningCandidate],
    *,
    image: ImageInfo,
    metadata: RunMetadata,
) -> FloorPlanResult:
    """Assign stable IDs after cleanup and produce validated source-pixel JSON."""
    nodes, node_id_by_index = _build_nodes(wall_graph)
    walls, wall_id_by_index = _build_walls(wall_graph, node_id_by_index)
    serialized_openings = _build_openings(openings, walls, wall_id_by_index)
    try:
        return FloorPlanResult(
            image=image,
            coordinate_system=CoordinateSystem(),
            nodes=nodes,
            walls=walls,
            openings=serialized_openings,
            metadata=metadata,
        )
    except ValidationError as exc:
        raise JsonContractError(
            f"result candidates violate JSON contract: {exc}"
        ) from exc


def export_result(result: FloorPlanResult, destination: Path) -> Path:
    """Validate and atomically write ``destination/result.json`` as UTF-8."""
    output = destination / "result.json"
    try:
        validated = FloorPlanResult.model_validate(result.model_dump(mode="python"))
    except ValidationError as exc:
        raise JsonContractError(f"{output}: invalid FloorPlanResult: {exc}") from exc

    descriptor: int | None = None
    temporary: Path | None = None
    try:
        destination.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=destination,
        )
        temporary = Path(temporary_name)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with stream as handle:
            handle.write(
                validated.model_dump_json(
                    indent=2,
                    exclude_none=True,
                )
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    except Exception as exc:
        raise JsonContractError(f"{output}: atomic export failed: {exc}") from exc
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _build_nodes(wall_graph: WallGraph) -> tuple[tuple[Node, ...], dict[int, str]]:
    ordered = sorted(
        enumerate(wall_graph.nodes),
        key=lambda item: (
            round(item[1].point.y, 4),
            round(item[1].point.x, 4),
            item[1].kind,
        ),
    )
    node_id_by_index = {
        original_index: f"node_{stable_index:06d}"
        for stable_index, (original_index, _candidate) in enumerate(ordered, start=1)
    }
    nodes = tuple(
        Node(
            id=node_id_by_index[original_index],
            point=candidate.point.as_tuple(),
            kind=candidate.kind,
            confidence=candidate.confidence,
        )
        for original_index, candidate in ordered
    )
    return nodes, node_id_by_index


def _build_walls(
    wall_graph: WallGraph,
    node_id_by_index: dict[int, str],
) -> tuple[tuple[Wall, ...], dict[int, str]]:
    records: list[_WallRecord] = []
    for original_index, candidate in enumerate(wall_graph.edges):
        try:
            start_node_id = node_id_by_index[candidate.start_node_index]
            end_node_id = node_id_by_index[candidate.end_node_index]
        except KeyError as exc:
            raise JsonContractError(
                f"wall candidate {original_index} references missing node index "
                f"{exc.args[0]}"
            ) from exc
        segment = candidate.segment
        if start_node_id > end_node_id:
            start_node_id, end_node_id = end_node_id, start_node_id
            segment = Segment2D(segment.end, segment.start)
        records.append(
            _WallRecord(
                original_index=original_index,
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                segment=segment,
                candidate=candidate,
            )
        )
    records.sort(
        key=lambda record: (
            record.start_node_id,
            record.end_node_id,
            record.original_index,
        )
    )
    wall_id_by_index = {
        record.original_index: f"wall_{stable_index:06d}"
        for stable_index, record in enumerate(records, start=1)
    }
    walls = tuple(
        Wall(
            id=wall_id_by_index[record.original_index],
            start_node_id=record.start_node_id,
            end_node_id=record.end_node_id,
            segment=(
                record.segment.start.as_tuple(),
                record.segment.end.as_tuple(),
            ),
            thickness_px=record.candidate.thickness_px,
            confidence=record.candidate.confidence,
            raster_support=record.candidate.raster_support,
        )
        for record in records
    )
    return walls, wall_id_by_index


def _build_openings(
    candidates: Sequence[OpeningCandidate],
    walls: tuple[Wall, ...],
    wall_id_by_index: dict[int, str],
) -> tuple[Opening, ...]:
    wall_by_id = {wall.id: wall for wall in walls}
    records: list[_OpeningRecord] = []
    for index, candidate in enumerate(candidates):
        if candidate.drop_reason is not None:
            continue
        if candidate.segment is None or candidate.host_wall_index is None:
            raise JsonContractError(
                f"active opening candidate {index} lacks segment or host wall"
            )
        try:
            host_wall_id = wall_id_by_index[candidate.host_wall_index]
        except KeyError as exc:
            raise JsonContractError(
                f"opening candidate {index} references missing wall index "
                f"{candidate.host_wall_index}"
            ) from exc
        host = wall_by_id[host_wall_id]
        records.append(
            _OpeningRecord(
                candidate=candidate,
                host_wall_id=host_wall_id,
                center_projection=_projection_parameter(
                    candidate.center.as_tuple(),
                    host.segment,
                ),
            )
        )
    records.sort(
        key=lambda record: (
            record.host_wall_id,
            record.center_projection,
            record.candidate.opening_type,
        )
    )
    return tuple(
        Opening(
            id=f"opening_{stable_index:06d}",
            type=record.candidate.opening_type,
            segment=(
                record.candidate.segment.start.as_tuple(),
                record.candidate.segment.end.as_tuple(),
            ),
            center=record.candidate.center.as_tuple(),
            length_px=record.candidate.length_px,
            host_wall_id=record.host_wall_id,
            confidence=record.candidate.confidence,
            attachment_score=record.candidate.attachment_score,
        )
        for stable_index, record in enumerate(records, start=1)
        if record.candidate.segment is not None
    )


def _projection_parameter(
    point: tuple[float, float],
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    delta_x = segment[1][0] - segment[0][0]
    delta_y = segment[1][1] - segment[0][1]
    denominator = delta_x * delta_x + delta_y * delta_y
    return (
        (point[0] - segment[0][0]) * delta_x + (point[1] - segment[0][1]) * delta_y
    ) / denominator
