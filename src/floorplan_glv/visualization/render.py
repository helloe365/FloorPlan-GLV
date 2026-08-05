"""Deterministic source-pixel debug visualization rendering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from floorplan_glv.data.output_schema import FloorPlanResult, Segment
from floorplan_glv.geometry.primitives import GeometryError
from floorplan_glv.postprocess.openings import OpeningCandidate

_WALL_COLOR = (30, 110, 220)
_WALL_BAND_COLOR = (210, 225, 245)
_NODE_COLOR = (220, 40, 40)
_DOOR_COLOR = (30, 170, 70)
_WINDOW_COLOR = (20, 150, 220)
_DROP_COLOR = (220, 30, 180)
_TEXT_COLOR = (20, 20, 20)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_debug_images(
    image: np.ndarray,
    result: FloorPlanResult,
    *,
    dropped_openings: Sequence[OpeningCandidate] = (),
) -> dict[str, np.ndarray]:
    """Render deterministic RGB wall, opening, and combined debug images."""
    source = _validated_source_image(image, result)
    wall_graph = source.copy()
    openings = source.copy()
    overlay = source.copy()
    _draw_walls(wall_graph, result)
    _draw_walls(overlay, result)
    _draw_openings(openings, result, dropped_openings)
    _draw_openings(overlay, result, dropped_openings)
    return {
        "wall_graph.png": wall_graph,
        "openings.png": openings,
        "overlay.png": overlay,
    }


def write_debug_visualizations(
    image: np.ndarray,
    result: FloorPlanResult,
    destination: Path,
    *,
    dropped_openings: Sequence[OpeningCandidate] = (),
) -> dict[str, Path]:
    """Write byte-deterministic PNG debug images and return their paths."""
    rendered = render_debug_images(
        image,
        result,
        dropped_openings=dropped_openings,
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, rgb_image in rendered.items():
        success, encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        if not success:
            raise GeometryError(f"failed to encode visualization {name}")
        output = destination / name
        output.write_bytes(encoded.tobytes())
        outputs[name] = output
    return outputs


def _validated_source_image(
    image: np.ndarray,
    result: FloorPlanResult,
) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise GeometryError("visualization image must be uint8 RGB [height, width, 3]")
    expected_shape = (result.image.height_px, result.image.width_px)
    if image.shape[:2] != expected_shape:
        raise GeometryError(
            f"visualization image shape must be {expected_shape}, got {image.shape[:2]}"
        )
    return image.copy()


def _draw_walls(canvas: np.ndarray, result: FloorPlanResult) -> None:
    for wall in result.walls:
        polygon = _segment_band(wall.segment, wall.thickness_px)
        cv2.fillConvexPoly(canvas, polygon, _WALL_BAND_COLOR, lineType=cv2.LINE_8)
        cv2.line(
            canvas,
            _point(wall.segment[0]),
            _point(wall.segment[1]),
            _WALL_COLOR,
            2,
            cv2.LINE_8,
        )
        midpoint = _midpoint(wall.segment)
        _label(
            canvas,
            f"{wall.id} {wall.confidence:.2f}",
            (midpoint[0] + 2, midpoint[1] - 4),
        )
    for node in result.nodes:
        point = _point(node.point)
        cv2.circle(canvas, point, 3, _NODE_COLOR, -1, cv2.LINE_8)
        _label(
            canvas,
            f"{node.id} {node.confidence:.2f}",
            (point[0] + 3, point[1] + 10),
        )


def _draw_openings(
    canvas: np.ndarray,
    result: FloorPlanResult,
    dropped_openings: Sequence[OpeningCandidate],
) -> None:
    for opening in result.openings:
        color = _DOOR_COLOR if opening.type == "door" else _WINDOW_COLOR
        cv2.line(
            canvas,
            _point(opening.segment[0]),
            _point(opening.segment[1]),
            color,
            4,
            cv2.LINE_8,
        )
        center = _point(opening.center)
        cv2.circle(canvas, center, 3, color, -1, cv2.LINE_8)
        _label(
            canvas,
            (
                f"{opening.id} {opening.type} {opening.host_wall_id} "
                f"{opening.confidence:.2f}"
            ),
            (center[0] + 3, center[1] - 5),
        )
    for candidate in dropped_openings:
        center = _point(candidate.center.as_tuple())
        if candidate.segment is not None:
            cv2.line(
                canvas,
                _point(candidate.segment.start.as_tuple()),
                _point(candidate.segment.end.as_tuple()),
                _DROP_COLOR,
                2,
                cv2.LINE_8,
            )
        cv2.line(
            canvas,
            (center[0] - 4, center[1] - 4),
            (center[0] + 4, center[1] + 4),
            _DROP_COLOR,
            2,
            cv2.LINE_8,
        )
        cv2.line(
            canvas,
            (center[0] - 4, center[1] + 4),
            (center[0] + 4, center[1] - 4),
            _DROP_COLOR,
            2,
            cv2.LINE_8,
        )
        _label(
            canvas,
            f"dropped: {candidate.drop_reason or 'unknown'}",
            (center[0] + 4, center[1] + 12),
            color=_DROP_COLOR,
        )


def _segment_band(segment: Segment, thickness_px: float) -> np.ndarray:
    delta_x = segment[1][0] - segment[0][0]
    delta_y = segment[1][1] - segment[0][1]
    length = math.hypot(delta_x, delta_y)
    normal_x = -delta_y / length
    normal_y = delta_x / length
    half_width = 0.5 * thickness_px
    return np.asarray(
        [
            _point(
                (
                    segment[0][0] + half_width * normal_x,
                    segment[0][1] + half_width * normal_y,
                )
            ),
            _point(
                (
                    segment[1][0] + half_width * normal_x,
                    segment[1][1] + half_width * normal_y,
                )
            ),
            _point(
                (
                    segment[1][0] - half_width * normal_x,
                    segment[1][1] - half_width * normal_y,
                )
            ),
            _point(
                (
                    segment[0][0] - half_width * normal_x,
                    segment[0][1] - half_width * normal_y,
                )
            ),
        ],
        dtype=np.int32,
    )


def _midpoint(segment: Segment) -> tuple[int, int]:
    return _point(
        (
            0.5 * (segment[0][0] + segment[1][0]),
            0.5 * (segment[0][1] + segment[1][1]),
        )
    )


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0]), round(point[1]))


def _label(
    canvas: np.ndarray,
    text: str,
    point: tuple[int, int],
    *,
    color: tuple[int, int, int] = _TEXT_COLOR,
) -> None:
    cv2.putText(
        canvas,
        text,
        point,
        _FONT,
        0.32,
        color,
        1,
        cv2.LINE_8,
    )
