"""Deterministic extraction of CubiCasa geometry from SVG files."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from floorplan_glv.geometry.primitives import GeometryError, Polygon2D

SvgCategory = Literal["wall", "door", "window", "fixed_furniture"]
Point = tuple[float, float]
Matrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

_IDENTITY: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
_PATH_TOKEN_RE = re.compile(rf"[A-Za-z]|{_NUMBER}")
_SEMANTIC_IDS: dict[str, SvgCategory] = {
    "Wall": "wall",
    "Door": "door",
    "Window": "window",
}
_SEMANTIC_CLASSES: dict[str, SvgCategory] = {
    "Wall": "wall",
    "Door": "door",
    "Window": "window",
    "FixedFurniture": "fixed_furniture",
}
_SHAPE_TAGS = {"polygon", "rect", "path"}


@dataclass(frozen=True, slots=True)
class SvgObject:
    """One supported semantic object in source-image pixel coordinates."""

    source_id: str
    category: SvgCategory
    polygon: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class SvgRejection:
    """One semantic SVG object that could not be represented in V1."""

    source_id: str
    category: SvgCategory
    reason: str


@dataclass(frozen=True, slots=True)
class ParsedSvg:
    """Parsed SVG dimensions, objects, and conversion audit information."""

    width: float
    height: float
    objects: tuple[SvgObject, ...]
    rejections: tuple[SvgRejection, ...]
    encountered_group_ids: tuple[str, ...]
    encountered_class_prefixes: tuple[str, ...]
    unknown_group_counts: dict[str, int]


def parse_svg_geometry(path: Path) -> ParsedSvg:
    """Parse supported CubiCasa SVG geometry.

    Coordinates are returned in the SVG raster's source-image pixel space.
    Unsupported geometry rejects only its containing semantic object.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GeometryError(f"failed to parse SVG {path}: {exc}") from exc

    width, height, root_matrix = _root_geometry(root)
    objects: list[SvgObject] = []
    rejections: list[SvgRejection] = []
    group_ids: set[str] = set()
    class_prefixes: set[str] = set()
    unknown_groups: Counter[str] = Counter()
    fallback_counts: Counter[SvgCategory] = Counter()

    def walk(element: ET.Element, parent_matrix: Matrix) -> None:
        matrix = _multiply(parent_matrix, _parse_transform(element.get("transform")))
        if _local_name(element.tag) == "g":
            group_id = element.get("id")
            class_prefix = _class_prefix(element)
            if group_id:
                group_ids.add(group_id)
            if class_prefix:
                class_prefixes.add(class_prefix)

            category = _semantic_category(element)
            if category is None:
                unknown_name = group_id or class_prefix
                if unknown_name:
                    unknown_groups[unknown_name] += 1
            else:
                fallback_counts[category] += 1
                shape = _find_primary_shape(element, category)
                if shape is not None:
                    shape_element, shape_matrix = shape
                    source_id = shape_element.get("id") or (
                        f"{category}_{fallback_counts[category]:06d}"
                    )
                    try:
                        polygon = _parse_shape(shape_element)
                        transformed = tuple(
                            _transform_point(
                                _multiply(matrix, shape_matrix),
                                point,
                            )
                            for point in polygon
                        )
                        Polygon2D(transformed)
                        objects.append(
                            SvgObject(
                                source_id=source_id,
                                category=category,
                                polygon=transformed,
                            )
                        )
                    except GeometryError as exc:
                        rejections.append(
                            SvgRejection(
                                source_id=source_id,
                                category=category,
                                reason=str(exc),
                            )
                        )

        for child in element:
            walk(child, matrix)

    walk(root, root_matrix)
    return ParsedSvg(
        width=width,
        height=height,
        objects=tuple(objects),
        rejections=tuple(rejections),
        encountered_group_ids=tuple(sorted(group_ids)),
        encountered_class_prefixes=tuple(sorted(class_prefixes)),
        unknown_group_counts=dict(sorted(unknown_groups.items())),
    )


def _root_geometry(root: ET.Element) -> tuple[float, float, Matrix]:
    view_box = _numbers(root.get("viewBox", ""))
    width = _dimension(root.get("width"))
    height = _dimension(root.get("height"))
    if (width is None or height is None) and len(view_box) == 4:
        width = view_box[2]
        height = view_box[3]
    if width is None or height is None or width <= 0.0 or height <= 0.0:
        raise GeometryError("SVG width and height must be positive")
    matrix = _IDENTITY
    if len(view_box) == 4:
        min_x, min_y, view_width, view_height = view_box
        if view_width <= 0.0 or view_height <= 0.0:
            raise GeometryError("SVG viewBox dimensions must be positive")
        matrix = _multiply(
            _scale(width / view_width, height / view_height),
            _translate(-min_x, -min_y),
        )
    return width, height, matrix


def _find_primary_shape(
    group: ET.Element,
    category: SvgCategory,
) -> tuple[ET.Element, Matrix] | None:
    if category == "fixed_furniture":
        for child in group:
            if (
                _local_name(child.tag) == "g"
                and _class_prefix(child) == "BoundaryPolygon"
            ):
                found = _find_shape(child, _IDENTITY)
                if found is not None:
                    return found
    for child in group:
        if _local_name(child.tag) in _SHAPE_TAGS:
            return child, _parse_transform(child.get("transform"))
    for child in group:
        if _local_name(child.tag) != "g" or _semantic_category(child) is not None:
            continue
        found = _find_shape(child, _IDENTITY)
        if found is not None:
            return found
    return None


def _find_shape(
    element: ET.Element,
    parent_matrix: Matrix,
) -> tuple[ET.Element, Matrix] | None:
    matrix = _multiply(parent_matrix, _parse_transform(element.get("transform")))
    for child in element:
        if _local_name(child.tag) in _SHAPE_TAGS:
            return child, _multiply(matrix, _parse_transform(child.get("transform")))
    for child in element:
        if _local_name(child.tag) != "g" or _semantic_category(child) is not None:
            continue
        found = _find_shape(child, matrix)
        if found is not None:
            return found
    return None


def _parse_shape(element: ET.Element) -> tuple[Point, ...]:
    tag = _local_name(element.tag)
    if tag == "polygon":
        values = _numbers(element.get("points", ""))
        if len(values) < 6 or len(values) % 2:
            raise GeometryError("polygon points must contain at least three pairs")
        return tuple(zip(values[::2], values[1::2], strict=True))
    if tag == "rect":
        x = _required_number(element, "x", default=0.0)
        y = _required_number(element, "y", default=0.0)
        width = _required_number(element, "width")
        height = _required_number(element, "height")
        if width <= 0.0 or height <= 0.0:
            raise GeometryError("rectangle width and height must be positive")
        return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
    if tag == "path":
        return _parse_path(element.get("d", ""))
    raise GeometryError(f"unsupported SVG shape {tag}")


def _parse_path(data: str) -> tuple[Point, ...]:
    tokens = _PATH_TOKEN_RE.findall(data)
    if not tokens:
        raise GeometryError("path data is empty")
    unsupported = sorted(
        {token for token in tokens if token.isalpha() and token not in "MmLlHhVvZz"}
    )
    if unsupported:
        raise GeometryError(f"unsupported path command {unsupported[0]}")

    points: list[Point] = []
    current = (0.0, 0.0)
    start: Point | None = None
    command: str | None = None
    index = 0
    move_pending = False
    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command in "Zz":
                if start is None:
                    raise GeometryError("path closes before it starts")
                current = start
                command = None
                continue
            move_pending = command in "Mm"
        if command is None:
            if index < len(tokens):
                raise GeometryError("path coordinates require a command")
            break
        relative = command.islower()
        upper = command.upper()
        needed = 2 if upper in {"M", "L"} else 1
        if index + needed > len(tokens) or any(
            tokens[position].isalpha() for position in range(index, index + needed)
        ):
            raise GeometryError(f"path command {command} lacks coordinates")
        values = tuple(float(tokens[index + offset]) for offset in range(needed))
        index += needed
        if upper in {"M", "L"}:
            x, y = values
            if relative:
                x += current[0]
                y += current[1]
            current = (x, y)
            if move_pending:
                start = current
                move_pending = False
                command = "l" if relative else "L"
        elif upper == "H":
            x = values[0] + current[0] if relative else values[0]
            current = (x, current[1])
        else:
            y = values[0] + current[1] if relative else values[0]
            current = (current[0], y)
        points.append(current)
    if len(points) > 1 and points[-1] == points[0]:
        points.pop()
    return tuple(points)


def _parse_transform(value: str | None) -> Matrix:
    if not value:
        return _IDENTITY
    result = _IDENTITY
    matches = tuple(_TRANSFORM_RE.finditer(value))
    if not matches:
        raise GeometryError(f"invalid SVG transform {value!r}")
    for match in matches:
        name = match.group(1)
        values = _numbers(match.group(2))
        if name == "translate" and len(values) in {1, 2}:
            transform = _translate(values[0], values[1] if len(values) == 2 else 0.0)
        elif name == "scale" and len(values) in {1, 2}:
            transform = _scale(values[0], values[1] if len(values) == 2 else values[0])
        elif name == "rotate" and len(values) in {1, 3}:
            transform = _rotate(*values)
        elif name == "matrix" and len(values) == 6:
            a, b, c, d, e, f = values
            transform = ((a, c, e), (b, d, f), (0.0, 0.0, 1.0))
        else:
            raise GeometryError(f"unsupported SVG transform {name}")
        result = _multiply(result, transform)
    return result


def _rotate(angle: float, center_x: float = 0.0, center_y: float = 0.0) -> Matrix:
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    rotation: Matrix = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    return _multiply(
        _multiply(_translate(center_x, center_y), rotation),
        _translate(-center_x, -center_y),
    )


def _translate(x: float, y: float) -> Matrix:
    return ((1.0, 0.0, x), (0.0, 1.0, y), (0.0, 0.0, 1.0))


def _scale(x: float, y: float) -> Matrix:
    return ((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, 1.0))


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            sum(left[row][offset] * right[offset][column] for offset in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _transform_point(matrix: Matrix, point: Point) -> Point:
    x, y = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
    )


def _semantic_category(element: ET.Element) -> SvgCategory | None:
    group_id = element.get("id")
    if group_id in _SEMANTIC_IDS:
        assert group_id is not None
        return _SEMANTIC_IDS[group_id]
    class_prefix = _class_prefix(element)
    return _SEMANTIC_CLASSES.get(class_prefix) if class_prefix is not None else None


def _class_prefix(element: ET.Element) -> str | None:
    value = element.get("class", "").strip()
    return value.split()[0] if value else None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _numbers(value: str) -> tuple[float, ...]:
    return tuple(float(number) for number in re.findall(_NUMBER, value))


def _dimension(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.match(_NUMBER, value.strip())
    return float(match.group()) if match else None


def _required_number(
    element: ET.Element,
    name: str,
    *,
    default: float | None = None,
) -> float:
    value = element.get(name)
    if value is None:
        if default is None:
            raise GeometryError(f"{_local_name(element.tag)} requires {name}")
        return default
    parsed = _dimension(value)
    if parsed is None:
        raise GeometryError(f"{_local_name(element.tag)} has invalid {name}")
    return parsed
