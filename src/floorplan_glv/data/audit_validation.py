"""Strict annotation checks and explicit unresolved-host waiver accounting."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from floorplan_glv.data.annotation_schema import AnnotationError


class UnresolvedHostSummary(BaseModel):
    """Raw unresolved-host counts for one normalized dataset source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opening_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    unresolved_rate: float = Field(ge=0.0, le=1.0)
    waived: bool


@dataclass(frozen=True)
class UnresolvedHostAccounting:
    """Raw and waiver-adjusted unresolved-host audit values."""

    by_source: dict[str, UnresolvedHostSummary]
    raw_rate: float
    excluding_waived_rate: float
    waived_sources: tuple[str, ...]


def coordinate_status(
    raw: Mapping[str, Any], width: int, height: int
) -> tuple[bool, bool, bool]:
    """Return malformed, non-finite, and decoded-image out-of-bounds flags."""
    invalid = nonfinite = out_of_bounds = False
    for group in ("walls", "openings", "hard_negatives"):
        objects = raw.get(group)
        if not isinstance(objects, list):
            continue
        for item in objects:
            if not isinstance(item, dict):
                continue
            for geometry in ("polygon", "segment"):
                value = item.get(geometry)
                if value is None:
                    continue
                if not isinstance(value, (list, tuple)):
                    invalid = True
                    continue
                for point in value:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        invalid = True
                        continue
                    x, y = point
                    if not (_is_number(x) and _is_number(y)):
                        invalid = True
                        continue
                    x, y = float(x), float(y)
                    if not (math.isfinite(x) and math.isfinite(y)):
                        nonfinite = True
                    elif not (0.0 <= x <= width and 0.0 <= y <= height):
                        out_of_bounds = True
    return invalid, nonfinite, out_of_bounds


def metadata_size_matches(raw: Mapping[str, Any], width: int, height: int) -> bool:
    """Return whether declared dimensions equal decoded image dimensions."""
    image = raw.get("image")
    return (
        isinstance(image, dict)
        and image.get("width") == width
        and image.get("height") == height
    )


def frequency(values: list[int] | list[float]) -> dict[str, int]:
    """Return stable literal value frequencies without thresholded bins."""
    counts = Counter(_canonical_number(float(value)) for value in values)
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def unresolved_host_accounting(
    unresolved_flags_by_source: Mapping[str, Sequence[bool]],
    present_sources: Sequence[str],
    requested_waivers: Sequence[str],
) -> UnresolvedHostAccounting:
    """Validate explicit waivers and compute raw/per-source/effective rates."""
    waived = _validated_waivers(requested_waivers, present_sources)
    waived_set = set(waived)
    by_source: dict[str, UnresolvedHostSummary] = {}
    raw_total = raw_unresolved = effective_total = effective_unresolved = 0
    for source in sorted(set(present_sources)):
        flags = tuple(unresolved_flags_by_source.get(source, ()))
        total = len(flags)
        unresolved = sum(flags)
        raw_total += total
        raw_unresolved += unresolved
        if source not in waived_set:
            effective_total += total
            effective_unresolved += unresolved
        by_source[source] = UnresolvedHostSummary(
            opening_count=total,
            unresolved_count=unresolved,
            unresolved_rate=unresolved / total if total else 0.0,
            waived=source in waived_set,
        )
    return UnresolvedHostAccounting(
        by_source=by_source,
        raw_rate=raw_unresolved / raw_total if raw_total else 0.0,
        excluding_waived_rate=(
            effective_unresolved / effective_total if effective_total else 0.0
        ),
        waived_sources=waived,
    )


def _validated_waivers(
    requested: Sequence[str], present_sources: Sequence[str]
) -> tuple[str, ...]:
    if any(not source.strip() for source in requested):
        raise AnnotationError("waived unresolved-host source must be non-empty")
    waived = tuple(sorted(set(requested)))
    missing = sorted(set(waived) - set(present_sources))
    if missing:
        raise AnnotationError(
            f"waived unresolved-host source is not present: {', '.join(missing)}"
        )
    return waived


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _canonical_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)
