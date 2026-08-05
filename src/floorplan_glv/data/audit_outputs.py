"""Deterministic artifact writers for normalized dataset audits."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from floorplan_glv.data.annotation_schema import AnnotationError, TrainingAnnotation
from floorplan_glv.data.index import _atomic_write_bytes
from floorplan_glv.training.checkpoint import atomic_write_json, atomic_write_text

if TYPE_CHECKING:
    from floorplan_glv.data.audit import DatasetAuditReport
    from floorplan_glv.data.audit_validation import UnresolvedHostSummary

_OVERLAY_PREFIX = "floorplan_glv_audit_"
_OVERLAY_SHA_LENGTH = 12
_OVERLAY_ID_LENGTH = 48
_SUMMARY_WIDTH = 960
_SUMMARY_LINE_HEIGHT = 18
_SUMMARY_PADDING = 24
_SUMMARY_MAX_FREQUENCY_ROWS = 20
_SUMMARY_MAX_SOURCE_ROWS = 20
_SUMMARY_MAX_LINES = 128
_SUMMARY_MAX_HEIGHT = _SUMMARY_PADDING * 2 + _SUMMARY_LINE_HEIGHT * _SUMMARY_MAX_LINES


@dataclass(frozen=True)
class _ValidSample:
    sample_id: str
    image_path: Path
    annotation: TrainingAnnotation


def build_overlay_selection(sample_ids: tuple[str, ...]) -> dict[str, str]:
    """Map stable owned filenames to selected sample IDs without collisions."""
    selection: dict[str, str] = {}
    for index, sample_id in enumerate(sample_ids, start=1):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
        safe_id = (safe_id or "sample")[:_OVERLAY_ID_LENGTH]
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        name = (
            f"{_OVERLAY_PREFIX}{index:06d}_{safe_id}_{digest[:_OVERLAY_SHA_LENGTH]}.png"
        )
        selection[name] = sample_id
    return selection


def write_audit_outputs(
    report: DatasetAuditReport, samples: list[_ValidSample], output: Path
) -> None:
    """Write deterministic report artifacts and selected geometry overlays."""
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "audit.json", report.model_dump(mode="json"))
    atomic_write_text(output / "audit.csv", _report_csv(report))
    _atomic_write_bytes(output / "audit_summary.png", _summary_png(report))
    overlays = output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for stale in overlays.glob(f"{_OVERLAY_PREFIX}*.png"):
        stale.unlink()
    samples_by_id = {sample.sample_id: sample for sample in samples}
    for name, sample_id in report.overlay_selection.items():
        _atomic_write_bytes(overlays / name, _overlay_png(samples_by_id[sample_id]))


def _report_csv(report: DatasetAuditReport) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("section", "name", "value"))
    for name, value in report.sample_counts.items():
        writer.writerow(("source_samples", name, value))
    for name, value in report.split_counts.items():
        writer.writerow(("split_samples", name, value))
    for name, value in report.image_size_counts.items():
        writer.writerow(("image_size_px", name, value))
    for name, value in report.class_counts.items():
        writer.writerow(("opening_class", name, value))
    frequencies = (
        ("wall_count_frequency", report.wall_count_frequency),
        ("opening_count_frequency", report.opening_count_frequency),
        ("wall_thickness_frequency_px", report.wall_thickness_frequency_px),
        ("opening_length_frequency_px", report.opening_length_frequency_px),
    )
    for section, values in frequencies:
        for name, value in values.items():
            writer.writerow((section, name, value))
    summaries: tuple[tuple[str, object], ...] = (
        ("indexed_samples", report.indexed_samples),
        ("valid_annotation_samples", report.valid_annotation_samples),
        ("invalid_annotation_samples", report.invalid_annotation_samples),
        ("wall_total", report.wall_count.total),
        ("wall_mean_per_sample", report.wall_count.mean_per_sample),
        ("opening_total", report.opening_count.total),
        ("opening_mean_per_sample", report.opening_count.mean_per_sample),
        ("wall_thickness_mean_px", report.wall_thickness_px.mean),
        ("opening_length_mean_px", report.opening_length_px.mean),
        ("unresolved_host_rate", report.unresolved_host_rate),
        (
            "unresolved_host_rate_excluding_waived",
            report.unresolved_host_rate_excluding_waived,
        ),
        ("invalid_json_samples", report.invalid_json_samples),
        ("invalid_coordinate_samples", report.invalid_coordinate_samples),
        ("nonfinite_coordinate_samples", report.nonfinite_coordinate_samples),
        ("out_of_bounds_samples", report.out_of_bounds_coordinate_samples),
        ("metadata_size_mismatches", report.metadata_size_mismatch_samples),
        ("door_window_type_missing", report.door_window_type_missing),
    )
    for name, summary_value in summaries:
        writer.writerow(("summary", name, summary_value))
    for source, summary in report.unresolved_host_by_source.items():
        writer.writerow(
            (
                "unresolved_host_source",
                source,
                f"{summary.unresolved_count}/{summary.opening_count}"
                f"|{summary.unresolved_rate}|waived={summary.waived}",
            )
        )
    for name, gate in report.hard_gates.items():
        writer.writerow(
            ("hard_gate", name, f"{gate.value}|{gate.requirement}|{gate.passed}")
        )
    for reason, value in report.rejection_reasons.items():
        writer.writerow(("rejection_reason", reason, value))
    return buffer.getvalue()


def _summary_png(report: DatasetAuditReport) -> bytes:
    lines = [
        "FloorPlan-GLV dataset audit",
        f"Samples: {report.valid_annotation_samples}/{report.indexed_samples}",
        f"Walls/openings: {report.wall_count.total}/{report.opening_count.total}",
        f"Door/window: {report.class_counts['door']}/{report.class_counts['window']}",
        f"Unresolved hosts raw: {report.unresolved_host_rate:.4f}",
        "Unresolved hosts excluding waived: "
        f"{report.unresolved_host_rate_excluding_waived:.4f}",
        "Waived sources: "
        + (", ".join(report.waived_unresolved_host_sources) or "none"),
        f"Duplicate candidates: {len(report.duplicate_candidates)}",
    ]
    frequency_sections = (
        ("Wall count frequency", report.wall_count_frequency),
        ("Opening count frequency", report.opening_count_frequency),
        ("Thickness frequency px", report.wall_thickness_frequency_px),
        ("Length frequency px", report.opening_length_frequency_px),
    )
    for title, values in frequency_sections:
        lines.extend(_frequency_summary_lines(title, values))
    lines.extend(_unresolved_host_summary_lines(report.unresolved_host_by_source))
    lines.extend(
        f"{name}: {'PASS' if gate.passed else 'FAIL'}"
        for name, gate in report.hard_gates.items()
    )
    lines = _absolutely_bounded_summary_lines(lines)
    height = min(
        _SUMMARY_MAX_HEIGHT,
        _SUMMARY_PADDING * 2 + _SUMMARY_LINE_HEIGHT * len(lines),
    )
    image = Image.new("RGB", (_SUMMARY_WIDTH, height), "white")
    ImageDraw.Draw(image).multiline_text(
        (_SUMMARY_PADDING, _SUMMARY_PADDING),
        "\n".join(lines),
        fill="black",
        spacing=6,
    )
    return _png_bytes(image)


def _frequency_summary_lines(title: str, values: dict[str, int]) -> list[str]:
    items = tuple(values.items())
    if len(items) <= _SUMMARY_MAX_FREQUENCY_ROWS:
        return [title, *(f"  {value}: {count}" for value, count in items)]
    side = _SUMMARY_MAX_FREQUENCY_ROWS // 2
    omitted = len(items) - _SUMMARY_MAX_FREQUENCY_ROWS
    return [
        title,
        *(f"  {value}: {count}" for value, count in items[:side]),
        f"  ... {omitted} exact values omitted from PNG ...",
        *(f"  {value}: {count}" for value, count in items[-side:]),
    ]


def _unresolved_host_summary_lines(
    values: dict[str, UnresolvedHostSummary],
) -> list[str]:
    items = tuple(sorted(values.items()))
    rows = tuple(
        f"  {source}: {summary.unresolved_count}/{summary.opening_count} "
        f"rate={summary.unresolved_rate:.4f} waived={summary.waived}"
        for source, summary in items
    )
    if len(rows) <= _SUMMARY_MAX_SOURCE_ROWS:
        return ["Unresolved hosts by source", *rows]
    side = _SUMMARY_MAX_SOURCE_ROWS // 2
    omitted = len(rows) - _SUMMARY_MAX_SOURCE_ROWS
    return [
        "Unresolved hosts by source",
        *rows[:side],
        f"  ... {omitted} sources omitted from PNG ...",
        *rows[-side:],
    ]


def _absolutely_bounded_summary_lines(lines: list[str]) -> list[str]:
    if len(lines) <= _SUMMARY_MAX_LINES:
        return lines
    head_count = _SUMMARY_MAX_LINES // 2
    tail_count = _SUMMARY_MAX_LINES - head_count - 1
    omitted = len(lines) - head_count - tail_count
    return [
        *lines[:head_count],
        f"... {omitted} summary lines omitted from PNG ...",
        *lines[-tail_count:],
    ]


def _overlay_png(sample: _ValidSample) -> bytes:
    try:
        with Image.open(sample.image_path) as encoded:
            image = encoded.convert("RGB")
    except OSError as exc:
        raise AnnotationError(
            f"sample {sample.sample_id} image {sample.image_path}: {exc}"
        ) from exc
    draw = ImageDraw.Draw(image)
    for wall in sample.annotation.walls:
        draw.line(wall.segment, fill=(255, 0, 0), width=2)
    for opening in sample.annotation.openings:
        if opening.segment is not None:
            color = (0, 120, 255) if opening.type == "door" else (0, 180, 0)
            draw.line(opening.segment, fill=color, width=2)
    return _png_bytes(image)


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
