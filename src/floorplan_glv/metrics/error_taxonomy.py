"""Source-stratified failure taxonomy and annotation-only crop manifests."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from floorplan_glv.training.checkpoint import atomic_write_json, atomic_write_text

ERROR_CATEGORIES = (
    "furniture mistaken for opening",
    "dimension line mistaken for wall",
    "thin wall missed",
    "thick wall split",
    "wall junction broken",
    "door/window type confusion",
    "opening not attached",
    "duplicate opening",
    "patch seam artifact",
    "non-Manhattan wall distorted",
)

ErrorCategory = Literal[
    "furniture mistaken for opening",
    "dimension line mistaken for wall",
    "thin wall missed",
    "thick wall split",
    "wall junction broken",
    "door/window type confusion",
    "opening not attached",
    "duplicate opening",
    "patch seam artifact",
    "non-Manhattan wall distorted",
]

FalsePositiveCategory = Literal[
    "furniture mistaken for opening",
    "dimension line mistaken for wall",
    "duplicate opening",
    "patch seam artifact",
]


class ErrorRecord(BaseModel):
    """One human- or evaluator-assigned failure in the documented taxonomy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    category: ErrorCategory


class ErrorTaxonomyReport(BaseModel):
    """Deterministic total and per-source counts for all ten categories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_records: int = Field(ge=0)
    total_counts: dict[str, int]
    by_source: dict[str, dict[str, int]]


class FalsePositiveCrop(BaseModel):
    """A source-pixel crop proposed for human hard-negative annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    prediction_id: str = Field(min_length=1)
    category: FalsePositiveCategory
    crop_box_xyxy: tuple[float, float, float, float]
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    provenance: dict[str, str]
    coordinate_system: Literal["source_pixel_xyxy"] = "source_pixel_xyxy"

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, value: dict[str, str]) -> dict[str, str]:
        """Require explicit non-empty provenance keys and values."""
        if not value or any(not key or not item for key, item in value.items()):
            raise ValueError("provenance requires non-empty keys and values")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_crop_box(self) -> Self:
        """Require a finite, non-empty source-pixel XYXY crop box."""
        x0, y0, x1, y1 = self.crop_box_xyxy
        if not all(math.isfinite(value) for value in self.crop_box_xyxy):
            raise ValueError("crop box coordinates must be finite")
        if x0 < 0.0 or y0 < 0.0 or x1 <= x0 or y1 <= y0:
            raise ValueError("crop box must be non-negative with x1>x0 and y1>y0")
        if x1 > self.image_width_px or y1 > self.image_height_px:
            raise ValueError("crop box must stay within source image bounds")
        return self


class SourceEvaluationRecord(BaseModel):
    """Finite named scalar metrics for one evaluated sample and source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    metrics: dict[str, float]

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, value: object) -> dict[str, float]:
        """Reject unnamed, nonnumeric, or non-finite metrics and sort keys."""
        if not isinstance(value, Mapping) or not value:
            raise ValueError("metrics must be a non-empty mapping")
        normalized: dict[str, float] = {}
        for name, raw_metric in value.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ValueError("metric names must be non-empty and trimmed")
            if type(raw_metric) not in (int, float):
                raise ValueError(f"metric {name!r} must be numeric")
            metric = float(raw_metric)
            if not math.isfinite(metric):
                raise ValueError(f"metric {name!r} must be finite")
            normalized[name] = metric
        return dict(sorted(normalized.items()))


class SourceEvaluationSummary(BaseModel):
    """Arithmetic metric means for one source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=1)
    metric_means: dict[str, float]


class SourceEvaluationReport(BaseModel):
    """Deterministic source-stratified evaluation summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_samples: int = Field(ge=0)
    metric_names: tuple[str, ...]
    by_source: dict[str, SourceEvaluationSummary]


def summarize_source_evaluation(
    records: Iterable[SourceEvaluationRecord],
) -> SourceEvaluationReport:
    """Aggregate a consistent metric set into deterministic per-source means."""
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                record.source,
                record.sample_id,
                tuple(record.metrics.items()),
            ),
        )
    )
    metric_names = tuple(ordered[0].metrics) if ordered else ()
    for record in ordered:
        if tuple(record.metrics) != metric_names:
            raise ValueError(
                f"metric names differ for sample {record.sample_id}: "
                f"expected {metric_names}, got {tuple(record.metrics)}"
            )
    by_source: dict[str, SourceEvaluationSummary] = {}
    for source in sorted({record.source for record in ordered}):
        source_records = tuple(record for record in ordered if record.source == source)
        means = {
            name: math.fsum(record.metrics[name] for record in source_records)
            / len(source_records)
            for name in metric_names
        }
        by_source[source] = SourceEvaluationSummary(
            sample_count=len(source_records), metric_means=means
        )
    return SourceEvaluationReport(
        total_samples=len(ordered), metric_names=metric_names, by_source=by_source
    )


def export_source_evaluation(
    records: Iterable[SourceEvaluationRecord], output: Path
) -> SourceEvaluationReport:
    """Write deterministic source-stratified evaluation JSON and CSV reports."""
    report = summarize_source_evaluation(records)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "source_evaluation.json", report.model_dump(mode="json"))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("source", "sample_count", "metric", "mean"))
    for source, summary in report.by_source.items():
        for name in report.metric_names:
            writer.writerow(
                (source, summary.sample_count, name, summary.metric_means[name])
            )
    atomic_write_text(output / "source_evaluation.csv", buffer.getvalue())
    return report


def summarize_errors(records: Iterable[ErrorRecord]) -> ErrorTaxonomyReport:
    """Count failures overall and by source, including explicit zero cells."""
    materialized = tuple(records)
    total: Counter[str] = Counter(record.category for record in materialized)
    sources = sorted({record.source for record in materialized})
    by_source: dict[str, dict[str, int]] = {}
    for source in sources:
        counts: Counter[str] = Counter(
            record.category for record in materialized if record.source == source
        )
        by_source[source] = {
            category: counts[category] for category in ERROR_CATEGORIES
        }
    return ErrorTaxonomyReport(
        total_records=len(materialized),
        total_counts={category: total[category] for category in ERROR_CATEGORIES},
        by_source=by_source,
    )


def export_error_taxonomy_report(
    records: Iterable[ErrorRecord], output: Path
) -> ErrorTaxonomyReport:
    """Write deterministic JSON and CSV source-stratified taxonomy reports."""
    report = summarize_errors(records)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "error_taxonomy.json", report.model_dump(mode="json"))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("source", "category", "count"))
    for source, counts in report.by_source.items():
        for category in ERROR_CATEGORIES:
            writer.writerow((source, category, counts[category]))
    for category in ERROR_CATEGORIES:
        writer.writerow(("__all__", category, report.total_counts[category]))
    atomic_write_text(output / "error_taxonomy.csv", buffer.getvalue())
    return report


def export_hard_negative_manifest(
    crops: Iterable[FalsePositiveCrop], destination: Path
) -> Path:
    """Export deterministic crop proposals without reading or editing annotations."""
    ordered = sorted(crops, key=_crop_sort_key)
    payload: Mapping[str, object] = {
        "schema_version": "1.0.0",
        "purpose": "human_annotation_only",
        "crops": [crop.model_dump(mode="json") for crop in ordered],
    }
    atomic_write_text(
        destination,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return destination


def _crop_sort_key(crop: FalsePositiveCrop) -> tuple[object, ...]:
    return (
        crop.sample_id,
        crop.prediction_id,
        crop.category,
        crop.crop_box_xyxy,
        crop.image_width_px,
        crop.image_height_px,
        crop.source,
        crop.image_path,
        tuple(sorted(crop.provenance.items())),
    )
