"""Read-only deterministic audits for normalized floor-plan datasets."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from floorplan_glv.data.annotation_schema import AnnotationError, TrainingAnnotation
from floorplan_glv.data.audit_outputs import (
    _ValidSample,
    build_overlay_selection,
    write_audit_outputs,
)
from floorplan_glv.data.audit_validation import (
    UnresolvedHostSummary,
    coordinate_status,
    frequency,
    metadata_size_matches,
    unresolved_host_accounting,
)
from floorplan_glv.data.index import IndexRecord

MAX_OVERLAY_SAMPLES = 100
_PERCEPTUAL_HASH_SIZE = 8
_CROSS_SPLIT_NAMES = frozenset(
    ("train", "val", "test", "val_public", "val_real", "test_real")
)


class NumericSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    count: int = Field(ge=0)
    minimum: float | None
    maximum: float | None
    mean: float | None


class CountSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total: int = Field(ge=0)
    minimum_per_sample: int | None
    maximum_per_sample: int | None
    mean_per_sample: float | None


class AuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    file: str
    category: str
    reason: str


class HardGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: int | float
    requirement: str
    passed: bool


class DatasetAuditReport(BaseModel):
    """Complete deterministic normalized-dataset audit boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    indexed_samples: int = Field(ge=0)
    valid_annotation_samples: int = Field(ge=0)
    sample_counts: dict[str, int]
    split_counts: dict[str, int]
    image_size_counts: dict[str, int]
    wall_count: CountSummary
    opening_count: CountSummary
    wall_count_frequency: dict[str, int]
    opening_count_frequency: dict[str, int]
    class_counts: dict[str, int]
    wall_thickness_px: NumericSummary
    opening_length_px: NumericSummary
    wall_thickness_frequency_px: dict[str, int]
    opening_length_frequency_px: dict[str, int]
    unresolved_host_rate: float = Field(ge=0.0, le=1.0)
    unresolved_host_rate_excluding_waived: float = Field(ge=0.0, le=1.0)
    unresolved_host_by_source: dict[str, UnresolvedHostSummary]
    waived_unresolved_host_sources: tuple[str, ...]
    invalid_json_samples: int = Field(ge=0)
    invalid_annotation_samples: int = Field(ge=0)
    invalid_coordinate_samples: int = Field(ge=0)
    nonfinite_coordinate_samples: int = Field(ge=0)
    out_of_bounds_coordinate_samples: int = Field(ge=0)
    metadata_size_mismatch_samples: int = Field(ge=0)
    door_window_type_missing: int = Field(ge=0)
    rejection_reasons: dict[str, int]
    duplicate_candidates: tuple[tuple[str, str], ...]
    overlay_selection: dict[str, str]
    hard_gates: dict[str, HardGate]
    issues: tuple[AuditIssue, ...]


def inspect_dataset(
    index_path: Path,
    output: Path,
    *,
    waived_unresolved_host_sources: tuple[str, ...] = (),
) -> DatasetAuditReport:
    """Audit normalized data read-only and write JSON, CSV, PNG, and overlays."""
    records = _read_index(index_path)
    root = index_path.parent
    split_by_sample = _read_splits(root)
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    issues: list[AuditIssue] = []
    valid_samples: list[_ValidSample] = []
    image_size_counts: Counter[str] = Counter()
    wall_counts: list[int] = []
    opening_counts: list[int] = []
    thicknesses: list[float] = []
    opening_lengths: list[float] = []
    unresolved_by_source: defaultdict[str, list[bool]] = defaultdict(list)
    hashes: dict[str, str] = {}
    invalid_json = invalid_annotations = invalid_coordinates = nonfinite = 0
    out_of_bounds = 0
    metadata_mismatches = 0
    missing_types = 0

    for record in records:
        image_path = _resolve_source_file(root, record.image, record.sample_id, "image")
        annotation_path = _resolve_source_file(
            root, record.annotation, record.sample_id, "annotation"
        )
        image = _read_image(image_path, record.sample_id)
        image_size_counts[f"{image.width}x{image.height}"] += 1
        hashes[record.sample_id] = _perceptual_hash(image)
        split_counts[split_by_sample.get(record.sample_id, "unspecified")] += 1
        try:
            raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid_json += 1
            source_counts[_fallback_source(record)] += 1
            _add_issue(
                issues, record.sample_id, annotation_path, "invalid_json", exc.msg
            )
            continue
        except OSError as exc:
            raise AnnotationError(
                f"sample {record.sample_id} annotation {annotation_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            invalid_json += 1
            source_counts[_fallback_source(record)] += 1
            _add_issue(
                issues,
                record.sample_id,
                annotation_path,
                "invalid_annotation",
                "JSON root is not an object",
            )
            continue

        source = _raw_source(raw, record)
        source_counts[source] += 1
        missing_types += _missing_opening_types(raw)
        if not metadata_size_matches(raw, image.width, image.height):
            metadata_mismatches += 1
            _add_issue(
                issues,
                record.sample_id,
                annotation_path,
                "image_metadata_size_mismatch",
            )
        invalid, has_nonfinite, has_out_of_bounds = coordinate_status(
            raw, image.width, image.height
        )
        if invalid:
            invalid_coordinates += 1
            _add_issue(issues, record.sample_id, annotation_path, "invalid_coordinates")
        if has_nonfinite:
            nonfinite += 1
            _add_issue(
                issues, record.sample_id, annotation_path, "non_finite_coordinates"
            )
        if has_out_of_bounds:
            out_of_bounds += 1
            _add_issue(
                issues, record.sample_id, annotation_path, "out_of_bounds_coordinates"
            )
        if invalid:
            continue
        try:
            annotation = TrainingAnnotation.model_validate(raw)
        except ValidationError as exc:
            invalid_annotations += 1
            if not (has_nonfinite or has_out_of_bounds):
                _add_issue(
                    issues,
                    record.sample_id,
                    annotation_path,
                    "invalid_annotation",
                    str(exc),
                )
            continue
        if annotation.sample_id != record.sample_id:
            invalid_annotations += 1
            _add_issue(
                issues,
                record.sample_id,
                annotation_path,
                "invalid_annotation",
                f"annotation sample_id is {annotation.sample_id!r}",
            )
            continue
        wall_counts.append(len(annotation.walls))
        opening_counts.append(len(annotation.openings))
        thicknesses.extend(wall.thickness_px for wall in annotation.walls)
        for opening in annotation.openings:
            class_counts[opening.type] += 1
            unresolved_by_source[source].append(opening.host_wall_source_id is None)
            if opening.segment is not None:
                opening_lengths.append(_segment_length(opening.segment))
        valid_samples.append(_ValidSample(record.sample_id, image_path, annotation))

    duplicates = _duplicate_pairs(hashes)
    cross_split_duplicates = sum(
        split_by_sample.get(left) in _CROSS_SPLIT_NAMES
        and split_by_sample.get(right) in _CROSS_SPLIT_NAMES
        and split_by_sample.get(left) != split_by_sample.get(right)
        for left, right in duplicates
    )
    host_accounting = unresolved_host_accounting(
        unresolved_by_source,
        tuple(source_counts),
        waived_unresolved_host_sources,
    )
    rejection_reasons = _read_rejection_reasons(root, issues)
    selected_ids = tuple(
        item.sample_id
        for item in sorted(valid_samples, key=_overlay_sort_key)[:MAX_OVERLAY_SAMPLES]
    )
    overlay_selection = build_overlay_selection(selected_ids)
    report = DatasetAuditReport(
        indexed_samples=len(records),
        valid_annotation_samples=len(valid_samples),
        sample_counts=dict(sorted(source_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        image_size_counts=dict(sorted(image_size_counts.items())),
        wall_count=_count_summary(wall_counts),
        opening_count=_count_summary(opening_counts),
        wall_count_frequency=frequency(wall_counts),
        opening_count_frequency=frequency(opening_counts),
        class_counts={name: class_counts[name] for name in ("door", "window")},
        wall_thickness_px=_numeric_summary(thicknesses),
        opening_length_px=_numeric_summary(opening_lengths),
        wall_thickness_frequency_px=frequency(thicknesses),
        opening_length_frequency_px=frequency(opening_lengths),
        unresolved_host_rate=host_accounting.raw_rate,
        unresolved_host_rate_excluding_waived=host_accounting.excluding_waived_rate,
        unresolved_host_by_source=host_accounting.by_source,
        waived_unresolved_host_sources=host_accounting.waived_sources,
        invalid_json_samples=invalid_json,
        invalid_annotation_samples=invalid_annotations,
        invalid_coordinate_samples=invalid_coordinates,
        nonfinite_coordinate_samples=nonfinite,
        out_of_bounds_coordinate_samples=out_of_bounds,
        metadata_size_mismatch_samples=metadata_mismatches,
        door_window_type_missing=missing_types,
        rejection_reasons=rejection_reasons,
        duplicate_candidates=duplicates,
        overlay_selection=overlay_selection,
        hard_gates=_hard_gates(
            invalid_json,
            invalid_annotations,
            invalid_coordinates,
            out_of_bounds,
            nonfinite,
            cross_split_duplicates,
            host_accounting.excluding_waived_rate,
            missing_types,
            host_accounting.waived_sources,
        ),
        issues=tuple(sorted(issues, key=_issue_key)),
    )
    write_audit_outputs(report, valid_samples, output)
    return report


def _read_index(path: Path) -> tuple[IndexRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnnotationError(f"index {path}: {exc}") from exc
    records: list[IndexRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(IndexRecord.model_validate_json(line))
        except ValidationError as exc:
            raise AnnotationError(f"index {path} line {line_number}: {exc}") from exc
    identifiers = [record.sample_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise AnnotationError(f"index {path}: duplicate sample_id values")
    return tuple(sorted(records, key=lambda record: record.sample_id))


def _read_splits(root: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    split_root = root / "splits"
    if not split_root.is_dir():
        return assignments
    for path in sorted(split_root.glob("*.txt"), key=lambda item: item.name):
        try:
            sample_ids = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AnnotationError(f"split file {path}: {exc}") from exc
        for sample_id in (item.strip() for item in sample_ids if item.strip()):
            previous = assignments.get(sample_id)
            if previous is not None and previous != path.stem:
                raise AnnotationError(
                    f"split file {path}: sample {sample_id} also belongs to {previous}"
                )
            assignments[sample_id] = path.stem
    return assignments


def _resolve_source_file(root: Path, value: str, sample_id: str, kind: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AnnotationError(
            f"sample {sample_id} {kind} path escapes dataset root: {value}"
        ) from exc
    if not path.is_file():
        raise AnnotationError(f"sample {sample_id} {kind} file does not exist: {path}")
    return path


def _read_image(path: Path, sample_id: str) -> Image.Image:
    try:
        with Image.open(path) as encoded:
            return encoded.convert("RGB")
    except OSError as exc:
        raise AnnotationError(f"sample {sample_id} image {path}: {exc}") from exc


def _perceptual_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize(
        (_PERCEPTUAL_HASH_SIZE, _PERCEPTUAL_HASH_SIZE), Image.Resampling.LANCZOS
    )
    values = gray.tobytes()
    average = sum(values) / len(values)
    bits = "".join("1" if value >= average else "0" for value in values)
    return f"{int(bits, 2):016x}"


def _missing_opening_types(raw: dict[str, Any]) -> int:
    openings = raw.get("openings")
    if not isinstance(openings, list):
        return 0
    return sum(
        not isinstance(opening, dict) or opening.get("type") not in ("door", "window")
        for opening in openings
    )


def _raw_source(raw: dict[str, Any], record: IndexRecord) -> str:
    image = raw.get("image")
    if isinstance(image, dict) and isinstance(image.get("source"), str):
        return str(image["source"])
    return _fallback_source(record)


def _fallback_source(record: IndexRecord) -> str:
    return record.source_sample.split("/", maxsplit=1)[0] or "unknown"


def _duplicate_pairs(hashes: dict[str, str]) -> tuple[tuple[str, str], ...]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for sample_id, value in hashes.items():
        grouped[value].append(sample_id)
    return tuple(
        sorted(
            pair
            for sample_ids in grouped.values()
            for pair in itertools.combinations(sorted(sample_ids), 2)
        )
    )


def _read_rejection_reasons(root: Path, issues: list[AuditIssue]) -> dict[str, int]:
    path = root / "rejected_samples.jsonl"
    if not path.is_file():
        return {}
    counts: Counter[str] = Counter()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnnotationError(f"rejection file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            _add_issue(
                issues, f"rejection_line_{line_number}", path, "invalid_rejection_json"
            )
            continue
        reason = raw.get("reason") if isinstance(raw, dict) else None
        counts[str(reason) if reason else "missing reason"] += 1
    return dict(sorted(counts.items()))


def _numeric_summary(values: list[float]) -> NumericSummary:
    return NumericSummary(
        count=len(values),
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
        mean=sum(values) / len(values) if values else None,
    )


def _count_summary(values: list[int]) -> CountSummary:
    return CountSummary(
        total=sum(values),
        minimum_per_sample=min(values) if values else None,
        maximum_per_sample=max(values) if values else None,
        mean_per_sample=sum(values) / len(values) if values else None,
    )


def _hard_gates(
    invalid_json: int,
    invalid_annotations: int,
    invalid_coordinates: int,
    out_of_bounds: int,
    nonfinite: int,
    cross_split_duplicates: int,
    unresolved_rate: float,
    missing_types: int,
    waived_sources: tuple[str, ...],
) -> dict[str, HardGate]:
    return {
        "door_window_type_missing": _zero_gate(missing_types),
        "invalid_annotation_samples": _zero_gate(invalid_annotations),
        "invalid_json_samples": _zero_gate(invalid_json),
        "invalid_coordinates": _zero_gate(invalid_coordinates),
        "non_finite_coordinates": _zero_gate(nonfinite),
        "out_of_bounds_coordinates": _zero_gate(out_of_bounds),
        "train_val_test_duplicate_samples": _zero_gate(cross_split_duplicates),
        "unresolved_host_rate": HardGate(
            value=unresolved_rate,
            requirement=(
                "< 0.05 excluding waived sources: "
                + (", ".join(waived_sources) if waived_sources else "none")
            ),
            passed=unresolved_rate < 0.05,
        ),
    }


def _segment_length(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return math.hypot(segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])


def _overlay_sort_key(sample: _ValidSample) -> tuple[str, str]:
    return hashlib.sha256(
        sample.sample_id.encode("utf-8")
    ).hexdigest(), sample.sample_id


def _add_issue(
    issues: list[AuditIssue],
    sample_id: str,
    path: Path,
    category: str,
    reason: str = "detected during annotation audit",
) -> None:
    issues.append(
        AuditIssue(
            sample_id=sample_id, file=str(path), category=category, reason=reason
        )
    )


def _zero_gate(value: int) -> HardGate:
    return HardGate(value=value, requirement="= 0", passed=value == 0)


def _issue_key(issue: AuditIssue) -> tuple[str, str, str, str]:
    return issue.sample_id, issue.category, issue.file, issue.reason
