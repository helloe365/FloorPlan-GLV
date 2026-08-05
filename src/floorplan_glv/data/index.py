"""Atomic CubiCasa conversion and deterministic normalized dataset indexing."""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from floorplan_glv.config.models import ConfigurationError
from floorplan_glv.data.annotation_schema import AnnotationError
from floorplan_glv.data.cubicasa_converter import (
    ConversionConfig,
    ConversionRejection,
    convert_cubicasa_sample,
)

NonEmptyString = Annotated[str, Field(min_length=1)]


class CubiCasaPreparationConfig(BaseModel):
    """Validated paths and conversion settings for dataset preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_root: Path
    output_root: Path
    svg_filename: NonEmptyString = "model.svg"
    image_filename: NonEmptyString = "F1_scaled.png"
    split_seed: Annotated[int, Field(ge=0)] = 1337
    train_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.80
    validation_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10
    test_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)

    @model_validator(mode="after")
    def validate_split_fractions(self) -> Self:
        """Require deterministic preparation splits to cover the dataset."""
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("dataset split fractions must sum to 1.0")
        return self


class IndexRecord(BaseModel):
    """One deterministic normalized dataset index record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    image: str
    annotation: str
    source_sample: str


class PreparationReport(BaseModel):
    """Aggregate conversion counts and observed SVG group vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discovered_samples: int
    converted_samples: int
    rejected_samples: int
    rejected_objects: int
    encountered_group_ids: tuple[str, ...]
    encountered_class_prefixes: tuple[str, ...]
    unknown_group_counts: dict[str, int]


def prepare_cubicasa_dataset(config_path: Path) -> PreparationReport:
    """Convert all CubiCasa samples selected by a YAML configuration file."""
    config = _load_config(config_path)
    if not config.source_root.is_dir():
        raise AnnotationError(
            f"CubiCasa source root does not exist: {config.source_root}"
        )

    sample_dirs = _discover_samples(
        config.source_root,
        svg_filename=config.svg_filename,
    )
    if not sample_dirs:
        raise AnnotationError(
            f"no {config.svg_filename} samples found under {config.source_root}"
        )

    config.output_root.mkdir(parents=True, exist_ok=True)
    records: list[IndexRecord] = []
    rejection_records: list[dict[str, str]] = []
    group_ids: set[str] = set()
    class_prefixes: set[str] = set()
    unknown_groups: Counter[str] = Counter()
    used_sample_ids: set[str] = set()
    rejected_sample_count = 0

    for sample_dir in sample_dirs:
        sample_id = _sample_id(sample_dir, config.source_root)
        if sample_id in used_sample_ids:
            raise AnnotationError(f"duplicate normalized sample ID: {sample_id}")
        used_sample_ids.add(sample_id)
        try:
            result = convert_cubicasa_sample(
                sample_dir,
                config=config.conversion,
                sample_id=sample_id,
                svg_filename=config.svg_filename,
                image_filename=config.image_filename,
            )
            sample_output = config.output_root / "samples" / sample_id
            sample_output.mkdir(parents=True, exist_ok=True)
            _write_normalized_image(
                sample_dir / config.image_filename,
                sample_output / "image.png",
            )
            _atomic_write_json(
                sample_output / "annotation.json",
                result.annotation.model_dump(mode="json"),
            )
            relative_source = sample_dir.relative_to(config.source_root).as_posix()
            records.append(
                IndexRecord(
                    sample_id=sample_id,
                    image=f"samples/{sample_id}/image.png",
                    annotation=f"samples/{sample_id}/annotation.json",
                    source_sample=relative_source or ".",
                )
            )
            group_ids.update(result.encountered_group_ids)
            class_prefixes.update(result.encountered_class_prefixes)
            unknown_groups.update(result.unknown_group_counts)
            rejection_records.extend(
                _object_rejection(sample_id, rejection)
                for rejection in result.rejections
            )
        except AnnotationError as exc:
            rejected_sample_count += 1
            rejection_records.append(
                {
                    "sample_id": sample_id,
                    "source_id": "",
                    "category": "sample",
                    "reason": str(exc),
                }
            )

    records.sort(key=lambda record: record.sample_id)
    rejection_records.sort(
        key=lambda record: (
            record["sample_id"],
            record["source_id"],
            record["category"],
            record["reason"],
        )
    )
    report = PreparationReport(
        discovered_samples=len(sample_dirs),
        converted_samples=len(records),
        rejected_samples=rejected_sample_count,
        rejected_objects=sum(
            record["category"] != "sample" for record in rejection_records
        ),
        encountered_group_ids=tuple(sorted(group_ids)),
        encountered_class_prefixes=tuple(sorted(class_prefixes)),
        unknown_group_counts=dict(sorted(unknown_groups.items())),
    )
    _atomic_write_text(
        config.output_root / "index.jsonl",
        _json_lines(record.model_dump(mode="json") for record in records),
    )
    _write_split_indexes(
        records,
        config.output_root,
        seed=config.split_seed,
        fractions=(
            config.train_fraction,
            config.validation_fraction,
            config.test_fraction,
        ),
    )
    _atomic_write_json(
        config.output_root / "conversion_report.json",
        report.model_dump(mode="json"),
    )
    _atomic_write_text(
        config.output_root / "rejected_samples.jsonl",
        _json_lines(rejection_records),
    )
    return report


def _write_split_indexes(
    records: list[IndexRecord],
    output_root: Path,
    *,
    seed: int,
    fractions: tuple[float, float, float],
) -> None:
    """Persist deterministic train/validation/test manifests and indexes."""
    names = ("train", "val", "test")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    raw_counts = [len(records) * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw_counts]
    remaining = len(records) - sum(counts)
    for index in sorted(
        range(len(names)),
        key=lambda item: (-(raw_counts[item] - counts[item]), item),
    )[:remaining]:
        counts[index] += 1
    split_root = output_root / "splits"
    start = 0
    for name, count in zip(names, counts, strict=True):
        split_records = sorted(
            shuffled[start : start + count], key=lambda record: record.sample_id
        )
        start += count
        _atomic_write_text(
            split_root / f"{name}.txt",
            "".join(f"{record.sample_id}\n" for record in split_records),
        )
        _atomic_write_text(
            output_root / f"{name}_index.jsonl",
            _json_lines(record.model_dump(mode="json") for record in split_records),
        )


def _load_config(path: Path) -> CubiCasaPreparationConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"failed to load configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration must be a mapping: {path}")
    try:
        return CubiCasaPreparationConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration {path}: {exc}") from exc


def _discover_samples(source_root: Path, *, svg_filename: str) -> tuple[Path, ...]:
    if (source_root / svg_filename).is_file():
        return (source_root,)
    return tuple(
        sorted(
            (path.parent for path in source_root.rglob(svg_filename)),
            key=lambda path: path.relative_to(source_root).as_posix(),
        )
    )


def _sample_id(sample_dir: Path, source_root: Path) -> str:
    relative = sample_dir.relative_to(source_root).as_posix()
    raw = source_root.name if relative == "." else relative.replace("/", "__")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not normalized:
        raise AnnotationError(f"cannot derive sample ID from {sample_dir}")
    return normalized


def _object_rejection(
    sample_id: str,
    rejection: ConversionRejection,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "source_id": rejection.source_id,
        "category": rejection.category,
        "reason": rejection.reason,
    }


def _write_normalized_image(source: Path, destination: Path) -> None:
    try:
        with Image.open(source) as image:
            normalized = image.convert("RGB")
            output = BytesIO()
            normalized.save(output, format="PNG")
    except OSError as exc:
        raise AnnotationError(f"failed to normalize image {source}: {exc}") from exc
    _atomic_write_bytes(destination, output.getvalue())


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _json_lines(values: Any) -> str:
    return "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for value in values
    )


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
