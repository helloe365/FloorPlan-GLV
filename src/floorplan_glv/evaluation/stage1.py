"""CPU-only primitives for Stage 1 dense-mask acceptance evaluation."""
# ruff: noqa: E501,B008

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from floorplan_glv.config.load import load_config
from floorplan_glv.config.models import ModelConfig
from floorplan_glv.data.collate import ModelBatch, collate_source_images
from floorplan_glv.data.dataset import FloorPlanDataset, SourceImageItem
from floorplan_glv.metrics.dense import BinaryMetrics, binary_metrics
from floorplan_glv.models.model import FloorPlanGLV
from floorplan_glv.training.checkpoint import _read_checkpoint, file_sha256
from floorplan_glv.training.runner import assert_no_production_test_split
from floorplan_glv.visualization.stage1 import (
    Stage1VisualizationInput,
    write_stage1_visualizations,
)

WeightSelection = Literal["raw", "ema", "both"]
HeadName = Literal["wall", "opening"]
_WEIGHTS = frozenset(("raw", "ema", "both"))
_RECORD_WEIGHTS = frozenset(("raw", "ema"))
_HEADS = frozenset(("wall", "opening"))


class Stage1EvaluationError(RuntimeError):
    """A reproducible Stage 1 evaluation could not be completed."""


@dataclass(frozen=True, slots=True)
class Stage1EvaluationOptions:
    weights: WeightSelection = "both"
    wall_threshold: float = 0.45
    opening_threshold: float = 0.50
    threshold_min: float = 0.30
    threshold_max: float = 0.70
    threshold_step: float = 0.05
    visual_samples: int = 12

    def __post_init__(self) -> None:
        if self.weights not in _WEIGHTS:
            raise Stage1EvaluationError(
                "weights must be one of 'raw', 'ema', or 'both'"
            )
        for name, value in (
            ("wall_threshold", self.wall_threshold),
            ("opening_threshold", self.opening_threshold),
            ("threshold_min", self.threshold_min),
            ("threshold_max", self.threshold_max),
        ):
            _validate_probability(value, name)
        if self.threshold_min > self.threshold_max:
            raise Stage1EvaluationError("threshold_min cannot exceed threshold_max")
        if (
            isinstance(self.threshold_step, bool)
            or not isinstance(self.threshold_step, (int, float))
            or not math.isfinite(float(self.threshold_step))
            or self.threshold_step <= 0.0
        ):
            raise Stage1EvaluationError("threshold_step must be finite and positive")
        if (
            isinstance(self.visual_samples, bool)
            or not isinstance(self.visual_samples, int)
            or self.visual_samples < 0
        ):
            raise Stage1EvaluationError("visual_samples must be a non-negative integer")
        # Validate that the configured sweep yields at least one threshold.
        build_thresholds(
            self.threshold_min,
            self.threshold_max,
            self.threshold_step,
        )

    @property
    def thresholds(self) -> tuple[float, ...]:
        return build_thresholds(
            self.threshold_min, self.threshold_max, self.threshold_step
        )


def _validate_probability(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise Stage1EvaluationError(f"{name} must be finite and between 0 and 1")


def build_thresholds(minimum: float, maximum: float, step: float) -> tuple[float, ...]:
    """Build a decimal-stable inclusive threshold sweep."""
    for name, value in (("threshold_min", minimum), ("threshold_max", maximum)):
        _validate_probability(value, name)
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(float(step))
        or step <= 0.0
    ):
        raise Stage1EvaluationError("threshold_step must be finite and positive")
    if minimum > maximum:
        raise Stage1EvaluationError("threshold_min cannot exceed threshold_max")

    try:
        current = Decimal(str(minimum))
        upper = Decimal(str(maximum))
        increment = Decimal(str(step))
    except (InvalidOperation, ValueError) as exc:
        raise Stage1EvaluationError("threshold sweep values must be numeric") from exc

    values: list[float] = []
    # A decimal loop avoids binary floating-point drift at endpoints. The
    # iteration guard protects callers from accidentally requesting an
    # impractically dense sweep.
    iterations = 0
    while current <= upper:
        values.append(float(current))
        current += increment
        iterations += 1
        if iterations > 1_000_000:
            raise Stage1EvaluationError("threshold sweep is too dense")
    if not values:
        raise Stage1EvaluationError("threshold sweep produced no thresholds")
    return tuple(values)


def masked_binary_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> BinaryMetrics:
    """Compute binary metrics using only pixels marked valid."""
    if not all(isinstance(mask, torch.Tensor) for mask in (predicted, target, valid)):
        raise Stage1EvaluationError("masked metrics require torch tensors")
    if predicted.shape != target.shape or predicted.shape != valid.shape:
        raise Stage1EvaluationError(
            "predicted, target, and valid masks must share shape"
        )
    if any(mask.dtype is not torch.bool for mask in (predicted, target, valid)):
        raise Stage1EvaluationError(
            "predicted, target, and valid masks must be boolean"
        )
    valid_count = int(torch.count_nonzero(valid).item())
    if valid_count == 0:
        raise Stage1EvaluationError("masked metrics require at least one valid pixel")
    return binary_metrics(predicted[valid], target[valid])


@dataclass(frozen=True, slots=True)
class PatchMetricRecord:
    weight: Literal["raw", "ema"]
    sample_id: str
    source_index: int
    patch_index: int
    head: HeadName
    threshold: float
    valid_pixel_count: int
    target_positive_pixels: int
    predicted_positive_pixels: int
    metrics: BinaryMetrics


@dataclass(frozen=True, slots=True)
class PatchMetricSummary:
    all_patch_count: int
    target_positive_patch_count: int
    positive_f1_p10: float | None
    positive_f1_median: float | None
    positive_f1_p90: float | None
    positive_iou_p10: float | None
    positive_iou_median: float | None
    positive_iou_p90: float | None
    target_empty_patch_count: int
    target_empty_false_positive_patch_count: int
    target_empty_false_positive_patch_rate: float | None


def summarize_patch_records(
    records: Sequence[PatchMetricRecord],
) -> PatchMetricSummary:
    """Summarize patch records, including positive-only quality percentiles."""
    records_tuple = tuple(records)
    if not records_tuple:
        return PatchMetricSummary(
            all_patch_count=0,
            target_positive_patch_count=0,
            positive_f1_p10=None,
            positive_f1_median=None,
            positive_f1_p90=None,
            positive_iou_p10=None,
            positive_iou_median=None,
            positive_iou_p90=None,
            target_empty_patch_count=0,
            target_empty_false_positive_patch_count=0,
            target_empty_false_positive_patch_rate=None,
        )
    if not all(isinstance(record, PatchMetricRecord) for record in records_tuple):
        raise Stage1EvaluationError("summary records must be PatchMetricRecord values")
    weights = {record.weight for record in records_tuple}
    heads = {record.head for record in records_tuple}
    if not weights <= _RECORD_WEIGHTS or len(weights) != 1:
        raise Stage1EvaluationError("summary records must use one weight")
    if not heads <= _HEADS or len(heads) != 1:
        raise Stage1EvaluationError("summary records must use one head")

    positive = [record for record in records_tuple if record.target_positive_pixels > 0]
    empty = [record for record in records_tuple if record.target_positive_pixels == 0]
    positive_f1 = _percentiles([record.metrics.f1 for record in positive])
    positive_iou = _percentiles([record.metrics.iou for record in positive])
    false_positive_empty = [
        record for record in empty if record.metrics.false_positive > 0
    ]
    empty_count = len(empty)
    return PatchMetricSummary(
        all_patch_count=len(records_tuple),
        target_positive_patch_count=len(positive),
        positive_f1_p10=None if positive_f1 is None else positive_f1[0],
        positive_f1_median=None if positive_f1 is None else positive_f1[1],
        positive_f1_p90=None if positive_f1 is None else positive_f1[2],
        positive_iou_p10=None if positive_iou is None else positive_iou[0],
        positive_iou_median=None if positive_iou is None else positive_iou[1],
        positive_iou_p90=None if positive_iou is None else positive_iou[2],
        target_empty_patch_count=empty_count,
        target_empty_false_positive_patch_count=len(false_positive_empty),
        target_empty_false_positive_patch_rate=(
            None if empty_count == 0 else len(false_positive_empty) / empty_count
        ),
    )


def _percentiles(values: Sequence[float]) -> tuple[float, float, float] | None:
    if not values:
        return None
    result = np.percentile(
        np.asarray(values, dtype=np.float64),
        (10, 50, 90),
        method="linear",
    )
    return (float(result[0]), float(result[1]), float(result[2]))


@dataclass(frozen=True, slots=True)
class VisualSelection:
    sample_id: str
    source_index: int
    patch_index: int
    reason: Literal[
        "wall_worst",
        "wall_median",
        "opening_worst",
        "opening_median",
        "wall_fallback",
        "opening_fallback",
    ]


def _record_identity(record: PatchMetricRecord) -> tuple[object, ...]:
    return (
        record.weight,
        record.sample_id,
        record.source_index,
        record.patch_index,
        record.head,
        record.threshold,
    )


def _selection_identity(record: PatchMetricRecord) -> tuple[str, int]:
    return (record.sample_id, record.patch_index)


def _stable_key(record: PatchMetricRecord) -> tuple[object, ...]:
    return (
        record.sample_id,
        record.patch_index,
        record.head,
        record.source_index,
        record.threshold,
    )


def select_visual_samples(
    records: Sequence[PatchMetricRecord],
    *,
    reference_weight: Literal["raw", "ema"],
    requested_count: int,
) -> tuple[VisualSelection, ...]:
    """Select deterministic worst/median examples with fallback filling."""
    if reference_weight not in _RECORD_WEIGHTS:
        raise Stage1EvaluationError("reference_weight must be 'raw' or 'ema'")
    if (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count < 0
    ):
        raise Stage1EvaluationError("requested_count must be a non-negative integer")
    if requested_count == 0:
        return ()

    records_tuple = tuple(records)
    if not all(isinstance(record, PatchMetricRecord) for record in records_tuple):
        raise Stage1EvaluationError(
            "selection records must be PatchMetricRecord values"
        )
    for record in records_tuple:
        if record.weight not in _RECORD_WEIGHTS:
            raise Stage1EvaluationError("selection records must use raw or ema weight")
        if record.head not in _HEADS:
            raise Stage1EvaluationError(
                "selection records must use wall or opening head"
            )

    candidates = tuple(
        record for record in records_tuple if record.weight == reference_weight
    )
    identities = [_record_identity(record) for record in candidates]
    if len(set(identities)) != len(identities):
        raise Stage1EvaluationError("selection records contain duplicate identities")
    if not candidates:
        return ()

    selected: list[VisualSelection] = []
    selected_ids: set[tuple[str, int]] = set()

    def add_candidate(record: PatchMetricRecord, reason: str) -> None:
        identity = _selection_identity(record)
        if identity in selected_ids or len(selected) >= requested_count:
            return
        selected_ids.add(identity)
        selected.append(
            VisualSelection(
                sample_id=record.sample_id,
                source_index=record.source_index,
                patch_index=record.patch_index,
                reason=reason,  # type: ignore[arg-type]
            )
        )

    groups: list[tuple[str, list[PatchMetricRecord]]] = []
    for head in ("wall", "opening"):
        head_records = [
            record
            for record in candidates
            if record.head == head and record.target_positive_pixels > 0
        ]
        if not head_records:
            continue
        worst = sorted(
            head_records,
            key=lambda record: (record.metrics.iou, _stable_key(record)),
        )
        median_iou = float(
            np.median(np.asarray([record.metrics.iou for record in head_records]))
        )
        median = sorted(
            head_records,
            key=lambda record: (
                abs(record.metrics.iou - median_iou),
                _stable_key(record),
            ),
        )
        groups.extend(((f"{head}_worst", worst), (f"{head}_median", median)))

    # Walk every candidate group in fixed order, repeating rounds until the
    # requested count is reached. A record that appears in multiple groups is
    # assigned only the first reason encountered.
    cursors = [0] * len(groups)
    while len(selected) < requested_count:
        made_progress = False
        for group_index, (reason, group_records) in enumerate(groups):
            cursor = cursors[group_index]
            while (
                cursor < len(group_records)
                and _selection_identity(group_records[cursor]) in selected_ids
            ):
                cursor += 1
            cursors[group_index] = cursor
            if cursor >= len(group_records):
                continue
            add_candidate(group_records[cursor], reason)
            cursors[group_index] += 1
            made_progress = True
            if len(selected) >= requested_count:
                break
        if not made_progress:
            break

    if len(selected) < requested_count:
        fallback = sorted(
            candidates,
            key=lambda record: (
                0 if record.head == "wall" else 1,
                record.metrics.iou,
                _stable_key(record),
            ),
        )
        for record in fallback:
            if len(selected) >= requested_count:
                break
            if _selection_identity(record) in selected_ids:
                continue
            add_candidate(record, f"{record.head}_fallback")
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class WeightState:
    """One validated model weight state selected from a checkpoint."""

    name: Literal["raw", "ema"]
    state_dict: Mapping[str, torch.Tensor]


def select_checkpoint_states(
    checkpoint: Mapping[str, object],
    model: nn.Module,
    *,
    selection: WeightSelection,
) -> tuple[WeightState, ...]:
    """Validate and select raw and/or EMA tensors without loading them."""
    if selection not in _WEIGHTS:
        raise Stage1EvaluationError(f"invalid weight selection {selection!r}")
    requested = (
        ("raw",)
        if selection == "raw"
        else ("ema",)
        if selection == "ema"
        else ("raw", "ema")
    )
    expected = model.state_dict()
    selected: list[WeightState] = []
    for name in requested:
        if name == "raw":
            if "model_state" not in checkpoint:
                raise Stage1EvaluationError(
                    "raw checkpoint state is missing raw model_state"
                )
            candidate: object = checkpoint["model_state"]
        else:
            if "ema_state" not in checkpoint:
                raise Stage1EvaluationError("ema checkpoint state is missing ema_state")
            ema_state = checkpoint["ema_state"]
            if not isinstance(ema_state, Mapping):
                raise Stage1EvaluationError(
                    "ema checkpoint state has incompatible ema_state contract"
                )
            if "shadow" not in ema_state:
                raise Stage1EvaluationError(
                    "ema checkpoint state is missing ema_state.shadow"
                )
            candidate = ema_state["shadow"]
        if not isinstance(candidate, Mapping):
            raise Stage1EvaluationError(
                f"{name} checkpoint state has incompatible mapping contract"
            )
        candidate_keys = set(candidate)
        expected_keys = set(expected)
        missing = sorted(expected_keys - candidate_keys)
        extra = sorted(candidate_keys - expected_keys)
        if missing or extra:
            raise Stage1EvaluationError(
                f"{name} checkpoint state has incompatible keys; "
                f"missing={missing}, extra={extra}"
            )
        copied: dict[str, torch.Tensor] = {}
        for key, destination in expected.items():
            value = candidate[key]
            if not isinstance(value, torch.Tensor):
                raise Stage1EvaluationError(
                    f"{name} checkpoint state has incompatible key {key}: "
                    "value is not a tensor"
                )
            if value.shape != destination.shape:
                raise Stage1EvaluationError(
                    f"{name} checkpoint state has incompatible key {key}: "
                    f"shape {tuple(value.shape)} != {tuple(destination.shape)}"
                )
            if value.dtype != destination.dtype:
                raise Stage1EvaluationError(
                    f"{name} checkpoint state has incompatible key {key}: "
                    f"dtype {value.dtype} != {destination.dtype}"
                )
            copied[key] = value
        selected.append(WeightState(name=name, state_dict=copied))  # type: ignore[arg-type]
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class HeadEvaluationSummary:
    micro: BinaryMetrics
    patches: PatchMetricSummary


@dataclass(frozen=True, slots=True)
class SweepMetricRecord:
    weight: Literal["raw", "ema"]
    head: HeadName
    threshold: float
    metrics: BinaryMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "weight": self.weight,
            "head": self.head,
            "threshold": self.threshold,
            "metrics": _metrics_dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class Stage1EvaluationResult:
    summary: Mapping[str, object]
    patch_records: tuple[PatchMetricRecord, ...]
    sweep_records: tuple[SweepMetricRecord, ...]
    visual_selections: tuple[VisualSelection, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": _json_safe(self.summary),
            "patch_records": [_patch_dict(record) for record in self.patch_records],
            "sweep_records": [record.as_dict() for record in self.sweep_records],
            "visual_selections": [
                _selection_dict(item) for item in self.visual_selections
            ],
        }


BatchIterable = Iterable[tuple[Sequence[str], ModelBatch]]
VisualizationWriter = Callable[
    [Sequence[Stage1VisualizationInput], Path], tuple[Path, ...]
]
ModelFactory = Callable[[ModelConfig], nn.Module]


def evaluate_stage1(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    options: Stage1EvaluationOptions = Stage1EvaluationOptions(),
    *,
    visualization_writer: VisualizationWriter | None = None,
    model_factory: ModelFactory = FloorPlanGLV,
    batches: BatchIterable | None = None,
    generated_at: datetime | None = None,
) -> Stage1EvaluationResult:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    try:
        config = load_config(config_path)
        checkpoint = _read_checkpoint(checkpoint_path, map_location="cpu")
        model = model_factory(config.model)
        states = select_checkpoint_states(checkpoint, model, selection=options.weights)
    except Stage1EvaluationError:
        raise
    except Exception as exc:
        raise Stage1EvaluationError(f"evaluation inputs {config_path}: {exc}") from exc
    resolved_config = config_path.expanduser().resolve()
    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    checkpoint_sha256 = (
        file_sha256(resolved_checkpoint) if resolved_checkpoint.is_file() else None
    )
    configured_index = getattr(config.train, "validation_index", None)
    resolved_index = (
        None
        if configured_index is None
        else Path(configured_index).expanduser().resolve()
    )
    index_sha256 = (
        file_sha256(resolved_index)
        if resolved_index is not None and resolved_index.is_file()
        else None
    )
    if batches is None:
        validation_index = getattr(config.train, "validation_index", None)
        if validation_index is None:
            raise Stage1EvaluationError(f"validation index is required: {config_path}")
        validation_index = Path(validation_index).expanduser().resolve()
        try:
            assert_no_production_test_split(validation_index, purpose="validation")
            dataset = FloorPlanDataset(
                validation_index,
                patches_per_image=config.train.patches_per_image,
                seed=config.train.seed,
                model_config=config.model,
                data_config=config.data,
                augmentation=None,
            )
            dataset.set_epoch(0)
            loader: Iterable[tuple[Sequence[str], ModelBatch]] = DataLoader(
                dataset,
                batch_size=getattr(
                    config.train, "batch_size", config.train.source_images_per_gpu
                ),
                shuffle=False,
                num_workers=config.train.num_workers,
                collate_fn=_collate_items,
            )
        except Exception as exc:
            raise Stage1EvaluationError(
                f"validation data {validation_index}: {exc}"
            ) from exc
        materialized = tuple(loader)

    if batches is not None:
        materialized = tuple(batches)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if materialized is not None and not materialized:
        raise Stage1EvaluationError("evaluation batches are empty")
    records: list[PatchMetricRecord] = []
    sweep_parts: list[SweepMetricRecord] = []
    model.eval()
    for state in states:
        try:
            model.load_state_dict(state.state_dict, strict=True)
            model.eval()
            with torch.inference_mode():
                for sample_ids, batch in materialized:
                    moved_batch = _move_batch(batch, device)
                    output = model(moved_batch)
                    _collect_batch(
                        output,
                        moved_batch,
                        tuple(sample_ids),
                        state.name,
                        options,
                        records,
                        sweep_parts,
                    )
        except Stage1EvaluationError:
            raise
        except Exception as exc:
            raise Stage1EvaluationError(
                f"{state.name} evaluation failed: {exc}"
            ) from exc
    if not records:
        raise Stage1EvaluationError("evaluation batches are empty")
    by_weight_head: dict[str, dict[str, HeadEvaluationSummary]] = {}
    for weight in {record.weight for record in records}:
        by_weight_head[weight] = {}
        for head in ("wall", "opening"):
            head_records = tuple(
                record
                for record in records
                if record.weight == weight and record.head == head
            )
            if head_records:
                by_weight_head[weight][head] = HeadEvaluationSummary(
                    micro=_micro_metrics(head_records),
                    patches=summarize_patch_records(head_records),
                )
    reference: Literal["raw", "ema"] = (
        "raw" if any(record.weight == "raw" for record in records) else "ema"
    )
    visual = select_visual_samples(
        records, reference_weight=reference, requested_count=options.visual_samples
    )
    visualization_published = False
    if visual:
        writer = visualization_writer
        if writer is None:

            def _default_writer(
                items: Sequence[Stage1VisualizationInput], destination: Path
            ) -> tuple[Path, ...]:
                return write_stage1_visualizations(
                    items,
                    destination,
                    wall_threshold=options.wall_threshold,
                    opening_threshold=options.opening_threshold,
                )

            writer = _default_writer
        try:
            visual_items = _render_selected_visualizations(
                model, states, materialized, visual, device
            )
            writer(visual_items, Path(output_dir))
            visualization_published = True
        except Stage1EvaluationError:
            raise
        except Exception as exc:
            raise Stage1EvaluationError(
                f"visualization output {output_dir}: {exc}"
            ) from exc
    generated = generated_at or datetime.now(UTC)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)
    summary = {
        "generated_at": generated.astimezone(UTC).isoformat(),
        "config_path": str(resolved_config),
        "checkpoint_path": str(resolved_checkpoint),
        "validation_index_path": None
        if resolved_index is None
        else str(resolved_index),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_index_sha256": index_sha256,
        "options": options.__dict__
        if hasattr(options, "__dict__")
        else {
            "weights": options.weights,
            "wall_threshold": options.wall_threshold,
            "opening_threshold": options.opening_threshold,
            "threshold_min": options.threshold_min,
            "threshold_max": options.threshold_max,
            "threshold_step": options.threshold_step,
            "visual_samples": options.visual_samples,
        },
        "source_count": len({record.sample_id for record in records}),
        "patch_count": len(
            {(record.sample_id, record.patch_index) for record in records}
        ),
        "selection": options.weights,
        "visual_selections": [_selection_dict(item) for item in visual],
        "metrics": by_weight_head,
        "artifacts": {
            "summary": "summary.json",
            "patch_metrics": "patch_metrics.jsonl",
            "threshold_sweep": "threshold_sweep.json",
            **({"visualizations": "visualizations"} if visualization_published else {}),
        },
    }
    return Stage1EvaluationResult(
        summary=summary,
        patch_records=tuple(records),
        sweep_records=_aggregate_sweep(sweep_parts),
        visual_selections=tuple(visual),
    )


def _render_selected_visualizations(
    model: nn.Module,
    states: Sequence[WeightState],
    materialized: Sequence[tuple[Sequence[str], ModelBatch]],
    selections: Sequence[VisualSelection],
    device: torch.device,
) -> tuple[Stage1VisualizationInput, ...]:
    selected_views = _selected_batch_views(materialized, selections)
    if not selected_views:
        raise Stage1EvaluationError("selected visual patches were not found")
    patch_data: dict[
        tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    probabilities: dict[
        tuple[str, int],
        dict[Literal["raw", "ema"], dict[Literal["wall", "opening"], np.ndarray]],
    ] = {}
    for identities, batch in selected_views:
        valid = batch.targets.get("valid_pixels")
        wall_target = batch.targets.get("wall_mask")
        opening_target = batch.targets.get("opening_mask")
        if not isinstance(valid, torch.Tensor):
            raise Stage1EvaluationError("selected visual valid targets are incomplete")
        if not isinstance(wall_target, torch.Tensor):
            raise Stage1EvaluationError("selected visual wall targets are incomplete")
        if not isinstance(opening_target, torch.Tensor):
            raise Stage1EvaluationError(
                "selected visual opening targets are incomplete"
            )
            raise Stage1EvaluationError("selected visual targets are incomplete")
        for patch_index, identity in enumerate(identities):
            patch_data[identity] = (
                _denormalize_patch(batch.local_patches[patch_index]),
                valid[patch_index, 0].detach().cpu().numpy().astype(bool),
                wall_target[patch_index, 0].detach().cpu().numpy().astype(bool),
                opening_target[patch_index, 0].detach().cpu().numpy().astype(bool),
            )
            probabilities[identity] = {}
    for state in states:
        model.load_state_dict(state.state_dict, strict=True)
        model.eval()
        for identities, batch in selected_views:
            moved_batch = _move_batch(batch, device)
            with torch.inference_mode():
                output = model(moved_batch)
            if not isinstance(output, Mapping):
                raise Stage1EvaluationError(
                    f"{state.name} visualization output is not a mapping"
                )
            for head, output_key in cast(
                tuple[tuple[Literal["wall", "opening"], str], ...],
                (
                    ("wall", "wall_mask_logits"),
                    ("opening", "opening_mask_logits"),
                ),
            ):
                logits = output.get(output_key)
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 4
                    or logits.shape[0] != len(identities)
                    or logits.shape[1] != 1
                ):
                    raise Stage1EvaluationError(
                        f"{state.name} visualization {head} output is incompatible"
                    )
                values = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()
                if not np.isfinite(values).all():
                    raise Stage1EvaluationError(
                        f"{state.name} visualization {head} output is non-finite"
                    )
                for patch_index, identity in enumerate(identities):
                    probabilities.setdefault(identity, {}).setdefault(state.name, {})[
                        head
                    ] = values[patch_index]
    result: list[Stage1VisualizationInput] = []
    for selection in selections:
        identity = (selection.sample_id, selection.patch_index)
        if identity not in patch_data or identity not in probabilities:
            raise Stage1EvaluationError(f"selected visual patch is missing: {identity}")
        image_rgb, valid_mask, wall_target_mask, opening_target_mask = patch_data[
            identity
        ]
        result.append(
            Stage1VisualizationInput(
                sample_id=selection.sample_id,
                patch_index=selection.patch_index,
                image_rgb=image_rgb,
                valid_mask=valid_mask,
                wall_target=wall_target_mask,
                opening_target=opening_target_mask,
                probabilities=probabilities[identity],
            )
        )
    return tuple(result)


def _selected_batch_views(
    materialized: Sequence[tuple[Sequence[str], ModelBatch]],
    selections: Sequence[VisualSelection],
) -> tuple[tuple[tuple[tuple[str, int], ...], ModelBatch], ...]:
    selected_ids = {(item.sample_id, item.patch_index) for item in selections}
    views: list[tuple[tuple[tuple[str, int], ...], ModelBatch]] = []
    seen: set[tuple[str, int]] = set()
    for sample_ids, batch in materialized:
        identities: list[tuple[str, int]] = []
        patch_indices: list[int] = []
        source_indices: list[int] = []
        patch_to_image = batch.patch_to_image.detach().cpu()
        local_numbers: dict[int, int] = {}
        for patch_index in range(batch.local_patches.shape[0]):
            source_index = int(patch_to_image[patch_index].item())
            local_index = local_numbers.get(source_index, 0)
            local_numbers[source_index] = local_index + 1
            identity = (str(sample_ids[source_index]), local_index)
            if identity in selected_ids:
                if identity in seen:
                    raise Stage1EvaluationError(
                        f"duplicate selected visual patch: {identity}"
                    )
                identities.append(identity)
                patch_indices.append(patch_index)
                source_indices.append(source_index)
                seen.add(identity)
        if not identities:
            continue
        unique_sources = tuple(dict.fromkeys(source_indices))
        source_remap = {source: index for index, source in enumerate(unique_sources)}
        patch_to_image_selected = torch.tensor(
            [source_remap[source] for source in source_indices], dtype=torch.int64
        )
        selected_targets = {
            key: value[patch_indices] for key, value in batch.targets.items()
        }
        views.append(
            (
                tuple(identities),
                ModelBatch(
                    global_images=batch.global_images[list(unique_sources)],
                    local_patches=batch.local_patches[patch_indices],
                    patch_to_image=patch_to_image_selected,
                    patch_boxes_global_xyxy=batch.patch_boxes_global_xyxy[
                        patch_indices
                    ],
                    patch_valid_masks=batch.patch_valid_masks[patch_indices],
                    targets=selected_targets,
                ),
            )
        )
    if seen != selected_ids:
        missing = sorted(selected_ids - seen)
        raise Stage1EvaluationError(f"selected visual patches are missing: {missing}")
    return tuple(views)


def _denormalize_patch(patch: torch.Tensor) -> np.ndarray:
    if patch.ndim != 3 or patch.shape[0] != 3:
        raise Stage1EvaluationError("selected local patch has incompatible image shape")
    values = patch.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    standard_deviation = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    image = np.clip(values * standard_deviation + mean, 0.0, 1.0)
    return np.ascontiguousarray(np.rint(image * 255.0).astype(np.uint8))


def _collect_batch(
    output: object,
    batch: ModelBatch,
    sample_ids: tuple[str, ...],
    weight: Literal["raw", "ema"],
    options: Stage1EvaluationOptions,
    records: list[PatchMetricRecord],
    sweep: list[SweepMetricRecord],
) -> None:
    if not isinstance(output, Mapping):
        raise Stage1EvaluationError(f"{weight} model output is not a mapping")
    if not sample_ids:
        raise Stage1EvaluationError("evaluation batch has no sample IDs")
    patch_to_image = batch.patch_to_image.detach().cpu()
    patch_numbers: dict[int, int] = {}
    valid = batch.targets.get("valid_pixels")
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
        raise Stage1EvaluationError("valid_pixels must be a boolean tensor")
    for patch_index in range(batch.local_patches.shape[0]):
        source_index = int(patch_to_image[patch_index].item())
        if source_index < 0 or source_index >= len(sample_ids):
            raise Stage1EvaluationError(
                "patch_to_image contains an invalid source index"
            )
        local_patch = patch_numbers.get(source_index, 0)
        patch_numbers[source_index] = local_patch + 1
        sample_id = sample_ids[source_index]
        for head, output_key, target_key, primary in (
            ("wall", "wall_mask_logits", "wall_mask", options.wall_threshold),
            (
                "opening",
                "opening_mask_logits",
                "opening_mask",
                options.opening_threshold,
            ),
        ):
            logits = output.get(output_key)
            target = batch.targets.get(target_key)
            if not isinstance(logits, torch.Tensor) or not isinstance(
                target, torch.Tensor
            ):
                raise Stage1EvaluationError(
                    f"{weight} {head} output is missing for sample {sample_id} patch {local_patch}"
                )
            if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
                raise Stage1EvaluationError(
                    f"{weight} {head} output shape is incompatible for sample {sample_id} patch {local_patch}"
                )
            if tuple(valid.shape) != tuple(target.shape):
                raise Stage1EvaluationError(
                    f"{weight} {head} valid target shape is incompatible for sample {sample_id} patch {local_patch}"
                )
            patch_logits = logits[patch_index, 0]
            if not torch.isfinite(patch_logits).all():
                raise Stage1EvaluationError(
                    f"{weight} {head} non-finite logit for sample {sample_id} patch {local_patch}"
                )
            probabilities = torch.sigmoid(patch_logits)
            if not torch.isfinite(probabilities).all():
                raise Stage1EvaluationError(
                    f"{weight} {head} non-finite probability for sample {sample_id} patch {local_patch}"
                )
            valid_patch = valid[patch_index, 0]
            if not torch.any(valid_patch):
                raise Stage1EvaluationError(
                    f"{weight} {head} sample {sample_id} patch {local_patch} has no valid pixels"
                )
            target_patch = target[patch_index, 0].bool()
            predicted = probabilities >= primary
            masked_predicted = predicted[valid_patch]
            masked_target = target_patch[valid_patch]
            metrics = binary_metrics(masked_predicted, masked_target)
            records.append(
                PatchMetricRecord(
                    weight=weight,
                    sample_id=sample_id,
                    source_index=source_index,
                    patch_index=local_patch,
                    head=head,  # type: ignore[arg-type]
                    threshold=primary,
                    valid_pixel_count=int(valid_patch.sum().item()),
                    target_positive_pixels=int(masked_target.sum().item()),
                    predicted_positive_pixels=int(masked_predicted.sum().item()),
                    metrics=metrics,
                )
            )
            for threshold in options.thresholds:
                sweep_predicted = probabilities >= threshold
                sweep.append(
                    SweepMetricRecord(
                        weight=weight,
                        head=head,  # type: ignore[arg-type]
                        threshold=threshold,
                        metrics=binary_metrics(
                            sweep_predicted[valid_patch], masked_target
                        ),
                    )
                )


def _aggregate_sweep(
    records: Sequence[SweepMetricRecord],
) -> tuple[SweepMetricRecord, ...]:
    grouped: dict[tuple[str, str, float], list[BinaryMetrics]] = {}
    for record in records:
        grouped.setdefault((record.weight, record.head, record.threshold), []).append(
            record.metrics
        )
    result: list[SweepMetricRecord] = []
    for (weight, head, threshold), values in sorted(grouped.items()):
        result.append(
            SweepMetricRecord(
                weight=cast(Literal["raw", "ema"], weight),
                head=cast(Literal["wall", "opening"], head),
                threshold=threshold,
                metrics=_metrics_from_values(values),
            )
        )
    return tuple(result)


def _metrics_from_values(values: Sequence[BinaryMetrics]) -> BinaryMetrics:
    return (
        BinaryMetrics(
            true_positive=sum(value.true_positive for value in values),
            true_negative=sum(value.true_negative for value in values),
            false_positive=sum(value.false_positive for value in values),
            false_negative=sum(value.false_negative for value in values),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            iou=0.0,
        )
        if not values
        else _metrics_from_counts(
            sum(value.true_positive for value in values),
            sum(value.true_negative for value in values),
            sum(value.false_positive for value in values),
            sum(value.false_negative for value in values),
        )
    )


def _metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> BinaryMetrics:
    predicted = tp + fp
    target = tp + fn
    precision = tp / predicted if predicted else (1.0 if target == 0 else 0.0)
    recall = tp / target if target else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    return BinaryMetrics(
        tp, tn, fp, fn, precision, recall, f1, 1.0 if union == 0 else tp / union
    )


def _micro_metrics(records: Sequence[PatchMetricRecord]) -> BinaryMetrics:
    return _metrics_from_counts(
        sum(record.metrics.true_positive for record in records),
        sum(record.metrics.true_negative for record in records),
        sum(record.metrics.false_positive for record in records),
        sum(record.metrics.false_negative for record in records),
    )


def _metrics_dict(metrics: BinaryMetrics) -> dict[str, object]:
    return {
        "true_positive": metrics.true_positive,
        "true_negative": metrics.true_negative,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "iou": metrics.iou,
    }


def _patch_dict(record: PatchMetricRecord) -> dict[str, object]:
    return {
        "weight": record.weight,
        "sample_id": record.sample_id,
        "source_index": record.source_index,
        "patch_index": record.patch_index,
        "head": record.head,
        "threshold": record.threshold,
        "valid_pixel_count": record.valid_pixel_count,
        "target_positive_pixels": record.target_positive_pixels,
        "predicted_positive_pixels": record.predicted_positive_pixels,
        "metrics": _metrics_dict(record.metrics),
    }


def _selection_dict(item: VisualSelection) -> dict[str, object]:
    return {
        "sample_id": item.sample_id,
        "source_index": item.source_index,
        "patch_index": item.patch_index,
        "reason": item.reason,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, BinaryMetrics):
        return _metrics_dict(value)
    if isinstance(value, PatchMetricSummary):
        return {
            "all_patch_count": value.all_patch_count,
            "target_positive_patch_count": value.target_positive_patch_count,
            "positive_f1_p10": value.positive_f1_p10,
            "positive_f1_median": value.positive_f1_median,
            "positive_f1_p90": value.positive_f1_p90,
            "positive_iou_p10": value.positive_iou_p10,
            "positive_iou_median": value.positive_iou_median,
            "positive_iou_p90": value.positive_iou_p90,
            "target_empty_patch_count": value.target_empty_patch_count,
            "target_empty_false_positive_patch_count": value.target_empty_false_positive_patch_count,
            "target_empty_false_positive_patch_rate": value.target_empty_false_positive_patch_rate,
        }
    if isinstance(value, HeadEvaluationSummary):
        return {
            "micro": _json_safe(value.micro),
            "patches": _json_safe(value.patches),
        }
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, SweepMetricRecord):
        return value.as_dict()
    if isinstance(value, PatchMetricRecord):
        return _patch_dict(value)
    if isinstance(value, VisualSelection):
        return _selection_dict(value)
    return value


def write_stage1_artifacts(
    result: Stage1EvaluationResult, output_dir: Path
) -> Mapping[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "summary": output_dir / "summary.json",
        "patch_metrics": output_dir / "patch_metrics.jsonl",
        "threshold_sweep": output_dir / "threshold_sweep.json",
    }
    try:
        summary_text = json.dumps(
            _json_safe(result.summary),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        patch_text = "".join(
            json.dumps(
                _patch_dict(record),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for record in result.patch_records
        )
        sweep_text = json.dumps(
            [record.as_dict() for record in result.sweep_records],
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        _atomic_write_text(paths["summary"], summary_text)
        _atomic_write_text(paths["patch_metrics"], patch_text)
        _atomic_write_text(paths["threshold_sweep"], sweep_text)
    except Stage1EvaluationError:
        raise
    except Exception as exc:
        raise Stage1EvaluationError(f"artifact output {output_dir}: {exc}") from exc
    return paths


def _atomic_write_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException as exc:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise Stage1EvaluationError(f"artifact output {destination}: {exc}") from exc


def _collate_items(
    items: Sequence[SourceImageItem],
) -> tuple[tuple[str, ...], ModelBatch]:
    return tuple(item.sample_id for item in items), collate_source_images(items)


def _move_batch(batch: ModelBatch, device: torch.device) -> ModelBatch:
    return ModelBatch(
        global_images=batch.global_images.to(device),
        local_patches=batch.local_patches.to(device),
        patch_to_image=batch.patch_to_image.to(device),
        patch_boxes_global_xyxy=batch.patch_boxes_global_xyxy.to(device),
        patch_valid_masks=batch.patch_valid_masks.to(device),
        targets={key: value.to(device) for key, value in batch.targets.items()},
    )
