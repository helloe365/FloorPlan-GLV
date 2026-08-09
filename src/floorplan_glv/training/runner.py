"""Configuration-driven training entry point and run artifact orchestration."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, TextIO

import torch
import yaml
from pydantic import ValidationError
from torch.utils.data import DataLoader

from floorplan_glv.config.load import load_config
from floorplan_glv.config.models import TrainConfig
from floorplan_glv.data.annotation_schema import AnnotationError
from floorplan_glv.data.collate import collate_source_images
from floorplan_glv.data.dataset import FloorPlanDataset, SourceImageItem
from floorplan_glv.data.index import IndexRecord
from floorplan_glv.geometry.rasterize import TargetMaps
from floorplan_glv.losses.combined import (
    LossConfig,
    LossReport,
    LossWeights,
    compute_losses,
)
from floorplan_glv.models.encoders import CheckpointError
from floorplan_glv.models.model import FloorPlanGLV
from floorplan_glv.models.types import FloorPlanModelOutput
from floorplan_glv.training.checkpoint import (
    atomic_write_json,
    atomic_write_text,
    build_run_metadata,
    capture_rng_state,
    file_sha256,
    load_checkpoint,
    load_ema_weights,
    save_checkpoint,
)
from floorplan_glv.training.distributed import (
    all_gather_objects,
    barrier,
    cleanup_distributed,
    distributed_sampler,
    initialize_distributed,
    wrap_ddp,
)
from floorplan_glv.training.ema import ExponentialMovingAverage
from floorplan_glv.training.engine import (
    TrainingEngine,
    build_optimizer,
    build_scheduler,
    seed_everything,
)
from floorplan_glv.training.progress import (
    TRAIN_PHASE,
    VALIDATION_PHASE,
    TrainingReporter,
)

_PRODUCTION_TEST_SPLITS = frozenset(("test", "test_real"))


class _TeeStream:
    """Fan one text stream out to a terminal and a log file.

    ``TrainingReporter`` redraws its batch line in place when the stream is a
    TTY, so ``isatty`` follows the terminal and interactive progress stays
    intact. Those redraws carry a carriage return; the log file takes only
    writes without one, so it keeps whole lines and no batch churn.
    """

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._terminal.write(text)
        if "\r" not in text:
            self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._terminal, "isatty", bool)())


def assert_no_production_test_split(index_path: Path, *, purpose: str) -> None:
    """Reject train/validation indexes that include production test samples."""
    split_root = index_path.parent / "splits"
    if not split_root.is_dir():
        raise AnnotationError(
            f"{purpose} dataset index {index_path} has no sibling splits directory"
        )
    assignments: dict[str, str] = {}
    for split_path in sorted(split_root.glob("*.txt"), key=lambda item: item.name):
        split_name = split_path.stem
        try:
            split_sample_ids = split_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AnnotationError(f"split file {split_path}: {exc}") from exc
        for sample_id in filter(None, (line.strip() for line in split_sample_ids)):
            previous = assignments.get(sample_id)
            if previous is not None and previous != split_name:
                raise AnnotationError(
                    f"split sample {sample_id} belongs to both {previous} and "
                    f"{split_name}"
                )
            assignments[sample_id] = split_name
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnnotationError(f"{purpose} dataset index {index_path}: {exc}") from exc
    sample_ids: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            sample_ids.append(IndexRecord.model_validate_json(line).sample_id)
        except ValidationError as exc:
            raise AnnotationError(
                f"invalid {purpose} dataset index record "
                f"{index_path}:{line_number}: {exc}"
            ) from exc
    leaked = sorted(
        (sample_id, assignments[sample_id])
        for sample_id in sample_ids
        if assignments.get(sample_id) in _PRODUCTION_TEST_SPLITS
    )
    unassigned = sorted(set(sample_ids) - set(assignments))
    if unassigned:
        raise AnnotationError(
            f"{purpose} dataset index {index_path} has missing split assignment: "
            + ", ".join(unassigned)
        )
    if leaked:
        details = ", ".join(f"{sample_id} ({split})" for sample_id, split in leaked)
        raise AnnotationError(
            f"{purpose} dataset index {index_path} includes production test split "
            f"samples: {details}"
        )


def run_training(config_path: Path) -> Path:
    """Run configured training and return the rank-zero ``last.pt`` path."""
    config = load_config(config_path)
    if not config.train.dataset_index.is_file():
        raise AnnotationError(
            f"training dataset index does not exist: {config.train.dataset_index}"
        )
    assert_no_production_test_split(
        config.train.dataset_index,
        purpose="training",
    )
    if (
        config.train.validation_index is not None
        and not config.train.validation_index.is_file()
    ):
        raise AnnotationError(
            f"validation dataset index does not exist: {config.train.validation_index}"
        )
    if config.train.validation_index is not None:
        assert_no_production_test_split(
            config.train.validation_index,
            purpose="validation",
        )
    for purpose, checkpoint in (
        ("initial", config.train.initial_checkpoint),
        ("resume", config.train.resume_checkpoint),
    ):
        if checkpoint is not None and not checkpoint.is_file():
            raise CheckpointError(f"{purpose} checkpoint does not exist: {checkpoint}")
    context = initialize_distributed()
    log_file: TextIO | None = None
    log_path: Path | None = None
    reporter_stream: TextIO = sys.stderr
    if context.is_rank_zero:
        config.train.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = config.train.output_dir / "training.log"
        log_file = log_path.open("a", encoding="utf-8")
        reporter_stream = cast(TextIO, _TeeStream(sys.stderr, log_file))
    reporter = TrainingReporter(
        stream=reporter_stream,
        enabled=context.is_rank_zero,
        log_interval=config.train.log_interval,
    )
    try:
        if log_path is not None:
            reporter.note(f"logging this run to {log_path}")
        seed_everything(
            config.train.seed + context.rank,
            deterministic_algorithms=config.train.deterministic_algorithms,
        )
        reporter.note(f"seeded run with {config.train.seed} (+rank)")
        train_dataset = FloorPlanDataset(
            config.train.dataset_index,
            patches_per_image=config.train.patches_per_image,
            seed=config.train.seed,
            model_config=config.model,
            data_config=config.data,
        )
        train_sampler = distributed_sampler(
            train_dataset,
            context=context,
            shuffle=True,
            seed=config.train.seed,
        )
        train_loader = _data_loader(
            train_dataset,
            config=config.train,
            sampler=train_sampler,
            shuffle=train_sampler is None,
        )
        validation_loader: DataLoader[SourceImageItem] | None = None
        validation_dataset_size: int | None = None
        if config.train.validation_index is not None:
            validation_dataset = FloorPlanDataset(
                config.train.validation_index,
                patches_per_image=config.train.patches_per_image,
                seed=config.train.seed,
                model_config=config.model,
                data_config=config.data,
            )
            validation_dataset_size = len(validation_dataset)
            validation_sampler = distributed_sampler(
                validation_dataset,
                context=context,
                shuffle=False,
                seed=config.train.seed,
            )
            validation_loader = _data_loader(
                validation_dataset,
                config=config.train,
                sampler=validation_sampler,
                shuffle=False,
            )
        reporter.note(
            f"loaded {len(train_dataset)} train"
            + (
                f" and {validation_dataset_size} validation"
                if validation_dataset_size is not None
                else ""
            )
            + " source images"
        )

        reporter.note(
            f"building model (local {config.model.local_checkpoint}"
            + (
                f", global {config.model.global_checkpoint}"
                if config.model.global_enabled
                else ", global disabled"
            )
            + ")"
        )
        base_model = FloorPlanGLV(config.model).to(context.device)
        _configure_gradient_checkpointing(base_model, config.train)
        if config.train.initial_checkpoint is not None:
            reporter.note(f"loading EMA weights from {config.train.initial_checkpoint}")
            load_ema_weights(
                config.train.initial_checkpoint,
                model=base_model,
                map_location=context.device,
            )
        optimizer = build_optimizer(base_model, config.train)
        optimizer_steps_per_epoch = math.ceil(
            len(train_loader) / config.train.gradient_accumulation_steps
        )
        scheduler = build_scheduler(
            optimizer,
            total_steps=optimizer_steps_per_epoch * config.train.epochs,
            config=config.train.scheduler,
        )
        ema = ExponentialMovingAverage(base_model, decay=config.train.ema_decay)
        loss_config = LossConfig(
            weights=LossWeights.model_validate(config.train.loss_weights.model_dump())
        )

        def task_losses(
            output: object,
            targets: dict[str, torch.Tensor],
        ) -> LossReport:
            return compute_losses(
                cast(FloorPlanModelOutput, output),
                cast(TargetMaps, targets),
                loss_config,
            )

        model = wrap_ddp(base_model, context)
        engine = TrainingEngine(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=task_losses,
            config=config.train,
            device=context.device,
            ema=ema,
            context=context,
            on_batch=reporter.batch_progress,
        )
        resolved_config = config.model_dump(mode="json")
        metadata = build_run_metadata(
            seed=config.train.seed,
            dataset_indexes=tuple(
                path
                for path in (
                    config.train.dataset_index,
                    config.train.validation_index,
                )
                if path is not None
            ),
            checkpoint=(
                config.train.resume_checkpoint or config.train.initial_checkpoint
            ),
            postprocess_config=config.postprocess.model_dump(mode="json"),
            deterministic_algorithms=config.train.deterministic_algorithms,
        )
        start_epoch = 0
        best_metrics: dict[str, float] = {}
        if config.train.resume_checkpoint is not None:
            reporter.note(f"resuming from {config.train.resume_checkpoint}")
            resume = load_checkpoint(
                config.train.resume_checkpoint,
                model=base_model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=engine.scaler,
                map_location=context.device,
                rank=context.rank,
            )
            start_epoch = resume.epoch + 1
            engine.global_step = resume.global_step
            best_metrics = resume.best_metrics

        last_checkpoint = config.train.output_dir / "last.pt"
        if context.is_rank_zero:
            atomic_write_text(
                config.train.output_dir / "resolved_config.yaml",
                yaml.safe_dump(resolved_config, sort_keys=True),
            )
            atomic_write_json(config.train.output_dir / "run_metadata.json", metadata)

        reporter.run_start(
            config_path=config_path,
            config=config,
            context=context,
            optimizer=optimizer,
            parameter_counts=_parameter_counts(base_model),
            amp_dtype=engine.amp_dtype,
            scaler_enabled=engine.scaler.is_enabled(),
            train_samples=len(train_dataset),
            train_batches=len(train_loader),
            validation_samples=validation_dataset_size,
            validation_batches=(
                len(validation_loader) if validation_loader is not None else None
            ),
            start_epoch=start_epoch,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        )

        for epoch in range(start_epoch, config.train.epochs):
            train_dataset.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            reporter.epoch_start(
                epoch=epoch,
                phase=TRAIN_PHASE,
                total_batches=len(train_loader),
                local_encoder_frozen=epoch < config.train.freeze_local_encoder_epochs,
            )
            train_report = engine.train_epoch(train_loader, epoch=epoch)
            reporter.epoch_end(
                epoch=epoch,
                phase=TRAIN_PHASE,
                metrics=train_report.metrics,
                batch_count=train_report.batch_count,
                optimizer_steps=train_report.optimizer_steps,
                skipped_steps=train_report.skipped_steps,
            )
            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "global_step": engine.global_step,
                "train": train_report.metrics,
            }
            if validation_loader is not None:
                reporter.epoch_start(
                    epoch=epoch,
                    phase=VALIDATION_PHASE,
                    total_batches=len(validation_loader),
                )
                validation_report = engine.evaluate(validation_loader, epoch=epoch)
                epoch_record["validation"] = validation_report.metrics
                validation_loss = validation_report.metrics["total"]
                previous_best = best_metrics.get("validation_loss", math.inf)
                is_best = validation_loss < previous_best
                best_metrics["validation_loss"] = min(previous_best, validation_loss)
                reporter.epoch_end(
                    epoch=epoch,
                    phase=VALIDATION_PHASE,
                    metrics=validation_report.metrics,
                    batch_count=validation_report.batch_count,
                    is_best=is_best,
                )
            if context.is_rank_zero:
                _append_json_line(
                    config.train.output_dir / "metrics.jsonl",
                    epoch_record,
                )
            rng_states = all_gather_objects(
                capture_rng_state(),
                context=context,
            )
            if context.is_rank_zero:
                save_checkpoint(
                    last_checkpoint,
                    model=base_model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=engine.scaler,
                    epoch=epoch,
                    global_step=engine.global_step,
                    best_metrics=best_metrics,
                    resolved_config=resolved_config,
                    run_metadata=metadata,
                    rng_states=cast(list[Mapping[str, Any]], rng_states),
                )
                metadata["checkpoint_sha256"] = file_sha256(last_checkpoint)
                atomic_write_json(
                    config.train.output_dir / "run_metadata.json",
                    metadata,
                )
                reporter.checkpoint_saved(
                    last_checkpoint,
                    epoch=epoch,
                    global_step=engine.global_step,
                )
            barrier(context)
        reporter.run_end(last_checkpoint=last_checkpoint, best_metrics=best_metrics)
        return last_checkpoint
    finally:
        cleanup_distributed(context)
        if log_file is not None:
            log_file.close()


def _data_loader(
    dataset: FloorPlanDataset,
    *,
    config: TrainConfig,
    sampler: torch.utils.data.Sampler[SourceImageItem] | None,
    shuffle: bool,
) -> DataLoader[SourceImageItem]:
    return DataLoader(
        dataset,
        batch_size=config.source_images_per_gpu,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_source_images,
        persistent_workers=False,
    )


def _parameter_counts(model: FloorPlanGLV) -> tuple[int, int]:
    """Return the total and trainable parameter counts for run reporting."""
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _configure_gradient_checkpointing(
    model: FloorPlanGLV,
    config: TrainConfig,
) -> None:
    encoders = (model.local_encoder, model.global_encoder)
    for encoder in encoders:
        method = getattr(encoder, "set_gradient_checkpointing", None)
        if callable(method):
            method(config.gradient_checkpointing)


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
