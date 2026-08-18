"""Configuration-driven training entry point and run artifact orchestration."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

import torch
import yaml
from pydantic import ValidationError
from torch.utils.data import DataLoader, DistributedSampler

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
from floorplan_glv.training.augmentation import build_augmentation
from floorplan_glv.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    ResumeState,
    atomic_write_json,
    atomic_write_text,
    build_run_metadata,
    capture_rng_state,
    file_sha256,
    load_checkpoint,
    load_checkpoint_weights,
    read_checkpoint_summary,
    save_checkpoint,
)
from floorplan_glv.training.distributed import (
    DistributedContext,
    all_gather_objects,
    barrier,
    cleanup_distributed,
    distributed_sampler,
    initialize_distributed,
    wrap_ddp,
)
from floorplan_glv.training.early_stopping import EarlyStopping
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


@dataclass(frozen=True, slots=True)
class TrainingLoopResult:
    """Checkpoint selection and completion state returned by the epoch loop."""

    selected_checkpoint: Path
    last_checkpoint: Path
    stopped_early: bool


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


def _build_early_stopper(
    *,
    config: TrainConfig,
    resume: ResumeState | None,
    best_checkpoint: Path,
) -> EarlyStopping | None:
    """Build and strictly restore the configured early-stopping policy."""
    policy_config = config.early_stopping
    if policy_config is None:
        validation_resume = resume is not None and config.validation_index is not None
        if validation_resume and not best_checkpoint.is_file():
            raise CheckpointError(
                "validation resume with early stopping disabled requires paired "
                f"best.pt at {best_checkpoint}; use the resume checkpoint as "
                "initial_checkpoint for a fresh run"
            )
        return None

    early_stopper = EarlyStopping(policy_config)
    if resume is None:
        return early_stopper

    if resume.schema_version == LEGACY_CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            "schema 1.0 resume checkpoints lack exact early-stopping state; "
            "use the file as initial_checkpoint for a fresh run or disable "
            "early stopping"
        )
    if resume.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            "enabled early stopping requires checkpoint schema "
            f"{CHECKPOINT_SCHEMA_VERSION}, got {resume.schema_version}"
        )

    saved_train = resume.resolved_config.get("train")
    saved_policy = (
        saved_train.get("early_stopping") if isinstance(saved_train, Mapping) else None
    )
    current_policy = policy_config.model_dump(mode="json")
    if saved_policy != current_policy:
        raise CheckpointError(
            "resume checkpoint early-stopping configuration differs from "
            "the current configuration"
        )

    state = resume.early_stopping_state
    if state is None:
        raise CheckpointError(
            "enabled early stopping requires saved early-stopping state"
        )
    try:
        early_stopper.load_state_dict(state)
    except ValueError as exc:
        raise CheckpointError(
            f"resume checkpoint has invalid early-stopping state: {exc}"
        ) from exc
    if early_stopper.best_value is None or early_stopper.best_epoch is None:
        raise CheckpointError(
            "enabled resume early-stopping state does not record a best epoch"
        )

    if not best_checkpoint.is_file():
        raise CheckpointError(f"paired best checkpoint {best_checkpoint} is missing")
    summary = read_checkpoint_summary(best_checkpoint)
    paired_state = summary.early_stopping_state
    if summary.schema_version != CHECKPOINT_SCHEMA_VERSION or paired_state is None:
        raise CheckpointError(
            f"paired best checkpoint {best_checkpoint} has incompatible state"
        )
    paired_policy = EarlyStopping(policy_config)
    try:
        paired_policy.load_state_dict(paired_state)
    except ValueError as exc:
        raise CheckpointError(
            f"paired best checkpoint {best_checkpoint} has invalid state: {exc}"
        ) from exc
    if (
        summary.epoch != early_stopper.best_epoch
        or paired_policy.best_value != early_stopper.best_value
        or paired_policy.best_epoch != early_stopper.best_epoch
    ):
        raise CheckpointError(
            f"paired best checkpoint {best_checkpoint} does not match "
            "the resume policy best value and epoch"
        )
    return early_stopper


def run_training(config_path: Path) -> Path:
    """Run configured training and return the selected checkpoint path."""
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
            augmentation=build_augmentation(config.train.augmentation),
        )
        if config.train.augmentation.enabled:
            reporter.note(
                "augmentation enabled: "
                f"flip_h={config.train.augmentation.horizontal_flip_prob:.2f}, "
                f"rotation={config.train.augmentation.rotation_prob:.2f}, "
                f"scale_prob={config.train.augmentation.scale_prob:.2f} "
                f"[{config.train.augmentation.scale_min:.2f},"
                f"{config.train.augmentation.scale_max:.2f}], "
                "brightness/contrast/gamma="
                f"{config.train.augmentation.brightness_prob:.2f}/"
                f"{config.train.augmentation.contrast_prob:.2f}/"
                f"{config.train.augmentation.gamma_prob:.2f}"
            )
        else:
            reporter.note("augmentation disabled")
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
            reporter.note(
                f"loading {config.train.initial_checkpoint_state} weights from "
                f"{config.train.initial_checkpoint}"
            )
            load_checkpoint_weights(
                config.train.initial_checkpoint,
                model=base_model,
                map_location=context.device,
                state=config.train.initial_checkpoint_state,
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
            weights=LossWeights.model_validate(config.train.loss_weights.model_dump()),
            focal_alpha=config.train.focal_alpha,
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
        resume: ResumeState | None = None
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

        best_checkpoint = config.train.output_dir / "best.pt"
        early_stopper = _build_early_stopper(
            config=config.train,
            resume=resume,
            best_checkpoint=best_checkpoint,
        )
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

        loop_result = _run_epochs(
            base_model=base_model,
            engine=engine,
            train_dataset=train_dataset,
            train_loader=train_loader,
            validation_loader=validation_loader,
            train_sampler=train_sampler,
            ema=ema,
            context=context,
            reporter=reporter,
            config=config.train,
            resolved_config=resolved_config,
            metadata=metadata,
            start_epoch=start_epoch,
            best_metrics=best_metrics,
            early_stopper=early_stopper,
        )
        reporter.run_end(
            selected_checkpoint=loop_result.selected_checkpoint,
            last_checkpoint=loop_result.last_checkpoint,
            best_metrics=best_metrics,
            stopped_early=loop_result.stopped_early,
        )
        return loop_result.selected_checkpoint
    finally:
        cleanup_distributed(context)
        if log_file is not None:
            log_file.close()


def _run_epochs(
    *,
    base_model: torch.nn.Module,
    engine: TrainingEngine,
    train_dataset: FloorPlanDataset,
    train_loader: DataLoader[SourceImageItem],
    validation_loader: DataLoader[SourceImageItem] | None,
    train_sampler: DistributedSampler[SourceImageItem] | None,
    ema: ExponentialMovingAverage,
    context: DistributedContext,
    reporter: TrainingReporter,
    config: TrainConfig,
    resolved_config: Mapping[str, Any],
    metadata: dict[str, Any],
    start_epoch: int,
    best_metrics: dict[str, float],
    early_stopper: EarlyStopping | None,
) -> TrainingLoopResult:
    """Run configured epochs and return selected and resumable artifacts."""
    last_checkpoint = config.output_dir / "last.pt"
    best_checkpoint = config.output_dir / "best.pt"
    selected_checkpoint = (
        best_checkpoint if validation_loader is not None else last_checkpoint
    )
    if early_stopper is not None and early_stopper.should_stop:
        best_epoch = early_stopper.best_epoch
        if best_epoch is None:
            raise CheckpointError(
                "triggered early-stopping state does not record a best epoch"
            )
        reporter.early_stopping_triggered(
            epoch=max(0, start_epoch - 1),
            best_epoch=best_epoch,
        )
        if not best_checkpoint.is_file():
            raise CheckpointError(
                f"selected checkpoint does not exist: {best_checkpoint}"
            )
        return TrainingLoopResult(
            selected_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            stopped_early=True,
        )

    stopped_early = False
    for epoch in range(start_epoch, config.epochs):
        train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        reporter.epoch_start(
            epoch=epoch,
            phase=TRAIN_PHASE,
            total_batches=len(train_loader),
            local_encoder_frozen=epoch < config.freeze_local_encoder_epochs,
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
        is_best = False
        should_stop = False
        early_stopping_state: Mapping[str, object] | None = None
        if validation_loader is not None:
            reporter.epoch_start(
                epoch=epoch,
                phase=VALIDATION_PHASE,
                total_batches=len(validation_loader),
            )
            with ema.average_parameters(base_model):
                validation_report = engine.evaluate(validation_loader, epoch=epoch)
            epoch_record["validation"] = validation_report.metrics
            monitor = (
                early_stopper.config.monitor
                if early_stopper is not None
                else "validation.total"
            )
            metric_name = monitor.removeprefix("validation.")
            validation_loss = validation_report.metrics.get(metric_name)
            if validation_loss is None:
                raise ValueError(
                    f"validation metrics lack configured monitor {monitor}"
                )
            best_metric_key = monitor
            previous_best = best_metrics.get(
                best_metric_key,
                best_metrics.get("validation_loss", math.inf),
            )
            if early_stopper is None:
                is_best = validation_loss < previous_best
                best_metrics[best_metric_key] = min(
                    previous_best,
                    validation_loss,
                )
            else:
                decision = early_stopper.update(epoch=epoch, value=validation_loss)
                is_best = decision.improved
                should_stop = decision.should_stop
                best_metrics[best_metric_key] = decision.best_value
                early_stopping_state = early_stopper.state_dict()
                epoch_record["early_stopping"] = {
                    "monitor": early_stopper.config.monitor,
                    "value": decision.value,
                    "best_value": decision.best_value,
                    "best_epoch": decision.best_epoch,
                    "bad_epochs": decision.bad_epochs,
                    "patience": early_stopper.config.patience,
                    "improved": decision.improved,
                    "should_stop": decision.should_stop,
                }
                reporter.early_stopping_progress(
                    monitor=early_stopper.config.monitor,
                    value=decision.value,
                    best_value=decision.best_value,
                    best_epoch=decision.best_epoch,
                    bad_epochs=decision.bad_epochs,
                    patience=early_stopper.config.patience,
                )
            reporter.epoch_end(
                epoch=epoch,
                phase=VALIDATION_PHASE,
                metrics=validation_report.metrics,
                batch_count=validation_report.batch_count,
                is_best=is_best,
            )
        if context.is_rank_zero:
            _append_json_line(
                config.output_dir / "metrics.jsonl",
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
                optimizer=engine.optimizer,
                scheduler=engine.scheduler,
                scaler=engine.scaler,
                epoch=epoch,
                global_step=engine.global_step,
                best_metrics=best_metrics,
                resolved_config=resolved_config,
                run_metadata=metadata,
                early_stopping_state=early_stopping_state,
                rng_states=cast(list[Mapping[str, Any]], rng_states),
            )
            metadata["checkpoint_sha256"] = file_sha256(last_checkpoint)
            atomic_write_json(
                config.output_dir / "run_metadata.json",
                metadata,
            )
            reporter.checkpoint_saved(
                last_checkpoint,
                epoch=epoch,
                global_step=engine.global_step,
            )
            if is_best:
                save_checkpoint(
                    best_checkpoint,
                    model=base_model,
                    ema=ema,
                    optimizer=engine.optimizer,
                    scheduler=engine.scheduler,
                    scaler=engine.scaler,
                    epoch=epoch,
                    global_step=engine.global_step,
                    best_metrics=best_metrics,
                    resolved_config=resolved_config,
                    run_metadata=metadata,
                    early_stopping_state=early_stopping_state,
                    rng_states=cast(list[Mapping[str, Any]], rng_states),
                )
                reporter.checkpoint_saved(
                    best_checkpoint,
                    epoch=epoch,
                    global_step=engine.global_step,
                )
        barrier(context)
        if should_stop:
            if early_stopper is None or early_stopper.best_epoch is None:
                raise RuntimeError("early stopping triggered without a best epoch")
            reporter.early_stopping_triggered(
                epoch=epoch,
                best_epoch=early_stopper.best_epoch,
            )
            stopped_early = True
            break
    if not selected_checkpoint.is_file():
        raise CheckpointError(
            f"selected checkpoint does not exist: {selected_checkpoint}"
        )
    return TrainingLoopResult(
        selected_checkpoint=selected_checkpoint,
        last_checkpoint=last_checkpoint,
        stopped_early=stopped_early,
    )


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
