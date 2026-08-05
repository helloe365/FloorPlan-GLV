"""Configuration-driven training entry point and run artifact orchestration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

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

_PRODUCTION_TEST_SPLITS = frozenset(("test", "test_real"))


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
    try:
        seed_everything(
            config.train.seed + context.rank,
            deterministic_algorithms=config.train.deterministic_algorithms,
        )
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
        if config.train.validation_index is not None:
            validation_dataset = FloorPlanDataset(
                config.train.validation_index,
                patches_per_image=config.train.patches_per_image,
                seed=config.train.seed,
                model_config=config.model,
                data_config=config.data,
            )
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

        base_model = FloorPlanGLV(config.model).to(context.device)
        _configure_gradient_checkpointing(base_model, config.train)
        if config.train.initial_checkpoint is not None:
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

        for epoch in range(start_epoch, config.train.epochs):
            train_dataset.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_report = engine.train_epoch(train_loader, epoch=epoch)
            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "global_step": engine.global_step,
                "train": train_report.metrics,
            }
            if validation_loader is not None:
                validation_report = engine.evaluate(validation_loader)
                epoch_record["validation"] = validation_report.metrics
                validation_loss = validation_report.metrics["total"]
                best_metrics["validation_loss"] = min(
                    best_metrics.get("validation_loss", math.inf),
                    validation_loss,
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
            barrier(context)
        return last_checkpoint
    finally:
        cleanup_distributed(context)


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
