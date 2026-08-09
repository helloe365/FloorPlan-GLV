"""Reproducible single-process and DDP training orchestration."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sized
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from floorplan_glv.config.models import (
    SchedulerConfig,
    TrainConfig,
)
from floorplan_glv.data.collate import ModelBatch
from floorplan_glv.losses.combined import LossReport
from floorplan_glv.training.distributed import DistributedContext, unwrap_model
from floorplan_glv.training.ema import ExponentialMovingAverage
from floorplan_glv.training.progress import (
    TRAIN_PHASE,
    VALIDATION_PHASE,
    BatchProgress,
)

LossFunction = Callable[[object, dict[str, torch.Tensor]], LossReport]
BatchObserver = Callable[[BatchProgress], None]


@dataclass(frozen=True, slots=True)
class EpochReport:
    """Aggregated metrics and progress from one train or validation epoch."""

    metrics: dict[str, float]
    batch_count: int
    optimizer_steps: int
    global_step: int
    skipped_steps: int = 0


def seed_everything(seed: int, *, deterministic_algorithms: bool) -> None:
    """Seed Python, NumPy, and PyTorch for one reproducible run."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    _apply_cudnn_workaround()


def _apply_cudnn_workaround() -> None:
    """Disable cuDNN on Blackwell GPUs (sm_120) to avoid an illegal-memory-access
    kernel bug in cuDNN 9.2.3 when 1x1 Conv2d projections consume SegFormer
    hidden states.

    Set ``FLOORPLAN_ALLOW_CUDNN=1`` to force cuDNN on regardless of GPU arch.
    """

    if os.environ.get("FLOORPLAN_ALLOW_CUDNN") == "1":
        return
    if not torch.cuda.is_available():
        return
    major = torch.cuda.get_device_capability(0)[0]
    if major >= 12:  # sm_120 = Blackwell; sm_120a = Blackwell with FP8
        torch.backends.cudnn.enabled = False


def build_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.AdamW:
    """Create named AdamW groups for every approved model section."""
    raw_model = unwrap_model(model)
    sections = (
        (
            "global_encoder",
            getattr(raw_model, "global_encoder", None),
            config.learning_rates.global_encoder,
        ),
        (
            "local_encoder",
            getattr(raw_model, "local_encoder", None),
            config.learning_rates.local_encoder,
        ),
        (
            "fusion",
            getattr(raw_model, "fusion_blocks", None),
            config.learning_rates.fusion,
        ),
        (
            "decoder",
            getattr(raw_model, "decoder", None),
            config.learning_rates.decoder,
        ),
        (
            "heads",
            getattr(raw_model, "heads", None),
            config.learning_rates.heads,
        ),
    )
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name, module, learning_rate in sections:
        if not isinstance(module, nn.Module):
            continue
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in seen
        ]
        if not parameters:
            continue
        seen.update(id(parameter) for parameter in parameters)
        groups.append(
            {
                "name": name,
                "params": parameters,
                "lr": learning_rate,
            }
        )
    trainable = {
        id(parameter) for parameter in raw_model.parameters() if parameter.requires_grad
    }
    if seen != trainable:
        missing_count = len(trainable - seen)
        raise ValueError(
            f"optimizer grouping missed {missing_count} trainable parameters"
        )
    if not groups:
        raise ValueError("model contains no trainable parameter groups")
    return torch.optim.AdamW(
        groups,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    config: SchedulerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the configured warmup-plus-cosine learning-rate schedule."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def scale(step: int) -> float:
        if config.warmup_steps > 0 and step < config.warmup_steps:
            return float(step + 1) / float(config.warmup_steps)
        decay_steps = max(1, total_steps - config.warmup_steps)
        progress = min(
            1.0,
            max(0.0, (step - config.warmup_steps) / decay_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


class TrainingEngine:
    """Train and validate a model with AMP, accumulation, clipping, and EMA."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_function: LossFunction,
        config: TrainConfig,
        device: torch.device,
        ema: ExponentialMovingAverage,
        context: DistributedContext | None = None,
        on_batch: BatchObserver | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.config = config
        self.device = device
        self.ema = ema
        self.context = context
        self.on_batch = on_batch
        self.global_step = 0
        self._amp_dtype, self._amp_enabled = _resolve_amp(
            device,
            config.precision,
        )
        scaler_enabled = (
            self._amp_enabled
            and self._amp_dtype == torch.float16
            and device.type == "cuda"
        )
        self.scaler = GradScaler(
            device.type,
            enabled=scaler_enabled,
        )

    @property
    def amp_dtype(self) -> torch.dtype | None:
        """Resolved autocast dtype, or ``None`` when autocast is disabled."""
        return self._amp_dtype

    @property
    def amp_enabled(self) -> bool:
        """Whether autocast wraps the forward pass."""
        return self._amp_enabled

    def train_epoch(
        self,
        batches: Iterable[ModelBatch],
        *,
        epoch: int,
    ) -> EpochReport:
        """Train one epoch over flattened global-local batches."""
        self.model.train()
        local_is_frozen = epoch < self.config.freeze_local_encoder_epochs
        _set_local_encoder_mode(
            self.model,
            trainable=not local_is_frozen,
        )
        self.optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        batch_count = 0
        optimizer_steps = 0
        skipped_steps = 0
        accumulation_count = 0
        total_batches = len(batches) if isinstance(batches, Sized) else None

        for batch_index, batch in enumerate(batches):
            batch_count += 1
            accumulation_count += 1
            is_last = total_batches is not None and batch_index + 1 == total_batches
            should_step = (
                accumulation_count == self.config.gradient_accumulation_steps or is_last
            )
            sync_context = _gradient_sync_context(self.model, should_step)
            moved = _move_batch(batch, self.device)
            with (
                sync_context,
                _autocast_context(
                    self.device,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ),
            ):
                output = self.model(moved)
                report = self.loss_function(output, moved.targets)
                scaled_loss = report.total / self.config.gradient_accumulation_steps
            scaled = self.scaler.scale(scaled_loss)
            scaled.backward()  # type: ignore[no-untyped-call]
            batch_loss = _accumulate_metrics(metric_sums, report)

            grad_norm: float | None = None
            skipped_step = False
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if accumulation_count < self.config.gradient_accumulation_steps:
                    correction = (
                        self.config.gradient_accumulation_steps / accumulation_count
                    )
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                if local_is_frozen:
                    _clear_local_encoder_gradients(self.model)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )
                )
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_updated = self.scaler.get_scale() >= previous_scale
                self.optimizer.zero_grad(set_to_none=True)
                if optimizer_updated:
                    self.scheduler.step()
                    self.ema.update(unwrap_model(self.model))
                    self.global_step += 1
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
                    skipped_step = True
                accumulation_count = 0
            if self.on_batch is not None:
                self.on_batch(
                    BatchProgress(
                        phase=TRAIN_PHASE,
                        epoch=epoch,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        global_step=self.global_step,
                        loss=batch_loss,
                        running_loss=metric_sums["total"] / batch_count,
                        learning_rate=_current_learning_rate(self.optimizer),
                        grad_norm=grad_norm,
                        skipped_step=skipped_step,
                    )
                )

        if batch_count == 0:
            raise ValueError("training epoch received no batches")
        if accumulation_count:
            raise RuntimeError(
                "unknown-length iterables must end on an accumulation step"
            )
        metrics = _finalize_metrics(
            metric_sums,
            batch_count=batch_count,
            context=self.context,
            device=self.device,
        )
        return EpochReport(
            metrics=metrics,
            batch_count=batch_count,
            optimizer_steps=optimizer_steps,
            global_step=self.global_step,
            skipped_steps=skipped_steps,
        )

    @torch.no_grad()
    def evaluate(
        self,
        batches: Iterable[ModelBatch],
        *,
        epoch: int = 0,
    ) -> EpochReport:
        """Evaluate one loader without mutating optimizer or EMA state."""
        was_training = self.model.training
        self.model.eval()
        metric_sums: dict[str, float] = {}
        batch_count = 0
        total_batches = len(batches) if isinstance(batches, Sized) else None
        for batch_index, batch in enumerate(batches):
            batch_count += 1
            moved = _move_batch(batch, self.device)
            with _autocast_context(
                self.device,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                output = self.model(moved)
                report = self.loss_function(output, moved.targets)
            batch_loss = _accumulate_metrics(metric_sums, report)
            if self.on_batch is not None:
                self.on_batch(
                    BatchProgress(
                        phase=VALIDATION_PHASE,
                        epoch=epoch,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        global_step=self.global_step,
                        loss=batch_loss,
                        running_loss=metric_sums["total"] / batch_count,
                        learning_rate=None,
                        grad_norm=None,
                        skipped_step=False,
                    )
                )
        if was_training:
            self.model.train()
        if batch_count == 0:
            raise ValueError("validation received no batches")
        metrics = _finalize_metrics(
            metric_sums,
            batch_count=batch_count,
            context=self.context,
            device=self.device,
        )
        return EpochReport(
            metrics=metrics,
            batch_count=batch_count,
            optimizer_steps=0,
            global_step=self.global_step,
        )


def _move_batch(batch: ModelBatch, device: torch.device) -> ModelBatch:
    return ModelBatch(
        global_images=batch.global_images.to(device, non_blocking=True),
        local_patches=batch.local_patches.to(device, non_blocking=True),
        patch_to_image=batch.patch_to_image.to(device, non_blocking=True),
        patch_boxes_global_xyxy=batch.patch_boxes_global_xyxy.to(
            device,
            non_blocking=True,
        ),
        patch_valid_masks=batch.patch_valid_masks.to(device, non_blocking=True),
        targets={
            key: value.to(device, non_blocking=True)
            for key, value in batch.targets.items()
        },
    )


def _resolve_amp(
    device: torch.device,
    precision: str,
) -> tuple[torch.dtype | None, bool]:
    if precision == "fp32":
        return None, False
    if precision == "bf16":
        if device.type == "cpu" or (
            device.type == "cuda" and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16, True
        if device.type == "cuda":
            return torch.float16, True
        return None, False
    if precision == "fp16" and device.type == "cuda":
        return torch.float16, True
    return None, False


def _autocast_context(
    device: torch.device,
    *,
    dtype: torch.dtype | None,
    enabled: bool,
) -> AbstractContextManager[None]:
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def _gradient_sync_context(
    model: nn.Module,
    should_step: bool,
) -> AbstractContextManager[Any]:
    no_sync = getattr(model, "no_sync", None)
    if not should_step and callable(no_sync):
        return cast(AbstractContextManager[Any], no_sync())
    return nullcontext()


def _set_local_encoder_mode(model: nn.Module, *, trainable: bool) -> None:
    local_encoder = getattr(unwrap_model(model), "local_encoder", None)
    if not isinstance(local_encoder, nn.Module):
        return
    local_encoder.train(trainable)


def _clear_local_encoder_gradients(model: nn.Module) -> None:
    local_encoder = getattr(unwrap_model(model), "local_encoder", None)
    if not isinstance(local_encoder, nn.Module):
        return
    for parameter in local_encoder.parameters():
        parameter.grad = None


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float | None:
    """Return the largest active group learning rate for progress reporting."""
    rates = [float(group["lr"]) for group in optimizer.param_groups]
    return max(rates) if rates else None


def _accumulate_metrics(
    sums: dict[str, float],
    report: LossReport,
) -> float:
    """Add one batch report into ``sums`` and return its total loss."""
    values = {
        "total": float(report.total.detach().float().cpu()),
        **{
            f"raw/{name}": float(value.detach().float().cpu())
            for name, value in report.raw_losses.items()
        },
        **{
            f"weighted/{name}": float(value.detach().float().cpu())
            for name, value in report.weighted_losses.items()
        },
        **{
            f"positive/{name}": float(value.detach().float().cpu())
            for name, value in report.positive_counts.items()
        },
    }
    for name, value in values.items():
        sums[name] = sums.get(name, 0.0) + value
    return values["total"]


def _finalize_metrics(
    sums: Mapping[str, float],
    *,
    batch_count: int,
    context: DistributedContext | None,
    device: torch.device,
) -> dict[str, float]:
    names = tuple(sorted(sums))
    values = torch.tensor(
        [sums[name] for name in names] + [float(batch_count)],
        dtype=torch.float64,
        device=device,
    )
    if context is not None and context.is_distributed:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    total_batches = float(values[-1].item())
    return {
        name: float(values[index].item()) / total_batches
        for index, name in enumerate(names)
    }
