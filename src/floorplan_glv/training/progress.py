"""Human-readable terminal reporting for configuration-driven training runs."""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch

from floorplan_glv.config.models import AppConfig
from floorplan_glv.training.distributed import DistributedContext

TRAIN_PHASE = "train"
VALIDATION_PHASE = "val"

_HEADER_RULE = "=" * 78
_MINIMUM_REDRAW_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class BatchProgress:
    """One observed train or validation batch handed to the reporter."""

    phase: str
    epoch: int
    batch_index: int
    total_batches: int | None
    global_step: int
    loss: float
    running_loss: float
    learning_rate: float | None
    grad_norm: float | None
    skipped_step: bool


def format_duration(seconds: float) -> str:
    """Render a positive duration as ``H:MM:SS`` or ``MM:SS``."""
    if not math.isfinite(seconds) or seconds < 0.0:
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_parameters(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _format_bytes(count: float) -> str:
    return f"{count / float(1 << 30):.1f}GiB"


class TrainingReporter:
    """Stream run, epoch, and batch progress to a terminal or a log file.

    Only the rank-zero process should be enabled so distributed runs emit one
    stream. Interactive terminals receive an in-place batch line; redirected
    streams receive one plain line per ``log_interval`` batches so ``nohup``
    logs stay readable.
    """

    def __init__(
        self,
        *,
        stream: TextIO,
        enabled: bool = True,
        log_interval: int = 10,
    ) -> None:
        self._stream = stream
        self._enabled = enabled
        self._log_interval = max(0, log_interval)
        self._is_tty = bool(getattr(stream, "isatty", bool)())
        self._pending_width = 0
        self._total_epochs = 0
        self._device = torch.device("cpu")
        self._run_start = time.monotonic()
        self._phase_start = time.monotonic()
        self._anchor_time: float | None = None
        self._anchor_done = 0
        self._last_redraw = 0.0
        self._nonfinite_batches = 0

    def _write_line(self, text: str) -> None:
        if not self._enabled:
            return
        if self._pending_width:
            self._stream.write("\r" + " " * self._pending_width + "\r")
            self._pending_width = 0
        self._stream.write(text + "\n")
        self._stream.flush()

    def _write_transient(self, text: str) -> None:
        if not self._enabled:
            return
        if not self._is_tty:
            self._write_line(text)
            return
        width = shutil.get_terminal_size((120, 24)).columns
        line = text[: max(1, width - 1)]
        self._stream.write("\r" + line.ljust(self._pending_width))
        self._pending_width = len(line)
        self._stream.flush()

    def note(self, message: str) -> None:
        """Emit one standalone status line outside the batch stream."""
        self._write_line(f"[setup] {message}")

    def warn(self, message: str) -> None:
        """Emit one standalone warning line outside the batch stream."""
        self._write_line(f"[warn]  {message}")

    def run_start(
        self,
        *,
        config_path: Path,
        config: AppConfig,
        context: DistributedContext,
        optimizer: torch.optim.Optimizer,
        parameter_counts: tuple[int, int],
        amp_dtype: torch.dtype | None,
        scaler_enabled: bool,
        train_samples: int,
        train_batches: int,
        validation_samples: int | None,
        validation_batches: int | None,
        start_epoch: int,
        optimizer_steps_per_epoch: int,
    ) -> None:
        """Print the resolved run configuration before the first epoch."""
        train = config.train
        self._total_epochs = train.epochs
        self._device = context.device
        self._run_start = time.monotonic()
        total_parameters, trainable_parameters = parameter_counts
        patches_per_step = (
            train.source_images_per_gpu
            * train.patches_per_image
            * train.gradient_accumulation_steps
            * context.world_size
        )
        rows: list[tuple[str, str]] = [
            ("config", str(config_path)),
            ("output dir", str(train.output_dir)),
            ("device", self._describe_device(context)),
            ("precision", self._describe_precision(train, amp_dtype, scaler_enabled)),
            (
                "epochs",
                f"{train.epochs}"
                + (f" (resuming at {start_epoch})" if start_epoch else ""),
            ),
            ("train data", f"{train_samples} images, {train_batches} batches/epoch"),
        ]
        if validation_samples is not None and validation_batches is not None:
            rows.append(
                (
                    "val data",
                    f"{validation_samples} images, {validation_batches} batches",
                )
            )
        else:
            rows.append(("val data", "disabled"))
        rows.extend(
            [
                (
                    "batching",
                    f"{train.source_images_per_gpu} img/gpu x "
                    f"{train.patches_per_image} patches x "
                    f"{train.gradient_accumulation_steps} accum x "
                    f"{context.world_size} gpu = {patches_per_step} patches/step",
                ),
                (
                    "optimizer steps",
                    f"{optimizer_steps_per_epoch}/epoch, "
                    f"{optimizer_steps_per_epoch * train.epochs} total",
                ),
                (
                    "parameters",
                    f"{_format_parameters(total_parameters)} total, "
                    f"{_format_parameters(trainable_parameters)} trainable",
                ),
                ("ema decay", f"{train.ema_decay}"),
                (
                    "scheduler",
                    f"{train.scheduler.name}, warmup {train.scheduler.warmup_steps} "
                    f"steps, min lr ratio {train.scheduler.min_lr_ratio}",
                ),
                ("grad clip", f"{train.gradient_clip_norm}"),
                ("log interval", f"{self._log_interval or 'off'} batches"),
            ]
        )
        if train.freeze_local_encoder_epochs:
            rows.append(
                ("local freeze", f"first {train.freeze_local_encoder_epochs} epochs")
            )
        for source, checkpoint in (
            ("initial checkpoint", train.initial_checkpoint),
            ("resume checkpoint", train.resume_checkpoint),
        ):
            if checkpoint is not None:
                rows.append((source, str(checkpoint)))
        self._write_line(_HEADER_RULE)
        self._write_line("FloorPlan-GLV training")
        self._write_line(_HEADER_RULE)
        for label, value in rows:
            self._write_line(f"  {label:<17} {value}")
        for group in optimizer.param_groups:
            numel = sum(int(item.numel()) for item in group["params"])
            self._write_line(
                f"  {'lr.' + str(group.get('name', '?')):<17} "
                f"{float(group['lr']):.2e}  ({_format_parameters(numel)} params)"
            )
        self._write_line(_HEADER_RULE)

    def _describe_device(self, context: DistributedContext) -> str:
        device = context.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return f"{device} (world size {context.world_size})"
        properties = torch.cuda.get_device_properties(device)
        return (
            f"{device} {properties.name}, "
            f"{_format_bytes(float(properties.total_memory))}, "
            f"world size {context.world_size}"
        )

    @staticmethod
    def _describe_precision(
        train: object,
        amp_dtype: torch.dtype | None,
        scaler_enabled: bool,
    ) -> str:
        requested = str(getattr(train, "precision", "?"))
        if amp_dtype is None:
            return f"{requested} (autocast off)"
        name = str(amp_dtype).removeprefix("torch.")
        scaler = "grad scaler on" if scaler_enabled else "grad scaler off"
        return f"{requested} (autocast {name}, {scaler})"

    def epoch_start(
        self,
        *,
        epoch: int,
        phase: str,
        total_batches: int,
        local_encoder_frozen: bool = False,
    ) -> None:
        """Mark the beginning of a train or validation epoch."""
        self._phase_start = time.monotonic()
        self._anchor_time = None
        self._anchor_done = 0
        self._nonfinite_batches = 0
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device)
        frozen = " [local encoder frozen]" if local_encoder_frozen else ""
        self._write_line(
            f"[{phase}] epoch {epoch + 1}/{self._total_epochs} start, "
            f"{total_batches} batches{frozen}"
        )

    def batch_progress(self, progress: BatchProgress) -> None:
        """Report one completed batch (train or validation)."""
        if progress.batch_index == 0:
            self._anchor_time = time.monotonic()
            self._anchor_done = 0
        if not math.isfinite(progress.loss):
            self._nonfinite_batches += 1
        phase_label = "train" if progress.phase == TRAIN_PHASE else "val"
        parts: list[str] = [
            f"[{phase_label}]",
            f"epoch {progress.epoch + 1}/{self._total_epochs}",
            f"batch {progress.batch_index + 1}",
        ]
        if progress.total_batches is not None:
            parts[-1] += f"/{progress.total_batches}"
        parts.append(f"step {progress.global_step}")
        parts.append(f"loss {progress.running_loss:.4f}")
        if progress.learning_rate is not None:
            parts.append(f"lr {progress.learning_rate:.2e}")
        if progress.grad_norm is not None:
            parts.append(f"grad {progress.grad_norm:.3f}")
        if progress.skipped_step:
            parts.append("(skipped)")
        if self._device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self._device)
            parts.append(f"mem {_format_bytes(float(allocated))}")
        elapsed = time.monotonic() - self._phase_start
        if progress.total_batches is not None and self._anchor_time is not None:
            done_since_anchor = progress.batch_index + 1 - self._anchor_done
            if done_since_anchor > 0:
                elapsed_since = time.monotonic() - self._anchor_time
                per_batch = elapsed_since / done_since_anchor
                pending = progress.total_batches - progress.batch_index - 1
                parts.append(f"eta {format_duration(pending * per_batch)}")
        parts.append(f"elapsed {format_duration(elapsed)}")
        line = " ".join(parts)
        if (
            self._log_interval
            and (progress.batch_index + 1) % self._log_interval == 0
        ) or progress.batch_index + 1 == progress.total_batches:
            self._write_transient(line)

    def epoch_end(
        self,
        *,
        epoch: int,
        phase: str,
        metrics: Mapping[str, float],
        batch_count: int,
        optimizer_steps: int | None = None,
        skipped_steps: int = 0,
        is_best: bool = False,
    ) -> None:
        """Print aggregated metrics after a complete train or validation epoch."""
        elapsed = time.monotonic() - self._phase_start
        parts: list[str] = [
            f"[{phase}]",
            f"epoch {epoch + 1}/{self._total_epochs} done",
            f"loss {metrics.get('total', float('nan')):.6f}",
            f"{batch_count} batches",
        ]
        if optimizer_steps is not None:
            steps = f"{optimizer_steps} steps"
            if skipped_steps:
                steps += f" ({skipped_steps} skipped)"
            parts.append(steps)
        parts.append(format_duration(elapsed))
        if self._device.type == "cuda" and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(self._device)
            parts.append(f"peak mem {_format_bytes(float(peak))}")
        if is_best:
            parts.append("** best **")
        self._write_line(" | ".join(parts))
        for line in _wrap_metrics(metrics, prefix="raw/"):
            self._write_line(f"  {line}")
        if self._nonfinite_batches:
            self.warn(
                f"{self._nonfinite_batches} of {batch_count} batches had a "
                "non-finite loss this epoch"
            )
        empty = sorted(
            key.removeprefix("positive/")
            for key, value in metrics.items()
            if key.startswith("positive/") and value <= 0.0
        )
        if empty:
            self.warn("no positive targets this epoch: " + ", ".join(empty))

    def checkpoint_saved(self, path: Path, *, epoch: int, global_step: int) -> None:
        """Note that a checkpoint was written."""
        self._write_line(
            f"[ckpt] epoch {epoch + 1}/{self._total_epochs} step {global_step} "
            f"-> {path}"
        )

    def early_stopping_progress(
        self,
        *,
        monitor: str,
        value: float,
        best_value: float,
        best_epoch: int,
        bad_epochs: int,
        patience: int,
    ) -> None:
        """Report the latest early-stopping decision state."""
        self._write_line(
            f"[early-stop] {monitor}={value:.6f} best={best_value:.6f} "
            f"epoch={best_epoch + 1} no-improve={bad_epochs}/{patience}"
        )

    def early_stopping_triggered(self, *, epoch: int, best_epoch: int) -> None:
        """Report that early stopping ended the training loop."""
        self._write_line(
            f"[early-stop] stopping at epoch {epoch + 1}; "
            f"best epoch {best_epoch + 1}"
        )

    def run_end(
        self,
        *,
        selected_checkpoint: Path,
        last_checkpoint: Path,
        best_metrics: Mapping[str, float],
        stopped_early: bool,
    ) -> None:
        """Print the final run summary."""
        elapsed = time.monotonic() - self._run_start
        self._write_line(_HEADER_RULE)
        self._write_line(f"Training finished in {format_duration(elapsed)}")
        for name in sorted(best_metrics):
            self._write_line(f"  best {name:<17} {best_metrics[name]:.6f}")
        self._write_line(f"  selected checkpoint {selected_checkpoint}")
        self._write_line(f"  resume checkpoint   {last_checkpoint}")
        reason = "early stopping" if stopped_early else "epoch limit"
        self._write_line(f"  completion reason   {reason}")
        self._write_line(_HEADER_RULE)


def _wrap_metrics(
    metrics: Mapping[str, float],
    *,
    prefix: str,
    columns: int = 3,
    width: int = 26,
) -> list[str]:
    """Lay out one metric family as fixed-width columns for compact logs."""
    items = [
        f"{key.removeprefix(prefix)} {metrics[key]:.4f}"
        for key in sorted(metrics)
        if key.startswith(prefix)
    ]
    lines: list[str] = []
    for offset in range(0, len(items), columns):
        chunk = items[offset : offset + columns]
        lines.append("".join(item.ljust(width) for item in chunk).rstrip())
    return lines
