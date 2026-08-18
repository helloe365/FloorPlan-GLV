"""Atomic strict checkpoints and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from floorplan_glv.models.encoders import CheckpointError
from floorplan_glv.training.ema import ExponentialMovingAverage

CHECKPOINT_SCHEMA_VERSION = "2.0.0"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "1.0.0"
LEGACY_CHECKPOINT_KEYS = frozenset({
    "schema_version",
    "model_state",
    "ema_state",
    "optimizer_state",
    "scheduler_state",
    "grad_scaler_state",
    "epoch",
    "global_step",
    "best_metrics",
    "resolved_config",
    "run_metadata",
})
CHECKPOINT_KEYS_BY_VERSION = {
    LEGACY_CHECKPOINT_SCHEMA_VERSION: LEGACY_CHECKPOINT_KEYS,
    CHECKPOINT_SCHEMA_VERSION: LEGACY_CHECKPOINT_KEYS | {"early_stopping_state"},
}
RNG_STATE_KEY = "_rng_state_by_rank"


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Non-module state restored from a strict training checkpoint."""

    schema_version: str
    epoch: int
    global_step: int
    best_metrics: dict[str, float]
    resolved_config: dict[str, Any]
    run_metadata: dict[str, Any]
    early_stopping_state: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    """Checkpoint state needed to validate a paired training artifact."""

    schema_version: str
    epoch: int
    early_stopping_state: dict[str, object] | None


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_metrics: Mapping[str, float],
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    early_stopping_state: Mapping[str, object] | None,
    rng_states: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Atomically save all state required for exact training resume."""
    if epoch < 0 or global_step < 0:
        raise CheckpointError("checkpoint epoch and global_step must be non-negative")
    checkpoint_metadata = dict(run_metadata)
    checkpoint_metadata[RNG_STATE_KEY] = list(
        rng_states if rng_states is not None else (capture_rng_state(),)
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": dict(best_metrics),
        "resolved_config": dict(resolved_config),
        "run_metadata": checkpoint_metadata,
        "early_stopping_state": dict(early_stopping_state)
        if early_stopping_state is not None
        else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"failed to save checkpoint {path}: {exc}") from exc
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    map_location: str | torch.device,
    rank: int = 0,
) -> ResumeState:
    """Strictly restore every state needed to continue training."""
    raw = _read_checkpoint(path, map_location=map_location)
    try:
        model.load_state_dict(cast(Mapping[str, torch.Tensor], raw["model_state"]))
        ema.load_state_dict(cast(Mapping[str, object], raw["ema_state"]))
        optimizer.load_state_dict(cast(dict[str, Any], raw["optimizer_state"]))
        scheduler.load_state_dict(cast(dict[str, Any], raw["scheduler_state"]))
        scaler.load_state_dict(cast(dict[str, Any], raw["grad_scaler_state"]))
        schema_version = cast(str, raw["schema_version"])
        early_stopping_state = _copy_early_stopping_state(raw)
        epoch = int(raw["epoch"])
        global_step = int(raw["global_step"])
        best_metrics = {}
        for key, value in cast(Mapping[str, object], raw["best_metrics"]).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"best metric {key} is not numeric")
            best_metrics[str(key)] = float(value)
        resolved_config = dict(cast(Mapping[str, Any], raw["resolved_config"]))
        run_metadata = dict(cast(Mapping[str, Any], raw["run_metadata"]))
        rng_states = cast(Sequence[Mapping[str, Any]], run_metadata.pop(RNG_STATE_KEY))
        if not 0 <= rank < len(rng_states):
            raise ValueError(f"checkpoint has no RNG state for rank {rank}")
        restore_rng_state(rng_states[rank])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointError(
            f"checkpoint {path} contains invalid state: {exc}"
        ) from exc
    if epoch < 0 or global_step < 0:
        raise CheckpointError(f"checkpoint {path} has negative progress state")
    return ResumeState(
        schema_version=schema_version,
        epoch=epoch,
        global_step=global_step,
        best_metrics=best_metrics,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
        early_stopping_state=early_stopping_state,
    )


def read_checkpoint_summary(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointSummary:
    """Read the progress and policy state needed for artifact pairing checks."""
    raw = _read_checkpoint(path, map_location=map_location)
    try:
        schema_version = cast(str, raw["schema_version"])
        epoch = int(raw["epoch"])
        early_stopping_state = _copy_early_stopping_state(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            f"checkpoint {path} contains invalid summary state: {exc}"
        ) from exc
    if epoch < 0:
        raise CheckpointError(f"checkpoint {path} has negative epoch")
    return CheckpointSummary(
        schema_version=schema_version,
        epoch=epoch,
        early_stopping_state=early_stopping_state,
    )


def load_checkpoint_weights(
    path: Path,
    *,
    model: nn.Module,
    map_location: str | torch.device,
    state: Literal["ema", "model"] = "ema",
    source_prefix: str = "",
    destination_prefix: str = "",
) -> None:
    """Initialize matching model tensors from a selected checkpoint state.

    Source tensors are validated as a complete candidate state before any
    destination model tensor is modified. Destination-only tensors retain
    their existing initialization for staged architecture expansion.
    """
    raw = _read_checkpoint(path, map_location=map_location)
    state_label = str(state)
    try:
        if state == "ema":
            ema_state = raw["ema_state"]
            if not isinstance(ema_state, Mapping):
                raise TypeError("EMA state must be a mapping")
            selected_state = ema_state["shadow"]
            state_label = "EMA"
        elif state == "model":
            selected_state = raw["model_state"]
            state_label = "model"
        else:
            raise ValueError(f"unsupported initialization state {state!r}")
        if not isinstance(selected_state, Mapping):
            raise TypeError(f"{state_label} state must be a mapping")

        current = model.state_dict()
        restored = dict(current)
        matched = 0
        for source_name, value in selected_state.items():
            if not isinstance(source_name, str):
                raise TypeError(f"{state_label} tensor name must be a string")
            if not source_name.startswith(source_prefix):
                continue
            destination_name = destination_prefix + source_name[len(source_prefix) :]
            if destination_name not in current:
                raise KeyError(
                    f"{state_label} tensor {source_name} has no destination "
                    f"{destination_name}"
                )
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{state_label} tensor {source_name} is not a tensor")
            destination = current[destination_name]
            if value.shape != destination.shape or value.dtype != destination.dtype:
                raise ValueError(
                    f"{state_label} tensor contract differs for {destination_name}"
                )
            if value.layout != destination.layout:
                raise ValueError(
                    f"{state_label} tensor layout differs for {destination_name}"
                )
            restored[destination_name] = value.detach().to(device=destination.device)
            matched += 1
        if matched == 0:
            raise ValueError(f"no {state_label} tensors matched the destination model")
        model.load_state_dict(restored, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
        error_label = "EMA" if state_label == "EMA" else repr(state_label)
        raise CheckpointError(
            f"checkpoint {path} contains invalid {error_label} initialization "
            f"state: {exc}"
        ) from exc


def load_ema_weights(
    path: Path,
    *,
    model: nn.Module,
    map_location: str | torch.device,
    source_prefix: str = "",
    destination_prefix: str = "",
) -> None:
    """Initialize matching model tensors from a prior stage's EMA state."""
    load_checkpoint_weights(
        path,
        model=model,
        map_location=map_location,
        state="ema",
        source_prefix=source_prefix,
        destination_prefix=destination_prefix,
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, torch CPU, and all CUDA RNG streams."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`."""
    random.setstate(cast(tuple[Any, ...], state["python"]))
    np.random.set_state(cast(tuple[Any, ...], state["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, state["torch_cpu"]).cpu())
    cuda_states = cast(list[torch.Tensor], state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointError(f"failed to hash file {path}: {exc}") from exc
    return digest.hexdigest()


def build_run_metadata(
    *,
    seed: int,
    dataset_indexes: Sequence[Path],
    checkpoint: Path | None,
    postprocess_config: Mapping[str, Any],
    deterministic_algorithms: bool,
) -> dict[str, Any]:
    """Collect the reproducibility metadata required for a training run."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    dataset_hashes = {str(path): file_sha256(path) for path in dataset_indexes}
    canonical_postprocess = json.dumps(
        postprocess_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(device_index),
            "device_index": device_index,
            "count": torch.cuda.device_count(),
        }
    return {
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
        "random_seed": seed,
        "deterministic_algorithms": deterministic_algorithms,
        "dataset_index_hashes": dataset_hashes,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "postprocess_config_sha256": hashlib.sha256(canonical_postprocess).hexdigest(),
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a deterministic UTF-8 JSON mapping."""
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def atomic_write_text(path: Path, value: str) -> None:
    """Atomically write UTF-8 text and fsync before replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _copy_early_stopping_state(
    raw: Mapping[str, Any],
) -> dict[str, object] | None:
    state = raw.get("early_stopping_state")
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise TypeError("early_stopping_state must be a mapping or null")
    return dict(cast(Mapping[str, object], state))


def _read_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CheckpointError(f"checkpoint {path} root must be a mapping")
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str):
        raise CheckpointError(f"checkpoint {path} schema_version must be a string")
    expected_keys = CHECKPOINT_KEYS_BY_VERSION.get(schema_version)
    if expected_keys is None:
        raise CheckpointError(
            f"checkpoint {path} has unsupported schema version {schema_version!r}; "
            f"supported versions are {sorted(CHECKPOINT_KEYS_BY_VERSION)}"
        )
    missing = expected_keys - set(raw)
    extra = set(raw) - expected_keys
    if missing or extra:
        raise CheckpointError(
            f"checkpoint {path} keys differ; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return cast(dict[str, Any], raw)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
