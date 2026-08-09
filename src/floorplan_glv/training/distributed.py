"""Small standard-PyTorch distributed helpers for single process and torchrun."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler

SampleT = TypeVar("SampleT")


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Resolved rank, world size, and device for one training process."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None
    initialized_here: bool

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    *,
    preferred_device: str | None = None,
) -> DistributedContext:
    """Initialize an env:// process group when launched by ``torchrun``."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise RuntimeError("invalid torchrun rank environment")
    if preferred_device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        backend = "gloo" if world_size > 1 else None
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if world_size > 1 else None
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        assert backend is not None
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        initialized_here = True
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
        initialized_here=initialized_here,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    """Synchronize and destroy a process group initialized by this module."""
    if context.is_distributed and dist.is_initialized():
        dist.barrier()
        if context.initialized_here:
            dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    """Synchronize ranks when distributed training is active."""
    if context.is_distributed and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(
    value: torch.Tensor,
    *,
    context: DistributedContext,
) -> torch.Tensor:
    """Return the world-size mean of a scalar or tensor value."""
    reduced = value.detach().clone().to(context.device)
    if context.is_distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced.div_(context.world_size)
    return reduced


def all_gather_objects(
    value: Any,
    *,
    context: DistributedContext,
) -> list[Any]:
    """Gather one checkpoint-safe object from every rank in rank order."""
    if not context.is_distributed:
        return [value]
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def distributed_sampler(
    dataset: Dataset[SampleT],
    *,
    context: DistributedContext,
    shuffle: bool,
    seed: int,
) -> DistributedSampler[SampleT] | None:
    """Build a standard DistributedSampler only when multiple ranks exist."""
    if not context.is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
    )


def wrap_ddp(model: nn.Module, context: DistributedContext) -> nn.Module:
    """Wrap ``model`` in standard DDP when multiple ranks are active."""
    if not context.is_distributed:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module from DDP or the input model itself."""
    if isinstance(model, DistributedDataParallel):
        return cast(nn.Module, model.module)
    return model
