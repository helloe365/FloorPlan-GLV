"""Exponential moving averages for model parameters and buffers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TypedDict, cast

import torch
from torch import nn


class EMAState(TypedDict):
    """Serializable EMA state stored inside training checkpoints."""

    decay: float
    num_updates: int
    shadow: dict[str, torch.Tensor]


class ExponentialMovingAverage:
    """Track a detached exponential moving average of a model state."""

    def __init__(self, model: nn.Module, *, decay: float) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be in [0, 1]")
        self.decay = float(decay)
        self.num_updates = 0
        self._shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update floating tensors and directly copy non-floating buffers."""
        current = model.state_dict()
        if set(current) != set(self._shadow):
            raise ValueError("EMA model state keys do not match")
        for name, value in current.items():
            shadow = self._shadow[name]
            if shadow.shape != value.shape or shadow.dtype != value.dtype:
                raise ValueError(f"EMA tensor contract changed for {name}")
            source = value.detach().to(device=shadow.device)
            if shadow.is_floating_point() or shadow.is_complex():
                shadow.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                shadow.copy_(source)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy the averaged state into ``model`` strictly."""
        model.load_state_dict(self._shadow, strict=True)

    def _validate_model_state(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        """Return a model state only when it matches the EMA tensor contract."""
        current = model.state_dict()
        if set(current) != set(self._shadow):
            raise ValueError("EMA model state keys do not match")
        seen_storages: dict[tuple[torch.device, int], str] = {}
        for name, value in current.items():
            shadow = self._shadow[name]
            if shadow.shape != value.shape or shadow.dtype != value.dtype:
                raise ValueError(f"EMA tensor contract changed for {name}")
            if value.numel() == 0:
                continue
            storage_key = (value.device, value.untyped_storage().data_ptr())
            aliased_name = seen_storages.get(storage_key)
            if aliased_name is not None:
                raise ValueError(
                    f"EMA model state tensors {aliased_name} and {name} are aliased"
                )
            seen_storages[storage_key] = name
        return current

    @torch.no_grad()
    def _swap_with_model(self, model: nn.Module) -> None:
        """Swap EMA and model tensors with one temporary tensor at a time."""
        current = self._validate_model_state(model)
        swapped: list[str] = []
        try:
            for name, value in current.items():
                shadow = self._shadow[name]
                temporary = value.detach().clone()
                copied_to_model = False
                try:
                    value.copy_(shadow)
                    copied_to_model = True
                    shadow.copy_(temporary)
                except BaseException:
                    if copied_to_model:
                        shadow.copy_(value)
                        value.copy_(temporary)
                    raise
                finally:
                    del temporary
                swapped.append(name)
        except BaseException:
            for name in reversed(swapped):
                value = current[name]
                shadow = self._shadow[name]
                temporary = value.detach().clone()
                try:
                    value.copy_(shadow)
                    shadow.copy_(temporary)
                finally:
                    del temporary
            raise

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily expose EMA tensors on ``model`` for evaluation."""
        self._swap_with_model(model)
        try:
            yield
        finally:
            self._swap_with_model(model)

    def state_dict(self) -> EMAState:
        """Return a detached checkpoint-safe EMA state."""
        return EMAState(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow={
                name: value.detach().clone() for name, value in self._shadow.items()
            },
        )

    def load_state_dict(
        self,
        state: Mapping[str, object],
        *,
        strict: bool = True,
    ) -> None:
        """Restore decay, update count, and shadow tensors."""
        required = {"decay", "num_updates", "shadow"}
        if strict and set(state) != required:
            raise ValueError(
                f"EMA state keys differ; missing={sorted(required - set(state))}, "
                f"extra={sorted(set(state) - required)}"
            )
        try:
            decay = float(cast(float, state["decay"]))
            num_updates = int(cast(int, state["num_updates"]))
            raw_shadow = cast(Mapping[str, object], state["shadow"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid EMA state: {exc}") from exc
        if not 0.0 <= decay <= 1.0 or num_updates < 0:
            raise ValueError("invalid EMA decay or update count")
        if strict and set(raw_shadow) != set(self._shadow):
            raise ValueError("EMA shadow keys do not match the model")
        restored: dict[str, torch.Tensor] = {}
        for name, current in self._shadow.items():
            value = raw_shadow.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"EMA shadow {name} is not a tensor")
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f"EMA tensor contract changed for {name}")
            restored[name] = value.detach().clone().to(device=current.device)
        self.decay = decay
        self.num_updates = num_updates
        self._shadow = restored
