"""Resumable early stopping for EMA validation losses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from floorplan_glv.config.models import EarlyStoppingConfig


class EarlyStoppingState(TypedDict):
    """Serializable state required to resume an early-stopping policy."""

    best_value: float | None
    best_epoch: int | None
    bad_epochs: int


@dataclass(frozen=True, slots=True)
class EarlyStoppingDecision:
    """The result of processing one EMA validation loss."""

    value: float
    best_value: float
    best_epoch: int
    bad_epochs: int
    improved: bool
    should_stop: bool


class EarlyStopping:
    """Stop training after eligible EMA-loss plateaus."""

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    @property
    def should_stop(self) -> bool:
        """Whether the configured plateau patience has been exhausted."""
        return self.bad_epochs >= self.config.patience

    def update(self, *, epoch: int, value: float) -> EarlyStoppingDecision:
        """Record one validation loss and return its stopping decision."""
        if self.should_stop:
            raise RuntimeError("early stopping has already triggered")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("early-stopping epoch must be a non-negative integer")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                "early-stopping value must be a finite non-negative number"
            )

        observed_value = float(value)
        if self.best_value is None:
            improved = True
        elif self.best_value == 0.0:
            improved = False
        else:
            threshold = self.best_value * (1.0 - self.config.min_delta_relative)
            improved = observed_value <= threshold

        if improved:
            self.best_value = observed_value
            self.best_epoch = epoch
            self.bad_epochs = 0
        elif epoch + 1 > self.config.min_epochs:
            self.bad_epochs += 1

        best_value = self.best_value
        best_epoch = self.best_epoch
        if best_value is None or best_epoch is None:
            raise RuntimeError("early-stopping policy has no best observation")
        return EarlyStoppingDecision(
            value=observed_value,
            best_value=best_value,
            best_epoch=best_epoch,
            bad_epochs=self.bad_epochs,
            improved=improved,
            should_stop=self.should_stop,
        )

    def state_dict(self) -> EarlyStoppingState:
        """Return the complete state required to resume this policy."""
        return EarlyStoppingState(
            best_value=self.best_value,
            best_epoch=self.best_epoch,
            bad_epochs=self.bad_epochs,
        )

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Strictly and atomically restore a serialized policy state."""
        required = {"best_value", "best_epoch", "bad_epochs"}
        if set(state) != required:
            raise ValueError("early-stopping state keys differ")

        raw_best_value = state["best_value"]
        raw_best_epoch = state["best_epoch"]
        raw_bad_epochs = state["bad_epochs"]
        if (
            isinstance(raw_bad_epochs, bool)
            or not isinstance(raw_bad_epochs, int)
            or raw_bad_epochs < 0
            or raw_bad_epochs > self.config.patience
        ):
            raise ValueError("early-stopping bad_epochs is invalid")

        if raw_best_value is None or raw_best_epoch is None:
            if not (
                raw_best_value is None
                and raw_best_epoch is None
                and raw_bad_epochs == 0
            ):
                raise ValueError("early-stopping best value and epoch are unpaired")
            best_value: float | None = None
            best_epoch: int | None = None
        else:
            if (
                isinstance(raw_best_value, bool)
                or not isinstance(raw_best_value, (int, float))
                or not math.isfinite(raw_best_value)
                or raw_best_value < 0.0
            ):
                raise ValueError("early-stopping best_value is invalid")
            if (
                isinstance(raw_best_epoch, bool)
                or not isinstance(raw_best_epoch, int)
                or raw_best_epoch < 0
            ):
                raise ValueError("early-stopping best_epoch is invalid")
            best_value = float(raw_best_value)
            best_epoch = raw_best_epoch

        self.best_value = best_value
        self.best_epoch = best_epoch
        self.bad_epochs = raw_bad_epochs
