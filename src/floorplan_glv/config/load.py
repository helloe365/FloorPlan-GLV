"""YAML configuration loading with domain-specific errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from floorplan_glv.config.models import AppConfig, ConfigurationError


def load_config(path: Path) -> AppConfig:
    """Load and validate an application YAML file.

    Args:
        path: UTF-8 YAML file containing an application configuration mapping.

    Returns:
        A fully resolved immutable application configuration.

    Raises:
        ConfigurationError: If the file cannot be read, parsed, or validated.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"invalid YAML in configuration {path}: {exc}"
        ) from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration {path} root must be a mapping")

    try:
        return AppConfig.model_validate(_string_key_mapping(raw))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration {path}: {exc}") from exc


def _string_key_mapping(raw: dict[Any, Any]) -> dict[str, Any]:
    """Reject non-string root keys before Pydantic validation."""
    if not all(isinstance(key, str) for key in raw):
        raise ConfigurationError("configuration root keys must be strings")
    return raw
