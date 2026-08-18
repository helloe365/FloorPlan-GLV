"""Deterministic Stage 1 patch review visualizations."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import cv2
import numpy as np

HeadName = Literal["wall", "opening"]
WeightName = Literal["raw", "ema"]

TITLE_HEIGHT = 24
_WEIGHT_ORDER: tuple[WeightName, ...] = ("raw", "ema")
_HEAD_ORDER: tuple[HeadName, ...] = ("wall", "opening")
_TP_COLOR = (0, 180, 0)
_FP_COLOR = (220, 30, 30)
_FN_COLOR = (30, 80, 220)
_TN_COLOR = (0, 0, 0)
_INVALID_COLOR = (96, 96, 96)
_TARGET_COLOR = (255, 255, 255)
_TITLE_BACKGROUND = (32, 32, 32)
_TITLE_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True, slots=True)
class Stage1VisualizationInput:
    """One selected normalized patch and its raw/EMA predictions."""

    sample_id: str
    patch_index: int
    image_rgb: np.ndarray
    valid_mask: np.ndarray
    wall_target: np.ndarray
    opening_target: np.ndarray
    probabilities: Mapping[WeightName, Mapping[HeadName, np.ndarray]]


def visualization_filename(sample_id: str, patch_index: int) -> str:
    """Return a path-safe, deterministic filename for one selected patch."""
    if (
        isinstance(sample_id, bool)
        or not isinstance(sample_id, str)
        or isinstance(patch_index, bool)
        or not isinstance(patch_index, int)
        or patch_index < 0
    ):
        raise ValueError("sample_id must be text and patch_index must be non-negative")
    normalized = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)
    normalized = normalized or "sample"
    return f"{normalized}__patch-{patch_index:03d}.png"


def render_stage1_visualization(
    item: Stage1VisualizationInput,
    *,
    wall_threshold: float,
    opening_threshold: float,
) -> np.ndarray:
    """Render wall/opening image, target, probability, and error panels."""
    image = _validate_image(item.image_rgb)
    height, width = image.shape[:2]
    valid = _validate_mask(item.valid_mask, "valid_mask")
    wall_target = _validate_mask(item.wall_target, "wall_target")
    opening_target = _validate_mask(item.opening_target, "opening_target")
    weights = _validated_weights(item.probabilities)
    if not weights:
        raise ValueError("probabilities must contain at least one weight")

    panels_by_head: list[np.ndarray] = []
    for head, target, threshold in cast(
        tuple[tuple[HeadName, np.ndarray, float], ...],
        (
            ("wall", wall_target, wall_threshold),
            ("opening", opening_target, opening_threshold),
        ),
    ):
        image_panel = _title_panel(image, f"{head} image")
        target_panel = _title_panel(
            _mask_panel(target, valid, (height, width)), f"{head} target"
        )
        panels = [image_panel, target_panel]
        for weight in weights:
            probability = _resize_probability(
                item.probabilities[weight][head], (height, width)
            )
            panels.append(
                _title_panel(
                    _probability_panel(probability, valid, (height, width)),
                    f"{weight} {head} probability",
                )
            )
            panels.append(
                _title_panel(
                    _error_panel(
                        probability, target, valid, threshold, (height, width)
                    ),
                    f"{weight} {head} error",
                )
            )
        panels_by_head.append(np.hstack(panels))
    return np.ascontiguousarray(np.vstack(panels_by_head), dtype=np.uint8)


def write_stage1_visualizations(
    items: Sequence[Stage1VisualizationInput],
    output_dir: Path,
    *,
    wall_threshold: float,
    opening_threshold: float,
) -> tuple[Path, ...]:
    """Publish selected patch images as one transactional directory."""
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise _evaluation_error(f"visualization output is a symlink: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _evaluation_error(f"visualization output {output_dir}: {exc}") from exc
    destination = output_dir / "visualizations"
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise _evaluation_error(
            f"visualizations path is not a real directory: {destination}"
        )

    ordered = tuple(
        sorted(
            items,
            key=lambda item: visualization_filename(item.sample_id, item.patch_index),
        )
    )
    names = tuple(
        visualization_filename(item.sample_id, item.patch_index) for item in ordered
    )
    if len(names) != len(set(names)):
        raise _evaluation_error("visualization filenames are not unique")

    temporary = output_dir / f".visualizations.{uuid4().hex}.tmp"
    backup: Path | None = None
    try:
        temporary.mkdir()
        for item, name in zip(ordered, names, strict=True):
            try:
                rendered = render_stage1_visualization(
                    item,
                    wall_threshold=wall_threshold,
                    opening_threshold=opening_threshold,
                )
                encoded = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(
                    str(temporary / name),
                    encoded,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                ):
                    raise _evaluation_error(f"failed to write visualization {name}")
            except RuntimeError:
                raise
            except Exception as exc:
                raise _evaluation_error(
                    f"failed to write visualization {name}: {exc}"
                ) from exc

        if destination.exists():
            backup = output_dir / f".visualizations.{uuid4().hex}.bak"
            _replace(destination, backup)
        try:
            _replace(temporary, destination)
        except Exception as exc:
            if backup is not None and backup.exists() and not destination.exists():
                try:
                    _replace(backup, destination)
                    backup = None
                except Exception as restore_exc:
                    raise _evaluation_error(
                        "failed to publish visualizations and restore backup: "
                        f"{restore_exc}"
                    ) from exc
            raise _evaluation_error(f"failed to publish visualizations: {exc}") from exc
        temporary = Path()
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        return tuple(destination / name for name in names)
    except BaseException:
        _cleanup_directory(temporary)
        if backup is not None and backup.exists() and not destination.exists():
            with suppress(OSError):
                _replace(backup, destination)
        raise
    finally:
        if backup is not None:
            _cleanup_directory(backup)


def _replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _cleanup_directory(path: Path) -> None:
    if not str(path) or path == Path(".") or path.is_symlink():
        return
    if path.is_dir():
        shutil.rmtree(path)


def _evaluation_error(message: str) -> RuntimeError:
    from floorplan_glv.evaluation.stage1 import Stage1EvaluationError

    return Stage1EvaluationError(message)


def _validate_image(image: np.ndarray) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise ValueError("image_rgb must be uint8 RGB [height, width, 3]")
    return np.ascontiguousarray(image)


def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != bool:
        raise ValueError(f"{name} must be a boolean [height, width] array")
    return mask


def _validated_weights(
    probabilities: Mapping[WeightName, Mapping[HeadName, np.ndarray]],
) -> tuple[WeightName, ...]:
    if not isinstance(probabilities, Mapping):
        raise ValueError("probabilities must be a mapping")
    unknown = set(probabilities) - set(_WEIGHT_ORDER)
    if unknown:
        raise ValueError(f"unknown probability weights: {sorted(unknown)}")
    result: list[WeightName] = []
    for weight in _WEIGHT_ORDER:
        if weight not in probabilities:
            continue
        heads = probabilities[weight]
        if not isinstance(heads, Mapping) or set(heads) != set(_HEAD_ORDER):
            raise ValueError(f"{weight} probabilities must contain wall and opening")
        for head in _HEAD_ORDER:
            value = heads[head]
            if (
                not isinstance(value, np.ndarray)
                or value.ndim != 2
                or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"{weight} {head} probabilities must be finite 2-D floats"
                )
        result.append(weight)
    return tuple(result)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def _resize_probability(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if probability.shape == shape:
        return np.clip(probability.astype(np.float32, copy=False), 0.0, 1.0)
    resized = cv2.resize(
        probability.astype(np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0)


def _mask_panel(
    mask: np.ndarray, valid: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    resized_mask = _resize_mask(mask, shape)
    resized_valid = _resize_mask(valid, shape)
    panel = np.zeros((*shape, 3), dtype=np.uint8)
    panel[resized_mask] = _TARGET_COLOR
    panel[~resized_valid] = _INVALID_COLOR
    return panel


def _probability_panel(
    probability: np.ndarray,
    valid: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    heatmap = np.rint(probability * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    panel = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    panel[~_resize_mask(valid, shape)] = _INVALID_COLOR
    return panel


def _error_panel(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    shape: tuple[int, int],
) -> np.ndarray:
    valid_resized = _resize_mask(valid, shape)
    target_resized = _resize_mask(target, shape)
    predicted = probability >= threshold
    panel = np.empty((*shape, 3), dtype=np.uint8)
    panel[:] = _TN_COLOR
    panel[predicted & target_resized & valid_resized] = _TP_COLOR
    panel[predicted & ~target_resized & valid_resized] = _FP_COLOR
    panel[~predicted & target_resized & valid_resized] = _FN_COLOR
    panel[~valid_resized] = _INVALID_COLOR
    return panel


def _title_panel(panel: np.ndarray, title: str) -> np.ndarray:
    title_strip = np.full(
        (TITLE_HEIGHT, panel.shape[1], 3), _TITLE_BACKGROUND, dtype=np.uint8
    )
    cv2.putText(
        title_strip,
        title,
        (3, 17),
        _FONT,
        0.32,
        _TITLE_COLOR,
        1,
        cv2.LINE_8,
    )
    return np.vstack((title_strip, panel))


__all__ = [
    "TITLE_HEIGHT",
    "HeadName",
    "Stage1VisualizationInput",
    "render_stage1_visualization",
    "visualization_filename",
    "write_stage1_visualizations",
]
