"""Stochastic training-time augmentation policy.

Composes the existing :class:`GeometricTransform` primitives (D4 orientation
plus optional uniform scale) into a single affine so image pixels and vector
annotations stay synchronized via :func:`apply_transform`. Photometric ops
(brightness, contrast, gamma) act on the image only and leave annotations
untouched; they are disabled by default (see :class:`AugmentationConfig`).

Flip and rotation are independently gated. When a rotation fires its
quarter-turn count is uniform over ``{1, 2, 3}``, so ``rotation_prob=0.75``
yields the equiprobable D4 group and ``rotation_prob=0.0`` disables rotation.

The returned callable consumes a :class:`numpy.random.Generator` and never
touches global RNG state, preserving the dataset's per-sample
``SeedSequence((seed, epoch, index))`` reproducibility contract. The same
generator is later advanced into the patch sampler downstream, so enabling
augmentation shifts patch-sampling streams relative to the un-augmented
baseline — intentional, but worth noting when comparing runs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

from floorplan_glv.config.models import AugmentationConfig
from floorplan_glv.data.annotation_schema import TrainingAnnotation
from floorplan_glv.data.dataset import Augmentation
from floorplan_glv.geometry.transforms import (
    GeometricTransform,
    Matrix,
    apply_transform,
)


def build_augmentation(config: AugmentationConfig) -> Augmentation | None:
    """Build a stochastic augmentation callable from ``config``.

    Returns ``None`` when ``config.enabled`` is ``False`` so callers can pass
    the result directly to :class:`FloorPlanDataset` without a conditional.
    """
    if not config.enabled:
        return None

    def augment(
        image: Image.Image,
        annotation: TrainingAnnotation,
        rng: np.random.Generator,
    ) -> tuple[Image.Image, TrainingAnnotation]:
        image, annotation = _apply_geometric(image, annotation, rng, config)
        image = _apply_photometric(image, rng, config)
        return image, annotation

    return augment


def _apply_geometric(
    image: Image.Image,
    annotation: TrainingAnnotation,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> tuple[Image.Image, TrainingAnnotation]:
    flip_h = bool(rng.random() < config.horizontal_flip_prob)
    apply_rotation = bool(rng.random() < config.rotation_prob)
    rot_k = int(rng.integers(1, 4)) if apply_rotation else 0
    apply_scale = bool(rng.random() < config.scale_prob)
    scale_factor = (
        float(rng.uniform(config.scale_min, config.scale_max))
        if apply_scale
        else 1.0
    )

    if not flip_h and rot_k == 0 and scale_factor == 1.0:
        return image, annotation

    image_size = (annotation.image.width, annotation.image.height)
    current_size = image_size
    ops: list[GeometricTransform] = []

    if flip_h:
        op = GeometricTransform.horizontal_flip(current_size)
        ops.append(op)
        current_size = op.output_size

    for _ in range(rot_k):
        op = GeometricTransform.rotate90_clockwise(current_size)
        ops.append(op)
        current_size = op.output_size

    if apply_scale:
        op = GeometricTransform.uniform_scale(current_size, scale_factor)
        ops.append(op)
        current_size = op.output_size

    composed_matrix = ops[0].matrix
    for op in ops[1:]:
        composed_matrix = _matmul_3x3(op.matrix, composed_matrix)

    transform = GeometricTransform(
        matrix=composed_matrix,
        input_size=image_size,
        output_size=current_size,
        minimum_retained_area_ratio=config.minimum_retained_area_ratio,
    )
    return apply_transform(image, annotation, transform)


def _apply_photometric(
    image: Image.Image,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> Image.Image:
    if rng.random() < config.brightness_prob:
        factor = float(rng.uniform(*config.brightness_range))
        image = ImageEnhance.Brightness(image).enhance(factor)
    if rng.random() < config.contrast_prob:
        factor = float(rng.uniform(*config.contrast_range))
        image = ImageEnhance.Contrast(image).enhance(factor)
    if rng.random() < config.gamma_prob:
        gamma = float(rng.uniform(*config.gamma_range))
        image = _apply_gamma(image, gamma)
    return image


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    """Apply a gamma correction via a per-channel lookup table."""
    channels = len(image.getbands())
    single = tuple(round((value / 255.0) ** gamma * 255.0) for value in range(256))
    return image.point(single * channels)


def _matmul_3x3(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two 3x3 affine matrices, returning ``a @ b``."""
    return tuple(
        tuple(
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
            for j in range(3)
        )
        for i in range(3)
    )
