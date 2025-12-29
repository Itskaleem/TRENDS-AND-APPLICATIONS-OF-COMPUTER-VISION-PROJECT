from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def crop_divisible(img: np.ndarray, divisor: int = 4) -> np.ndarray:
    """Crop an image so its height/width are divisible by ``divisor``.

    Discarded pixels are taken from the image end; this is intended for small
    divisors used in stride-based feature extraction.
    """
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")

    height, width = img.shape[:2]
    rem_h = height % divisor
    rem_w = width % divisor
    return img[: height - rem_h, : width - rem_w]


def cv_centercrop(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Return a center crop of ``size`` from the given image."""
    crop_h, crop_w = size
    height, width = img.shape[:2]
    if crop_h > height or crop_w > width:
        raise ValueError(f"Crop size {size} larger than image {img.shape[:2]}")

    center_h = height // 2
    center_w = width // 2
    half_h = crop_h // 2
    half_w = crop_w // 2
    return img[center_h - half_h : center_h + half_h, center_w - half_w : center_w + half_w]


def cv_randomcrops(
    img: np.ndarray,
    size: Tuple[int, int] = (224, 224),
    num: int = 5,
    rng: np.random.Generator | None = None,
) -> Iterable[np.ndarray]:
    """Return ``num`` random crops of ``size`` within the image."""
    crop_h, crop_w = size
    height, width = img.shape[:2]
    if crop_h > height or crop_w > width:
        raise ValueError(f"Crop size {size} larger than image {img.shape[:2]}")
    if num <= 0:
        raise ValueError("num must be positive")

    rng = rng or np.random.default_rng()
    max_h = height - crop_h
    max_w = width - crop_w
    rand_h = rng.integers(0, max_h + 1, size=num)
    rand_w = rng.integers(0, max_w + 1, size=num)
    return [img[h : h + crop_h, w : w + crop_w] for h, w in zip(rand_h, rand_w)]
