"""Image fingerprint utilities used to match an uploaded preview to a known ROI.

This is a faithful port of the legacy algorithm that produced the existing
manifest fingerprints (``Classification_legacy/src/utils/fingerprint.py``).
The legacy steps are:

1. Open the file as 8-bit grayscale ("L"), divide by 255 if max > 1.
2. Min-max normalise to ``[0, 1]`` (handling all-equal / non-positive cases).
3. Resize to the fingerprint size with ``scipy.ndimage.zoom`` order=1, then
   center-crop/pad in case the zoom output is off by a pixel.
4. Flatten as ``float32``.

The bridge expects ``load_fingerprint``, ``fingerprint_score`` and
``match_confidence``; those wrap the legacy helpers under their original
names so manifests produced by the legacy pipeline keep matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage


ArrayLike2D = Union[np.ndarray, Sequence[Sequence[float]]]


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

    if image.size == 0:
        raise ValueError("Empty image cannot be used to build a fingerprint.")

    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    elif max_value > 0.0:
        image = np.ones_like(image, dtype=np.float32)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    return image


def load_reference_image(image_path: Union[str, Path]) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if float(np.max(image)) > 1.0:
        image = image / 255.0
    return normalize_image(image)


def resize_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return image.astype(np.float32)

    zoom_factors = (
        target_shape[0] / max(image.shape[0], 1),
        target_shape[1] / max(image.shape[1], 1),
    )
    resized = ndimage.zoom(image, zoom_factors, order=1)

    output = np.zeros(target_shape, dtype=np.float32)
    src_h, src_w = resized.shape[:2]
    dst_h, dst_w = target_shape

    src_y0 = max((src_h - dst_h) // 2, 0)
    src_x0 = max((src_w - dst_w) // 2, 0)
    dst_y0 = max((dst_h - src_h) // 2, 0)
    dst_x0 = max((dst_w - src_w) // 2, 0)

    copy_h = min(src_h, dst_h)
    copy_w = min(src_w, dst_w)
    output[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = resized[
        src_y0:src_y0 + copy_h,
        src_x0:src_x0 + copy_w,
    ]
    return output


def compute_image_fingerprint(
    image_or_path: Union[str, Path, ArrayLike2D],
    fingerprint_size: Tuple[int, int] = (16, 16),
) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        image = load_reference_image(image_or_path)
    else:
        image = normalize_image(np.asarray(image_or_path, dtype=np.float32))

    resized = resize_image(image, fingerprint_size)
    return resized.reshape(-1).astype(np.float32)


def score_fingerprints(query: np.ndarray, reference: np.ndarray) -> float:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.shape != reference.shape:
        raise ValueError(
            f"Fingerprint shape mismatch: {query.shape} vs {reference.shape}"
        )
    diff = query - reference
    return float(np.mean(diff * diff))


# ---------------------------------------------------------------------------
# Public API used by the V2 bridge.
# ---------------------------------------------------------------------------


def load_fingerprint(image_path: str | Path, size: Sequence[int] = (16, 16)) -> np.ndarray:
    """Compute a fingerprint for the file at ``image_path``."""
    target = (int(size[0]), int(size[1]))
    return compute_image_fingerprint(image_path, fingerprint_size=target)


def fingerprint_score(query: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared error between two fingerprints (lower is better)."""
    return score_fingerprints(query, reference)


def match_confidence(score: float) -> float:
    """Map MSE score to a [0, 1] confidence; clipped linearly."""
    return float(np.clip(1.0 - score * 3.0, 0.0, 1.0))


__all__ = [
    "normalize_image",
    "load_reference_image",
    "resize_image",
    "compute_image_fingerprint",
    "score_fingerprints",
    "load_fingerprint",
    "fingerprint_score",
    "match_confidence",
]
