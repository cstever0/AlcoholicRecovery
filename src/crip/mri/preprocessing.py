"""Cardiac MRI Preprocessing and Spatial Restoration Pipeline.

Handles intensity normalization, slice resizing/padding, and exact spatial
mask restoration for automated 2D U-Net segmentation.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("crip.mri.preprocessing")

DEFAULT_TARGET_SPATIAL_SIZE: Tuple[int, int] = (224, 224)


def normalize_intensity(
    image: np.ndarray,
    p_min: float = 1.0,
    p_max: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Robust percentile-based intensity normalization to [0.0, 1.0].

    Clips intense outliers (e.g., chest fat, blood pool spikes) to the given
    percentiles before scaling to [0, 1].

    Args:
        image: 2D or 3D numeric array.
        p_min: Lower percentile cutoff (default: 1.0).
        p_max: Upper percentile cutoff (default: 99.0).
        eps: Small constant to avoid division by zero.

    Returns:
        Float32 array scaled to [0.0, 1.0].
    """
    img_float = np.asarray(image, dtype=np.float32)
    if img_float.size == 0:
        return img_float

    v_min = float(np.percentile(img_float, p_min))
    v_max = float(np.percentile(img_float, p_max))

    if v_max - v_min < eps:
        # Fallback to min-max if percentiles collapse (e.g., uniform region)
        v_min = float(img_float.min())
        v_max = float(img_float.max())
        if v_max - v_min < eps:
            return np.zeros_like(img_float, dtype=np.float32)

    clipped = np.clip(img_float, v_min, v_max)
    normalized = (clipped - v_min) / (v_max - v_min + eps)
    return normalized.astype(np.float32)


def resize_slice(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
    is_mask: bool = False,
) -> np.ndarray:
    """Resize a 2D slice to target dimensions.

    Uses bilinear interpolation for intensity images and strict nearest-neighbor
    interpolation for segmentation masks to preserve discrete label identities.

    Args:
        image: 2D array of shape (H, W).
        target_size: Desired (target_H, target_W), default (224, 224).
        is_mask: If True, uses nearest neighbor interpolation; else bilinear.

    Returns:
        Resized 2D array of shape target_size.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array for slice resize, got shape {image.shape}")

    orig_h, orig_w = image.shape
    target_h, target_w = target_size

    if (orig_h, orig_w) == (target_h, target_w):
        return image.copy()

    # Convert to PyTorch tensor [1, 1, H, W] for fast, deterministic interpolation
    tensor = torch.from_numpy(image.astype(np.float32 if not is_mask else np.int64))
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    if is_mask:
        # Nearest neighbor interpolation for discrete integer labels
        resized = F.interpolate(
            tensor.float(),
            size=(target_h, target_w),
            mode="nearest",
        )
        return resized.squeeze(0).squeeze(0).to(torch.int64).numpy()
    else:
        resized = F.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def restore_mask_to_original_shape(
    pred_mask: np.ndarray,
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Restore a predicted 2D mask from target size back to original patient slice shape.

    Uses strict nearest-neighbor interpolation to ensure label boundaries are
    integer-valued (0, 1, 2, 3) without fractional blending.

    Args:
        pred_mask: 2D predicted mask array of shape (target_H, target_W).
        original_shape: Original patient acquisition shape (H, W).

    Returns:
        2D integer array of shape original_shape with values in {0, 1, 2, 3}.
    """
    if pred_mask.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {pred_mask.shape}")

    orig_h, orig_w = original_shape
    if pred_mask.shape == (orig_h, orig_w):
        return pred_mask.astype(np.int64)

    return resize_slice(pred_mask, target_size=(orig_h, orig_w), is_mask=True)


def restore_volume_mask(
    pred_volume: np.ndarray,
    original_volume_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Restore a stack of 2D predicted masks back to the original 3D volume dimensions.

    Args:
        pred_volume: 3D array of shape (num_slices, target_H, target_W).
        original_volume_shape: Target 3D shape (num_slices, orig_H, orig_W).

    Returns:
        3D integer array of shape original_volume_shape.
    """
    num_slices, orig_h, orig_w = original_volume_shape
    if pred_volume.shape[0] != num_slices:
        raise ValueError(
            f"Slice count mismatch: pred has {pred_volume.shape[0]}, original has {num_slices}"
        )

    restored = np.zeros(original_volume_shape, dtype=np.int64)
    for s in range(num_slices):
        restored[s] = restore_mask_to_original_shape(
            pred_volume[s],
            original_shape=(orig_h, orig_w),
        )
    return restored
