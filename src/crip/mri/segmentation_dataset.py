"""PyTorch Dataset for Cardiac MRI Short-Axis Slice Segmentation.

Extracts 2D slices from 3D cine volumes (ED and ES frames), applies
standardized preprocessing and spatial normalization, and provides
deterministic and augmented slice streams.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from crip.mri.acdc_loader import ACDCDataLoader, DEFAULT_ACDC_TRAINING_DIR
from crip.mri.preprocessing import (
    DEFAULT_TARGET_SPATIAL_SIZE,
    normalize_intensity,
    resize_slice,
)

logger = logging.getLogger("crip.mri.segmentation_dataset")


class ACDCSegmentationDataset(Dataset):
    """2D Slice Dataset for Cardiac MRI multi-class segmentation.

    Extracts short-axis slices from ED and ES phases of specified patients.
    Labels:
        0: Background
        1: Right Ventricular Cavity (RV)
        2: Myocardium (MYO)
        3: Left Ventricular Cavity (LV)
    """

    def __init__(
        self,
        patient_ids: Sequence[str],
        data_dir: Path | str = DEFAULT_ACDC_TRAINING_DIR,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
        augment: bool = False,
        seed: Optional[int] = None,
        cache_data: bool = True,
    ) -> None:
        """Initialize segmentation dataset.

        Args:
            patient_ids: List of patient IDs (e.g. ['patient001', 'patient002']).
            data_dir: Path to raw ACDC training directory.
            target_size: Target (H, W) spatial resolution for 2D slices.
            augment: Whether to apply online data augmentation.
            seed: Optional random seed for reproducible augmentation.
            cache_data: Whether to pre-extract and cache slices in memory.
        """
        self.patient_ids = list(patient_ids)
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augment = augment
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.loader = ACDCDataLoader(self.data_dir)

        # Slice index metadata list
        self.slices: List[Dict[str, Any]] = []

        # Load and index slices
        self._build_slice_index(cache_data=cache_data)

    def _build_slice_index(self, cache_data: bool) -> None:
        """Inspect patient volumes and build slice manifest."""
        logger.info(f"Indexing slices for {len(self.patient_ids)} patients...")
        for pid in self.patient_ids:
            try:
                meta = self.loader.get_patient_metadata(pid)
                ed_frame_obj = self.loader.load_cardiac_frame(pid, phase="ED", load_mask=True)
                es_frame_obj = self.loader.load_cardiac_frame(pid, phase="ES", load_mask=True)
            except Exception as e:
                logger.warning(f"Skipping patient {pid} due to load failure: {e}")
                continue

            phases = [
                ("ED", ed_frame_obj.image_data, ed_frame_obj.mask_data, ed_frame_obj.voxel_spacing),
                ("ES", es_frame_obj.image_data, es_frame_obj.mask_data, es_frame_obj.voxel_spacing),
            ]

            for phase_name, frame, mask, voxel_spacing in phases:
                if frame is None or mask is None:
                    continue

                num_slices = frame.shape[0]
                orig_shape = (frame.shape[1], frame.shape[2])

                for s in range(num_slices):
                    slice_img_raw = frame[s]
                    slice_mask_raw = mask[s]

                    slice_entry: Dict[str, Any] = {
                        "patient_id": pid,
                        "phase": phase_name,
                        "slice_idx": s,
                        "num_slices": num_slices,
                        "original_shape": orig_shape,
                        "voxel_spacing": voxel_spacing,
                        "group": meta.group,
                    }

                    if cache_data:
                        norm_img = normalize_intensity(slice_img_raw)
                        resized_img = resize_slice(norm_img, self.target_size, is_mask=False)
                        resized_mask = resize_slice(slice_mask_raw, self.target_size, is_mask=True)

                        slice_entry["image"] = resized_img
                        slice_entry["mask"] = resized_mask

                    self.slices.append(slice_entry)

        logger.info(f"Successfully indexed {len(self.slices)} 2D slices.")

    def __len__(self) -> int:
        return len(self.slices)

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric and intensity augmentations preserving label integrity."""
        # 1. Random Horizontal Flip
        if self.rng.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # 2. Random Vertical Flip
        if self.rng.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # 3. Random 90-degree rotations (0, 90, 180, 270)
        k = self.rng.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        # 4. Random Gamma / Intensity Adjustment (image only)
        if self.rng.rand() > 0.5:
            gamma = self.rng.uniform(0.8, 1.2)
            image = np.clip(image ** gamma, 0.0, 1.0)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.slices[idx]

        if "image" in entry and "mask" in entry:
            image = entry["image"].copy()
            mask = entry["mask"].copy()
        else:
            # Fallback if un-cached
            frame_obj = self.loader.load_cardiac_frame(entry["patient_id"], phase=entry["phase"], load_mask=True)
            slice_raw = frame_obj.image_data[entry["slice_idx"]]
            mask_raw = frame_obj.mask_data[entry["slice_idx"]] if frame_obj.mask_data is not None else np.zeros_like(slice_raw, dtype=np.int64)

            norm_img = normalize_intensity(slice_raw)
            image = resize_slice(norm_img, self.target_size, is_mask=False)
            mask = resize_slice(mask_raw, self.target_size, is_mask=True)

        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Convert to PyTorch tensors:
        # image: [1, H, W] float32
        # mask: [H, W] int64
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patient_id": entry["patient_id"],
            "phase": entry["phase"],
            "slice_idx": entry["slice_idx"],
            "original_shape": entry["original_shape"],
            "voxel_spacing": entry["voxel_spacing"],
            "group": entry["group"],
        }
