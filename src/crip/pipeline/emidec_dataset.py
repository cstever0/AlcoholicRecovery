"""EMIDEC Delayed-Enhancement CMR Segmentation Dataset.

Provides slice-by-slice 2D indexing, physical geometry preservation,
percentile-based intensity normalization, training data augmentations,
and strict 5-class label handling for LGE tissue characterization:
  - 0: Background
  - 1: Left Ventricular Cavity
  - 2: Normal Myocardium
  - 3: Myocardial Infarction (Hyperenhancement)
  - 4: Microvascular Obstruction (No-Reflow Core)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from crip.datasets.emidec_loader import DEFAULT_EMIDEC_ROOT, EMIDECDataLoader
from crip.datasets.labels import EMIDECLabel
from crip.mri.preprocessing import normalize_intensity, resize_slice

logger = logging.getLogger("crip.pipeline.emidec_dataset")

DEFAULT_TARGET_SPATIAL_SIZE: Tuple[int, int] = (224, 224)


@dataclass
class EMIDECSliceRecord:
    """Metadata descriptor for an individual 2D short-axis slice."""
    case_id: str
    slice_index: int
    total_slices: int
    voxel_spacing: Tuple[float, float, float]
    has_infarct: bool
    has_mvo: bool
    pathology_group: str


class EMIDECSegmentationDataset(Dataset):
    """PyTorch Dataset indexing 2D short-axis slices from 3D EMIDEC LGE volumes."""

    def __init__(
        self,
        case_ids: Sequence[str],
        loader: Optional[EMIDECDataLoader] = None,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
        is_training: bool = False,
        augment: bool = False,
        cache_data: bool = True,
    ) -> None:
        """Initialize dataset with patient case IDs and preprocessing configuration.

        Args:
            case_ids: List of case IDs belonging strictly to one split partition.
            loader: Optional EMIDECDataLoader instance.
            target_size: 2D target spatial dimensions (H, W).
            is_training: Whether dataset is used in training mode.
            augment: Whether to apply spatial and intensity data augmentation.
            cache_data: Whether to pre-load and cache processed slices in memory.
        """
        self.case_ids = list(case_ids)
        self.loader = loader or EMIDECDataLoader()
        self.target_size = target_size
        self.is_training = is_training
        self.augment = augment and is_training
        self.cache_data = cache_data

        self.slice_records: List[EMIDECSliceRecord] = []
        self._cached_images: List[np.ndarray] = []
        self._cached_masks: List[np.ndarray] = []

        self._index_cases()

    def _index_cases(self) -> None:
        """Scan specified cases and index all short-axis slices."""
        logger.info(f"Indexing EMIDEC slices across {len(self.case_ids)} cases...")
        for cid in self.case_ids:
            try:
                case_dict = self.loader.load_case(cid)
            except Exception as e:
                logger.warning(f"Failed to load case {cid}: {e}")
                continue

            img = case_dict["image"]
            mask = case_dict["contour"]
            zooms = case_dict["voxel_spacing"]
            pathology = case_dict.get("pathology", "Unknown")

            if mask is None:
                logger.warning(f"Case {cid} has no contour mask; skipping.")
                continue

            # EMIDEC volumes have shape (H, W, num_slices)
            # Transpose to (num_slices, H, W) for slice-by-slice processing
            if img.ndim == 3 and mask.ndim == 3:
                num_slices = img.shape[2]
                for s in range(num_slices):
                    sl_img = img[:, :, s].astype(np.float32)
                    sl_mask = mask[:, :, s].astype(np.int64)

                    has_infarct = bool(int(EMIDECLabel.MYOCARDIAL_INFARCTION) in sl_mask)
                    has_mvo = bool(int(EMIDECLabel.NO_REFLOW) in sl_mask)

                    rec = EMIDECSliceRecord(
                        case_id=cid,
                        slice_index=s,
                        total_slices=num_slices,
                        voxel_spacing=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
                        has_infarct=has_infarct,
                        has_mvo=has_mvo,
                        pathology_group=pathology,
                    )
                    self.slice_records.append(rec)

                    if self.cache_data:
                        # Normalize and resize to target size
                        norm_img = normalize_intensity(sl_img)
                        # Zero-mean unit-variance
                        std = float(norm_img.std())
                        norm_img = (norm_img - float(norm_img.mean())) / (std + 1e-6)

                        resized_img = resize_slice(norm_img, target_size=self.target_size, is_mask=False)
                        resized_mask = resize_slice(sl_mask, target_size=self.target_size, is_mask=True)

                        self._cached_images.append(resized_img.astype(np.float32))
                        self._cached_masks.append(resized_mask.astype(np.int64))

            else:
                logger.warning(f"Case {cid} has non-3D shape: img {img.shape}, mask {mask.shape}")

        logger.info(
            f"Indexed {len(self.slice_records)} slices from {len(self.case_ids)} cases. "
            f"Infarct slices: {sum(1 for r in self.slice_records if r.has_infarct)}, "
            f"MVO slices: {sum(1 for r in self.slice_records if r.has_mvo)}"
        )

    def __len__(self) -> int:
        return len(self.slice_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve a single preprocessed 2D slice sample."""
        rec = self.slice_records[idx]

        if self.cache_data:
            img = self._cached_images[idx].copy()
            mask = self._cached_masks[idx].copy()
        else:
            case_dict = self.loader.load_case(rec.case_id)
            sl_img = case_dict["image"][:, :, rec.slice_index].astype(np.float32)
            sl_mask = case_dict["contour"][:, :, rec.slice_index].astype(np.int64)

            norm_img = normalize_intensity(sl_img)
            std = float(norm_img.std())
            norm_img = (norm_img - float(norm_img.mean())) / (std + 1e-6)

            img = resize_slice(norm_img, target_size=self.target_size, is_mask=False)
            mask = resize_slice(sl_mask, target_size=self.target_size, is_mask=True)

        # Apply training-only data augmentation
        if self.augment:
            img, mask = self._apply_augmentations(img, mask)

        # Convert to PyTorch tensors [1, H, W]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "case_id": rec.case_id,
            "slice_index": rec.slice_index,
            "total_slices": rec.total_slices,
            "has_infarct": rec.has_infarct,
            "has_mvo": rec.has_mvo,
            "pathology": rec.pathology_group,
            "voxel_spacing": list(rec.voxel_spacing),
        }

    def _apply_augmentations(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flips, rotations, and intensity jittering."""
        # 1. Random horizontal flip (p=0.5)
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # 2. Random vertical flip (p=0.5)
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # 3. Random 90-degree rotation (k in {0, 1, 2, 3})
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k=k).copy()
            mask = np.rot90(mask, k=k).copy()

        # 4. Intensity scaling (0.9 to 1.1) and shifting (-0.1 to 0.1)
        scale = float(np.random.uniform(0.9, 1.1))
        shift = float(np.random.uniform(-0.1, 0.1))
        img = (img * scale) + shift

        return img, mask
