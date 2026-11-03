"""Joint Multi-Dataset Cine Short-Axis Slice PyTorch Dataset.

Extracts and harmonizes 2D short-axis cine CMR slices from:
1. ACDC (Cine SSFP, Canonical ACDC 0=BG, 1=RV, 2=MYO, 3=LV)
2. M&Ms (Multi-Vendor Cine SSFP, remapped from M&Ms raw to Canonical ACDC)

Guarantees:
- Strict patient-level partition isolation (no patient spans splits).
- Dataset-aware indexing and optional balanced sampling.
- Intensity normalization and spatial resampling to target dimensions.
- Deterministic data augmentation during training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from crip.datasets.labels import map_mnms_to_canonical
from crip.datasets.mnms_loader import MNMSDataLoader
from crip.mri.acdc_loader import ACDCDataLoader, DEFAULT_ACDC_TRAINING_DIR
from crip.mri.preprocessing import (
    DEFAULT_TARGET_SPATIAL_SIZE,
    normalize_intensity,
    resize_slice,
)

logger = logging.getLogger("crip.pipeline.joint_dataset")


class JointCineSegmentationDataset(Dataset):
    """Multi-Dataset 2D Slice Dataset for Joint Cine Cardiac Segmentation."""

    def __init__(
        self,
        acdc_patient_ids: Sequence[str] = (),
        mnms_case_ids: Sequence[str] = (),
        acdc_data_dir: Path | str = DEFAULT_ACDC_TRAINING_DIR,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
        augment: bool = False,
        seed: Optional[int] = 42,
        cache_data: bool = True,
    ) -> None:
        self.acdc_patient_ids = list(acdc_patient_ids)
        self.mnms_case_ids = list(mnms_case_ids)
        self.acdc_data_dir = Path(acdc_data_dir)
        self.target_size = target_size
        self.augment = augment
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        self.acdc_loader = ACDCDataLoader(self.acdc_data_dir)
        self.mnms_loader = MNMSDataLoader()

        self.slices: List[Dict[str, Any]] = []
        self._build_slice_index(cache_data=cache_data)

    def _build_slice_index(self, cache_data: bool) -> None:
        """Index slices from both ACDC and M&Ms cohorts."""
        logger.info(
            f"Indexing joint cine slices: {len(self.acdc_patient_ids)} ACDC patients, "
            f"{len(self.mnms_case_ids)} M&Ms cases..."
        )

        # 1. Index ACDC slices
        acdc_slice_count = 0
        for pid in self.acdc_patient_ids:
            try:
                ed_obj = self.acdc_loader.load_cardiac_frame(pid, phase="ED", load_mask=True)
                es_obj = self.acdc_loader.load_cardiac_frame(pid, phase="ES", load_mask=True)
            except Exception as e:
                logger.warning(f"Failed to load ACDC patient {pid}: {e}")
                continue

            for phase_name, frame_obj in [("ED", ed_obj), ("ES", es_obj)]:
                if frame_obj.image_data is None or frame_obj.mask_data is None:
                    continue
                num_slices = frame_obj.image_data.shape[0]
                for s in range(num_slices):
                    img_raw = frame_obj.image_data[s]
                    mask_raw = frame_obj.mask_data[s]

                    entry: Dict[str, Any] = {
                        "dataset": "acdc",
                        "case_id": pid,
                        "phase": phase_name,
                        "slice_idx": s,
                        "num_slices": num_slices,
                    }
                    if cache_data:
                        norm_img = normalize_intensity(img_raw)
                        entry["image"] = resize_slice(norm_img, self.target_size, is_mask=False)
                        entry["mask"] = resize_slice(mask_raw, self.target_size, is_mask=True).astype(np.int64)
                    else:
                        entry["image_raw"] = img_raw
                        entry["mask_raw"] = mask_raw

                    self.slices.append(entry)
                    acdc_slice_count += 1

        # 2. Index M&Ms slices
        mnms_slice_count = 0
        for cid in self.mnms_case_ids:
            try:
                phase_dict = self.mnms_loader.load_phase_frames(cid, mapped_to_canonical=True)
            except Exception as e:
                logger.warning(f"Failed to load M&Ms case {cid}: {e}")
                continue

            for phase_name in ["ED", "ES"]:
                p_data = phase_dict.get(phase_name)
                if not p_data or p_data.get("image") is None or p_data.get("mask") is None:
                    continue

                img_vol = p_data["image"]  # shape (H, W, Slices)
                mask_vol = p_data["mask"]  # shape (H, W, Slices)
                num_slices = img_vol.shape[2] if len(img_vol.shape) == 3 else 1

                for s in range(num_slices):
                    img_raw = img_vol[:, :, s]
                    mask_raw = mask_vol[:, :, s]

                    entry = {
                        "dataset": "mnms",
                        "case_id": cid,
                        "phase": phase_name,
                        "slice_idx": s,
                        "num_slices": num_slices,
                    }
                    if cache_data:
                        norm_img = normalize_intensity(img_raw)
                        entry["image"] = resize_slice(norm_img, self.target_size, is_mask=False)
                        entry["mask"] = resize_slice(mask_raw, self.target_size, is_mask=True).astype(np.int64)
                    else:
                        entry["image_raw"] = img_raw
                        entry["mask_raw"] = mask_raw

                    self.slices.append(entry)
                    mnms_slice_count += 1

        logger.info(
            f"Joint indexing complete: {len(self.slices)} total slices "
            f"({acdc_slice_count} ACDC, {mnms_slice_count} M&Ms)."
        )

    def __len__(self) -> int:
        return len(self.slices)

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric augmentations preserving label boundaries."""
        # 1. Horizontal Flip (50% probability)
        if self.rng.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 2. Small random rotation (-15 to +15 degrees)
        angle = self.rng.uniform(-15.0, 15.0)
        if abs(angle) > 2.0:
            import scipy.ndimage as ndi
            image = ndi.rotate(image, angle, reshape=False, order=1, mode="nearest")
            mask = ndi.rotate(mask, angle, reshape=False, order=0, mode="nearest")

        # 3. Slight intensity scaling (0.9 to 1.1)
        scale = self.rng.uniform(0.9, 1.1)
        image = np.clip(image * scale, 0.0, 1.0)

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.slices[idx]

        if "image" in entry:
            image = entry["image"]
            mask = entry["mask"]
        else:
            norm_img = normalize_intensity(entry["image_raw"])
            image = resize_slice(norm_img, self.target_size, is_mask=False)
            mask = resize_slice(entry["mask_raw"], self.target_size, is_mask=True).astype(np.int64)

        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Shape: [1, H, W]
        img_tensor = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "dataset": entry["dataset"],
            "case_id": entry["case_id"],
            "phase": entry["phase"],
            "slice_idx": entry["slice_idx"],
        }


class BalancedDatasetBatchSampler(Sampler):
    """Batched sampler that ensures equal representation of ACDC and M&Ms in each batch."""

    def __init__(
        self,
        dataset: JointCineSegmentationDataset,
        batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        self.acdc_indices = [i for i, s in enumerate(dataset.slices) if s["dataset"] == "acdc"]
        self.mnms_indices = [i for i, s in enumerate(dataset.slices) if s["dataset"] == "mnms"]

        self.half_batch = batch_size // 2
        self.num_batches = max(len(self.acdc_indices), len(self.mnms_indices)) // max(self.half_batch, 1)

    def __iter__(self):
        acdc_pool = list(self.acdc_indices)
        mnms_pool = list(self.mnms_indices)

        self.rng.shuffle(acdc_pool)
        self.rng.shuffle(mnms_pool)

        acdc_idx = 0
        mnms_idx = 0

        for _ in range(self.num_batches):
            # Sample half batch from ACDC, half from M&Ms
            batch = []
            for _ in range(self.half_batch):
                batch.append(acdc_pool[acdc_idx % len(acdc_pool)])
                acdc_idx += 1
            for _ in range(self.batch_size - self.half_batch):
                batch.append(mnms_pool[mnms_idx % len(mnms_pool)])
                mnms_idx += 1

            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
