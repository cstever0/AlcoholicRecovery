"""Automated Cardiac MRI Segmentation Inference Engine.

Provides slice-by-slice 2D U-Net prediction, exact nearest-neighbor spatial
restoration to native short-axis resolution, 3D volume reconstruction,
and volumetric calculation comparing automated vs. expert measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from crip.mri.acdc_loader import ACDCDataLoader, ACDCCardiacFrame, DEFAULT_ACDC_TRAINING_DIR
from crip.mri.cardiac_metrics import (
    BiventricularCardiacMeasurements,
    compute_biventricular_measurements,
)
from crip.mri.preprocessing import (
    DEFAULT_TARGET_SPATIAL_SIZE,
    normalize_intensity,
    resize_slice,
    restore_mask_to_original_shape,
)
from crip.mri.segmentation_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_unet_model,
    get_optimal_device,
    load_checkpoint,
)

logger = logging.getLogger("crip.mri.segmentation_inference")


@dataclass
class PatientInferenceResult:
    """Encapsulates full-volume automated segmentation predictions and volumetrics."""
    patient_id: str
    pred_ed_mask: np.ndarray  # Shape: (slices, height, width), uint8/int64
    pred_es_mask: np.ndarray  # Shape: (slices, height, width), uint8/int64
    pred_metrics: BiventricularCardiacMeasurements
    gt_ed_mask: Optional[np.ndarray] = None
    gt_es_mask: Optional[np.ndarray] = None
    gt_metrics: Optional[BiventricularCardiacMeasurements] = None
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    device_used: str = "cpu"
    gt_ed_mask: Optional[np.ndarray] = None
    gt_es_mask: Optional[np.ndarray] = None
    gt_metrics: Optional[CardiacVolumetrics] = None
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    device_used: str = "cpu"


class SegmentationInferenceEngine:
    """Inference engine managing model caching, slice preprocessing, and 3D reconstruction."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        device: Optional[torch.device] = None,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SPATIAL_SIZE,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.target_size = target_size
        self.device = device or get_optimal_device()
        self._model: Optional[torch.nn.Module] = None
        self._is_loaded = False

    @property
    def is_model_available(self) -> bool:
        """Check if model checkpoint exists on disk."""
        return self.checkpoint_path.is_file()

    def load_model(self) -> torch.nn.Module:
        """Load and cache the trained 2D U-Net model."""
        if self._model is not None and self._is_loaded:
            return self._model

        if not self.is_model_available:
            raise FileNotFoundError(
                f"Segmentation checkpoint not found at {self.checkpoint_path}. "
                "Please train the model first using train_segmentation.py."
            )

        model = build_unet_model().to(self.device)
        load_checkpoint(model, path=self.checkpoint_path, device=self.device)
        model.eval()
        self._model = model
        self._is_loaded = True
        return self._model

    @torch.no_grad()
    def predict_slice_mask(
        self,
        slice_2d: np.ndarray,
        model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """Predict multi-structure mask for a single 2D short-axis slice.

        Pipeline:
          1. Robust intensity normalization
          2. Bilinear resize to target input size (224, 224)
          3. Forward pass through U-Net on active device
          4. Argmax logit classification to integer labels {0, 1, 2, 3}
          5. Strict nearest-neighbor restoration back to original slice shape

        Args:
            slice_2d: 2D numpy array of shape (orig_H, orig_W).
            model: Optional model instance. If None, uses loaded engine model.

        Returns:
            2D integer array of shape (orig_H, orig_W) with labels in {0, 1, 2, 3}.
        """
        net = model or self.load_model()
        orig_h, orig_w = slice_2d.shape

        # Step 1: Normalize
        norm_slice = normalize_intensity(slice_2d)

        # Step 2: Resize to input dimensions
        resized_slice = resize_slice(norm_slice, target_size=self.target_size, is_mask=False)

        # Step 3: Model tensor preparation [1, 1, H, W]
        input_tensor = torch.from_numpy(resized_slice).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Step 4: Inference
        logits = net(input_tensor)
        pred_labels = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        # Step 5: Spatial restoration to native acquisition shape
        restored_mask = restore_mask_to_original_shape(pred_labels, original_shape=(orig_h, orig_w))
        return restored_mask

    @torch.no_grad()
    def predict_cardiac_frame_volume(
        self,
        frame_volume: np.ndarray,
        model: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """Segment an entire 3D cardiac volume (num_slices, H, W) slice-by-slice."""
        net = model or self.load_model()
        num_slices, orig_h, orig_w = frame_volume.shape
        pred_volume = np.zeros((num_slices, orig_h, orig_w), dtype=np.int64)

        for s in range(num_slices):
            slice_img = frame_volume[s]
            pred_volume[s] = self.predict_slice_mask(slice_img, model=net)

        return pred_volume

    def predict_patient(
        self,
        loader: ACDCDataLoader,
        patient_id: str,
    ) -> PatientInferenceResult:
        """Run complete biventricular segmentation and volumetric analysis for a patient."""
        model = self.load_model()

        # Load raw ED and ES frames
        ed_frame_obj = loader.load_cardiac_frame(patient_id, phase="ED", load_mask=True)
        es_frame_obj = loader.load_cardiac_frame(patient_id, phase="ES", load_mask=True)

        voxel_spacing = ed_frame_obj.voxel_spacing

        # Predict 3D masks
        pred_ed_mask = self.predict_cardiac_frame_volume(ed_frame_obj.image_data, model=model)
        pred_es_mask = self.predict_cardiac_frame_volume(es_frame_obj.image_data, model=model)

        # Compute automated volumetrics
        pred_metrics = compute_biventricular_measurements(
            ed_mask=pred_ed_mask,
            es_mask=pred_es_mask,
            voxel_spacing_mm=voxel_spacing,
            patient_id=patient_id,
        )

        # Compute ground truth volumetrics if masks exist
        gt_ed_mask = ed_frame_obj.mask_data
        gt_es_mask = es_frame_obj.mask_data
        gt_metrics: Optional[BiventricularCardiacMeasurements] = None

        if gt_ed_mask is not None and gt_es_mask is not None:
            try:
                gt_metrics = compute_biventricular_measurements(
                    ed_mask=gt_ed_mask,
                    es_mask=gt_es_mask,
                    voxel_spacing_mm=voxel_spacing,
                    patient_id=patient_id,
                )
            except Exception as e:
                logger.warning(f"Could not compute ground truth volumetrics for {patient_id}: {e}")

        return PatientInferenceResult(
            patient_id=patient_id,
            pred_ed_mask=pred_ed_mask,
            pred_es_mask=pred_es_mask,
            pred_metrics=pred_metrics,
            gt_ed_mask=gt_ed_mask,
            gt_es_mask=gt_es_mask,
            gt_metrics=gt_metrics,
            voxel_spacing=voxel_spacing,
            device_used=str(self.device),
        )
