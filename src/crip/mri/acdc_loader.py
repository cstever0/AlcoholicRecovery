"""ACDC Cardiac MRI Data Ingestion and Preprocessing Loader.

This module provides structured, validated access to the Automated Cardiac Diagnosis
Challenge (ACDC) dataset, supporting 3D end-diastolic (ED) and end-systolic (ES)
cine frames, expert multi-structure segmentation masks, and 4D cine MRI series.

Coordinate & Array Convention:
------------------------------
Arrays returned for 3D frames are shaped as (num_slices, height, width):
  - Axis 0 (z): Slice index along the short-axis stack (base to apex)
  - Axis 1 (y): Image height (rows)
  - Axis 2 (x): Image width (columns)
Voxel spacing is reported as (spacing_x, spacing_y, spacing_z) in millimeters (mm).

Segmentation Label Semantics:
----------------------------
  - Label 0: Background
  - Label 1: Right Ventricle (RV) cavity
  - Label 2: Myocardium (LV myocardium)
  - Label 3: Left Ventricle (LV) cavity

DISCLAIMER:
Research proof-of-concept utility. Does not alter original data or fabricate images.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

DEFAULT_ACDC_TRAINING_DIR = Path("data/raw/acdc/training")
DEFAULT_ACDC_TESTING_DIR = Path("data/raw/acdc/testing")

VALID_SEGMENTATION_LABELS: Set[int] = {0, 1, 2, 3}

LABEL_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Right Ventricle Cavity (RV)",
    2: "Myocardium",
    3: "Left Ventricle Cavity (LV)",
}


class ACDCDataError(Exception):
    """Exception raised when an ACDC file is missing, corrupt, or inconsistent."""
    pass


@dataclass(frozen=True)
class ACDCPatientMetadata:
    """Strongly-typed metadata for an ACDC patient parsed from Info.cfg."""
    patient_id: str
    ed_frame: int
    es_frame: int
    group: str
    height_cm: Optional[float]
    weight_kg: Optional[float]
    num_frames: Optional[int]
    patient_dir: Path


@dataclass
class ACDCCardiacFrame:
    """Encapsulates a loaded 3D cardiac phase (ED or ES) with optional expert mask."""
    patient_id: str
    phase_name: str  # "ED" or "ES"
    frame_index: int
    image_data: np.ndarray  # Shape: (slices, height, width), float32
    raw_image_data: np.ndarray  # Original intensity array
    mask_data: Optional[np.ndarray]  # Shape: (slices, height, width), uint8, labels {0, 1, 2, 3}
    voxel_spacing: Tuple[float, float, float]  # (spacing_x, spacing_y, spacing_z) in mm
    affine: np.ndarray  # 4x4 coordinate transformation matrix
    origin: Tuple[float, float, float]  # (x, y, z) physical coordinate origin
    direction: Tuple[float, ...]  # Direction cosine matrix components
    num_slices: int
    height: int
    width: int

    def get_slice_image(self, slice_idx: int, normalized: bool = True) -> np.ndarray:
        """Retrieve a 2D slice for visualization."""
        if not (0 <= slice_idx < self.num_slices):
            raise IndexError(f"Slice index {slice_idx} out of range [0, {self.num_slices - 1}]")
        img = self.image_data[slice_idx]
        if normalized:
            return normalize_for_display(img)
        return img

    def get_slice_mask(self, slice_idx: int) -> Optional[np.ndarray]:
        """Retrieve a 2D segmentation mask slice."""
        if self.mask_data is None:
            return None
        if not (0 <= slice_idx < self.num_slices):
            raise IndexError(f"Slice index {slice_idx} out of range [0, {self.num_slices - 1}]")
        return self.mask_data[slice_idx]


@dataclass
class ACDCCineSeries:
    """Encapsulates a loaded 4D cine cardiac MRI sequence."""
    patient_id: str
    cine_data: np.ndarray  # Shape: (frames, slices, height, width) or (slices, height, width, frames)
    voxel_spacing: Tuple[float, ...]
    num_frames: int
    num_slices: int


def normalize_for_display(
    img_2d: np.ndarray,
    p_min: float = 1.0,
    p_max: float = 99.0,
) -> np.ndarray:
    """Robust percentile-based intensity normalization to [0.0, 1.0] for UI display."""
    if img_2d.size == 0:
        return img_2d
    v_min, v_max = np.percentile(img_2d, (p_min, p_max))
    if v_max <= v_min:
        v_min, v_max = float(img_2d.min()), float(img_2d.max())
    if v_max <= v_min:
        return np.zeros_like(img_2d, dtype=np.float32)
    clipped = np.clip(img_2d, v_min, v_max)
    return ((clipped - v_min) / (v_max - v_min)).astype(np.float32)


class ACDCDataLoader:
    """Loader providing systematic access to the ACDC dataset directory."""

    def __init__(self, dataset_root: Path | str = DEFAULT_ACDC_TRAINING_DIR):
        self.dataset_root = Path(dataset_root)

    def is_available(self) -> bool:
        """Check if dataset root directory exists and contains patient folders."""
        if not self.dataset_root.is_dir():
            return False
        return len(self.list_patients()) > 0

    def list_patients(self) -> List[str]:
        """Return sorted list of patient identifiers present in the dataset."""
        if not self.dataset_root.is_dir():
            return []
        patients = [
            d.name
            for d in self.dataset_root.iterdir()
            if d.is_dir() and d.name.startswith("patient")
        ]
        return sorted(patients)

    def get_patient_dir(self, patient_id: str) -> Path:
        """Resolve and validate directory for a given patient identifier."""
        p_dir = self.dataset_root / patient_id
        if not p_dir.is_dir():
            raise ACDCDataError(f"Patient directory does not exist: {p_dir}")
        return p_dir

    def get_patient_metadata(self, patient_id: str) -> ACDCPatientMetadata:
        """Parse Info.cfg for a patient into ACDCPatientMetadata."""
        p_dir = self.get_patient_dir(patient_id)
        info_file = p_dir / "Info.cfg"
        if not info_file.is_file():
            raise ACDCDataError(f"Missing Info.cfg for patient: {patient_id}")

        data: Dict[str, str] = {}
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        k, v = line.split(":", 1)
                        data[k.strip()] = v.strip()
        except Exception as e:
            raise ACDCDataError(f"Error reading Info.cfg for {patient_id}: {e}") from e

        try:
            ed = int(data["ED"])
            es = int(data["ES"])
            group = data.get("Group", "UNKNOWN")
            height = float(data["Height"]) if "Height" in data and data["Height"] else None
            weight = float(data["Weight"]) if "Weight" in data and data["Weight"] else None
            nb_frame = int(data["NbFrame"]) if "NbFrame" in data and data["NbFrame"] else None
        except KeyError as e:
            raise ACDCDataError(f"Missing required key in Info.cfg for {patient_id}: {e}") from e
        except ValueError as e:
            raise ACDCDataError(f"Malformed value in Info.cfg for {patient_id}: {e}") from e

        if ed <= 0 or es <= 0:
            raise ACDCDataError(f"Invalid cardiac frame indices in Info.cfg (ED={ed}, ES={es})")

        return ACDCPatientMetadata(
            patient_id=patient_id,
            ed_frame=ed,
            es_frame=es,
            group=group,
            height_cm=height,
            weight_kg=weight,
            num_frames=nb_frame,
            patient_dir=p_dir,
        )

    def load_cardiac_frame(
        self,
        patient_id: str,
        phase: str = "ED",
        load_mask: bool = True,
    ) -> ACDCCardiacFrame:
        """Load a 3D cardiac frame (ED or ES) and its corresponding ground-truth segmentation.

        Args:
            patient_id: Patient folder name (e.g. 'patient001')
            phase: 'ED' for end-diastolic frame, 'ES' for end-systolic frame
            load_mask: If True, load and validate the matching ground-truth mask

        Returns:
            ACDCCardiacFrame dataclass with image array, mask array, and metadata.
        """
        phase_upper = phase.upper()
        if phase_upper not in ("ED", "ES"):
            raise ValueError(f"Invalid cardiac phase '{phase}'. Must be 'ED' or 'ES'.")

        meta = self.get_patient_metadata(patient_id)
        frame_num = meta.ed_frame if phase_upper == "ED" else meta.es_frame

        p_dir = meta.patient_dir
        img_file = p_dir / f"{patient_id}_frame{frame_num:02d}.nii.gz"
        if not img_file.is_file():
            img_file = p_dir / f"{patient_id}_frame{frame_num:02d}.nii"
        if not img_file.is_file():
            raise ACDCDataError(f"Cardiac image file not found for {patient_id} frame {frame_num:02d}")

        # Load image via SimpleITK to obtain consistent (slices, height, width) ordering
        try:
            sitk_img = sitk.ReadImage(str(img_file))
            raw_img_arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
            spacing_sitk = sitk_img.GetSpacing()  # (x, y, z)
            origin = sitk_img.GetOrigin()
            direction = sitk_img.GetDirection()
        except Exception as e:
            raise ACDCDataError(f"Failed to load image file {img_file}: {e}") from e

        # Also obtain 4x4 affine via NiBabel for coordinate transformations
        try:
            nib_img = nib.load(str(img_file))
            affine = nib_img.affine
        except Exception:
            affine = np.eye(4)

        num_slices, height, width = raw_img_arr.shape

        mask_arr: Optional[np.ndarray] = None
        if load_mask:
            gt_file = p_dir / f"{patient_id}_frame{frame_num:02d}_gt.nii.gz"
            if not gt_file.is_file():
                gt_file = p_dir / f"{patient_id}_frame{frame_num:02d}_gt.nii"
            if not gt_file.is_file():
                raise ACDCDataError(f"Expert mask file not found: {gt_file}")

            try:
                sitk_gt = sitk.ReadImage(str(gt_file))
                mask_arr = sitk.GetArrayFromImage(sitk_gt).astype(np.uint8)
            except Exception as e:
                raise ACDCDataError(f"Failed to load mask file {gt_file}: {e}") from e

            # Shape compatibility validation
            if mask_arr.shape != raw_img_arr.shape:
                raise ACDCDataError(
                    f"Shape mismatch: Image shape {raw_img_arr.shape} != "
                    f"Mask shape {mask_arr.shape} for {patient_id} frame {frame_num:02d}"
                )

            # Unexpected segmentation label detection
            unique_labels = set(np.unique(mask_arr).tolist())
            invalid = unique_labels - VALID_SEGMENTATION_LABELS
            if invalid:
                raise ACDCDataError(
                    f"Unexpected segmentation label(s) {invalid} in {gt_file}. "
                    f"Expected only {VALID_SEGMENTATION_LABELS}"
                )

        return ACDCCardiacFrame(
            patient_id=patient_id,
            phase_name=phase_upper,
            frame_index=frame_num,
            image_data=raw_img_arr,
            raw_image_data=raw_img_arr.copy(),
            mask_data=mask_arr,
            voxel_spacing=(float(spacing_sitk[0]), float(spacing_sitk[1]), float(spacing_sitk[2])),
            affine=affine,
            origin=origin,
            direction=direction,
            num_slices=num_slices,
            height=height,
            width=width,
        )

    def load_cine_4d(self, patient_id: str) -> ACDCCineSeries:
        """Load 4D cine MRI sequence for a patient."""
        meta = self.get_patient_metadata(patient_id)
        p_dir = meta.patient_dir
        cine_file = p_dir / f"{patient_id}_4d.nii.gz"
        if not cine_file.is_file():
            cine_file = p_dir / f"{patient_id}_4d.nii"
        if not cine_file.is_file():
            raise ACDCDataError(f"4D cine file not found for {patient_id}: {cine_file}")

        try:
            sitk_cine = sitk.ReadImage(str(cine_file))
            cine_arr = sitk.GetArrayFromImage(sitk_cine).astype(np.float32)
            spacing = sitk_cine.GetSpacing()
        except Exception as e:
            raise ACDCDataError(f"Failed to load 4D cine file {cine_file}: {e}") from e

        # SimpleITK 4D array shape is typically (frames, slices, height, width)
        if cine_arr.ndim != 4:
            raise ACDCDataError(f"Expected 4D array for cine sequence, got {cine_arr.ndim}D")

        num_frames = cine_arr.shape[0]
        num_slices = cine_arr.shape[1]

        return ACDCCineSeries(
            patient_id=patient_id,
            cine_data=cine_arr,
            voxel_spacing=spacing,
            num_frames=num_frames,
            num_slices=num_slices,
        )
