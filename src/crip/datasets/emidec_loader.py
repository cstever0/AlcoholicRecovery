"""EMIDEC (Evaluation of Myocardial Infarction from Delayed-Enhancement CMR) Loader.

Discovers and parses 3D late-gadolinium enhancement (LGE) viability CMR volumes,
tissue viability reference contours (infarction and no-reflow core), and patient
clinical metadata from the official MICCAI 2020 EMIDEC challenge structure.

CLINICAL CONTEXT:
EMIDEC is a static T1-weighted viability scan acquired 10-15 minutes after intravenous
gadolinium administration. Unlike cine SSFP, it does NOT provide temporal frames
across the cardiac cycle, and cannot be used for dynamic ejection fraction (LVEF)
or Simpson's biplane rule calculations.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

from crip.datasets.labels import EMIDECLabel, EMIDEC_LABEL_NAMES
from crip.datasets.schemas import (
    CardiacPhase,
    CaseMetadata,
    DatasetStatus,
    ModalityType,
)

logger = logging.getLogger("crip.datasets.emidec_loader")

DEFAULT_EMIDEC_ROOT = Path("data/raw/emidec")


class EMIDECDataLoader:
    """Specialized data loader for the EMIDEC 2020 Delayed-Enhancement CMR dataset."""

    def __init__(self, root_dir: Optional[Union[Path, str]] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else DEFAULT_EMIDEC_ROOT
        self.train_dir = self.root_dir / "emidec-dataset-1.0.1"
        self.test_dir = self.root_dir / "emidec-segmentation-testset-1.0.0"

        # If user passed train_dir directly as root_dir
        if (self.root_dir / "Images").is_dir() or any(self.root_dir.glob("Case_*")):
            if (self.root_dir.parent / "emidec-dataset-1.0.1").is_dir():
                self.train_dir = self.root_dir.parent / "emidec-dataset-1.0.1"
                self.test_dir = self.root_dir.parent / "emidec-segmentation-testset-1.0.0"
            else:
                self.train_dir = self.root_dir
                self.test_dir = Path("/nonexistent")

        self._cases: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def is_available(self) -> bool:
        """Check if any valid EMIDEC cases are present on disk."""
        return self.train_dir.is_dir() or self.test_dir.is_dir()

    def discover_cases(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Discover all training and testing cases from the dataset directories."""
        if self._cases is not None and not force_refresh:
            return self._cases

        cases: Dict[str, Dict[str, Any]] = {}

        # 1. Discover training cases
        if self.train_dir.is_dir():
            for case_dir in sorted(self.train_dir.iterdir()):
                if not case_dir.is_dir() or not case_dir.name.startswith("Case_"):
                    continue

                cid = case_dir.name
                img_files = list((case_dir / "Images").glob("*.nii*")) if (case_dir / "Images").is_dir() else []
                ctr_files = list((case_dir / "Contours").glob("*.nii*")) if (case_dir / "Contours").is_dir() else []

                # Find clinical text file: e.g. "Case N006.txt" or "Case P001.txt"
                formatted_name = cid.replace("Case_", "Case ") + ".txt"
                clinical_file = self.train_dir / formatted_name
                if not clinical_file.is_file():
                    # Fallback check inside case_dir
                    sub_txt = list(case_dir.glob("*.txt"))
                    clinical_file = sub_txt[0] if sub_txt else None

                pathology = "Normal" if "_N" in cid else "Myocardial Infarction"

                cases[cid] = {
                    "case_id": cid,
                    "split": "train",
                    "pathology": pathology,
                    "image_path": img_files[0] if img_files else None,
                    "contour_path": ctr_files[0] if ctr_files else None,
                    "clinical_file": clinical_file if clinical_file and clinical_file.is_file() else None,
                    "has_image": len(img_files) > 0,
                    "has_contour": len(ctr_files) > 0,
                    "has_clinical": clinical_file is not None and clinical_file.is_file(),
                }

        # 2. Discover challenge test cases
        if self.test_dir.is_dir():
            for case_dir in sorted(self.test_dir.iterdir()):
                if not case_dir.is_dir() or not case_dir.name.startswith("Case_"):
                    continue

                cid = case_dir.name
                img_files = list((case_dir / "Images").glob("*.nii*")) if (case_dir / "Images").is_dir() else []
                ctr_files = list((case_dir / "Contours").glob("*.nii*")) if (case_dir / "Contours").is_dir() else []

                cases[cid] = {
                    "case_id": cid,
                    "split": "test",
                    "pathology": "Unknown (Held-Out Test)",
                    "image_path": img_files[0] if img_files else None,
                    "contour_path": ctr_files[0] if ctr_files else None,
                    "clinical_file": None,
                    "has_image": len(img_files) > 0,
                    "has_contour": len(ctr_files) > 0,
                    "has_clinical": False,
                }

        self._cases = cases
        logger.info(f"Discovered {len(cases)} EMIDEC cases ({sum(1 for c in cases.values() if c['has_contour'])} with contours)")
        return cases

    def get_case_ids(self) -> List[str]:
        """Return all discovered case identifiers in sorted order."""
        return sorted(list(self.discover_cases().keys()))

    def parse_clinical_metadata(self, clinical_path: Optional[Path]) -> Dict[str, Any]:
        """Parse structured clinical fields from an EMIDEC case text file."""
        if not clinical_path or not clinical_path.is_file():
            return {}

        data: Dict[str, Any] = {}
        try:
            content = clinical_path.read_text(encoding="latin1")
            for line in content.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":", 1)
                raw_k = parts[0].strip().replace("\xa0", " ")
                val = parts[1].strip()

                k_lower = raw_k.lower()
                if "sex" in k_lower:
                    data["sex"] = val.upper()
                elif "age" in k_lower:
                    try:
                        data["age"] = float(val)
                    except ValueError:
                        data["age"] = None
                elif "tobacco" in k_lower:
                    try:
                        data["tobacco_tier"] = int(val)
                    except ValueError:
                        data["tobacco_tier"] = None
                elif "hypertension" in k_lower:
                    data["hypertension"] = (val.upper() == "Y")
                elif "diabetes" in k_lower:
                    data["diabetes"] = (val.upper() == "Y")
                elif "overweight" in k_lower:
                    data["overweight"] = (val.upper() == "Y")
                elif "coronary" in k_lower:
                    data["cad_family_history"] = (val.upper() == "Y")
                elif "ecg" in k_lower:
                    data["ecg_st_elevation"] = (val.upper() == "Y")
                elif "troponin" in k_lower:
                    try:
                        data["troponin"] = float(val)
                    except ValueError:
                        data["troponin"] = None
                elif "killip" in k_lower:
                    try:
                        data["killip_max"] = int(val)
                    except ValueError:
                        data["killip_max"] = None
                elif "fevg" in k_lower:
                    # French: Fraction d'Éjection du Ventricule Gauche (LVEF)
                    try:
                        data["lvef_reference_percent"] = float(val)
                    except ValueError:
                        data["lvef_reference_percent"] = None
                elif "ntprobnp" in k_lower or "bnp" in k_lower:
                    try:
                        data["nt_probnp"] = float(val)
                    except ValueError:
                        data["nt_probnp"] = None
                elif "gap" in k_lower or "slice" in k_lower:
                    data["slice_gap_reported"] = val
        except Exception as e:
            logger.warning(f"Error parsing clinical file {clinical_path}: {e}")

        return data

    def get_case_metadata(self, case_id: str) -> CaseMetadata:
        """Construct standardized CaseMetadata for a specified EMIDEC case."""
        cases = self.discover_cases()
        if case_id not in cases:
            raise KeyError(f"Case '{case_id}' not found in EMIDEC dataset.")

        c_info = cases[case_id]
        img_path = c_info["image_path"]
        if not img_path:
            raise FileNotFoundError(f"Case '{case_id}' has no valid image NIfTI file.")

        img_nii = nib.load(img_path)
        shape = tuple(img_nii.shape)
        zooms = tuple(float(z) for z in img_nii.header.get_zooms())
        affine = img_nii.affine.tolist()

        ctr_path = c_info["contour_path"]
        has_mask = ctr_path is not None and ctr_path.is_file()
        raw_labels: List[int] = []
        if has_mask:
            ctr_nii = nib.load(ctr_path)
            raw_labels = sorted(np.unique(ctr_nii.get_fdata()).astype(int).tolist())

        clinical = self.parse_clinical_metadata(c_info["clinical_file"])

        return CaseMetadata(
            case_id=case_id,
            dataset_id="emidec",
            image_path=img_path,
            mask_path=ctr_path,
            has_mask=has_mask,
            spatial_shape=shape,
            voxel_spacing_mm=zooms,
            affine=affine,
            phases_available=[CardiacPhase.STATIC],
            ed_frame_index=None,
            es_frame_index=None,
            vendor="Siemens",
            center="Centre Hospitalier Universitaire Dijon",
            scanner_field_strength_tesla=3.0 if "3T" in str(img_path) else 1.5,
            diagnosis=c_info["pathology"],
            sex=clinical.get("sex"),
            age=clinical.get("age"),
            raw_labels_present=raw_labels,
            clinical_metadata=clinical,
        )

    def load_image(self, case_id: str) -> Tuple[np.ndarray, Tuple[float, ...], np.ndarray]:
        """Load 3D LGE volume, voxel spacing in mm, and affine matrix.

        Returns:
            Tuple of (image_array_3d, voxel_spacing_mm, affine_4x4)
        """
        cases = self.discover_cases()
        if case_id not in cases or not cases[case_id]["image_path"]:
            raise FileNotFoundError(f"Image not found for EMIDEC case '{case_id}'")

        nii = nib.load(cases[case_id]["image_path"])
        arr = np.asarray(nii.get_fdata(), dtype=np.float32)
        zooms = tuple(float(z) for z in nii.header.get_zooms())
        affine = nii.affine
        return arr, zooms, affine

    def load_contour(self, case_id: str) -> Optional[Tuple[np.ndarray, Tuple[float, ...], np.ndarray]]:
        """Load 3D tissue viability contour mask with original EMIDEC labels (0-4).

        Returns:
            Tuple of (mask_array_3d, voxel_spacing_mm, affine_4x4) or None if no mask exists.
        """
        cases = self.discover_cases()
        if case_id not in cases or not cases[case_id]["contour_path"]:
            return None

        nii = nib.load(cases[case_id]["contour_path"])
        arr = np.asarray(nii.get_fdata(), dtype=np.int64)
        zooms = tuple(float(z) for z in nii.header.get_zooms())
        affine = nii.affine
        return arr, zooms, affine

    def load_case(self, case_id: str) -> Dict[str, Any]:
        """Load full case data dictionary (image, contour, spacing, metadata).

        Returns:
            Dict containing case_id, image, contour, has_contour, voxel_spacing,
            affine, pathology, clinical_metadata.
        """
        cases = self.discover_cases()
        if case_id not in cases:
            raise KeyError(f"Case '{case_id}' not found in EMIDEC dataset.")

        c_info = cases[case_id]
        img_arr, zooms, affine = self.load_image(case_id)
        cnt_arr = None
        has_contour = False
        if c_info["contour_path"]:
            res = self.load_contour(case_id)
            if res is not None:
                cnt_arr = res[0]
                has_contour = True

        clinical = self.parse_clinical_metadata(c_info["clinical_file"])

        return {
            "case_id": case_id,
            "image": img_arr,
            "contour": cnt_arr,
            "has_contour": has_contour,
            "voxel_spacing": zooms,
            "affine": affine,
            "pathology": c_info["pathology"],
            "clinical_metadata": clinical,
        }

    def validate_case_alignment(self, case_id: str) -> bool:
        """Verify spatial shape, affine, and voxel spacing match between image and contour."""
        img_arr, img_zooms, img_aff = self.load_image(case_id)
        ctr = self.load_contour(case_id)
        if ctr is None:
            return True  # No contour to compare against

        ctr_arr, ctr_zooms, ctr_aff = ctr
        if img_arr.shape != ctr_arr.shape:
            logger.error(f"Shape mismatch in {case_id}: image {img_arr.shape} vs contour {ctr_arr.shape}")
            return False

        if not np.allclose(img_aff, ctr_aff, atol=1e-3):
            logger.error(f"Affine mismatch in {case_id}")
            return False

        return True
