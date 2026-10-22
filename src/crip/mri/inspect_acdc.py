"""ACDC Cardiac MRI Dataset Inspection and Validation Utility.

This script scans an ACDC dataset directory (training and/or testing split),
validates metadata files (Info.cfg), checks all NIfTI image and segmentation mask
files, verifies segmentation labels (0=Background, 1=RV, 2=Myocardium, 3=LV),
computes diagnostic category distributions, and outputs both a human-readable
summary and a structured JSON validation report.

DISCLAIMER:
Research proof-of-concept utility. Does not alter or overwrite source dataset files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("crip.mri.inspect_acdc")

VALID_LABELS: Set[int] = {0, 1, 2, 3}


@dataclass
class PatientMetadata:
    patient_id: str
    ed_frame: int
    es_frame: int
    group: str
    height: Optional[float]
    weight: Optional[float]
    nb_frame: Optional[int]


@dataclass
class FrameValidationResult:
    frame_name: str
    frame_number: int
    image_exists: bool
    image_path: Optional[str]
    gt_exists: bool
    gt_path: Optional[str]
    image_shape: Optional[Tuple[int, ...]]
    gt_shape: Optional[Tuple[int, ...]]
    shapes_match: bool
    voxel_spacing: Optional[Tuple[float, ...]]
    labels_found: List[int]
    invalid_labels: List[int]
    error_message: Optional[str] = None


@dataclass
class PatientValidationReport:
    patient_id: str
    directory_path: str
    info_cfg_exists: bool
    metadata: Optional[Dict[str, Any]]
    cine_4d_exists: bool
    cine_4d_path: Optional[str]
    cine_4d_shape: Optional[Tuple[int, ...]]
    ed_validation: Optional[Dict[str, Any]]
    es_validation: Optional[Dict[str, Any]]
    is_complete: bool
    errors: List[str]
    warnings: List[str]


def parse_info_cfg(info_file: Path, patient_id: str) -> Tuple[Optional[PatientMetadata], List[str]]:
    """Parse patient Info.cfg file into structured metadata."""
    errors = []
    if not info_file.is_file():
        errors.append(f"Info.cfg not found for {patient_id}")
        return None, errors

    data: Dict[str, str] = {}
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    except Exception as e:
        errors.append(f"Failed to read Info.cfg: {e}")
        return None, errors

    try:
        ed = int(data.get("ED", "0"))
        es = int(data.get("ES", "0"))
        group = data.get("Group", "UNKNOWN")
        height = float(data["Height"]) if "Height" in data and data["Height"] else None
        weight = float(data["Weight"]) if "Weight" in data and data["Weight"] else None
        nb_frame = int(data["NbFrame"]) if "NbFrame" in data and data["NbFrame"] else None

        if ed <= 0 or es <= 0:
            errors.append(f"Invalid ED ({ed}) or ES ({es}) frame numbers in Info.cfg")

        meta = PatientMetadata(
            patient_id=patient_id,
            ed_frame=ed,
            es_frame=es,
            group=group,
            height=height,
            weight=weight,
            nb_frame=nb_frame,
        )
        return meta, errors
    except Exception as e:
        errors.append(f"Failed parsing values in Info.cfg: {e}")
        return None, errors


def validate_frame_pair(
    patient_dir: Path,
    patient_id: str,
    frame_name: str,
    frame_num: int,
) -> FrameValidationResult:
    """Validate a 3D cardiac frame and its corresponding ground-truth segmentation mask."""
    img_filename = f"{patient_id}_frame{frame_num:02d}.nii.gz"
    gt_filename = f"{patient_id}_frame{frame_num:02d}_gt.nii.gz"

    img_path = patient_dir / img_filename
    gt_path = patient_dir / gt_filename

    # Fallback to .nii if .nii.gz does not exist
    if not img_path.exists() and (patient_dir / f"{patient_id}_frame{frame_num:02d}.nii").exists():
        img_path = patient_dir / f"{patient_id}_frame{frame_num:02d}.nii"
    if not gt_path.exists() and (patient_dir / f"{patient_id}_frame{frame_num:02d}_gt.nii").exists():
        gt_path = patient_dir / f"{patient_id}_frame{frame_num:02d}_gt.nii"

    img_exists = img_path.is_file()
    gt_exists = gt_path.is_file()

    img_shape = None
    gt_shape = None
    voxel_spacing = None
    labels_found: List[int] = []
    invalid_labels: List[int] = []
    error_msg = None

    if img_exists:
        try:
            nib_img = nib.load(str(img_path))
            img_shape = tuple(nib_img.shape)
            zooms = nib_img.header.get_zooms()
            voxel_spacing = tuple(float(z) for z in zooms[:len(img_shape)])
        except Exception as e:
            error_msg = f"Failed to load image NIfTI: {e}"

    if gt_exists:
        try:
            nib_gt = nib.load(str(gt_path))
            gt_shape = tuple(nib_gt.shape)
            gt_data = nib_gt.get_fdata(dtype=np.float32)
            unique_vals = np.unique(gt_data)
            unique_ints = [int(round(v)) for v in unique_vals]
            labels_found = sorted(unique_ints)
            invalid_labels = [lbl for lbl in labels_found if lbl not in VALID_LABELS]
        except Exception as e:
            msg = f"Failed to load mask NIfTI: {e}"
            error_msg = f"{error_msg}; {msg}" if error_msg else msg

    shapes_match = bool(img_shape is not None and gt_shape is not None and img_shape == gt_shape)

    return FrameValidationResult(
        frame_name=frame_name,
        frame_number=frame_num,
        image_exists=img_exists,
        image_path=str(img_path) if img_exists else None,
        gt_exists=gt_exists,
        gt_path=str(gt_path) if gt_exists else None,
        image_shape=img_shape,
        gt_shape=gt_shape,
        shapes_match=shapes_match,
        voxel_spacing=voxel_spacing,
        labels_found=labels_found,
        invalid_labels=invalid_labels,
        error_message=error_msg,
    )


def validate_patient(patient_dir: Path) -> PatientValidationReport:
    """Inspect and validate an individual patient directory."""
    patient_id = patient_dir.name
    errors: List[str] = []
    warnings: List[str] = []

    info_file = patient_dir / "Info.cfg"
    info_exists = info_file.is_file()
    meta, parse_errors = parse_info_cfg(info_file, patient_id)
    errors.extend(parse_errors)

    # 4D cine inspection
    cine_file = patient_dir / f"{patient_id}_4d.nii.gz"
    if not cine_file.exists() and (patient_dir / f"{patient_id}_4d.nii").exists():
        cine_file = patient_dir / f"{patient_id}_4d.nii"

    cine_exists = cine_file.is_file()
    cine_shape = None
    if cine_exists:
        try:
            nib_cine = nib.load(str(cine_file))
            cine_shape = tuple(nib_cine.shape)
        except Exception as e:
            warnings.append(f"4D cine could not be loaded: {e}")
    else:
        warnings.append(f"4D cine file {patient_id}_4d.nii.gz not found")

    ed_res = None
    es_res = None

    if meta:
        ed_res = validate_frame_pair(patient_dir, patient_id, "ED", meta.ed_frame)
        es_res = validate_frame_pair(patient_dir, patient_id, "ES", meta.es_frame)

        # Validate ED
        if not ed_res.image_exists:
            errors.append(f"ED image missing: frame{meta.ed_frame:02d}")
        if not ed_res.gt_exists:
            errors.append(f"ED mask missing: frame{meta.ed_frame:02d}_gt")
        elif not ed_res.shapes_match:
            errors.append(f"ED image shape {ed_res.image_shape} != mask shape {ed_res.gt_shape}")
        if ed_res.invalid_labels:
            errors.append(f"ED mask contains invalid labels: {ed_res.invalid_labels}")

        # Validate ES
        if not es_res.image_exists:
            errors.append(f"ES image missing: frame{meta.es_frame:02d}")
        if not es_res.gt_exists:
            errors.append(f"ES mask missing: frame{meta.es_frame:02d}_gt")
        elif not es_res.shapes_match:
            errors.append(f"ES image shape {es_res.image_shape} != mask shape {es_res.gt_shape}")
        if es_res.invalid_labels:
            errors.append(f"ES mask contains invalid labels: {es_res.invalid_labels}")

        if ed_res.error_message:
            errors.append(f"ED error: {ed_res.error_message}")
        if es_res.error_message:
            errors.append(f"ES error: {es_res.error_message}")

    is_complete = bool(info_exists and meta is not None and not errors)

    return PatientValidationReport(
        patient_id=patient_id,
        directory_path=str(patient_dir),
        info_cfg_exists=info_exists,
        metadata=asdict(meta) if meta else None,
        cine_4d_exists=cine_exists,
        cine_4d_path=str(cine_file) if cine_exists else None,
        cine_4d_shape=cine_shape,
        ed_validation=asdict(ed_res) if ed_res else None,
        es_validation=asdict(es_res) if es_res else None,
        is_complete=is_complete,
        errors=errors,
        warnings=warnings,
    )


def inspect_acdc_dataset(
    dataset_dir: Path | str,
    output_report_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Inspect all patient directories within an ACDC dataset split."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"ACDC dataset directory not found at: {root}")

    patient_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("patient")])
    total_patients = len(patient_dirs)

    reports: List[PatientValidationReport] = []
    groups: Counter = Counter()
    complete_count = 0
    incomplete_count = 0
    corrupted_count = 0
    total_images_loaded = 0
    total_masks_loaded = 0

    for p_dir in patient_dirs:
        rep = validate_patient(p_dir)
        reports.append(rep)

        if rep.metadata and "group" in rep.metadata:
            groups[rep.metadata["group"]] += 1

        if rep.is_complete:
            complete_count += 1
        else:
            incomplete_count += 1

        if rep.errors:
            corrupted_count += 1

        if rep.ed_validation and rep.ed_validation["image_exists"]:
            total_images_loaded += 1
        if rep.es_validation and rep.es_validation["image_exists"]:
            total_images_loaded += 1
        if rep.ed_validation and rep.ed_validation["gt_exists"]:
            total_masks_loaded += 1
        if rep.es_validation and rep.es_validation["gt_exists"]:
            total_masks_loaded += 1

    summary = {
        "dataset_path": str(root.resolve()),
        "total_patient_directories": total_patients,
        "successfully_validated_patients": complete_count,
        "incomplete_or_invalid_patients": incomplete_count,
        "patients_with_errors": corrupted_count,
        "total_ed_es_images_verified": total_images_loaded,
        "total_ed_es_masks_verified": total_masks_loaded,
        "diagnostic_groups_distribution": dict(groups),
        "validation_status": "PASS" if incomplete_count == 0 else "WARNING_INCOMPLETE",
    }

    full_report = {
        "summary": summary,
        "patients": [asdict(r) for r in reports],
    }

    if output_report_path:
        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Structured validation report saved to: {out_path}")

    return full_report


def print_summary(summary: Dict[str, Any]) -> None:
    """Print human-readable summary to console."""
    print("=" * 70)
    print("      ACDC CARDIAC MRI DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Dataset Path:                   {summary['dataset_path']}")
    print(f"Total Patient Folders Found:    {summary['total_patient_directories']}")
    print(f"Complete & Validated Patients:  {summary['successfully_validated_patients']}")
    print(f"Incomplete / Error Patients:    {summary['incomplete_or_invalid_patients']}")
    print(f"Total 3D Cardiac Images Verified: {summary['total_ed_es_images_verified']}")
    print(f"Total 3D Expert Masks Verified:   {summary['total_ed_es_masks_verified']}")
    print("-" * 70)
    print("Diagnostic Group Breakdown:")
    for group, count in summary["diagnostic_groups_distribution"].items():
        print(f"  - {group:<10}: {count} patients")
    print("-" * 70)
    print(f"Overall Validation Result:      {summary['validation_status']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate ACDC cardiac MRI dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/acdc/training",
        help="Path to ACDC dataset split directory (default: data/raw/acdc/training)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/acdc/acdc_validation_report.json",
        help="Destination path for JSON validation report",
    )
    args = parser.parse_args()

    report = inspect_acdc_dataset(args.data_dir, args.output)
    print_summary(report["summary"])


if __name__ == "__main__":
    main()
