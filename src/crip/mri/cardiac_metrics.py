"""Cardiac Ventricular Volumetrics and Functional Metric Calculations.

This module provides initial computational interfaces for extracting physical
ventricular volumes (EDV, ESV), stroke volume (SV), and left ventricular ejection
fraction (LVEF) from 3D cardiac segmentation masks and physical voxel spacing.

Mathematical Principles:
------------------------
1. Physical Voxel Volume (mL):
     V_voxel = (spacing_x * spacing_y * spacing_z) / 1000.0  [since 1000 mm^3 = 1 mL]

2. Cavity Volume (mL):
     V_cavity = N_voxels * V_voxel
   where N_voxels is the count of voxels belonging to the target anatomical label
   (e.g., Left Ventricle cavity = label 3).

3. Stroke Volume (SV, mL):
     SV = EDV - ESV

4. Left Ventricular Ejection Fraction (LVEF, %):
     LVEF = ((EDV - ESV) / EDV) * 100.0

DISCLAIMER:
Preliminary research calculation interface prepared for Milestone 2B.
Measurement accuracy has not yet undergone formal clinical validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CardiacVolumetrics:
    """Quantitative Left Ventricular functional parameters."""
    edv_ml: float
    esv_ml: float
    stroke_volume_ml: float
    lvef_percent: float
    voxel_volume_mm3: float
    ed_voxel_count: int
    es_voxel_count: int
    is_validated: bool = False


def calculate_structure_volume_ml(
    mask_3d: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
    target_label: int = 3,
) -> Tuple[float, int, float]:
    """Calculate the physical volume in milliliters (mL) for a target segmentation label.

    Args:
        mask_3d: 3D integer array of segmentation labels (e.g. {0, 1, 2, 3})
        voxel_spacing_mm: Physical spacing tuple (dx, dy, dz) in millimeters
        target_label: Segmentation label index (default 3 for LV cavity)

    Returns:
        Tuple of (volume_ml, voxel_count, voxel_volume_mm3)
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask array, got {mask_3d.ndim}D")

    dx, dy, dz = voxel_spacing_mm
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError(f"Voxel spacings must be strictly positive: {voxel_spacing_mm}")

    voxel_volume_mm3 = float(dx * dy * dz)
    voxel_volume_ml = voxel_volume_mm3 / 1000.0  # 1 mL = 1000 mm^3

    voxel_count = int(np.sum(mask_3d == target_label))
    volume_ml = float(voxel_count * voxel_volume_ml)

    return volume_ml, voxel_count, voxel_volume_mm3


def calculate_ejection_fraction(edv_ml: float, esv_ml: float) -> Tuple[float, float]:
    """Calculate stroke volume (mL) and LVEF (%) from EDV and ESV.

    Formula:
        SV = EDV - ESV
        LVEF = ((EDV - ESV) / EDV) * 100.0

    Returns:
        Tuple of (stroke_volume_ml, lvef_percent)
    """
    if edv_ml <= 0:
        raise ValueError(f"EDV must be strictly positive, got {edv_ml}")
    if esv_ml < 0:
        raise ValueError(f"ESV cannot be negative, got {esv_ml}")
    if esv_ml > edv_ml:
        # Clinical pathology or non-physiological contour
        pass

    stroke_volume = edv_ml - esv_ml
    lvef = (stroke_volume / edv_ml) * 100.0
    return stroke_volume, lvef


def compute_cardiac_volumetrics_from_masks(
    ed_mask: np.ndarray,
    es_mask: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
    lv_label: int = 3,
) -> CardiacVolumetrics:
    """Compute complete Left Ventricular functional parameters from ED and ES masks.

    Note: This interface represents foundation work for Milestone 2B validation.
    """
    edv_ml, ed_count, vox_vol = calculate_structure_volume_ml(
        ed_mask, voxel_spacing_mm, target_label=lv_label
    )
    esv_ml, es_count, _ = calculate_structure_volume_ml(
        es_mask, voxel_spacing_mm, target_label=lv_label
    )

    stroke_volume, lvef = calculate_ejection_fraction(edv_ml, esv_ml)

    return CardiacVolumetrics(
        edv_ml=edv_ml,
        esv_ml=esv_ml,
        stroke_volume_ml=stroke_volume,
        lvef_percent=lvef,
        voxel_volume_mm3=vox_vol,
        ed_voxel_count=ed_count,
        es_voxel_count=es_count,
        is_validated=False,
    )
