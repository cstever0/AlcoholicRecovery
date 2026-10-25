"""Cardiac Ventricular Volumetrics and Functional Metric Calculations.

This module provides quantitative interfaces for calculating Left Ventricular (LV)
and Right Ventricular (RV) cavity volumes, stroke volumes, ejection fractions,
and myocardial mass from 3D cardiac segmentation masks and physical voxel spacing.

Mathematical Principles:
------------------------
1. Physical Voxel Volume (mL):
     V_voxel = (spacing_x * spacing_y * spacing_z) / 1000.0  [since 1000 mm^3 = 1 mL]
   Note: Correctly accounts for non-isotropic in-plane pixel spacing (dx, dy)
   and slice thickness / spacing (dz).

2. Cavity Volume (mL):
     V_cavity = N_voxels * V_voxel
   where N_voxels is the count of voxels belonging to the target anatomical label.
   ACDC Labels:
     - 0: Background
     - 1: Right Ventricular (RV) cavity
     - 2: Myocardium
     - 3: Left Ventricular (LV) cavity

3. Stroke Volume (SV, mL):
     SV = EDV - ESV

4. Ejection Fraction (EF, %):
     EF = (SV / EDV) * 100.0 = ((EDV - ESV) / EDV) * 100.0

5. Estimated Myocardial Mass (g):
     Mass = Volume_myocardium * Density_myocardium
   Density Assumption:
     MYOCARDIAL_DENSITY_G_PER_ML = 1.05 g/mL (or 1.05 g/cm³), the standard clinical
     consensus established by the American Heart Association (AHA) and Society for
     Cardiovascular Magnetic Resonance (SCMR) (Cerqueira et al., Circulation 2002;
     Schulz-Menger et al., J Cardiovasc Magn Reson 2020).

DISCLAIMER:
All measurements are derived directly from expert-provided segmentation masks.
Measurement accuracy has not undergone independent external clinical calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# Standard clinical tissue density constant for human myocardium
MYOCARDIAL_DENSITY_G_PER_ML: float = 1.05

# ACDC Segmentation Label Definitions
LABEL_BACKGROUND: int = 0
LABEL_RV_CAVITY: int = 1
LABEL_MYOCARDIUM: int = 2
LABEL_LV_CAVITY: int = 3

VALID_SEGMENTATION_LABELS: Set[int] = {
    LABEL_BACKGROUND,
    LABEL_RV_CAVITY,
    LABEL_MYOCARDIUM,
    LABEL_LV_CAVITY,
}


@dataclass(frozen=True)
class CardiacVolumetrics:
    """Quantitative Left Ventricular functional parameters (Milestone 2A legacy)."""
    edv_ml: float
    esv_ml: float
    stroke_volume_ml: float
    lvef_percent: float
    voxel_volume_mm3: float
    ed_voxel_count: int
    es_voxel_count: int
    is_validated: bool = False


@dataclass
class BiventricularCardiacMeasurements:
    """Comprehensive biventricular and myocardial functional metrics for a patient."""
    patient_id: Optional[str]
    lvedv_ml: float
    lvesv_ml: float
    lvsv_ml: float
    lvef_percent: float
    rvedv_ml: float
    rvesv_ml: float
    rvsv_ml: float
    rvef_percent: float
    myocardial_volume_ed_ml: float
    myocardial_volume_es_ml: float
    myocardial_mass_ed_g: float
    myocardial_mass_es_g: float
    voxel_spacing_mm: Tuple[float, float, float]
    voxel_volume_mm3: float
    is_physiologically_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        d = asdict(self)
        # Ensure tuple converted to list for clean JSON serialization
        d["voxel_spacing_mm"] = list(self.voxel_spacing_mm)
        return d


def calculate_structure_volume_ml(
    mask_3d: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
    target_label: int = LABEL_LV_CAVITY,
) -> Tuple[float, int, float]:
    """Calculate the physical volume in milliliters (mL) for a target segmentation label.

    Handles non-isotropic voxel spacing:
        Volume (mL) = N_voxels * (dx * dy * dz) / 1000.0

    Args:
        mask_3d: 3D integer array of segmentation labels (e.g. {0, 1, 2, 3})
        voxel_spacing_mm: Physical spacing tuple (dx, dy, dz) in millimeters
        target_label: Segmentation label index (default 3 for LV cavity)

    Returns:
        Tuple of (volume_ml, voxel_count, voxel_volume_mm3)
    """
    if mask_3d is None:
        raise ValueError("Mask array cannot be None")
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


def calculate_ejection_fraction(
    edv_ml: float,
    esv_ml: float,
    strict: bool = True,
) -> Tuple[float, float]:
    """Calculate stroke volume (mL) and ejection fraction (%) from EDV and ESV.

    Formulas:
        SV = EDV - ESV
        EF = ((EDV - ESV) / EDV) * 100.0

    Args:
        edv_ml: End-diastolic volume in mL
        esv_ml: End-systolic volume in mL
        strict: If True, raise ValueError on non-physiologic inputs (EDV <= 0 or ESV > EDV).
                If False, calculate and return values while permitting caller to record warnings.

    Returns:
        Tuple of (stroke_volume_ml, ejection_fraction_percent)
    """
    if edv_ml <= 0:
        raise ValueError(f"EDV must be strictly positive to compute EF, got {edv_ml} mL")
    if esv_ml < 0:
        raise ValueError(f"ESV cannot be negative, got {esv_ml} mL")

    if esv_ml > edv_ml and strict:
        raise ValueError(
            f"End-systolic volume ({esv_ml:.1f} mL) exceeds end-diastolic volume ({edv_ml:.1f} mL), "
            "producing a negative stroke volume."
        )

    stroke_volume = edv_ml - esv_ml
    ef = (stroke_volume / edv_ml) * 100.0
    return stroke_volume, ef


def calculate_myocardial_mass_g(
    myocardial_volume_ml: float,
    density_g_per_ml: float = MYOCARDIAL_DENSITY_G_PER_ML,
) -> float:
    """Calculate estimated myocardial mass in grams.

    Formula:
        Mass (g) = Volume (mL) * Density (g/mL)
    Default density is 1.05 g/mL per AHA/SCMR guidelines.
    """
    if myocardial_volume_ml < 0:
        raise ValueError(f"Myocardial volume cannot be negative, got {myocardial_volume_ml}")
    if density_g_per_ml <= 0:
        raise ValueError(f"Myocardial density must be strictly positive, got {density_g_per_ml}")

    return float(myocardial_volume_ml * density_g_per_ml)


def compute_cardiac_volumetrics_from_masks(
    ed_mask: np.ndarray,
    es_mask: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
    lv_label: int = LABEL_LV_CAVITY,
) -> CardiacVolumetrics:
    """Compute Left Ventricular functional parameters from ED and ES masks (legacy interface)."""
    edv_ml, ed_count, vox_vol = calculate_structure_volume_ml(
        ed_mask, voxel_spacing_mm, target_label=lv_label
    )
    esv_ml, es_count, _ = calculate_structure_volume_ml(
        es_mask, voxel_spacing_mm, target_label=lv_label
    )

    stroke_volume, lvef = calculate_ejection_fraction(edv_ml, esv_ml, strict=False)

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


def compute_biventricular_measurements(
    ed_mask: np.ndarray,
    es_mask: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
    patient_id: Optional[str] = None,
) -> BiventricularCardiacMeasurements:
    """Compute complete biventricular (LV and RV) volumes, EFs, and myocardial mass.

    Args:
        ed_mask: 3D array of segmentation labels at End-Diastole
        es_mask: 3D array of segmentation labels at End-Systole
        voxel_spacing_mm: Physical voxel dimensions (dx, dy, dz) in mm
        patient_id: Optional patient identifier

    Returns:
        BiventricularCardiacMeasurements dataclass with complete measurements
        and physiological validation diagnostics.
    """
    warnings: List[str] = []
    is_valid = True

    if ed_mask is None or es_mask is None:
        raise ValueError("Both ED and ES masks must be provided")
    if ed_mask.shape != es_mask.shape:
        raise ValueError(
            f"ED mask shape {ed_mask.shape} does not match ES mask shape {es_mask.shape}"
        )

    # Check for invalid labels
    ed_labels = set(np.unique(ed_mask).tolist())
    es_labels = set(np.unique(es_mask).tolist())
    invalid_ed = ed_labels - VALID_SEGMENTATION_LABELS
    invalid_es = es_labels - VALID_SEGMENTATION_LABELS

    if invalid_ed:
        raise ValueError(f"ED mask contains invalid labels: {invalid_ed}")
    if invalid_es:
        raise ValueError(f"ES mask contains invalid labels: {invalid_es}")

    # Calculate LV Cavity Volumes
    lvedv, lv_ed_count, vox_vol = calculate_structure_volume_ml(
        ed_mask, voxel_spacing_mm, target_label=LABEL_LV_CAVITY
    )
    lvesv, lv_es_count, _ = calculate_structure_volume_ml(
        es_mask, voxel_spacing_mm, target_label=LABEL_LV_CAVITY
    )

    # Calculate RV Cavity Volumes
    rvedv, rv_ed_count, _ = calculate_structure_volume_ml(
        ed_mask, voxel_spacing_mm, target_label=LABEL_RV_CAVITY
    )
    rvesv, rv_es_count, _ = calculate_structure_volume_ml(
        es_mask, voxel_spacing_mm, target_label=LABEL_RV_CAVITY
    )

    # Calculate Myocardial Volumes and Masses
    myo_ed_vol, myo_ed_count, _ = calculate_structure_volume_ml(
        ed_mask, voxel_spacing_mm, target_label=LABEL_MYOCARDIUM
    )
    myo_es_vol, myo_es_count, _ = calculate_structure_volume_ml(
        es_mask, voxel_spacing_mm, target_label=LABEL_MYOCARDIUM
    )
    myo_ed_mass = calculate_myocardial_mass_g(myo_ed_vol)
    myo_es_mass = calculate_myocardial_mass_g(myo_es_vol)

    # Validate LV Ejection Fraction
    if lvedv <= 0:
        is_valid = False
        warnings.append("LVEDV is zero; LVEF cannot be calculated.")
        lvsv = 0.0
        lvef = 0.0
    else:
        if lvesv > lvedv:
            is_valid = False
            warnings.append(
                f"LVESV ({lvesv:.1f} mL) > LVEDV ({lvedv:.1f} mL): negative stroke volume."
            )
        lvsv, lvef = calculate_ejection_fraction(lvedv, lvesv, strict=False)

    # Validate RV Ejection Fraction
    if rvedv <= 0:
        is_valid = False
        warnings.append("RVEDV is zero; RVEF cannot be calculated.")
        rvsv = 0.0
        rvef = 0.0
    else:
        if rvesv > rvedv:
            is_valid = False
            warnings.append(
                f"RVESV ({rvesv:.1f} mL) > RVEDV ({rvedv:.1f} mL): negative stroke volume."
            )
        rvsv, rvef = calculate_ejection_fraction(rvedv, rvesv, strict=False)

    # Check for extreme, non-physiological ejection fractions
    if is_valid:
        if lvef < 5.0 or lvef > 90.0:
            warnings.append(f"LVEF of {lvef:.1f}% is outside typical physiological bounds [5%, 90%].")
        if rvef < 5.0 or rvef > 90.0:
            warnings.append(f"RVEF of {rvef:.1f}% is outside typical physiological bounds [5%, 90%].")

    return BiventricularCardiacMeasurements(
        patient_id=patient_id,
        lvedv_ml=lvedv,
        lvesv_ml=lvesv,
        lvsv_ml=lvsv,
        lvef_percent=lvef,
        rvedv_ml=rvedv,
        rvesv_ml=rvesv,
        rvsv_ml=rvsv,
        rvef_percent=rvef,
        myocardial_volume_ed_ml=myo_ed_vol,
        myocardial_volume_es_ml=myo_es_vol,
        myocardial_mass_ed_g=myo_ed_mass,
        myocardial_mass_es_g=myo_es_mass,
        voxel_spacing_mm=voxel_spacing_mm,
        voxel_volume_mm3=vox_vol,
        is_physiologically_valid=is_valid,
        validation_warnings=warnings,
    )
