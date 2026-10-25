"""MRI analysis module for CRIP.

Provides ACDC dataset ingestion, 3D cardiac frame extraction,
expert segmentation visualization, and quantitative volumetric calculation interfaces.
"""

from crip.mri.acdc_loader import (
    ACDCDataError,
    ACDCDataLoader,
    ACDCCardiacFrame,
    ACDCCineSeries,
    ACDCPatientMetadata,
    LABEL_NAMES,
    normalize_for_display,
)
from crip.mri.cardiac_metrics import (
    BiventricularCardiacMeasurements,
    CardiacVolumetrics,
    MYOCARDIAL_DENSITY_G_PER_ML,
    LABEL_BACKGROUND,
    LABEL_RV_CAVITY,
    LABEL_MYOCARDIUM,
    LABEL_LV_CAVITY,
    VALID_SEGMENTATION_LABELS,
    calculate_ejection_fraction,
    calculate_myocardial_mass_g,
    calculate_structure_volume_ml,
    compute_biventricular_measurements,
    compute_cardiac_volumetrics_from_masks,
)

__all__ = [
    "ACDCDataError",
    "ACDCDataLoader",
    "ACDCCardiacFrame",
    "ACDCCineSeries",
    "ACDCPatientMetadata",
    "LABEL_NAMES",
    "normalize_for_display",
    "BiventricularCardiacMeasurements",
    "CardiacVolumetrics",
    "MYOCARDIAL_DENSITY_G_PER_ML",
    "LABEL_BACKGROUND",
    "LABEL_RV_CAVITY",
    "LABEL_MYOCARDIUM",
    "LABEL_LV_CAVITY",
    "VALID_SEGMENTATION_LABELS",
    "calculate_ejection_fraction",
    "calculate_myocardial_mass_g",
    "calculate_structure_volume_ml",
    "compute_biventricular_measurements",
    "compute_cardiac_volumetrics_from_masks",
]
