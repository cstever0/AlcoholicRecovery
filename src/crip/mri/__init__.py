"""MRI analysis module for CRIP.

Provides ACDC dataset ingestion, 3D cardiac frame extraction,
expert segmentation visualization, and volumetric calculation interfaces.
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
    CardiacVolumetrics,
    calculate_ejection_fraction,
    calculate_structure_volume_ml,
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
    "CardiacVolumetrics",
    "calculate_ejection_fraction",
    "calculate_structure_volume_ml",
    "compute_cardiac_volumetrics_from_masks",
]
