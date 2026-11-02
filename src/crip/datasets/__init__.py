"""Cardiac MRI Datasets Package.

Modular dataset registry, standardized schemas, label transformations,
and dataset-specific loaders for multi-cohort cardiac MRI integration:
- ACDC 2017 (Cine SSFP, 4D/3D)
- EMIDEC 2020 (Delayed-enhancement LGE viability CMR, 3D static)
- M&Ms 2020 (Multi-centre, multi-vendor cine SSFP, 4D/3D)
- Sunnybrook (Historic cine benchmark, future)
- MyoPS 2020 (Multi-sequence viability CMR, future)
"""

from __future__ import annotations

from crip.datasets.emidec_loader import EMIDECDataLoader
from crip.datasets.labels import (
    ACDC_LABEL_NAMES,
    ACDCLabel,
    EMIDEC_LABEL_NAMES,
    EMIDECLabel,
    MNMS_LABEL_NAMES,
    MNMS_TO_CANONICAL_MAP,
    CANONICAL_TO_MNMS_MAP,
    MNMSLabel,
    map_canonical_to_mnms,
    map_mnms_to_canonical,
    validate_label_set,
)
from crip.datasets.mnms_loader import MNMSDataLoader
from crip.datasets.registry import (
    DatasetRegistry,
    get_dataset_registry,
)
from crip.datasets.schemas import (
    CardiacPhase,
    CaseMetadata,
    DatasetDefinition,
    DatasetStatus,
    ModalityType,
)
from crip.datasets.validation import (
    validate_emidec_dataset,
    validate_mnms_dataset,
)

__all__ = [
    # Schemas
    "ModalityType",
    "CardiacPhase",
    "DatasetStatus",
    "DatasetDefinition",
    "CaseMetadata",
    # Registry
    "DatasetRegistry",
    "get_dataset_registry",
    # Labels
    "ACDCLabel",
    "MNMSLabel",
    "EMIDECLabel",
    "ACDC_LABEL_NAMES",
    "MNMS_LABEL_NAMES",
    "EMIDEC_LABEL_NAMES",
    "MNMS_TO_CANONICAL_MAP",
    "CANONICAL_TO_MNMS_MAP",
    "map_mnms_to_canonical",
    "map_canonical_to_mnms",
    "validate_label_set",
    # Loaders
    "EMIDECDataLoader",
    "MNMSDataLoader",
    # Validation
    "validate_emidec_dataset",
    "validate_mnms_dataset",
]
