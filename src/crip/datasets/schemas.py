"""Standardized Schemas and Data Models for Multi-Dataset Cardiac MRI Integration.

Defines unified data models, metadata contracts, modality classifications,
and label mapping schemas across heterogeneous cardiac MRI datasets:
- ACDC (Cine SSFP, 4D/3D, multi-phase)
- EMIDEC (Delayed-enhancement LGE MRI, 3D static volume, tissue necrosis)
- M&Ms (Cine SSFP, 4D/3D multi-vendor, multi-center)
- Sunnybrook (Cine SSFP, future extension)
- MyoPS 2020 (Multi-sequence LGE/T2/bSSFP, future extension)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ModalityType(str, Enum):
    """Cardiac MRI imaging modality category."""
    CINE_MRI = "Cine SSFP Dynamic CMR"
    LGE_MRI = "Late Gadolinium Enhancement (LGE) Viability CMR"
    MULTI_SEQUENCE_MRI = "Multi-Sequence Multi-Contrast CMR"


class CardiacPhase(str, Enum):
    """Cardiac cycle phase representation."""
    ED = "End-Diastole"
    ES = "End-Systole"
    STATIC = "Static Acquisition (Non-cine)"
    FULL_CINE = "Full Cardiac Cycle (4D Cine)"


class DatasetStatus(str, Enum):
    """Availability status of dataset within the platform."""
    AVAILABLE = "AVAILABLE"  # Fully present with images and annotations
    PARTIAL = "PARTIAL"      # Present with incomplete annotations or subset
    BLOCKED = "BLOCKED"      # Download prerequisite required / restricted access


@dataclass(frozen=True)
class DatasetDefinition:
    """Master registry entry defining a cardiac MRI dataset and its clinical scope."""
    dataset_id: str
    name: str
    modality: ModalityType
    root_path: Path
    status: DatasetStatus
    raw_label_names: Dict[int, str]
    canonical_label_map: Optional[Dict[int, int]] = None
    has_phases: bool = False
    has_reference_masks: bool = False
    has_clinical_metadata: bool = False
    supports_lvef_evaluation: bool = False
    compatible_tasks: List[str] = field(default_factory=list)
    license_type: str = "Research Only"
    citation: str = ""
    provenance: str = ""
    total_cases_expected: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert definition to JSON-serializable dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "modality": self.modality.value,
            "root_path": str(self.root_path),
            "status": self.status.value,
            "raw_label_names": {str(k): v for k, v in self.raw_label_names.items()},
            "canonical_label_map": {str(k): v for k, v in self.canonical_label_map.items()} if self.canonical_label_map else None,
            "has_phases": self.has_phases,
            "has_reference_masks": self.has_reference_masks,
            "has_clinical_metadata": self.has_clinical_metadata,
            "supports_lvef_evaluation": self.supports_lvef_evaluation,
            "compatible_tasks": list(self.compatible_tasks),
            "license_type": self.license_type,
            "citation": self.citation,
            "provenance": self.provenance,
            "total_cases_expected": self.total_cases_expected,
            "description": self.description,
        }


@dataclass
class CaseMetadata:
    """Standardized metadata for an individual patient/case across all datasets."""
    case_id: str
    dataset_id: str
    image_path: Path
    mask_path: Optional[Path] = None
    has_mask: bool = False
    spatial_shape: Tuple[int, ...] = field(default_factory=tuple)
    voxel_spacing_mm: Tuple[float, ...] = field(default_factory=tuple)
    affine: List[List[float]] = field(default_factory=list)
    phases_available: List[CardiacPhase] = field(default_factory=list)
    ed_frame_index: Optional[int] = None
    es_frame_index: Optional[int] = None
    vendor: Optional[str] = None
    center: Optional[str] = None
    scanner_field_strength_tesla: Optional[float] = None
    diagnosis: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    raw_labels_present: List[int] = field(default_factory=list)
    clinical_metadata: Dict[str, Any] = field(default_factory=dict)
    checksum_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert case metadata to JSON-serializable dictionary."""
        return {
            "case_id": self.case_id,
            "dataset_id": self.dataset_id,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "has_mask": self.has_mask,
            "spatial_shape": list(self.spatial_shape),
            "voxel_spacing_mm": [round(float(v), 4) for v in self.voxel_spacing_mm],
            "phases_available": [p.value for p in self.phases_available],
            "ed_frame_index": self.ed_frame_index,
            "es_frame_index": self.es_frame_index,
            "vendor": self.vendor,
            "center": self.center,
            "scanner_field_strength_tesla": self.scanner_field_strength_tesla,
            "diagnosis": self.diagnosis,
            "sex": self.sex,
            "age": self.age,
            "weight_kg": self.weight_kg,
            "height_cm": self.height_cm,
            "raw_labels_present": list(self.raw_labels_present),
            "clinical_metadata": self.clinical_metadata,
            "checksum_sha256": self.checksum_sha256,
        }
