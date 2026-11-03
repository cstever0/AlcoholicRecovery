"""Unified Task and Data Schemas for Multi-Dataset CRIP Architecture.

Defines standardized task identifiers, modality definitions, patient-level
records, split definitions, and empirical feature schemas across:
- ACDC (Cine SSFP CMR)
- M&Ms (Multi-center, multi-vendor Cine SSFP CMR)
- EMIDEC (Delayed-enhancement LGE CMR)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


class PipelineTask(str, Enum):
    """Supported prediction and orchestration tasks in the unified pipeline."""
    CINE_VENTRICULAR_SEGMENTATION = "CINE_VENTRICULAR_SEGMENTATION"
    LGE_TISSUE_SEGMENTATION = "LGE_TISSUE_SEGMENTATION"
    EMPIRICAL_FEATURE_EXTRACTION = "EMPIRICAL_FEATURE_EXTRACTION"
    SYNTHETIC_COHORT_GENERATION = "SYNTHETIC_COHORT_GENERATION"
    SYNTHETIC_RECOVERY_MODELING = "SYNTHETIC_RECOVERY_MODELING"


class Modality(str, Enum):
    """Cardiac MRI imaging modality category."""
    CINE_CMR = "CINE_CMR"
    LGE_CMR = "LGE_CMR"


class PartitionSplit(str, Enum):
    """Standardized patient-level experimental partition."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HELD_OUT_EXTERNAL = "held_out_external"
    CROSS_VAL = "cross_val"


@dataclass
class UnifiedCaseRecord:
    """Master metadata record for an individual patient / study in the unified pipeline.

    Encapsulates imaging modality, file paths, scanner attributes, diagnostic classification,
    partitioning assignments, supported tasks, clinical variables, and data provenance.
    """
    dataset_id: str
    patient_id: str
    modality: Modality
    image_path: str
    source_partition: str
    assigned_split: PartitionSplit
    has_mask: bool = False
    annotation_path: Optional[str] = None
    study_id: Optional[str] = None
    acquisition_id: Optional[str] = None
    scanner_vendor: Optional[str] = None
    scanner_model: Optional[str] = None
    magnetic_field_t: Optional[float] = None
    center_id: Optional[str] = None
    diagnostic_group: Optional[str] = None
    available_labels: List[int] = field(default_factory=list)
    supported_tasks: List[PipelineTask] = field(default_factory=list)
    clinical_variables: Dict[str, Any] = field(default_factory=dict)
    missingness: Dict[str, bool] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        d = asdict(self)
        d["modality"] = self.modality.value
        d["assigned_split"] = self.assigned_split.value
        d["supported_tasks"] = [t.value for t in self.supported_tasks]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedCaseRecord:
        """Construct record from dictionary."""
        data_copy = dict(data)
        if isinstance(data_copy.get("modality"), str):
            data_copy["modality"] = Modality(data_copy["modality"])
        if isinstance(data_copy.get("assigned_split"), str):
            data_copy["assigned_split"] = PartitionSplit(data_copy["assigned_split"])
        if "supported_tasks" in data_copy:
            data_copy["supported_tasks"] = [
                PipelineTask(t) if isinstance(t, str) else t
                for t in data_copy["supported_tasks"]
            ]
        return cls(**data_copy)


@dataclass
class EmpiricalFeatureRecord:
    """Standardized record for an extracted empirical measurement.

    Maintains source attribution, measurement method, units, and clinical context
    without merging synthetic or cross-dataset artificial matches.
    """
    case_id: str
    dataset_id: str
    modality: Modality
    mask_provenance: str  # "expert_ground_truth" or "model_predicted"
    measurement_method: str
    diagnostic_group: Optional[str] = None
    scanner_vendor: Optional[str] = None
    center: Optional[str] = None
    magnetic_field_t: Optional[float] = None

    # Cine functional & structural measurements (ACDC / M&Ms)
    lvedv_ml: Optional[float] = None
    lvesv_ml: Optional[float] = None
    lvsv_ml: Optional[float] = None
    lvef_percent: Optional[float] = None
    rvedv_ml: Optional[float] = None
    rvesv_ml: Optional[float] = None
    rvsv_ml: Optional[float] = None
    rvef_percent: Optional[float] = None
    myocardial_mass_g: Optional[float] = None

    # LGE viability measurements (EMIDEC)
    normal_myo_vol_ml: Optional[float] = None
    infarct_vol_ml: Optional[float] = None
    no_reflow_vol_ml: Optional[float] = None
    total_myo_vol_ml: Optional[float] = None
    infarct_burden_percent: Optional[float] = None
    no_reflow_burden_percent: Optional[float] = None
    has_microvascular_obstruction: Optional[bool] = None

    # Available clinical variables & demographics
    sex: Optional[str] = None
    age: Optional[float] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    hypertension: Optional[bool] = None
    diabetes: Optional[bool] = None
    cad_family_history: Optional[bool] = None

    # Missingness tracking flags
    is_cine_complete: bool = False
    is_viability_complete: bool = False
    is_clinical_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature record to dictionary."""
        d = asdict(self)
        d["modality"] = self.modality.value
        return d
