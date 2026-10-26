"""Unified Patient Feature Layer for CRIP Multimodal Integration.

DISCLAIMER:
Provides standard schema and builder for combining real ACDC cardiac MRI metrics
with simulated or entered clinical biomarkers, genetics, and demographics.
Real ACDC MRI data and synthetic clinical data are never represented as belonging to
the same actual human subject; constructed cases are assigned hypothetical IDs.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from crip.integration.provenance import (
    CaseSourceType,
    FeatureSourceOrigin,
    generate_hypothetical_case_id,
)
from crip.integration.validation import FeatureValidationResult, validate_unified_features
from crip.synthetic.schemas import (
    LGEPattern,
    LGEStatus,
    PEthRiskTier,
    Sex,
    TTNClassification,
)

logger = logging.getLogger("crip.integration.patient_features")

# Exact 20 features expected by recovery model pipeline v1.0-m3b in order
MODEL_FEATURE_WHITELIST: List[str] = [
    "age",
    "baseline_lvef",
    "edv_ml",
    "esv_ml",
    "stroke_volume_ml",
    "rvedv_ml",
    "rvef_percent",
    "myocardial_mass_g",
    "peth_value_ng_ml",
    "reported_drinks_per_week",
    "time_since_last_drink_days",
    "years_heavy_drinking",
    "lge_extent_percent",
    "biological_sex",
    "ttn_classification",
    "lge_status",
    "lge_pattern",
    "peth_available",
    "alcohol_history_available",
    "lge_available",
]


@dataclass
class UnifiedPatientProfile:
    """Standardized multimodal patient profile unifying imaging, clinical, and genetic features."""

    case_id: str
    source_type: str = CaseSourceType.CONSTRUCTED_MULTIMODAL_SCENARIO.value
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    # Demographics
    age: int = 50
    biological_sex: str = "MALE"

    # Cardiac MRI Volumetrics
    mri_source: str = FeatureSourceOrigin.REAL_ACDC_MRI_AUTOMATED_UNET.value
    acdc_patient_id: Optional[str] = None
    baseline_lvef: float = 35.0
    edv_ml: float = 210.0
    esv_ml: float = 136.5
    stroke_volume_ml: float = 73.5
    rvedv_ml: Optional[float] = 160.0
    rvef_percent: Optional[float] = 45.0
    myocardial_mass_g: Optional[float] = 130.0

    # Alcohol Exposure & Biomarkers
    peth_value_ng_ml: Optional[float] = 250.0
    peth_available: float = 1.0
    peth_risk_tier: str = PEthRiskTier.HEAVY_CHRONIC.value
    reported_drinks_per_week: Optional[float] = 35.0
    time_since_last_drink_days: Optional[float] = 7.0
    years_heavy_drinking: Optional[float] = 12.0
    alcohol_history_available: float = 1.0

    # Genetics
    ttn_classification: str = TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC.value
    ttn_provenance: str = FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value

    # Tissue Characterization (CMR LGE)
    lge_status: str = LGEStatus.NEGATIVE.value
    lge_pattern: str = LGEPattern.NONE.value
    lge_extent_percent: Optional[float] = 0.0
    lge_available: float = 1.0

    # Granular Feature Provenance Map
    provenance_map: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None

    def validate(self) -> FeatureValidationResult:
        """Run standard physical, biological, and clinical consistency checks."""
        raw_dict = self.to_dict()
        return validate_unified_features(raw_dict)

    def to_model_input_df(self) -> pd.DataFrame:
        """Construct a single-row Pandas DataFrame formatted specifically for the recovery pipeline.

        Ensures exact column presence, order, and proper data types (such as float indicator flags).
        """
        validation = self.validate()
        vfeat = validation.validated_features

        data = {
            "age": float(vfeat.get("age", self.age)),
            "baseline_lvef": float(vfeat.get("baseline_lvef", self.baseline_lvef)),
            "edv_ml": float(vfeat.get("edv_ml", self.edv_ml)),
            "esv_ml": float(vfeat.get("esv_ml", self.esv_ml)),
            "stroke_volume_ml": float(vfeat.get("stroke_volume_ml", self.stroke_volume_ml)),
            "rvedv_ml": (
                float(vfeat["rvedv_ml"]) if vfeat.get("rvedv_ml") is not None else float("nan")
            ),
            "rvef_percent": (
                float(vfeat["rvef_percent"])
                if vfeat.get("rvef_percent") is not None
                else float("nan")
            ),
            "myocardial_mass_g": (
                float(vfeat["myocardial_mass_g"])
                if vfeat.get("myocardial_mass_g") is not None
                else float("nan")
            ),
            "peth_value_ng_ml": (
                float(vfeat["peth_value_ng_ml"])
                if vfeat.get("peth_value_ng_ml") is not None
                else float("nan")
            ),
            "reported_drinks_per_week": (
                float(self.reported_drinks_per_week)
                if self.reported_drinks_per_week is not None
                else float("nan")
            ),
            "time_since_last_drink_days": (
                float(self.time_since_last_drink_days)
                if self.time_since_last_drink_days is not None
                else float("nan")
            ),
            "years_heavy_drinking": (
                float(self.years_heavy_drinking)
                if self.years_heavy_drinking is not None
                else float("nan")
            ),
            "lge_extent_percent": (
                float(vfeat["lge_extent_percent"])
                if vfeat.get("lge_extent_percent") is not None
                else float("nan")
            ),
            "biological_sex": str(vfeat.get("biological_sex", self.biological_sex)),
            "ttn_classification": str(vfeat.get("ttn_classification", self.ttn_classification)),
            "lge_status": str(vfeat.get("lge_status", self.lge_status)),
            "lge_pattern": str(vfeat.get("lge_pattern", self.lge_pattern)),
            "peth_available": float(vfeat.get("peth_available", self.peth_available)),
            "alcohol_history_available": float(
                vfeat.get("alcohol_history_available", self.alcohol_history_available)
            ),
            "lge_available": float(vfeat.get("lge_available", self.lge_available)),
        }

        df = pd.DataFrame([data], columns=MODEL_FEATURE_WHITELIST)
        return df

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedPatientProfile:
        """Instantiate profile from dictionary with type safety."""
        clean = dict(data)
        # Filter keys to match class fields
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in clean.items() if k in valid_fields}
        return cls(**filtered)


def build_unified_patient_features(
    mri_metrics: Dict[str, Any],
    clinical_profile: Optional[Dict[str, Any]] = None,
    case_id: Optional[str] = None,
    source_type: CaseSourceType = CaseSourceType.CONSTRUCTED_MULTIMODAL_SCENARIO,
    mri_source_origin: FeatureSourceOrigin = FeatureSourceOrigin.REAL_ACDC_MRI_AUTOMATED_UNET,
    acdc_patient_id: Optional[str] = None,
) -> UnifiedPatientProfile:
    """Construct and validate a UnifiedPatientProfile combining MRI measurements and clinical data.

    Assigns explicit provenance metadata indicating that imaging features originate from
    ACDC cardiac MRI while clinical biomarkers, genetics, and history originate from
    simulated or entered clinical sources.
    """
    if case_id is None:
        case_id = generate_hypothetical_case_id()

    clin = clinical_profile or {}

    # Extract MRI Volumetrics
    edv = float(mri_metrics.get("edv_ml", mri_metrics.get("lv_edv_ml", 210.0)))
    esv = float(mri_metrics.get("esv_ml", mri_metrics.get("lv_esv_ml", 136.5)))
    sv = float(mri_metrics.get("stroke_volume_ml", mri_metrics.get("lv_stroke_volume_ml", edv - esv)))
    ef = float(mri_metrics.get("baseline_lvef", mri_metrics.get("lvef_percent", (sv / edv) * 100.0 if edv > 0 else 35.0)))
    rvedv = mri_metrics.get("rvedv_ml", mri_metrics.get("rv_edv_ml"))
    rvef = mri_metrics.get("rvef_percent", mri_metrics.get("rv_ef_percent"))
    mass = mri_metrics.get("myocardial_mass_g", mri_metrics.get("lv_myocardial_mass_g"))

    # Extract Demographics
    age = int(clin.get("age", mri_metrics.get("age", 50)))
    sex = str(clin.get("biological_sex", mri_metrics.get("biological_sex", "MALE"))).upper()

    # Extract Biomarkers
    peth_val = clin.get("peth_value_ng_ml")
    peth_avail = float(1.0 if clin.get("peth_available", peth_val is not None) else 0.0)

    # Establish Provenance Map
    prov_map: Dict[str, str] = {
        "case_id": source_type.value,
        "edv_ml": mri_source_origin.value,
        "esv_ml": mri_source_origin.value,
        "stroke_volume_ml": mri_source_origin.value,
        "baseline_lvef": mri_source_origin.value,
        "rvedv_ml": mri_source_origin.value,
        "rvef_percent": mri_source_origin.value,
        "myocardial_mass_g": mri_source_origin.value,
        "age": FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value if "age" not in mri_metrics else FeatureSourceOrigin.REAL_ACDC_MRI_EXPERT_MASK.value,
        "biological_sex": FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value if "biological_sex" not in mri_metrics else FeatureSourceOrigin.REAL_ACDC_MRI_EXPERT_MASK.value,
        "peth_value_ng_ml": FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value if peth_avail else FeatureSourceOrigin.MISSING_CLINICAL_DATA.value,
        "reported_drinks_per_week": FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value,
        "ttn_classification": FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value,
        "lge_status": FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value,
    }

    profile = UnifiedPatientProfile(
        case_id=case_id,
        source_type=source_type.value,
        age=age,
        biological_sex=sex,
        mri_source=mri_source_origin.value,
        acdc_patient_id=acdc_patient_id,
        baseline_lvef=round(ef, 2),
        edv_ml=round(edv, 2),
        esv_ml=round(esv, 2),
        stroke_volume_ml=round(sv, 2),
        rvedv_ml=round(float(rvedv), 2) if rvedv is not None else None,
        rvef_percent=round(float(rvef), 2) if rvef is not None else None,
        myocardial_mass_g=round(float(mass), 2) if mass is not None else None,
        peth_value_ng_ml=round(float(peth_val), 2) if peth_val is not None else None,
        peth_available=peth_avail,
        peth_risk_tier=clin.get("peth_risk_tier", PEthRiskTier.HEAVY_CHRONIC.value if (peth_val or 0) > 200 else PEthRiskTier.ABSTINENT_OR_LIGHT.value),
        reported_drinks_per_week=clin.get("reported_drinks_per_week", 28.0),
        time_since_last_drink_days=clin.get("time_since_last_drink_days", 5.0),
        years_heavy_drinking=clin.get("years_heavy_drinking", 10.0),
        alcohol_history_available=float(1.0 if clin.get("alcohol_history_available", True) else 0.0),
        ttn_classification=clin.get("ttn_classification", TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC.value),
        ttn_provenance=FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value,
        lge_status=clin.get("lge_status", LGEStatus.NEGATIVE.value),
        lge_pattern=clin.get("lge_pattern", LGEPattern.NONE.value),
        lge_extent_percent=clin.get("lge_extent_percent", 0.0),
        lge_available=float(1.0 if clin.get("lge_available", True) else 0.0),
        provenance_map=prov_map,
        notes=clin.get("notes", "Constructed multimodal scenario for clinical research demonstration."),
    )

    return profile
