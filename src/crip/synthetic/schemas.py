"""Synthetic patient data schemas for the Cardiomyopathy Recovery Intelligence Platform.

DISCLAIMER:
All schemas and data representations herein are strictly designed for research
prototyping and synthetic benchmark investigation. They MUST NOT be used for
clinical diagnosis, patient risk stratification, or clinical treatment decisions.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Sex(str, Enum):
    """Biological sex."""
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


class ProvenanceType(str, Enum):
    """Data origin and generation provenance tracking."""
    ACTUAL_ACDC_MRI = "ACTUAL_ACDC_MRI"
    CALCULATED_ACDC_MRI = "CALCULATED_ACDC_MRI"
    SYNTHETIC_SIMULATED = "SYNTHETIC_SIMULATED"
    HYPOTHETICAL_MANUAL = "HYPOTHETICAL_MANUAL"
    LITERATURE_BASED_ASSUMPTION = "LITERATURE_BASED_ASSUMPTION"


class TTNVariant(str, Enum):
    """Titin (TTN) genetic variant classification (Legacy / Unified).

    TTN truncating variants (TTNtv) represent a frequent genetic predisposition
    in dilated cardiomyopathy and can interact with toxic alcohol exposure.
    """
    NOT_DETECTED = "NOT_DETECTED"
    PATHOGENIC_TRUNCATING = "PATHOGENIC_TRUNCATING"  # TTNtv in constitutively expressed A-band
    VARIANT_UNCERTAIN_SIGNIFICANCE = "VARIANT_UNCERTAIN_SIGNIFICANCE"  # VUS
    BENIGN_OR_LIKELY_BENIGN = "BENIGN_OR_LIKELY_BENIGN"
    LIKELY_PATHOGENIC = "LIKELY_PATHOGENIC"
    NOT_TESTED = "NOT_TESTED"


class TTNClassification(str, Enum):
    """ACMG-AMP 2015 Tiered Titin (TTN) sequence variant classification.

    CRITICAL CLINICAL GENETICS SAFETY RULES:
    1. A TTN VUS must NEVER be treated as a confirmed inherited cardiomyopathy etiology.
    2. Pathogenic truncating variants (TTNtv) are distinct from missense VUS variants.
    """
    PATHOGENIC = "PATHOGENIC"
    LIKELY_PATHOGENIC = "LIKELY_PATHOGENIC"
    VARIANT_OF_UNCERTAIN_SIGNIFICANCE = "VARIANT_OF_UNCERTAIN_SIGNIFICANCE"  # VUS
    NO_PATHOGENIC_OR_LIKELY_PATHOGENIC = "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"
    NOT_TESTED = "NOT_TESTED"


def map_ttn_variant_to_classification(variant: TTNVariant) -> TTNClassification:
    """Map legacy TTNVariant enum to 5-tier ACMG-AMP classification."""
    if variant == TTNVariant.PATHOGENIC_TRUNCATING:
        return TTNClassification.PATHOGENIC
    if variant == TTNVariant.LIKELY_PATHOGENIC:
        return TTNClassification.LIKELY_PATHOGENIC
    if variant == TTNVariant.VARIANT_UNCERTAIN_SIGNIFICANCE:
        return TTNClassification.VARIANT_OF_UNCERTAIN_SIGNIFICANCE
    if variant == TTNVariant.NOT_TESTED:
        return TTNClassification.NOT_TESTED
    return TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC


def map_ttn_classification_to_variant(classification: TTNClassification) -> TTNVariant:
    """Map 5-tier ACMG-AMP classification back to TTNVariant enum."""
    if classification == TTNClassification.PATHOGENIC:
        return TTNVariant.PATHOGENIC_TRUNCATING
    if classification == TTNClassification.LIKELY_PATHOGENIC:
        return TTNVariant.LIKELY_PATHOGENIC
    if classification == TTNClassification.VARIANT_OF_UNCERTAIN_SIGNIFICANCE:
        return TTNVariant.VARIANT_UNCERTAIN_SIGNIFICANCE
    if classification == TTNClassification.NOT_TESTED:
        return TTNVariant.NOT_TESTED
    return TTNVariant.NOT_DETECTED


class LGEStatus(str, Enum):
    """Late Gadolinium Enhancement (LGE) myocardial fibrosis status on cardiac MRI.

    Identifies presence and distribution of replacement fibrosis in the left ventricle.
    """
    NEGATIVE = "NEGATIVE"
    MIDWALL = "MIDWALL"  # Characteristic non-ischemic DCM pattern
    SUBEPICARDIAL = "SUBEPICARDIAL"
    SUBENDOCARDIAL = "SUBENDOCARDIAL"  # Typically ischemic
    PATCHY_TRANSMURAL = "PATCHY_TRANSMURAL"
    POSITIVE = "POSITIVE"
    INDETERMINATE = "INDETERMINATE"
    NOT_PERFORMED = "NOT_PERFORMED"


class LGEPattern(str, Enum):
    """Spatial anatomical distribution pattern of replacement fibrosis."""
    NONE = "NONE"
    MIDWALL = "MIDWALL"
    SUBEPICARDIAL = "SUBEPICARDIAL"
    SUBENDOCARDIAL = "SUBENDOCARDIAL"
    PATCHY_TRANSMURAL = "PATCHY_TRANSMURAL"


class PEthRiskTier(str, Enum):
    """Phosphatidylethanol (PEth) clinical exposure interpretation tier."""
    ABSTINENT_OR_LIGHT = "ABSTINENT_OR_LIGHT"      # < 20 ng/mL
    SIGNIFICANT_MODERATE = "SIGNIFICANT_MODERATE"  # 20 - 200 ng/mL
    HEAVY_CHRONIC = "HEAVY_CHRONIC"                # > 200 ng/mL


class AbstinenceScenario(str, Enum):
    """Simulated longitudinal alcohol consumption scenario."""
    SUSTAINED_ABSTINENCE = "SUSTAINED_ABSTINENCE"
    ONGOING_HEAVY_CONSUMPTION = "ONGOING_HEAVY_CONSUMPTION"
    REDUCED_CONSUMPTION = "REDUCED_CONSUMPTION"
    UNKNOWN_OR_UNSPECIFIED = "UNKNOWN_OR_UNSPECIFIED"


class RecoveryCategory(str, Enum):
    """Standardized definition of 6-month left ventricular recovery."""
    COMPLETE_RECOVERY = "COMPLETE_RECOVERY"      # LVEF >= 50%
    PARTIAL_RECOVERY = "PARTIAL_RECOVERY"        # Delta LVEF >= 10% but LVEF < 50%
    STABLE_NON_RECOVERY = "STABLE_NON_RECOVERY"  # Delta LVEF between -5% and +10%
    DETERIORATION = "DETERIORATION"              # Delta LVEF < -5%
    PENDING_EVALUATION = "PENDING_EVALUATION"


class PEthMeasurement(BaseModel):
    """Phosphatidylethanol (PEth) blood biomarker measurement and observation date.

    PEth is a direct metabolite of ethanol with an elimination half-life of 4-10 days,
    reflecting alcohol exposure over the preceding 2-4 weeks.
    """
    value_ng_ml: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="PEth concentration in blood (ng/mL).",
    )
    measurement_date: date = Field(
        ...,
        description="Date when blood sample was drawn for PEth quantification.",
    )

    @property
    def risk_tier(self) -> PEthRiskTier:
        """Classifies PEth into established clinical exposure risk tiers."""
        if self.value_ng_ml < 20.0:
            return PEthRiskTier.ABSTINENT_OR_LIGHT
        if self.value_ng_ml <= 200.0:
            return PEthRiskTier.SIGNIFICANT_MODERATE
        return PEthRiskTier.HEAVY_CHRONIC


class FollowUpOutcomes(BaseModel):
    """Six-month longitudinal clinical follow-up outcomes.

    Captures simulated or observed recovery metrics.
    EXPERIMENTAL RESEARCH ENDPOINT:
      Recovery = 1 if delta_lvef >= 10.0 percentage points, else 0.
    """
    follow_up_date: Optional[date] = Field(
        default=None,
        description="Date of 6-month follow-up assessment.",
    )
    follow_up_lvef: Optional[float] = Field(
        default=None,
        ge=5.0,
        le=85.0,
        description="Left ventricular ejection fraction at 6-month follow-up (%).",
    )
    recovery_category: RecoveryCategory = Field(
        default=RecoveryCategory.PENDING_EVALUATION,
        description="Categorical recovery outcome at 6 months.",
    )
    recovery_binary: Optional[int] = Field(
        default=None,
        description="Binary recovery endpoint: 1 if delta LVEF >= 10.0 percentage points, else 0.",
    )
    delta_lvef: Optional[float] = Field(
        default=None,
        description="Absolute change in LVEF (follow_up_lvef - baseline_lvef) in percentage points.",
    )
    adverse_cardiac_events: List[str] = Field(
        default_factory=list,
        description="Adverse cardiac events recorded during follow-up period (e.g. HF hospitalization).",
    )
    sustained_abstinence_confirmed: Optional[bool] = Field(
        default=None,
        description="Whether sustained alcohol abstinence was confirmed through follow-up PEth / history.",
    )
    assumed_abstinence_scenario: AbstinenceScenario = Field(
        default=AbstinenceScenario.SUSTAINED_ABSTINENCE,
        description="Hypothetical alcohol exposure scenario evaluated for simulation.",
    )


class PatientProfile(BaseModel):
    """Comprehensive synthetic patient record schema for CRIP research modeling.

    Captures baseline demographic, structural MRI volumetrics, genetic predispositions,
    toxicology biomarkers, tissue characterization (LGE), and follow-up trajectory.
    """
    patient_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique pseudo-anonymized synthetic patient identifier.",
        examples=["CRIP-SYN-00101"],
    )
    age: int = Field(
        ...,
        ge=18,
        le=105,
        description="Patient age in completed years.",
    )
    biological_sex: Sex = Field(
        default=Sex.MALE,
        description="Biological sex.",
    )
    baseline_lvef: float = Field(
        ...,
        ge=5.0,
        le=75.0,
        description="Baseline Left Ventricular Ejection Fraction (%).",
    )
    edv_ml: float = Field(
        ...,
        ge=40.0,
        le=600.0,
        description="End-Diastolic Volume measured via cardiac MRI (mL).",
    )
    esv_ml: float = Field(
        ...,
        ge=15.0,
        le=500.0,
        description="End-Systolic Volume measured via cardiac MRI (mL).",
    )
    peth: PEthMeasurement = Field(
        ...,
        description="Baseline PEth blood test measurement and timestamp.",
    )
    ttn_variant: TTNVariant = Field(
        default=TTNVariant.NOT_DETECTED,
        description="Titin (TTN) variant sequencing classification (legacy enum).",
    )
    ttn_classification: Optional[TTNClassification] = Field(
        default=None,
        description="ACMG-AMP 5-tier classification of Titin (TTN) genetic findings.",
    )
    ttn_provenance: str = Field(
        default="ClinVar / ACMG-AMP 2015 Tiered Assessment",
        description="Provenance and classification standard for TTN genetic call.",
    )
    lge_status: LGEStatus = Field(
        default=LGEStatus.NEGATIVE,
        description="Late Gadolinium Enhancement myocardial fibrosis status.",
    )
    lge_pattern: LGEPattern = Field(
        default=LGEPattern.NONE,
        description="Late Gadolinium Enhancement spatial distribution pattern.",
    )
    lge_extent_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50.0,
        description="Quantified replacement fibrosis extent as a percentage of total LV myocardial mass.",
    )
    lge_available: bool = Field(
        default=True,
        description="Whether LGE tissue characterization was successfully acquired.",
    )
    rvedv_ml: Optional[float] = Field(
        default=None,
        ge=30.0,
        le=600.0,
        description="Right Ventricular End-Diastolic Volume (mL).",
    )
    rvesv_ml: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=500.0,
        description="Right Ventricular End-Systolic Volume (mL).",
    )
    rvef_percent: Optional[float] = Field(
        default=None,
        ge=5.0,
        le=85.0,
        description="Right Ventricular Ejection Fraction (%).",
    )
    myocardial_mass_g: Optional[float] = Field(
        default=None,
        ge=40.0,
        le=500.0,
        description="Estimated Left Ventricular Myocardial Mass (g) assuming density = 1.05 g/mL.",
    )
    reported_drinks_per_week: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Self-reported average weekly standard alcoholic drinks.",
    )
    time_since_last_drink_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=365,
        description="Reported elapsed time in days since last alcohol consumption.",
    )
    years_heavy_drinking: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=70.0,
        description="Cumulative estimated years of heavy alcohol exposure (>80 g/day).",
    )
    alcohol_history_available: bool = Field(
        default=True,
        description="Whether self-reported alcohol consumption history is documented.",
    )
    peth_available: bool = Field(
        default=True,
        description="Whether PEth blood quantification was successfully acquired at baseline.",
    )
    mri_source: str = Field(
        default="SYNTHETIC_SIMULATED",
        description="Origin of structural cardiac dimensions (e.g. ACDC_DERIVED vs SYNTHETIC).",
    )
    measurement_date: Optional[date] = Field(
        default=None,
        description="Date of baseline cardiac MRI acquisition.",
    )
    provenance: ProvenanceType = Field(
        default=ProvenanceType.SYNTHETIC_SIMULATED,
        description="Primary data origin and simulation tier.",
    )
    generation_version: str = Field(
        default="v1.0-m3a",
        description="Version string of the synthetic cohort generation model.",
    )
    generation_seed: Optional[int] = Field(
        default=None,
        description="Random seed used to generate this synthetic patient record.",
    )
    variable_provenance: Dict[str, str] = Field(
        default_factory=dict,
        description="Granular provenance tracking for each individual clinical feature.",
    )
    follow_up: FollowUpOutcomes = Field(
        default_factory=FollowUpOutcomes,
        description="Six-month follow-up assessment and recovery status.",
    )

    @property
    def volume_derived_lvef(self) -> float:
        """Calculate Left Ventricular Ejection Fraction (%) strictly from MRI volumes.

        Formula: LVEF = ((EDV - ESV) / EDV) * 100
        """
        sv = self.edv_ml - self.esv_ml
        return (sv / self.edv_ml) * 100.0

    @property
    def stroke_volume_ml(self) -> float:
        """Calculate Stroke Volume (SV = EDV - ESV) in mL."""
        return self.edv_ml - self.esv_ml

    @property
    def delta_lvef(self) -> Optional[float]:
        """Calculate absolute change in LVEF between baseline and 6-month follow-up."""
        if self.follow_up.follow_up_lvef is not None:
            return round(self.follow_up.follow_up_lvef - self.baseline_lvef, 2)
        return None

    @property
    def recovery_binary(self) -> Optional[int]:
        """Binary recovery endpoint: 1 if delta LVEF >= 10.0% points, else 0."""
        d = self.delta_lvef
        if d is not None:
            return 1 if d >= 10.0 else 0
        return None

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Patient identifier cannot be empty or whitespace.")
        return v_stripped

    @model_validator(mode="after")
    def validate_cardiac_volumetrics(self) -> PatientProfile:
        """Validate volumetric consistency between EDV and ESV and synchronize classifications."""
        if self.esv_ml >= self.edv_ml:
            raise ValueError(
                f"End-systolic volume ({self.esv_ml} mL) must be strictly less than "
                f"end-diastolic volume ({self.edv_ml} mL)."
            )

        # Confirm consistency between reported baseline LVEF and volume-derived LVEF
        calc_lvef = self.volume_derived_lvef
        if abs(calc_lvef - self.baseline_lvef) > 5.0:
            raise ValueError(
                f"Reported baseline LVEF ({self.baseline_lvef:.1f}%) deviates by >5% "
                f"from volume-derived LVEF ({calc_lvef:.1f}%)."
            )

        # Synchronize TTN classification and TTN variant
        if self.ttn_classification is None:
            self.ttn_classification = map_ttn_variant_to_classification(self.ttn_variant)
        elif self.ttn_variant == TTNVariant.NOT_DETECTED and self.ttn_classification != TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC:
            self.ttn_variant = map_ttn_classification_to_variant(self.ttn_classification)

        # Synchronize follow-up outcomes delta and binary label if follow-up LVEF is set
        if self.follow_up.follow_up_lvef is not None:
            calc_delta = round(self.follow_up.follow_up_lvef - self.baseline_lvef, 2)
            if self.follow_up.delta_lvef is None:
                self.follow_up.delta_lvef = calc_delta
            if self.follow_up.recovery_binary is None:
                self.follow_up.recovery_binary = 1 if calc_delta >= 10.0 else 0

            # Update recovery category if still PENDING_EVALUATION
            if self.follow_up.recovery_category == RecoveryCategory.PENDING_EVALUATION:
                if self.follow_up.follow_up_lvef >= 50.0:
                    self.follow_up.recovery_category = RecoveryCategory.COMPLETE_RECOVERY
                elif calc_delta >= 10.0:
                    self.follow_up.recovery_category = RecoveryCategory.PARTIAL_RECOVERY
                elif calc_delta >= -5.0:
                    self.follow_up.recovery_category = RecoveryCategory.STABLE_NON_RECOVERY
                else:
                    self.follow_up.recovery_category = RecoveryCategory.DETERIORATION

        return self
