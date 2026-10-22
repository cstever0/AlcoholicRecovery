"""Synthetic patient data schemas for the Cardiomyopathy Recovery Intelligence Platform.

DISCLAIMER:
All schemas and data representations herein are strictly designed for research
prototyping and synthetic benchmark investigation. They MUST NOT be used for
clinical diagnosis, patient risk stratification, or clinical treatment decisions.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Sex(str, Enum):
    """Biological sex."""
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


class TTNVariant(str, Enum):
    """Titin (TTN) genetic variant classification.
    
    TTN truncating variants (TTNtv) represent a frequent genetic predisposition
    in dilated cardiomyopathy and can interact with toxic alcohol exposure.
    """
    NOT_DETECTED = "NOT_DETECTED"
    PATHOGENIC_TRUNCATING = "PATHOGENIC_TRUNCATING"  # TTNtv
    VARIANT_UNCERTAIN_SIGNIFICANCE = "VARIANT_UNCERTAIN_SIGNIFICANCE"  # VUS
    BENIGN_OR_LIKELY_BENIGN = "BENIGN_OR_LIKELY_BENIGN"


class LGEStatus(str, Enum):
    """Late Gadolinium Enhancement (LGE) myocardial fibrosis status on cardiac MRI.
    
    Identifies presence and distribution of replacement fibrosis in the left ventricle.
    """
    NEGATIVE = "NEGATIVE"
    MIDWALL = "MIDWALL"  # Characteristic non-ischemic DCM pattern
    SUBEPICARDIAL = "SUBEPICARDIAL"
    SUBENDOCARDIAL = "SUBENDOCARDIAL"  # Typically ischemic
    PATCHY_TRANSMURAL = "PATCHY_TRANSMURAL"


class PEthRiskTier(str, Enum):
    """Phosphatidylethanol (PEth) clinical exposure interpretation tier."""
    ABSTINENT_OR_LIGHT = "ABSTINENT_OR_LIGHT"      # < 20 ng/mL
    SIGNIFICANT_MODERATE = "SIGNIFICANT_MODERATE"  # 20 - 200 ng/mL
    HEAVY_CHRONIC = "HEAVY_CHRONIC"                # > 200 ng/mL


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
    """Six-month longitudinal clinical follow-up outcomes."""
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
    adverse_cardiac_events: List[str] = Field(
        default_factory=list,
        description="Adverse cardiac events recorded during follow-up period (e.g. HF hospitalization).",
    )
    sustained_abstinence_confirmed: Optional[bool] = Field(
        default=None,
        description="Whether sustained alcohol abstinence was confirmed through follow-up PEth / history.",
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
        description="Titin (TTN) variant sequencing classification.",
    )
    lge_status: LGEStatus = Field(
        default=LGEStatus.NEGATIVE,
        description="Late Gadolinium Enhancement myocardial fibrosis status.",
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

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Patient identifier cannot be empty or whitespace.")
        return v_stripped

    @model_validator(mode="after")
    def validate_cardiac_volumetrics(self) -> PatientProfile:
        """Validate volumetric consistency between EDV and ESV."""
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

        return self
