"""Provenance tracking and hypothetical case identifier generation for CRIP integration.

DISCLAIMER:
Real ACDC MRI data and synthetic clinical records must never be represented as
belonging to the same actual patient. When combining them for demonstration, a new
hypothetical case identifier is generated and explicitly labeled as a constructed scenario.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Dict, Optional


class CaseSourceType(str, Enum):
    """Origin classification of a patient record or constructed multimodal case."""
    CONSTRUCTED_MULTIMODAL_SCENARIO = "CONSTRUCTED_MULTIMODAL_SCENARIO"
    SYNTHETIC_COHORT_PROFILE = "SYNTHETIC_COHORT_PROFILE"
    MOCK_FHIR_EHR_IMPORT = "MOCK_FHIR_EHR_IMPORT"
    MANUAL_HYPOTHETICAL_INPUT = "MANUAL_HYPOTHETICAL_INPUT"
    REAL_ACDC_MRI_ONLY = "REAL_ACDC_MRI_ONLY"


class FeatureSourceOrigin(str, Enum):
    """Granular origin of an individual clinical or imaging feature."""
    REAL_ACDC_MRI_AUTOMATED_UNET = "REAL_ACDC_MRI_AUTOMATED_UNET"
    REAL_ACDC_MRI_EXPERT_MASK = "REAL_ACDC_MRI_EXPERT_MASK"
    SYNTHETIC_LITERATURE_GROUNDED = "SYNTHETIC_LITERATURE_GROUNDED"
    MANUALLY_ENTERED_HYPOTHETICAL = "MANUALLY_ENTERED_HYPOTHETICAL"
    MOCK_FHIR_OBSERVATION = "MOCK_FHIR_OBSERVATION"
    MISSING_CLINICAL_DATA = "MISSING_CLINICAL_DATA"


def generate_hypothetical_case_id(prefix: str = "CRIP-SCENARIO") -> str:
    """Generate a unique pseudo-anonymized identifier for a constructed demonstration case."""
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"{prefix}-{short_uuid}"
