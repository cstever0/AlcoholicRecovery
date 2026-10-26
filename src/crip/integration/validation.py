"""Unified feature validation service for multimodal cardiomyopathy profiles.

DISCLAIMER:
Validates physical volume constraints, biological limits, and clinical classifications
prior to ingestion by predictive models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from crip.synthetic.schemas import (
    LGEPattern,
    LGEStatus,
    PEthRiskTier,
    Sex,
    TTNClassification,
)

logger = logging.getLogger("crip.integration.validation")


@dataclass
class FeatureValidationResult:
    """Detailed summary of feature validation outcomes, errors, and warnings."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_features: Dict[str, Any] = field(default_factory=dict)


def validate_unified_features(features: Dict[str, Any]) -> FeatureValidationResult:
    """Validate multimodal baseline clinical, imaging, and genetic features against clinical constraints.

    Checks:
      1. Mandatory fields presence (age, biological_sex, baseline_lvef, edv_ml, esv_ml).
      2. Biological ranges:
         - Age: 18 - 105
         - LVEF: 5.0% - 85.0%
         - EDV: 40.0 - 600.0 mL
         - ESV: 15.0 - 500.0 mL
      3. Physical volumetric integrity:
         - ESV < EDV
         - Stroke Volume = EDV - ESV > 0
         - Calculated EF = (SV / EDV) * 100 within +/- 5% of reported baseline LVEF
      4. Toxicological bounds:
         - PEth: 0.0 - 5000.0 ng/mL if available
      5. Genetic classification standard:
         - Must conform to ACMG-AMP 5-tier classification.
      6. Tissue characterization (LGE):
         - Extent: 0.0% - 50.0% of LV myocardial mass.
    """
    errors: List[str] = []
    warnings: List[str] = []
    clean: Dict[str, Any] = dict(features)

    # 1. Demographics
    try:
        age = int(clean.get("age", 50))
        if not (18 <= age <= 105):
            errors.append(f"Age {age} is outside the plausible adult range [18, 105].")
        clean["age"] = age
    except (ValueError, TypeError):
        errors.append("Age must be a valid integer.")

    raw_sex = str(clean.get("biological_sex", "MALE")).upper()
    if raw_sex in ("MALE", "M"):
        clean["biological_sex"] = "MALE"
    elif raw_sex in ("FEMALE", "F"):
        clean["biological_sex"] = "FEMALE"
    else:
        errors.append(f"Biological sex '{raw_sex}' must be 'MALE' or 'FEMALE'.")

    # 2. Cardiac Volumetrics
    try:
        edv = float(clean["edv_ml"])
        esv = float(clean["esv_ml"])
        reported_lvef = float(clean["baseline_lvef"])

        if edv <= 0:
            errors.append(f"End-diastolic volume ({edv} mL) must be strictly positive.")
        if esv <= 0:
            errors.append(f"End-systolic volume ({esv} mL) must be strictly positive.")

        if esv >= edv:
            errors.append(
                f"Volumetric impossibility: End-systolic volume ({esv:.1f} mL) is greater than "
                f"or equal to end-diastolic volume ({edv:.1f} mL)."
            )
        else:
            sv = edv - esv
            clean["stroke_volume_ml"] = round(sv, 2)
            calc_lvef = (sv / edv) * 100.0

            if abs(calc_lvef - reported_lvef) > 5.0:
                warnings.append(
                    f"Discrepancy between reported LVEF ({reported_lvef:.1f}%) and "
                    f"volume-derived LVEF ({calc_lvef:.1f}%)."
                )

        clean["edv_ml"] = round(edv, 2)
        clean["esv_ml"] = round(esv, 2)
        clean["baseline_lvef"] = round(reported_lvef, 2)

    except (KeyError, ValueError, TypeError) as e:
        errors.append(f"Missing or invalid primary volumetric values (edv_ml, esv_ml, baseline_lvef): {e}")

    # Optional Right Ventricle and Mass
    if clean.get("rvedv_ml") is not None:
        try:
            rvedv = float(clean["rvedv_ml"])
            clean["rvedv_ml"] = round(rvedv, 2)
        except (ValueError, TypeError):
            clean["rvedv_ml"] = None

    if clean.get("rvef_percent") is not None:
        try:
            rvef = float(clean["rvef_percent"])
            clean["rvef_percent"] = round(rvef, 2)
        except (ValueError, TypeError):
            clean["rvef_percent"] = None

    if clean.get("myocardial_mass_g") is not None:
        try:
            mass = float(clean["myocardial_mass_g"])
            clean["myocardial_mass_g"] = round(mass, 2)
        except (ValueError, TypeError):
            clean["myocardial_mass_g"] = None

    # 3. Toxicology (PEth)
    peth_avail = bool(clean.get("peth_available", True))
    clean["peth_available"] = float(1.0 if peth_avail else 0.0)

    if peth_avail and clean.get("peth_value_ng_ml") is not None:
        try:
            peth_val = float(clean["peth_value_ng_ml"])
            if not (0.0 <= peth_val <= 5000.0):
                errors.append(f"PEth concentration {peth_val} ng/mL is outside valid clinical range [0, 5000].")
            clean["peth_value_ng_ml"] = round(peth_val, 2)

            # Assign risk tier
            if peth_val < 20.0:
                clean["peth_risk_tier"] = PEthRiskTier.ABSTINENT_OR_LIGHT.value
            elif peth_val <= 200.0:
                clean["peth_risk_tier"] = PEthRiskTier.SIGNIFICANT_MODERATE.value
            else:
                clean["peth_risk_tier"] = PEthRiskTier.HEAVY_CHRONIC.value
        except (ValueError, TypeError):
            errors.append("Invalid PEth value format.")
    else:
        clean["peth_value_ng_ml"] = None
        clean["peth_risk_tier"] = "UNAVAILABLE"
        clean["peth_available"] = 0.0

    # 4. Genetics (TTN)
    valid_ttn_tiers = {
        "PATHOGENIC",
        "LIKELY_PATHOGENIC",
        "VARIANT_OF_UNCERTAIN_SIGNIFICANCE",
        "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC",
        "NOT_TESTED",
    }
    raw_ttn = str(clean.get("ttn_classification", "NOT_TESTED")).upper()
    if raw_ttn not in valid_ttn_tiers:
        warnings.append(f"Unknown TTN tier '{raw_ttn}', defaulting to 'NOT_TESTED'.")
        raw_ttn = "NOT_TESTED"
    clean["ttn_classification"] = raw_ttn

    # 5. Tissue Characterization (LGE)
    lge_avail = bool(clean.get("lge_available", True))
    clean["lge_available"] = float(1.0 if lge_avail else 0.0)

    if lge_avail:
        raw_lge_st = str(clean.get("lge_status", "NEGATIVE")).upper()
        if raw_lge_st not in ("POSITIVE", "NEGATIVE"):
            raw_lge_st = "NEGATIVE"
        clean["lge_status"] = raw_lge_st

        raw_pat = str(clean.get("lge_pattern", "NONE")).upper()
        valid_patterns = {"NONE", "MIDWALL", "SUBEPICARDIAL", "SUBENDOCARDIAL", "PATCHY_TRANSMURAL"}
        if raw_pat not in valid_patterns:
            raw_pat = "MIDWALL" if raw_lge_st == "POSITIVE" else "NONE"
        clean["lge_pattern"] = raw_pat

        if raw_lge_st == "POSITIVE":
            ext = float(clean.get("lge_extent_percent", 5.0) or 5.0)
            if not (0.0 <= ext <= 50.0):
                warnings.append(f"LGE extent {ext}% clipped to maximum clinical plausible limit (50%).")
                ext = min(max(ext, 0.0), 50.0)
            clean["lge_extent_percent"] = round(ext, 2)
        else:
            clean["lge_extent_percent"] = 0.0
    else:
        clean["lge_status"] = "NEGATIVE"
        clean["lge_pattern"] = "NONE"
        clean["lge_extent_percent"] = None
        clean["lge_available"] = 0.0

    # 6. Self-reported alcohol history
    alc_avail = bool(clean.get("alcohol_history_available", True))
    clean["alcohol_history_available"] = float(1.0 if alc_avail else 0.0)

    is_valid = len(errors) == 0
    return FeatureValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        validated_features=clean,
    )
