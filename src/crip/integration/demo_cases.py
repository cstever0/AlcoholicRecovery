"""Pre-configured reproducible multimodal demonstration cases for CRIP.

DISCLAIMER:
Provides 3 curated demonstration cases pairing real ACDC cardiac MRI with
representative synthetic clinical, toxicological, and genetic features:
  - Case A (CRIP-DEMO-CASE-A): High reversible toxic insult, negative TTN, negative LGE (High recovery).
  - Case B (CRIP-DEMO-CASE-B): Genetic & fibrotic substrate (TTNtv+, LGE midwall+), attenuated recovery.
  - Case C (CRIP-DEMO-CASE-C): Real-world missing data (missing PEth, untested TTN, missing LGE).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from crip.integration.provenance import CaseSourceType
from crip.integration.workflow import run_end_to_end_prediction_workflow
from crip.synthetic.schemas import (
    LGEPattern,
    LGEStatus,
    PEthRiskTier,
    TTNClassification,
)

logger = logging.getLogger("crip.integration.demo_cases")

DEMO_SCENARIO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "CRIP-DEMO-CASE-A": {
        "case_id": "CRIP-DEMO-CASE-A",
        "title": "High Reversible Toxic Insult (Favorable Substrate)",
        "description": (
            "Real ACDC cine MRI (patient001, dilated LV) paired with high PEth (420 ng/mL), "
            "negative TTN variant, and absent myocardial scar (LGE negative). Under sustained "
            "abstinence, this profile exhibits high recovery propensity."
        ),
        "acdc_patient_id": "patient001",
        "clinical_profile": {
            "age": 48,
            "biological_sex": "MALE",
            "peth_value_ng_ml": 420.0,
            "peth_available": 1.0,
            "peth_risk_tier": PEthRiskTier.HEAVY_CHRONIC.value,
            "reported_drinks_per_week": 42.0,
            "time_since_last_drink_days": 4.0,
            "years_heavy_drinking": 14.0,
            "alcohol_history_available": 1.0,
            "ttn_classification": TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC.value,
            "lge_status": LGEStatus.NEGATIVE.value,
            "lge_pattern": LGEPattern.NONE.value,
            "lge_extent_percent": 0.0,
            "lge_available": 1.0,
            "notes": "Demonstration Case A: High alcohol exposure with intact myocardium and wildtype TTN.",
        },
    },
    "CRIP-DEMO-CASE-B": {
        "case_id": "CRIP-DEMO-CASE-B",
        "title": "Genetic Substrate & Midwall Fibrosis (Attenuated Recovery)",
        "description": (
            "Real ACDC cine MRI (patient002) paired with moderate PEth (140 ng/mL), "
            "pathogenic TTN truncating variant (TTNtv), and midwall replacement fibrosis (LGE 12.5%). "
            "The genetic cardiomyopathy substrate and irreversible fibrosis attenuate recovery potential."
        ),
        "acdc_patient_id": "patient002",
        "clinical_profile": {
            "age": 56,
            "biological_sex": "MALE",
            "peth_value_ng_ml": 140.0,
            "peth_available": 1.0,
            "peth_risk_tier": PEthRiskTier.SIGNIFICANT_MODERATE.value,
            "reported_drinks_per_week": 18.0,
            "time_since_last_drink_days": 6.0,
            "years_heavy_drinking": 8.0,
            "alcohol_history_available": 1.0,
            "ttn_classification": TTNClassification.PATHOGENIC.value,
            "lge_status": LGEStatus.POSITIVE.value,
            "lge_pattern": LGEPattern.MIDWALL.value,
            "lge_extent_percent": 12.5,
            "lge_available": 1.0,
            "notes": "Demonstration Case B: Dual toxic and genetic cardiomyopathy with midwall scar.",
        },
    },
    "CRIP-DEMO-CASE-C": {
        "case_id": "CRIP-DEMO-CASE-C",
        "title": "Clinical Incompleteness & Missing Biomarkers",
        "description": (
            "Real ACDC cine MRI (patient003) with missing toxicology (PEth unperformed), "
            "unperformed genetic sequencing, and absent gadolinium administration (LGE unavailable). "
            "Tests model robust median/mode imputation and missingness indicator handling."
        ),
        "acdc_patient_id": "patient003",
        "clinical_profile": {
            "age": 62,
            "biological_sex": "FEMALE",
            "peth_value_ng_ml": None,
            "peth_available": 0.0,
            "peth_risk_tier": "UNAVAILABLE",
            "reported_drinks_per_week": None,
            "time_since_last_drink_days": None,
            "years_heavy_drinking": None,
            "alcohol_history_available": 0.0,
            "ttn_classification": TTNClassification.NOT_TESTED.value,
            "lge_status": "NEGATIVE",
            "lge_pattern": "NONE",
            "lge_extent_percent": None,
            "lge_available": 0.0,
            "notes": "Demonstration Case C: Incomplete real-world records with missing indicators.",
        },
    },
}


def list_demonstration_cases() -> List[Dict[str, Any]]:
    """Return available demonstration scenario summaries."""
    return [
        {
            "case_id": cfg["case_id"],
            "title": cfg["title"],
            "description": cfg["description"],
            "acdc_patient_id": cfg["acdc_patient_id"],
        }
        for cfg in DEMO_SCENARIO_CONFIGS.values()
    ]


def execute_demonstration_case(
    case_key: str = "CRIP-DEMO-CASE-A",
    use_automated_mri: bool = False,
) -> Dict[str, Any]:
    """Execute one of the pre-configured demonstration cases end-to-end."""
    if case_key not in DEMO_SCENARIO_CONFIGS:
        raise KeyError(
            f"Unknown demonstration case '{case_key}'. Available: {list(DEMO_SCENARIO_CONFIGS.keys())}"
        )

    cfg = DEMO_SCENARIO_CONFIGS[case_key]
    return run_end_to_end_prediction_workflow(
        patient_id_acdc=cfg["acdc_patient_id"],
        clinical_profile=cfg["clinical_profile"],
        use_automated_mri=use_automated_mri,
        case_id=cfg["case_id"],
        source_type=CaseSourceType.CONSTRUCTED_MULTIMODAL_SCENARIO,
    )
