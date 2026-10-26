"""Integration layer for CRIP multimodal workflow, FHIR interoperability, and reporting."""

from crip.integration.demo_cases import (
    DEMO_SCENARIO_CONFIGS,
    execute_demonstration_case,
    list_demonstration_cases,
)
from crip.integration.mock_fhir import (
    CLINVAR_SYSTEM,
    CRIP_SYSTEM,
    LOINC_SYSTEM,
    RADLEX_SYSTEM,
    SNOMED_SYSTEM,
    UCUM_SYSTEM,
    export_to_fhir_bundle,
    import_from_fhir_bundle,
    validate_fhir_bundle,
)
from crip.integration.patient_features import (
    MODEL_FEATURE_WHITELIST,
    UnifiedPatientProfile,
    build_unified_patient_features,
)
from crip.integration.provenance import (
    CaseSourceType,
    FeatureSourceOrigin,
    generate_hypothetical_case_id,
)
from crip.integration.reporting import (
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
)
from crip.integration.validation import (
    FeatureValidationResult,
    validate_unified_features,
)
from crip.integration.workflow import run_end_to_end_prediction_workflow

__all__ = [
    # Provenance
    "CaseSourceType",
    "FeatureSourceOrigin",
    "generate_hypothetical_case_id",
    # Validation
    "FeatureValidationResult",
    "validate_unified_features",
    # Unified Patient Features
    "MODEL_FEATURE_WHITELIST",
    "UnifiedPatientProfile",
    "build_unified_patient_features",
    # Workflow
    "run_end_to_end_prediction_workflow",
    # FHIR Interoperability
    "export_to_fhir_bundle",
    "import_from_fhir_bundle",
    "validate_fhir_bundle",
    "LOINC_SYSTEM",
    "UCUM_SYSTEM",
    "CLINVAR_SYSTEM",
    "RADLEX_SYSTEM",
    "SNOMED_SYSTEM",
    "CRIP_SYSTEM",
    # Reporting
    "generate_markdown_report",
    "generate_html_report",
    "generate_json_report",
    # Demo Scenarios
    "DEMO_SCENARIO_CONFIGS",
    "execute_demonstration_case",
    "list_demonstration_cases",
]
