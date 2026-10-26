"""Comprehensive Integration Test Suite for CRIP Milestone 4.

Verifies end-to-end integration across:
  1. Provenance tracking and hypothetical case ID generation
  2. Physical volume and clinical bounds validation
  3. Unified patient profile and feature whitelist alignment
  4. End-to-end prediction workflow execution
  5. Mock HL7 FHIR R4 export, import, and round-trip integrity
  6. Structured research reporting (Markdown, HTML, JSON)
  7. Pre-configured demonstration scenarios (Cases A, B, and C)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from crip.integration.demo_cases import (
    DEMO_SCENARIO_CONFIGS,
    execute_demonstration_case,
    list_demonstration_cases,
)
from crip.integration.mock_fhir import (
    CLINVAR_SYSTEM,
    CRIP_SYSTEM,
    LOINC_SYSTEM,
    export_to_fhir_bundle,
    import_from_fhir_bundle,
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
from crip.integration.validation import FeatureValidationResult, validate_unified_features
from crip.integration.workflow import run_end_to_end_prediction_workflow
from crip.modeling.artifacts import load_recovery_model_artifacts


class TestMilestone4ProvenanceAndValidation:
    """Tests for hypothetical case ID generation, provenance, and clinical bounds validation."""

    def test_hypothetical_case_id_format(self):
        cid1 = generate_hypothetical_case_id()
        cid2 = generate_hypothetical_case_id()
        assert cid1.startswith("CRIP-SCENARIO-")
        assert cid2.startswith("CRIP-SCENARIO-")
        assert cid1 != cid2

    def test_provenance_enums_complete(self):
        assert CaseSourceType.CONSTRUCTED_MULTIMODAL_SCENARIO.value == "CONSTRUCTED_MULTIMODAL_SCENARIO"
        assert CaseSourceType.MOCK_FHIR_EHR_IMPORT.value == "MOCK_FHIR_EHR_IMPORT"
        assert FeatureSourceOrigin.REAL_ACDC_MRI_AUTOMATED_UNET.value == "REAL_ACDC_MRI_AUTOMATED_UNET"
        assert FeatureSourceOrigin.REAL_ACDC_MRI_EXPERT_MASK.value == "REAL_ACDC_MRI_EXPERT_MASK"
        assert FeatureSourceOrigin.SYNTHETIC_LITERATURE_GROUNDED.value == "SYNTHETIC_LITERATURE_GROUNDED"

    def test_valid_profile_passes_validation(self):
        data = {
            "age": 45,
            "biological_sex": "MALE",
            "edv_ml": 220.0,
            "esv_ml": 150.0,
            "baseline_lvef": 31.8,
            "peth_value_ng_ml": 350.0,
            "ttn_classification": "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC",
            "lge_status": "NEGATIVE",
        }
        res = validate_unified_features(data)
        assert res.is_valid is True
        assert len(res.errors) == 0
        assert res.validated_features["stroke_volume_ml"] == 70.0

    def test_impossible_volumes_rejected(self):
        # ESV >= EDV
        data = {
            "age": 50,
            "biological_sex": "MALE",
            "edv_ml": 150.0,
            "esv_ml": 160.0,
            "baseline_lvef": 30.0,
        }
        res = validate_unified_features(data)
        assert res.is_valid is False
        assert any("Volumetric impossibility" in err for err in res.errors)

    def test_negative_volume_rejected(self):
        data = {
            "age": 50,
            "biological_sex": "MALE",
            "edv_ml": -20.0,
            "esv_ml": 100.0,
            "baseline_lvef": 30.0,
        }
        res = validate_unified_features(data)
        assert res.is_valid is False
        assert any("strictly positive" in err for err in res.errors)

    def test_age_out_of_bounds_rejected(self):
        data = {
            "age": 12,  # Pediatric outside adult platform scope
            "biological_sex": "FEMALE",
            "edv_ml": 180.0,
            "esv_ml": 110.0,
            "baseline_lvef": 38.9,
        }
        res = validate_unified_features(data)
        assert res.is_valid is False
        assert any("Age" in err for err in res.errors)

    def test_peth_out_of_bounds_rejected(self):
        data = {
            "age": 50,
            "biological_sex": "MALE",
            "edv_ml": 200.0,
            "esv_ml": 130.0,
            "baseline_lvef": 35.0,
            "peth_value_ng_ml": 9999.0,  # Impossible blood concentration
            "peth_available": True,
        }
        res = validate_unified_features(data)
        assert res.is_valid is False
        assert any("PEth concentration" in err for err in res.errors)

    def test_lge_extent_clipped_to_plausible_limit(self):
        data = {
            "age": 50,
            "biological_sex": "MALE",
            "edv_ml": 200.0,
            "esv_ml": 130.0,
            "baseline_lvef": 35.0,
            "lge_status": "POSITIVE",
            "lge_extent_percent": 85.0,  # Clipped to 50%
        }
        res = validate_unified_features(data)
        assert res.is_valid is True
        assert res.validated_features["lge_extent_percent"] == 50.0
        assert len(res.warnings) > 0


class TestMilestone4PatientProfileAndModelAlignment:
    """Tests for UnifiedPatientProfile, feature whitelist compatibility, and serialization."""

    def test_model_feature_whitelist_exact_length_and_order(self):
        assert len(MODEL_FEATURE_WHITELIST) == 20
        assert MODEL_FEATURE_WHITELIST[0] == "age"
        assert MODEL_FEATURE_WHITELIST[1] == "baseline_lvef"
        assert MODEL_FEATURE_WHITELIST[19] == "lge_available"

    def test_profile_to_model_input_df_structure(self):
        profile = build_unified_patient_features(
            {"lv_edv_ml": 215.0, "lv_esv_ml": 145.0, "lvef_percent": 32.5},
            clinical_profile={"age": 52, "biological_sex": "MALE", "peth_value_ng_ml": 310.0},
        )
        df = profile.to_model_input_df()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 20)
        assert list(df.columns) == MODEL_FEATURE_WHITELIST
        # Indicators must be float
        assert isinstance(df["peth_available"].iloc[0], (float, np.floating))
        assert isinstance(df["alcohol_history_available"].iloc[0], (float, np.floating))
        assert isinstance(df["lge_available"].iloc[0], (float, np.floating))

    def test_profile_dictionary_round_trip(self):
        profile1 = build_unified_patient_features(
            {"lv_edv_ml": 230.0, "lv_esv_ml": 150.0},
            clinical_profile={"age": 49, "ttn_classification": "PATHOGENIC"},
        )
        d = profile1.to_dict()
        profile2 = UnifiedPatientProfile.from_dict(d)
        assert profile1.case_id == profile2.case_id
        assert profile1.edv_ml == profile2.edv_ml
        assert profile1.ttn_classification == profile2.ttn_classification

    def test_recovery_pipeline_predicts_on_unified_profile(self):
        pipeline, meta = load_recovery_model_artifacts()
        profile = build_unified_patient_features(
            {"lv_edv_ml": 220.0, "lv_esv_ml": 140.0, "lvef_percent": 36.4},
            clinical_profile={"peth_value_ng_ml": 280.0},
        )
        df = profile.to_model_input_df()
        prob = pipeline.predict_proba(df)[0, 1]
        assert 0.0 <= prob <= 1.0


class TestMilestone4WorkflowExecution:
    """Tests for end-to-end multimodal workflow orchestration."""

    def test_workflow_execution_with_real_acdc_mri(self):
        res = run_end_to_end_prediction_workflow(
            patient_id_acdc="patient001",
            use_automated_mri=False,  # Test with expert mask baseline
        )
        assert res["status"] == "SUCCESS"
        assert res["case_id"].startswith("CRIP-SCENARIO-")
        assert res["validation"]["is_valid"] is True
        assert res["mri_metrics"]["method"] == "EXPERT_GROUND_TRUTH_MASK"
        assert "recovery_prediction" in res
        assert 0.0 <= res["recovery_prediction"]["recovery_probability"] <= 1.0
        assert "explanation" in res
        assert "disclaimer" in res
        assert "RESEARCH DEMONSTRATION DISCLAIMER" in res["disclaimer"]

    def test_workflow_invalid_input_fails_gracefully(self):
        # Provide impossible volumes (ESV > EDV)
        res = run_end_to_end_prediction_workflow(
            patient_id_acdc=None,
            clinical_profile={"edv_ml": 120.0, "esv_ml": 180.0, "baseline_lvef": 25.0},
        )
        assert res["status"] == "VALIDATION_FAILED"
        assert res["validation"]["is_valid"] is False
        assert len(res["validation"]["errors"]) > 0
        assert res["recovery_prediction"] is None


class TestMilestone4FHIRInteroperability:
    """Tests for mock HL7 FHIR R4 Bundle export, import, and round-trip preservation."""

    def test_export_to_fhir_bundle_structure(self):
        profile = build_unified_patient_features(
            {"lv_edv_ml": 220.0, "lv_esv_ml": 145.0, "lvef_percent": 34.1},
            clinical_profile={
                "age": 58,
                "biological_sex": "FEMALE",
                "peth_value_ng_ml": 290.0,
                "ttn_classification": "PATHOGENIC",
                "lge_status": "POSITIVE",
                "lge_pattern": "MIDWALL",
                "lge_extent_percent": 7.5,
            },
        )
        bundle = export_to_fhir_bundle(profile)
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"

        resource_types = [entry["resource"]["resourceType"] for entry in bundle["entry"]]
        assert "Patient" in resource_types
        assert "Observation" in resource_types
        assert "DiagnosticReport" in resource_types

    def test_fhir_round_trip_integrity(self):
        original = build_unified_patient_features(
            {"lv_edv_ml": 235.0, "lv_esv_ml": 160.0, "lvef_percent": 31.9},
            clinical_profile={
                "age": 51,
                "biological_sex": "MALE",
                "peth_value_ng_ml": 410.0,
                "ttn_classification": "LIKELY_PATHOGENIC",
                "lge_status": "POSITIVE",
                "lge_pattern": "MIDWALL",
                "lge_extent_percent": 9.2,
            },
        )
        bundle = export_to_fhir_bundle(original)
        imported = import_from_fhir_bundle(bundle)

        assert imported.case_id == original.case_id
        assert imported.edv_ml == original.edv_ml
        assert imported.esv_ml == original.esv_ml
        assert imported.baseline_lvef == original.baseline_lvef
        assert imported.peth_value_ng_ml == original.peth_value_ng_ml
        assert imported.ttn_classification == original.ttn_classification
        assert imported.lge_status == original.lge_status
        assert imported.lge_extent_percent == original.lge_extent_percent
        assert imported.source_type == CaseSourceType.MOCK_FHIR_EHR_IMPORT.value

    def test_validate_fhir_bundle_schema_and_integrity(self):
        from crip.integration.mock_fhir import validate_fhir_bundle

        profile = build_unified_patient_features(
            {"lv_edv_ml": 210.0, "lv_esv_ml": 140.0, "lvef_percent": 33.3},
            clinical_profile={"peth_value_ng_ml": 350.0},
        )
        bundle = export_to_fhir_bundle(profile)
        val_res = validate_fhir_bundle(bundle)
        assert val_res["is_valid"] is True
        assert len(val_res["errors"]) == 0
        assert val_res["resource_counts"]["Patient"] == 1
        assert val_res["resource_counts"]["DiagnosticReport"] == 1

        # Test invalid bundle detection
        bad_val = validate_fhir_bundle({"resourceType": "NotABundle"})
        assert bad_val["is_valid"] is False
        assert len(bad_val["errors"]) > 0


class TestMilestone4DemonstrationCases:
    """Tests for curated demonstration cases A, B, and C."""

    def test_list_demonstration_cases(self):
        cases = list_demonstration_cases()
        assert len(cases) == 3
        ids = [c["case_id"] for c in cases]
        assert "CRIP-DEMO-CASE-A" in ids
        assert "CRIP-DEMO-CASE-B" in ids
        assert "CRIP-DEMO-CASE-C" in ids

    def test_execute_demo_case_a(self):
        # Case A: Favorable substrate (high reversible toxic insult, WT TTN, negative LGE)
        res = execute_demonstration_case("CRIP-DEMO-CASE-A", use_automated_mri=False)
        assert res["status"] == "SUCCESS"
        prob = res["recovery_prediction"]["recovery_probability"]
        tier = res["recovery_prediction"]["propensity_tier"]
        assert prob > 0.70
        assert tier == "HIGH_PROBABILITY_RECOVERY"

    def test_execute_demo_case_b(self):
        # Case B: Genetic & fibrotic substrate (TTNtv+, LGE midwall+) -> Attenuated recovery
        res = execute_demonstration_case("CRIP-DEMO-CASE-B", use_automated_mri=False)
        assert res["status"] == "SUCCESS"
        prob = res["recovery_prediction"]["recovery_probability"]
        tier = res["recovery_prediction"]["propensity_tier"]
        assert prob < 0.40
        assert tier == "LOW_PROBABILITY_RECOVERY"

    def test_execute_demo_case_c_missing_data(self):
        # Case C: Real-world missing data handles gracefully
        res = execute_demonstration_case("CRIP-DEMO-CASE-C", use_automated_mri=False)
        assert res["status"] == "SUCCESS"
        assert res["profile"]["peth_available"] == 0.0
        assert res["profile"]["alcohol_history_available"] == 0.0
        assert res["profile"]["lge_available"] == 0.0
        assert res["recovery_prediction"] is not None

        # Verify Plausible Sensitivity Analysis
        sens = res.get("sensitivity_analysis", {})
        assert sens["has_missing_modalities"] is True
        assert len(sens["missing_modalities"]) >= 3
        psr = sens["plausible_sensitivity_range"]
        assert psr is not None
        assert psr["pessimistic_probability"] <= res["recovery_prediction"]["recovery_probability"]
        assert psr["optimistic_probability"] >= res["recovery_prediction"]["recovery_probability"]
        assert "MUST NOT be interpreted as a clinically validated confidence interval" in psr["disclaimer"]


class TestMilestone4Reporting:
    """Tests for Markdown, HTML, and JSON structured research reporting."""

    def test_markdown_report_generation(self):
        res = execute_demonstration_case("CRIP-DEMO-CASE-A", use_automated_mri=False)
        md = generate_markdown_report(res)
        assert "# CRIP Multimodal Research Case Report" in md
        assert "CLINICAL RESEARCH DISCLAIMER" in md
        assert "CRIP-DEMO-CASE-A" in md
        assert "Simulated 6-Month Recovery Prediction" in md
        assert "Model Explainability & Attribution Decomposition" in md

    def test_html_report_generation(self):
        res = execute_demonstration_case("CRIP-DEMO-CASE-A", use_automated_mri=False)
        html = generate_html_report(res)
        assert "<!DOCTYPE html>" in html
        assert "CRIP Research Case Report" in html
        assert "disclaimer-box" in html
        assert "CRIP-DEMO-CASE-A" in html

    def test_json_report_generation(self):
        res = execute_demonstration_case("CRIP-DEMO-CASE-A", use_automated_mri=False)
        json_str = generate_json_report(res)
        parsed = json.loads(json_str)
        assert parsed["case_id"] == "CRIP-DEMO-CASE-A"
        assert "recovery_prediction" in parsed
