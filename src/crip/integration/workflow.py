"""End-to-end integration workflow orchestrating MRI segmentation, volumetrics, and recovery prediction.

DISCLAIMER:
Integrates real cardiac MRI imaging (MONAI U-Net segmentation / ACDC dataset)
with simulated alcohol biomarkers, genetics, and clinical records.
Real patient MRI and synthetic clinical records are strictly combined as
hypothetical research demonstration cases and must never be used clinically.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from crip.integration.patient_features import (
    UnifiedPatientProfile,
    build_unified_patient_features,
)
from crip.integration.provenance import (
    CaseSourceType,
    FeatureSourceOrigin,
    generate_hypothetical_case_id,
)
from crip.integration.validation import FeatureValidationResult, validate_unified_features
from crip.modeling.artifacts import DEFAULT_MODEL_DIR, load_recovery_model_artifacts
from crip.modeling.explain import explain_patient_prediction
from crip.mri.acdc_loader import (
    DEFAULT_ACDC_TRAINING_DIR,
    ACDCDataLoader,
)
from crip.mri.cardiac_metrics import compute_biventricular_measurements
from crip.mri.segmentation_inference import (
    DEFAULT_CHECKPOINT_PATH,
    SegmentationInferenceEngine,
)

logger = logging.getLogger("crip.integration.workflow")


def run_end_to_end_prediction_workflow(
    patient_id_acdc: Optional[str] = "patient001",
    clinical_profile: Optional[Dict[str, Any]] = None,
    use_automated_mri: bool = True,
    acdc_data_dir: Union[Path, str] = DEFAULT_ACDC_TRAINING_DIR,
    mri_checkpoint_path: Union[Path, str] = DEFAULT_CHECKPOINT_PATH,
    recovery_model_dir: Union[Path, str] = DEFAULT_MODEL_DIR,
    case_id: Optional[str] = None,
    source_type: CaseSourceType = CaseSourceType.CONSTRUCTED_MULTIMODAL_SCENARIO,
) -> Dict[str, Any]:
    """Execute complete end-to-end CRIP multimodal pipeline.

    Steps:
      1. Ingest MRI from ACDC cohort (or fallback to provided volumetric parameters).
      2. If automated MRI requested and checkpoint available, perform 2D U-Net slice-by-slice
         segmentation, nearest-neighbor spatial reconstruction, and 3D volumetric calculation.
      3. Fuse imaging measurements with clinical profile into UnifiedPatientProfile.
      4. Validate all physical and biological constraints.
      5. If valid, execute Scikit-learn recovery prediction pipeline (v1.0-m3b).
      6. Decompose prediction into local feature attributions and odds ratios.
      7. Return structured dictionary containing complete artifacts and clinical disclaimers.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clin = dict(clinical_profile or {})
    mri_metrics_dict: Dict[str, Any] = {}
    mri_source_origin = FeatureSourceOrigin.MANUALLY_ENTERED_HYPOTHETICAL

    # Step 1: MRI Processing
    if patient_id_acdc is not None:
        try:
            loader = ACDCDataLoader(dataset_root=acdc_data_dir)
            # Check if patient exists in loader
            available_pts = set(loader.list_patients())

            # If not in default dir, check test directory
            if patient_id_acdc not in available_pts:
                test_dir = Path(acdc_data_dir).parent / "testing"
                if test_dir.exists():
                    test_loader = ACDCDataLoader(dataset_root=test_dir)
                    if patient_id_acdc in set(test_loader.list_patients()):
                        loader = test_loader
                        available_pts = set(test_loader.list_patients())

            if patient_id_acdc in available_pts:
                ed_frame = loader.load_cardiac_frame(patient_id_acdc, phase="ED", load_mask=True)
                es_frame = loader.load_cardiac_frame(patient_id_acdc, phase="ES", load_mask=True)

                checkpoint_file = Path(mri_checkpoint_path)
                if use_automated_mri and checkpoint_file.is_file():
                    logger.info(f"Running automated MONAI U-Net segmentation for {patient_id_acdc}...")
                    engine = SegmentationInferenceEngine(checkpoint_path=checkpoint_file)
                    inference_res = engine.predict_patient(loader, patient_id_acdc)
                    meas = inference_res.pred_metrics

                    mri_metrics_dict = {
                        "edv_ml": meas.lvedv_ml,
                        "esv_ml": meas.lvesv_ml,
                        "stroke_volume_ml": meas.lvsv_ml,
                        "baseline_lvef": meas.lvef_percent,
                        "rvedv_ml": meas.rvedv_ml,
                        "rvef_percent": meas.rvef_percent,
                        "myocardial_mass_g": meas.myocardial_mass_ed_g,
                        "method": "MONAI_2D_UNET_AUTOMATED",
                    }
                    mri_source_origin = FeatureSourceOrigin.REAL_ACDC_MRI_AUTOMATED_UNET
                else:
                    # Fallback to expert ground truth masks from ACDC
                    logger.info(f"Computing volumetrics from expert masks for {patient_id_acdc}...")
                    if ed_frame.mask_data is not None and es_frame.mask_data is not None:
                        meas = compute_biventricular_measurements(
                            ed_mask=ed_frame.mask_data,
                            es_mask=es_frame.mask_data,
                            voxel_spacing_mm=ed_frame.voxel_spacing,
                            patient_id=patient_id_acdc,
                        )
                        mri_metrics_dict = {
                            "edv_ml": meas.lvedv_ml,
                            "esv_ml": meas.lvesv_ml,
                            "stroke_volume_ml": meas.lvsv_ml,
                            "baseline_lvef": meas.lvef_percent,
                            "rvedv_ml": meas.rvedv_ml,
                            "rvef_percent": meas.rvef_percent,
                            "myocardial_mass_g": meas.myocardial_mass_ed_g,
                            "method": "EXPERT_GROUND_TRUTH_MASK",
                        }
                        mri_source_origin = FeatureSourceOrigin.REAL_ACDC_MRI_EXPERT_MASK
                    else:
                        mri_metrics_dict = {
                            "edv_ml": 210.0,
                            "esv_ml": 136.5,
                            "stroke_volume_ml": 73.5,
                            "baseline_lvef": 35.0,
                            "method": "DEFAULT_REFERENCE_VOLUMETRICS",
                        }
            else:
                logger.warning(f"Patient {patient_id_acdc} not discovered in {acdc_data_dir}.")
        except Exception as e:
            logger.error(f"Error loading ACDC MRI for {patient_id_acdc}: {e}. Falling back to clinical inputs.")

    # If MRI metrics not populated from dataset, populate from clinical profile or defaults
    if not mri_metrics_dict:
        mri_metrics_dict = {
            "edv_ml": clin.get("edv_ml", 210.0),
            "esv_ml": clin.get("esv_ml", 136.5),
            "stroke_volume_ml": clin.get("stroke_volume_ml", 73.5),
            "baseline_lvef": clin.get("baseline_lvef", 35.0),
            "rvedv_ml": clin.get("rvedv_ml", 160.0),
            "rvef_percent": clin.get("rvef_percent", 45.0),
            "myocardial_mass_g": clin.get("myocardial_mass_g", 130.0),
            "method": "MANUAL_OR_SYNTHETIC_SPECIFICATION",
        }
        mri_source_origin = FeatureSourceOrigin.MANUALLY_ENTERED_HYPOTHETICAL

    # Step 2: Assemble Unified Patient Profile
    profile = build_unified_patient_features(
        mri_metrics=mri_metrics_dict,
        clinical_profile=clin,
        case_id=case_id,
        source_type=source_type,
        mri_source_origin=mri_source_origin,
        acdc_patient_id=patient_id_acdc,
    )

    # Step 3: Validate Features
    validation_res = profile.validate()
    if not validation_res.is_valid:
        logger.warning(f"Validation failed for scenario {profile.case_id}: {validation_res.errors}")
        return {
            "case_id": profile.case_id,
            "status": "VALIDATION_FAILED",
            "timestamp": timestamp,
            "profile": profile.to_dict(),
            "validation": {
                "is_valid": False,
                "errors": validation_res.errors,
                "warnings": validation_res.warnings,
            },
            "mri_metrics": mri_metrics_dict,
            "recovery_prediction": None,
            "explanation": None,
            "disclaimer": "CRIP is an academic research proof of concept. Not for clinical use.",
        }

    # Step 4: Recovery Prediction & Attribution
    pipeline, metadata = load_recovery_model_artifacts(model_dir=recovery_model_dir)
    input_df = profile.to_model_input_df()

    # Predict probability of recovery under sustained abstinence
    prob_recovery = float(pipeline.predict_proba(input_df)[0, 1])
    binary_pred = int(prob_recovery >= 0.50)

    # Risk / Recovery Propensity Tier
    if prob_recovery >= 0.70:
        recovery_propensity = "HIGH_PROBABILITY_RECOVERY"
    elif prob_recovery >= 0.40:
        recovery_propensity = "INTERMEDIATE_PROBABILITY_RECOVERY"
    else:
        recovery_propensity = "LOW_PROBABILITY_RECOVERY"

    # Step 5: Feature Attribution
    explanation = explain_patient_prediction(
        pipeline=pipeline,
        patient_row=input_df,
        baseline_lvef=profile.baseline_lvef,
    )

    # Step 6: Missing-Data Reliability & Plausible Sensitivity Analysis
    missing_modalities: list[str] = []
    if profile.peth_available < 0.5 or profile.peth_value_ng_ml is None:
        missing_modalities.append("PEth (Phosphatidylethanol Biomarker)")
    if profile.ttn_classification == "NOT_TESTED":
        missing_modalities.append("Titin (TTN) Genetic Sequencing")
    if profile.lge_available < 0.5 or profile.lge_extent_percent is None:
        missing_modalities.append("CMR Late Gadolinium Enhancement (LGE Scar)")
    if profile.alcohol_history_available < 0.5:
        missing_modalities.append("Self-Reported Alcohol Intake History")

    sensitivity_analysis: Dict[str, Any] = {
        "has_missing_modalities": len(missing_modalities) > 0,
        "missing_modalities": missing_modalities,
        "imputation_method": "Median for numerical features, Mode for categorical features, with binary missing indicator flags",
    }

    if missing_modalities:
        # Optimistic Plausible Scenario: Reversible toxic etiology, wildtype TTN, absent LGE
        df_opt = input_df.copy()
        df_opt["peth_value_ng_ml"] = 450.0
        df_opt["peth_available"] = 1.0
        df_opt["ttn_classification"] = "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"
        df_opt["lge_status"] = "NEGATIVE"
        df_opt["lge_pattern"] = "NONE"
        df_opt["lge_extent_percent"] = 0.0
        df_opt["lge_available"] = 1.0
        opt_prob = float(pipeline.predict_proba(df_opt)[0, 1])

        # Pessimistic Plausible Scenario: Non-toxic cardiomyopathy, pathogenic TTN, extensive midwall fibrosis
        df_pess = input_df.copy()
        df_pess["peth_value_ng_ml"] = 15.0
        df_pess["peth_available"] = 1.0
        df_pess["ttn_classification"] = "PATHOGENIC"
        df_pess["lge_status"] = "POSITIVE"
        df_pess["lge_pattern"] = "MIDWALL"
        df_pess["lge_extent_percent"] = 15.0
        df_pess["lge_available"] = 1.0
        pess_prob = float(pipeline.predict_proba(df_pess)[0, 1])

        sensitivity_analysis["plausible_sensitivity_range"] = {
            "pessimistic_probability": round(pess_prob, 4),
            "baseline_imputed_probability": round(prob_recovery, 4),
            "optimistic_probability": round(opt_prob, 4),
            "range_spread": round(opt_prob - pess_prob, 4),
            "disclaimer": (
                "This sensitivity range represents the spread of predictions across alternative plausible "
                "unobserved clinical assumptions. It MUST NOT be interpreted as a clinically validated confidence interval."
            ),
        }
    else:
        sensitivity_analysis["plausible_sensitivity_range"] = None

    result = {
        "case_id": profile.case_id,
        "status": "SUCCESS",
        "timestamp": timestamp,
        "profile": profile.to_dict(),
        "validation": {
            "is_valid": True,
            "errors": [],
            "warnings": validation_res.warnings,
        },
        "mri_metrics": mri_metrics_dict,
        "recovery_prediction": {
            "recovery_probability": round(prob_recovery, 4),
            "binary_prediction": binary_pred,
            "propensity_tier": recovery_propensity,
            "assumed_scenario": "SUSTAINED_ABSTINENCE",
            "model_version": metadata.get("model_version", "v1.0-m3b"),
        },
        "explanation": explanation,
        "sensitivity_analysis": sensitivity_analysis,
        "metadata": {
            "acdc_patient_id": patient_id_acdc,
            "mri_method": mri_metrics_dict.get("method"),
            "source_type": source_type.value,
            "model_metadata": metadata,
        },
        "disclaimer": (
            "CRIP RESEARCH DEMONSTRATION DISCLAIMER: "
            "This output is generated by a research proof of concept combining real cardiac MRI "
            "with synthetic biomarkers and clinical outcomes. This output MUST NEVER be used "
            "for clinical diagnosis, risk stratification, or treatment decisions."
        ),
    }

    return result
