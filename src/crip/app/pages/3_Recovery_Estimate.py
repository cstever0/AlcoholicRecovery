"""Recovery Estimate, Explainability & FHIR Interoperability page for CRIP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crip.app.components.disclaimer import render_clinical_disclaimer
from crip.integration.demo_cases import (
    DEMO_SCENARIO_CONFIGS,
    execute_demonstration_case,
    list_demonstration_cases,
)
from crip.integration.mock_fhir import export_to_fhir_bundle, import_from_fhir_bundle
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
from crip.integration.workflow import run_end_to_end_prediction_workflow
from crip.modeling.artifacts import load_recovery_model_artifacts, save_recovery_model_artifacts
from crip.modeling.dataset import (
    DEFAULT_COHORT_CSV,
    FEATURE_WHITELIST,
    load_modeling_dataset,
)
from crip.modeling.explain import (
    explain_patient_prediction,
    get_global_logistic_regression_weights,
)
from crip.modeling.train import train_baseline_models
from crip.evaluation.recovery_metrics import evaluate_final_model_on_test

MODEL_DIR = Path("models/recovery")
METRICS_PATH = Path("data/processed/recovery_test_metrics.json")


@st.cache_resource
def get_or_train_recovery_model() -> Tuple[Any, Dict[str, Any]]:
    """Load the trained recovery model artifact, or fit and save it if not yet serialized."""
    try:
        model, metadata = load_recovery_model_artifacts(MODEL_DIR)
        return model, metadata
    except (FileNotFoundError, Exception):
        # Auto-train and serialize default model if artifact does not exist
        dataset = load_modeling_dataset(DEFAULT_COHORT_CSV, scenario="SUSTAINED_ABSTINENCE")
        models = train_baseline_models(dataset)
        lr_model = models["logistic_regression"]
        report = evaluate_final_model_on_test(lr_model, dataset)
        save_recovery_model_artifacts(lr_model, dataset, report, output_dir=MODEL_DIR)
        model, metadata = load_recovery_model_artifacts(MODEL_DIR)
        return model, metadata


@st.cache_data
def load_cohort_records(csv_path: str) -> Optional[pd.DataFrame]:
    """Load cached cohort dataframe for patient inspection."""
    path = Path(csv_path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - Recovery Estimation & Interoperability",
        page_icon="📈",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("📈 Simulated 6-Month LVEF Recovery Estimation & Interoperability")
    st.markdown(
        """
        Multimodal interpretable predictive modeling framework estimating the simulated probability
        that a patient experiences functional cardiac recovery (**improvement in LVEF of at least 10 percentage points**)
        over six months, **under an explicitly assumed sustained-abstinence scenario**, combining real cardiac
        MRI measurements with synthetic clinical biomarkers, genomics, and tissue characterization.
        """
    )

    # Load model
    try:
        model, metadata = get_or_train_recovery_model()
    except Exception as e:
        st.error(f"⚠️ Predictive model could not be loaded: {e}")
        return

    # Load cohort data for selection
    df_cohort = load_cohort_records(str(DEFAULT_COHORT_CSV))
    if df_cohort is not None and "assumed_abstinence_scenario" in df_cohort.columns:
        df_eligible = df_cohort[df_cohort["assumed_abstinence_scenario"] == "SUSTAINED_ABSTINENCE"].copy()
    else:
        df_eligible = df_cohort.copy() if df_cohort is not None else pd.DataFrame()

    # Sidebar: Case Input Selection
    st.sidebar.header("👤 Case Input Selection")
    input_modes = [
        "Synthetic Cohort Patient",
        "Curated Demonstration Scenario",
        "Transferred Real MRI (from Page 2)",
        "Mock HL7 FHIR Bundle Import",
    ]
    input_mode = st.sidebar.radio("Select Case Mode", input_modes, index=0)

    patient_profile: Optional[UnifiedPatientProfile] = None
    case_title = ""
    is_constructed_multimodal = False

    if input_mode == "Synthetic Cohort Patient":
        if df_eligible.empty:
            st.info("Synthetic cohort data file not found on disk.")
            return
        patient_ids = list(df_eligible["patient_id"].astype(str))
        selected_id = st.sidebar.selectbox("Select Synthetic Patient ID", patient_ids, index=0)
        prow = df_eligible[df_eligible["patient_id"].astype(str) == str(selected_id)].iloc[0]

        patient_profile = UnifiedPatientProfile(
            case_id=str(selected_id),
            source_type=CaseSourceType.SYNTHETIC_COHORT_PROFILE.value,
            age=int(prow["age"]),
            biological_sex=str(prow["biological_sex"]),
            mri_source=str(prow.get("mri_source", "SYNTHETIC")),
            baseline_lvef=float(prow["baseline_lvef"]),
            edv_ml=float(prow["edv_ml"]),
            esv_ml=float(prow["esv_ml"]),
            stroke_volume_ml=float(prow["stroke_volume_ml"]),
            rvedv_ml=float(prow["rvedv_ml"]) if pd.notna(prow.get("rvedv_ml")) else None,
            rvef_percent=float(prow["rvef_percent"]) if pd.notna(prow.get("rvef_percent")) else None,
            myocardial_mass_g=float(prow["myocardial_mass_g"]) if pd.notna(prow.get("myocardial_mass_g")) else None,
            peth_value_ng_ml=float(prow["peth_value_ng_ml"]) if pd.notna(prow.get("peth_value_ng_ml")) else None,
            peth_available=float(prow.get("peth_available", 1.0)),
            peth_risk_tier=str(prow.get("peth_risk_tier", "UNAVAILABLE")),
            reported_drinks_per_week=float(prow["reported_drinks_per_week"]) if pd.notna(prow.get("reported_drinks_per_week")) else None,
            time_since_last_drink_days=float(prow["time_since_last_drink_days"]) if pd.notna(prow.get("time_since_last_drink_days")) else None,
            years_heavy_drinking=float(prow["years_heavy_drinking"]) if pd.notna(prow.get("years_heavy_drinking")) else None,
            alcohol_history_available=float(prow.get("alcohol_history_available", 1.0)),
            ttn_classification=str(prow.get("ttn_classification", "NOT_TESTED")),
            lge_status=str(prow.get("lge_status", "NEGATIVE")),
            lge_pattern=str(prow.get("lge_pattern", "NONE")),
            lge_extent_percent=float(prow["lge_extent_percent"]) if pd.notna(prow.get("lge_extent_percent")) else 0.0,
            lge_available=float(prow.get("lge_available", 1.0)),
            notes="Direct record from synthetic cardiomyopathy cohort.",
        )
        case_title = f"Synthetic Patient `{selected_id}`"

    elif input_mode == "Curated Demonstration Scenario":
        demo_keys = list(DEMO_SCENARIO_CONFIGS.keys())
        demo_choice = st.sidebar.selectbox(
            "Select Scenario",
            demo_keys,
            format_func=lambda k: f"{k}: {DEMO_SCENARIO_CONFIGS[k]['title']}",
        )
        cfg = DEMO_SCENARIO_CONFIGS[demo_choice]
        st.sidebar.info(cfg["description"])

        with st.spinner(f"Loading MRI & clinical data for {demo_choice}..."):
            workflow_res = execute_demonstration_case(demo_choice, use_automated_mri=False)
            patient_profile = UnifiedPatientProfile.from_dict(workflow_res["profile"])

        case_title = f"{demo_choice} ({cfg['title']})"
        is_constructed_multimodal = True

    elif input_mode == "Transferred Real MRI (from Page 2)":
        transferred = st.session_state.get("transferred_mri_features")
        if not transferred:
            st.warning("⚠️ No MRI features have been transferred yet from Page 2 (MRI Analysis).")
            st.info("Navigate to **Page 2 (MRI Analysis)**, inspect any ACDC patient scan, and click **🚀 Transfer to Recovery Predictor**.")
            st.markdown("Or select a fallback ACDC patient below to load directly:")
            fallback_pt = st.selectbox("Select ACDC Subject", ["patient001", "patient002", "patient003", "patient004"])
            if st.button("Load ACDC Subject"):
                with st.spinner("Executing end-to-end MRI pipeline..."):
                    wf = run_end_to_end_prediction_workflow(patient_id_acdc=fallback_pt, use_automated_mri=False)
                    patient_profile = UnifiedPatientProfile.from_dict(wf["profile"])
                case_title = f"Hypothetical Scenario with {fallback_pt} CMR"
                is_constructed_multimodal = True
            else:
                return
        else:
            pt_id = transferred["acdc_patient_id"]
            st.sidebar.success(f"Loaded MRI Volumetrics for `{pt_id}` ({transferred['mri_method']})")
            st.sidebar.markdown(f"- **LVEDV:** {transferred['edv_ml']:.1f} mL\n- **LVEF:** {transferred['baseline_lvef']:.1f}%")

            # Allow customizing simulated clinical biomarkers for this scenario
            st.sidebar.markdown("---")
            st.sidebar.subheader("Simulate Clinical Characteristics")
            age_in = st.sidebar.slider("Age", 20, 85, 52)
            sex_in = st.sidebar.selectbox("Biological Sex", ["MALE", "FEMALE"])
            peth_in = st.sidebar.slider("PEth Concentration (ng/mL)", 0.0, 1500.0, 380.0, step=10.0)
            ttn_in = st.sidebar.selectbox("Titin (*TTN*) Genetic Tier", [
                "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC",
                "PATHOGENIC",
                "LIKELY_PATHOGENIC",
                "VARIANT_OF_UNCERTAIN_SIGNIFICANCE",
                "NOT_TESTED",
            ])
            lge_st_in = st.sidebar.selectbox("LGE Scar Status", ["NEGATIVE", "POSITIVE"])
            lge_ext_in = st.sidebar.slider("LGE Extent (% LV Mass)", 0.0, 30.0, 0.0 if lge_st_in == "NEGATIVE" else 8.5) if lge_st_in == "POSITIVE" else 0.0
            lge_pat_in = st.sidebar.selectbox("LGE Pattern", ["NONE", "MIDWALL", "SUBEPICARDIAL"]) if lge_st_in == "POSITIVE" else "NONE"

            patient_profile = build_unified_patient_features(
                mri_metrics=transferred,
                clinical_profile={
                    "age": age_in,
                    "biological_sex": sex_in,
                    "peth_value_ng_ml": peth_in,
                    "ttn_classification": ttn_in,
                    "lge_status": lge_st_in,
                    "lge_pattern": lge_pat_in,
                    "lge_extent_percent": lge_ext_in,
                },
                acdc_patient_id=pt_id,
                source_type=CaseSourceType.CONSTRUCTED_MULTIMODAL_SCENARIO,
            )
            case_title = f"Scenario: Real ACDC `{pt_id}` + Custom Simulated Biomarkers"
            is_constructed_multimodal = True

    elif input_mode == "Mock HL7 FHIR Bundle Import":
        st.subheader("📥 Mock HL7 FHIR R4 Bundle Ingestion")
        st.caption("Paste a FHIR R4 JSON Bundle containing Patient, Observation (PEth, LVEF, LVEDV, TTN, LGE), and DiagnosticReport resources.")

        # Provide a sample button to load a valid bundle
        sample_profile = build_unified_patient_features(
            {"lv_edv_ml": 240.0, "lv_esv_ml": 170.0, "lv_stroke_volume_ml": 70.0, "lvef_percent": 29.2},
            clinical_profile={"age": 55, "biological_sex": "MALE", "peth_value_ng_ml": 480.0, "ttn_classification": "PATHOGENIC", "lge_status": "POSITIVE", "lge_pattern": "MIDWALL", "lge_extent_percent": 11.0},
        )
        sample_json_str = json.dumps(export_to_fhir_bundle(sample_profile), indent=2)

        bundle_input = st.text_area("FHIR Bundle JSON", value=sample_json_str, height=220)
        if st.button("Parse & Ingest FHIR Bundle"):
            try:
                parsed_json = json.loads(bundle_input)
                patient_profile = import_from_fhir_bundle(parsed_json)
                st.success(f"Successfully ingested FHIR Bundle for Patient ID: `{patient_profile.case_id}`!")
            except Exception as e:
                st.error(f"Failed to parse FHIR bundle: {e}")
                return
        else:
            try:
                parsed_json = json.loads(bundle_input)
                patient_profile = import_from_fhir_bundle(parsed_json)
            except Exception:
                st.info("Click 'Parse & Ingest FHIR Bundle' to process.")
                return
        case_title = f"FHIR Ingestion: `{patient_profile.case_id}`"

    if patient_profile is None:
        st.warning("Please select or load a patient profile.")
        return

    # Multimodal Scenario Disclaimer Alert
    if is_constructed_multimodal:
        st.warning(
            f"🔬 **CONSTRUCTED RESEARCH DEMONSTRATION SCENARIO (`{patient_profile.case_id}`):**\n\n"
            "This profile integrates **real cardiac MRI scans from the ACDC challenge** with **simulated alcohol exposure "
            "biomarkers (PEth), genetic classifications (*TTN*), and CMR tissue characterization (LGE)**.\n\n"
            "**Real patient MRI and synthetic clinical records do NOT belong to the same human subject.** "
            "This constructed scenario is strictly for clinical AI workflow demonstration and must never be used clinically."
        )

    # Validate feature profile
    validation_res = patient_profile.validate()
    if not validation_res.is_valid:
        st.error(f"🛑 **Feature Validation Failed:** {validation_res.errors}")
        return

    # Baseline Profile Summary Card
    st.subheader(f"Baseline Clinical & Imaging Profile: `{patient_profile.case_id}`")
    st.caption(f"**Classification:** {patient_profile.source_type} | **Origin:** {case_title}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Age / Biological Sex", f"{int(patient_profile.age)}y / {patient_profile.biological_sex}")
        st.caption(f"MRI Source: `{patient_profile.mri_source}`")
    with c2:
        st.metric("Baseline LVEF", f"{float(patient_profile.baseline_lvef):.1f}%")
        st.metric("End-Diastolic Vol", f"{float(patient_profile.edv_ml):.1f} mL")
    with c3:
        if patient_profile.peth_available > 0.5 and patient_profile.peth_value_ng_ml is not None:
            st.metric("Baseline PEth", f"{float(patient_profile.peth_value_ng_ml):.1f} ng/mL")
            st.caption(f"Tier: `{patient_profile.peth_risk_tier}`")
        else:
            st.metric("Baseline PEth", "MISSING")
            st.caption("Observation unavailable")
    with c4:
        ttn_val = str(patient_profile.ttn_classification)
        st.metric("Titin (*TTN*) Call", ttn_val.replace("_", " "))
        st.caption("ACMG-AMP 5-tier classification")
    with c5:
        ext_str = f"{float(patient_profile.lge_extent_percent):.1f}%" if patient_profile.lge_extent_percent is not None else "0.0%"
        st.metric("LGE Fibrosis", f"{patient_profile.lge_status}")
        st.caption(f"Pattern: {patient_profile.lge_pattern} ({ext_str})")

    st.divider()

    # Model Inference for Selected Patient
    df_patient_input = patient_profile.to_model_input_df()
    explanation = explain_patient_prediction(
        pipeline=model,
        patient_row=df_patient_input,
        baseline_lvef=float(patient_profile.baseline_lvef),
    )

    pred_prob = explanation["simulated_recovery_probability"]

    # Missing Modalities Audit & Plausible Sensitivity Analysis
    missing_mods: list[str] = []
    if patient_profile.peth_available < 0.5 or patient_profile.peth_value_ng_ml is None:
        missing_mods.append("PEth (Phosphatidylethanol Biomarker)")
    if patient_profile.ttn_classification == "NOT_TESTED":
        missing_mods.append("Titin (TTN) Genetic Variant")
    if patient_profile.lge_available < 0.5 or patient_profile.lge_extent_percent is None:
        missing_mods.append("CMR Late Gadolinium Enhancement (LGE Scar)")
    if patient_profile.alcohol_history_available < 0.5:
        missing_mods.append("Self-Reported Alcohol Intake History")

    sensitivity_analysis: Dict[str, Any] = {
        "has_missing_modalities": len(missing_mods) > 0,
        "missing_modalities": missing_mods,
        "imputation_method": "Median for numerical features, Mode for categorical features, with binary indicator flags",
    }

    if missing_mods:
        df_opt = df_patient_input.copy()
        df_opt["peth_value_ng_ml"] = 450.0
        df_opt["peth_available"] = 1.0
        df_opt["ttn_classification"] = "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"
        df_opt["lge_status"] = "NEGATIVE"
        df_opt["lge_pattern"] = "NONE"
        df_opt["lge_extent_percent"] = 0.0
        df_opt["lge_available"] = 1.0
        p_opt = float(model.predict_proba(df_opt)[0, 1])

        df_pess = df_patient_input.copy()
        df_pess["peth_value_ng_ml"] = 15.0
        df_pess["peth_available"] = 1.0
        df_pess["ttn_classification"] = "PATHOGENIC"
        df_pess["lge_status"] = "POSITIVE"
        df_pess["lge_pattern"] = "MIDWALL"
        df_pess["lge_extent_percent"] = 15.0
        df_pess["lge_available"] = 1.0
        p_pess = float(model.predict_proba(df_pess)[0, 1])

        sensitivity_analysis["plausible_sensitivity_range"] = {
            "pessimistic_probability": round(pess_prob, 4) if "pess_prob" in locals() else round(p_pess, 4),
            "baseline_imputed_probability": round(pred_prob, 4),
            "optimistic_probability": round(p_opt, 4),
            "range_spread": round(p_opt - p_pess, 4),
            "disclaimer": (
                "This sensitivity range represents the spread of predictions across alternative plausible "
                "unobserved clinical assumptions. It MUST NOT be interpreted as a clinically validated confidence interval."
            ),
        }
    else:
        sensitivity_analysis["plausible_sensitivity_range"] = None

    # Assemble comprehensive result dict
    case_result_dict = {
        "case_id": patient_profile.case_id,
        "status": "SUCCESS",
        "timestamp": patient_profile.created_at,
        "profile": patient_profile.to_dict(),
        "validation": {
            "is_valid": validation_res.is_valid,
            "errors": validation_res.errors,
            "warnings": validation_res.warnings,
        },
        "mri_metrics": {
            "edv_ml": patient_profile.edv_ml,
            "esv_ml": patient_profile.esv_ml,
            "stroke_volume_ml": patient_profile.stroke_volume_ml,
            "baseline_lvef": patient_profile.baseline_lvef,
            "rvedv_ml": patient_profile.rvedv_ml,
            "rvef_percent": patient_profile.rvef_percent,
            "myocardial_mass_g": patient_profile.myocardial_mass_g,
            "method": patient_profile.mri_source,
            "patient_id": patient_profile.acdc_patient_id,
        },
        "recovery_prediction": {
            "recovery_probability": round(pred_prob, 4),
            "binary_prediction": int(pred_prob >= 0.50),
            "propensity_tier": "HIGH_PROBABILITY_RECOVERY" if pred_prob >= 0.70 else ("INTERMEDIATE_PROBABILITY_RECOVERY" if pred_prob >= 0.40 else "LOW_PROBABILITY_RECOVERY"),
            "assumed_scenario": "SUSTAINED_ABSTINENCE",
            "model_version": metadata.get("model_version", "v1.0-m3b"),
        },
        "explanation": explanation,
        "sensitivity_analysis": sensitivity_analysis,
    }

    # Prominent Missing Data Warning & Sensitivity Display
    if missing_mods:
        st.warning(
            f"⚠️ **PROGNOSTIC MODALITY MISSINGNESS ({len(missing_mods)} unmeasured features):**\n\n"
            f"The primary point estimate ({pred_prob:.1%}) relies on empirical median/mode imputation because the following "
            f"modalities are unmeasured in this profile: **{', '.join(missing_mods)}**.\n\n"
            f"**Plausible Sensitivity Range:** Under alternative unobserved clinical assumptions (from pessimistic non-reversible genetic/fibrotic "
            f"etiology to optimistic reversible toxic exposure), the estimated recovery probability spans **{p_pess:.1%} to {p_opt:.1%}**.\n\n"
            f"*Disclaimer: This range illustrates sensitivity to unmeasured parameters and is NOT a clinically validated confidence interval.*"
        )

    # Main Estimation Display
    pred_col, narrative_col = st.columns([1, 1.6])
    with pred_col:
        st.subheader("Simulated Recovery Probability")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100.0,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Estimated Probability of ΔLVEF ≥ 10%", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1f77b4" if pred_prob >= 0.50 else "#d62728"},
                "steps": [
                    {"range": [0, 40], "color": "#f8d7da"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            f"**Model Version:** `{metadata.get('model_version', 'v1.0-m3b')}` | "
            f"**Estimator:** `{metadata.get('estimator_type', 'LogisticRegression')}` | "
            f"**Target:** $\\Delta\\text{{LVEF}} \\ge 10\\%$ at 6 months under sustained abstinence."
        )

    with narrative_col:
        st.subheader("Clinical Attribution Narrative")
        st.info(explanation["narrative_explanation"])
        st.warning(explanation["causal_disclaimer"])

    st.divider()

    # Detailed Tabs
    tab_explain, tab_explorer, tab_reporting, tab_validation = st.tabs([
        "🔍 Feature Attribution Breakdown",
        "🧪 Hypothetical Scenario Explorer",
        "📄 Research Report & Mock FHIR Export",
        "📊 Model Calibration & Held-Out Metrics",
    ])

    with tab_explain:
        st.subheader("Local Feature Attribution (Log-Odds Contribution)")
        st.markdown(
            "This table shows how each clinical and imaging characteristic contributes to the simulated log-odds of recovery. "
            "Positive values increase estimated recovery probability; negative values decrease it."
        )

        df_att = pd.DataFrame(explanation["feature_attributions"])
        if not df_att.empty:
            df_disp = df_att[["feature", "raw_value", "weight", "odds_ratio", "attribution", "direction"]].copy()
            df_disp.columns = ["Feature", "Value", "Weight (β)", "Odds Ratio", "Attribution (Δ Log-Odds)", "Direction"]
            st.dataframe(df_disp, use_container_width=True)

            # Waterfall bar chart of top attributions
            df_top = df_att.head(10).iloc[::-1]
            fig_bar = go.Figure(go.Bar(
                x=df_top["attribution"],
                y=df_top["feature"],
                orientation="h",
                marker_color=["#2ca02c" if a > 0 else "#d62728" for a in df_top["attribution"]],
                text=[f"{a:+.2f}" for a in df_top["attribution"]],
                textposition="auto",
            ))
            fig_bar.update_layout(
                title="Top 10 Feature Contributions to Prediction",
                xaxis_title="Attribution to Log-Odds (β · x)",
                yaxis_title="Feature",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_explorer:
        st.subheader("Hypothetical Clinical Modification Simulation")
        st.caption(
            "Explore how changes in baseline clinical markers would alter the simulated recovery probability under this model. "
            "Does not imply direct causal intervention efficacy."
        )

        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            mod_peth = st.slider(
                "Modify PEth (ng/mL)",
                min_value=0.0,
                max_value=1500.0,
                value=float(patient_profile.peth_value_ng_ml or 150.0),
                step=10.0,
            )
        with exp_col2:
            ttn_options = [
                "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC",
                "PATHOGENIC",
                "LIKELY_PATHOGENIC",
                "VARIANT_OF_UNCERTAIN_SIGNIFICANCE",
                "NOT_TESTED",
            ]
            mod_ttn = st.selectbox(
                "Modify Titin (*TTN*) Variant",
                ttn_options,
                index=ttn_options.index(patient_profile.ttn_classification) if patient_profile.ttn_classification in ttn_options else 0,
            )
        with exp_col3:
            lge_options = ["NEGATIVE", "POSITIVE"]
            mod_lge = st.selectbox(
                "Modify LGE Fibrosis",
                lge_options,
                index=0 if patient_profile.lge_status == "NEGATIVE" else 1,
            )

        df_mod = df_patient_input.copy()
        df_mod["peth_value_ng_ml"] = float(mod_peth)
        df_mod["ttn_classification"] = str(mod_ttn)
        df_mod["lge_status"] = str(mod_lge)
        if mod_lge == "NEGATIVE":
            df_mod["lge_pattern"] = "NONE"
            df_mod["lge_extent_percent"] = 0.0
        else:
            df_mod["lge_pattern"] = "MIDWALL"
            df_mod["lge_extent_percent"] = 10.0

        mod_prob = float(model.predict_proba(df_mod)[0, 1])
        delta_p = mod_prob - pred_prob

        st.metric(
            "Counterfactual Recovery Probability",
            f"{mod_prob:.1%}",
            delta=f"{delta_p:+.1%} vs baseline",
            delta_color="normal",
        )

    with tab_reporting:
        st.subheader("📄 Structured Research Demonstration Reporting & Interoperability")
        st.caption("Generate publication-ready Markdown and HTML reports, or export HL7 FHIR R4 Bundles for mock EHR systems.")

        md_report = generate_markdown_report(case_result_dict)
        html_report = generate_html_report(case_result_dict)
        json_report = generate_json_report(case_result_dict)
        fhir_bundle = export_to_fhir_bundle(patient_profile)
        fhir_bundle_json = json.dumps(fhir_bundle, indent=2)

        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        with dl_col1:
            st.download_button(
                "📥 Download Markdown Report (.md)",
                data=md_report,
                file_name=f"CRIP_Report_{patient_profile.case_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "📥 Download HTML Report (.html)",
                data=html_report,
                file_name=f"CRIP_Report_{patient_profile.case_id}.html",
                mime="text/html",
                use_container_width=True,
            )
        with dl_col3:
            st.download_button(
                "📥 Download JSON Artifact (.json)",
                data=json_report,
                file_name=f"CRIP_Result_{patient_profile.case_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_col4:
            st.download_button(
                "📥 Download HL7 FHIR Bundle (.json)",
                data=fhir_bundle_json,
                file_name=f"FHIR_Bundle_{patient_profile.case_id}.json",
                mime="application/json",
                use_container_width=True,
            )

        rep_view_mode = st.radio("Preview Report Format", ["Markdown Preview", "HL7 FHIR R4 Bundle Preview"], horizontal=True)
        if rep_view_mode == "Markdown Preview":
            st.markdown(md_report)
        else:
            st.json(fhir_bundle)

    with tab_validation:
        st.subheader("Model Performance on Held-Out Synthetic Test Partition")
        test_metrics_path = METRICS_PATH
        if test_metrics_path.is_file():
            with open(test_metrics_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            m = report["metrics"]
            ci = report["confidence_intervals"]

            vk1, vk2, vk3, vk4 = st.columns(4)
            with vk1:
                st.metric("Held-Out ROC-AUC", f"{m['roc_auc']:.3f}", delta=f"95% CI: [{ci['roc_auc_ci_95'][0]:.3f}, {ci['roc_auc_ci_95'][1]:.3f}]")
            with vk2:
                st.metric("PR-AUC (Avg Precision)", f"{m['pr_auc']:.3f}")
            with vk3:
                st.metric("Brier Score", f"{m['brier_score']:.3f}", delta=f"95% CI: [{ci['brier_score_ci_95'][0]:.3f}, {ci['brier_score_ci_95'][1]:.3f}]")
            with vk4:
                st.metric("Test Accuracy", f"{m['accuracy']:.1%}", delta=f"Prevalence: {m['prevalence']:.1%}")

            # Calibration curve plot
            calib_points = report.get("calibration_curve", [])
            if calib_points:
                st.markdown("#### Binned Reliability Calibration Diagram")
                df_cal = pd.DataFrame(calib_points)
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=df_cal["mean_predicted_prob"],
                    y=df_cal["observed_frequency"],
                    mode="lines+markers",
                    name="Model Reliability",
                    line=dict(color="#1f77b4", width=3),
                ))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(color="gray", dash="dash"),
                ))
                fig_cal.update_layout(
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Empirical Observed Recovery Rate",
                    xaxis_range=[0, 1],
                    yaxis_range=[0, 1],
                    height=340,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.info("Held-out evaluation report not yet written to disk.")


if __name__ == "__main__" or True:
    render_page()
