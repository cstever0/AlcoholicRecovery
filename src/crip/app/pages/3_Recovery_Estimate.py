"""Recovery Estimate & Explainability page for CRIP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crip.app.components.disclaimer import render_clinical_disclaimer
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
        page_title="CRIP - Recovery Estimation & Explainability",
        page_icon="📈",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("📈 Simulated 6-Month LVEF Recovery Estimation")
    st.markdown(
        """
        Multimodal interpretable predictive modeling framework estimating the simulated probability
        that a patient experiences functional cardiac recovery (**improvement in LVEF of at least 10 percentage points**)
        over six months, **under an explicitly assumed sustained-abstinence scenario**, using information
        available at the initial clinical presentation.
        """
    )

    # Load model
    try:
        model, metadata = get_or_train_recovery_model()
        model_loaded = True
    except Exception as e:
        model = None
        metadata = {}
        model_loaded = False
        st.error(f"⚠️ Predictive model could not be loaded: {e}")
        return

    # Load cohort data for selection
    df_cohort = load_cohort_records(str(DEFAULT_COHORT_CSV))
    if df_cohort is None:
        st.info("Synthetic cohort data file not found on disk. Please generate cohort via Milestone 3A generator.")
        return

    # Filter to eligible patients under sustained abstinence
    if "assumed_abstinence_scenario" in df_cohort.columns:
        df_eligible = df_cohort[df_cohort["assumed_abstinence_scenario"] == "SUSTAINED_ABSTINENCE"].copy()
    else:
        df_eligible = df_cohort.copy()

    # Sidebar: Patient Selection
    st.sidebar.header("👤 Patient Selection")
    patient_ids = list(df_eligible["patient_id"].astype(str))
    selected_id = st.sidebar.selectbox("Select Synthetic Patient ID", patient_ids, index=0)

    patient_row = df_eligible[df_eligible["patient_id"].astype(str) == str(selected_id)].iloc[0]

    # Baseline Profile Summary Card
    st.subheader(f"Baseline Clinical Profile: `{selected_id}`")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Age / Biological Sex", f"{int(patient_row['age'])}y / {patient_row['biological_sex']}")
        st.caption(f"MRI Source: `{patient_row.get('mri_source', 'SYNTHETIC')}`")
    with c2:
        st.metric("Baseline LVEF", f"{float(patient_row['baseline_lvef']):.1f}%")
        st.metric("End-Diastolic Vol", f"{float(patient_row['edv_ml']):.1f} mL")
    with c3:
        peth_val = patient_row.get("peth_value_ng_ml")
        peth_avail = bool(patient_row.get("peth_available", True))
        if peth_avail and pd.notna(peth_val):
            st.metric("Baseline PEth", f"{float(peth_val):.1f} ng/mL")
            st.caption(f"Tier: `{patient_row.get('peth_risk_tier', 'UNAVAILABLE')}`")
        else:
            st.metric("Baseline PEth", "MISSING")
            st.caption("Observation unavailable")
    with c4:
        ttn_val = str(patient_row.get("ttn_classification", "NOT_TESTED"))
        st.metric("Titin (*TTN*) Call", ttn_val.replace("_", " "))
        st.caption("ACMG-AMP 5-tier classification")
    with c5:
        lge_st = str(patient_row.get("lge_status", "NEGATIVE"))
        lge_ext = patient_row.get("lge_extent_percent")
        ext_str = f"{float(lge_ext):.1f}%" if pd.notna(lge_ext) else "0.0%"
        st.metric("LGE Fibrosis", lge_st)
        st.caption(f"Extent: {ext_str} of LV mass")

    st.divider()

    # Model Inference for Selected Patient
    df_patient_input = pd.DataFrame([patient_row.to_dict()])
    for col in ["peth_available", "alcohol_history_available", "lge_available"]:
        if col in df_patient_input.columns:
            df_patient_input[col] = df_patient_input[col].astype(float)
    explanation = explain_patient_prediction(
        pipeline=model,
        patient_row=df_patient_input,
        baseline_lvef=float(patient_row["baseline_lvef"]),
    )

    pred_prob = explanation["simulated_recovery_probability"]

    # Main Estimation Display
    pred_col, narrative_col = st.columns([1, 1.6])
    with pred_col:
        st.subheader("Simulated Recovery Probability")
        # Visual probability meter
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

    # Detailed Tabs: Local Explanation, Hypothetical Explorer, Model Validation
    tab_explain, tab_explorer, tab_validation = st.tabs([
        "🔍 Feature Attribution Breakdown",
        "🧪 Hypothetical Scenario Explorer",
        "📊 Model Calibration & Held-Out Metrics",
    ])

    with tab_explain:
        st.subheader("Patient-Level Feature Contribution Breakdown")
        st.markdown(
            """
            This waterfall chart displays the additive contribution of each baseline factor to the model's
            log-odds prediction. Positive bars (green) increase the estimated recovery probability under sustained
            abstinence, while negative bars (red) constrain recovery potential.
            """
        )

        attr_list = explanation["feature_attributions"]
        if attr_list:
            df_attr = pd.DataFrame(attr_list).head(12)
            df_attr = df_attr.sort_values(by="contribution_log_odds", ascending=True)

            colors = ["#2ca02c" if val >= 0 else "#d62728" for val in df_attr["contribution_log_odds"]]
            fig_bar = go.Figure(go.Bar(
                x=df_attr["contribution_log_odds"],
                y=df_attr["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in df_attr["contribution_log_odds"]],
                textposition="auto",
            ))
            fig_bar.update_layout(
                title=f"Log-Odds Feature Contributions for Patient {selected_id}",
                xaxis_title="Log-Odds Contribution (Additive to Intercept)",
                yaxis_title="Predictor Feature",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Tabular breakdown
            st.dataframe(pd.DataFrame(attr_list), use_container_width=True)

    with tab_explorer:
        st.subheader("Hypothetical What-If Scenario Explorer")
        st.markdown(
            """
            Modify hypothetical baseline characteristics below to observe how the statistical model's
            simulated prediction changes.

            > [!IMPORTANT]
            > **Non-Causal Notice:** This simulator computes observational statistical projections under
            > synthetic assumptions. Modifying a feature in this explorer does **NOT** reflect the biological
            > consequence of an intervention.
            """
        )

        scen_c1, scen_c2, scen_c3 = st.columns(3)
        with scen_c1:
            new_lvef = st.slider("Hypothetical Baseline LVEF (%)", 10.0, 50.0, float(patient_row["baseline_lvef"]), step=1.0)
            new_edv = st.slider("Hypothetical LVEDV (mL)", 120.0, 380.0, float(patient_row["edv_ml"]), step=5.0)
        with scen_c2:
            current_peth = float(patient_row["peth_value_ng_ml"]) if pd.notna(patient_row.get("peth_value_ng_ml")) else 200.0
            new_peth = st.slider("Hypothetical Baseline PEth (ng/mL)", 0.0, 1500.0, current_peth, step=25.0)
            ttn_options = ["NO_PATHOGENIC_OR_LIKELY_PATHOGENIC", "PATHOGENIC", "LIKELY_PATHOGENIC", "VARIANT_OF_UNCERTAIN_SIGNIFICANCE", "NOT_TESTED"]
            curr_ttn = str(patient_row.get("ttn_classification", "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"))
            idx_ttn = ttn_options.index(curr_ttn) if curr_ttn in ttn_options else 0
            new_ttn = st.selectbox("Hypothetical Titin Classification", ttn_options, index=idx_ttn)
        with scen_c3:
            lge_options = ["NEGATIVE", "POSITIVE"]
            curr_lge = str(patient_row.get("lge_status", "NEGATIVE"))
            idx_lge = lge_options.index(curr_lge) if curr_lge in lge_options else 0
            new_lge = st.selectbox("Hypothetical LGE Fibrosis Status", lge_options, index=idx_lge)
            new_extent = st.slider("Hypothetical Fibrosis Extent (% of LV mass)", 0.0, 30.0, float(patient_row.get("lge_extent_percent", 0.0) or 0.0), step=1.0)

        # Build hypothetical row
        hypo_dict = patient_row.to_dict()
        hypo_dict["baseline_lvef"] = new_lvef
        hypo_dict["edv_ml"] = new_edv
        hypo_dict["esv_ml"] = new_edv * (1.0 - (new_lvef / 100.0))
        hypo_dict["stroke_volume_ml"] = new_edv - hypo_dict["esv_ml"]
        hypo_dict["peth_value_ng_ml"] = new_peth
        hypo_dict["peth_available"] = 1.0
        hypo_dict["ttn_classification"] = new_ttn
        hypo_dict["lge_status"] = new_lge
        hypo_dict["lge_extent_percent"] = new_extent if new_lge == "POSITIVE" else 0.0
        hypo_dict["lge_pattern"] = "MIDWALL" if new_lge == "POSITIVE" else "NONE"
        hypo_dict["lge_available"] = 1.0

        df_hypo = pd.DataFrame([hypo_dict])
        for col in ["peth_available", "alcohol_history_available", "lge_available"]:
            if col in df_hypo.columns:
                df_hypo[col] = df_hypo[col].astype(float)
        hypo_prob = float(model.predict_proba(df_hypo)[0, 1])

        delta_prob = hypo_prob - pred_prob
        hc1, hc2 = st.columns(2)
        with hc1:
            st.metric(
                "Original Baseline Prediction",
                f"{pred_prob:.1%}",
            )
        with hc2:
            st.metric(
                "Hypothetical Scenario Prediction",
                f"{hypo_prob:.1%}",
                delta=f"{delta_prob:+.1%}",
            )

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
