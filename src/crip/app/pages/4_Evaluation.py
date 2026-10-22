"""Evaluation page for CRIP."""

import streamlit as st
from crip.app.components.disclaimer import render_clinical_disclaimer


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - Evaluation",
        page_icon="🔬",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("🔬 Model Evaluation & Clinical Metrics")
    st.markdown(
        """
        Rigorous, transparent validation protocols to evaluate model calibration, discrimination,
        and subgroup stability across synthetic cohorts.
        """
    )

    st.warning(
        "🛑 **Milestone 1 Constraint Notice:** Clinical validation results are strictly not fabricated. "
        "Per research specifications, model training, cross-validation, and hold-out evaluation will be "
        "executed during Milestone 4. Below is the predefined statistical evaluation protocol."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Planned Statistical Metrics")
        st.markdown(
            r"""
            - **Discrimination:**
              - Area Under Receiver Operating Characteristic Curve (ROC-AUC) for binary recovery ($\ge 50\%$ LVEF).
              - Sensitivity, Specificity, Positive Predictive Value (PPV), Negative Predictive Value (NPV).
            - **Calibration:**
              - Brier Score (mean squared error of probabilistic predictions).
              - Calibration slope and intercept with reliability curves.
            - **Continuous Trajectory Accuracy:**
              - Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on follow-up LVEF (%).
              - $R^2$ variance explained on independent validation sets.
            """
        )

    with col2:
        st.subheader("🛡️ Clinical Fairness & Subgroup Analysis")
        st.markdown(
            """
            Models will be scrutinized across key clinical stratified subgroups:
            - **Genetics:** *TTN* variant carrier vs. wild type.
            - **Fibrosis:** Presence vs. absence of Late Gadolinium Enhancement (LGE).
            - **Exposure:** Baseline PEth $> 200\\text{ ng/mL}$ vs. moderate exposure ($20-200\\text{ ng/mL}$).
            - **Demographics:** Biological sex and age groups.

            This prevents hidden disparate error rates across vulnerable phenotypes.
            """
        )

    st.divider()
    st.subheader("📋 Pre-registered Evaluation Checklist")
    checklist_items = [
        ("K-Fold Stratified Cross-Validation", "Enforces balanced outcome splits across folds"),
        ("Independent Hold-Out Test Set (20%)", "Protected test set untouched until final evaluation"),
        ("Permutation Feature Importance Verification", "Validates SHAP attribution consistency"),
        ("Brier Calibration Curves", "Ensures probabilities reflect empirical event rates"),
        ("Adversarial Perturbation Sensitivity", "Checks model stability under measurement noise"),
    ]

    for title, desc in checklist_items:
        st.markdown(f"- **{title}**: {desc}")


if __name__ == "__main__" or True:
    render_page()
