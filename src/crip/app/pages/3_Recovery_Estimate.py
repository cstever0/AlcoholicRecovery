"""Recovery Estimate page for CRIP."""

import streamlit as st
import plotly.graph_objects as go
from crip.app.components.disclaimer import render_clinical_disclaimer


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - Recovery Estimate",
        page_icon="📈",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("📈 6-Month LVEF Recovery Estimation")
    st.markdown(
        """
        Multimodal interpretable modeling framework designed to investigate left ventricular
        recovery trajectories under cessation of toxic exposure and clinical management.
        """
    )

    st.warning(
        "🛑 **Milestone 1 Constraint Notice:** In strict adherence to development guidelines, "
        "machine learning models have **not been trained yet**, and no synthetic patient probabilities "
        "are fabricated. The panels below demonstrate the planned interface architecture and feature "
        "attribution layout scheduled for Milestone 3."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Feature Attribution Pipeline (Design)")
        st.markdown(
            """
            The recovery estimation pipeline will evaluate five core feature axes:
            1. **Toxic Exposure:** PEth blood level (recent exposure quantification) + reported cessation.
            2. **Ventricular Remodeling:** Baseline LVEF, End-Diastolic Volume Index (EDVI), Stroke Volume.
            3. **Genetic Predisposition:** Titin (*TTN*) truncating variants (*TTNtv*).
            4. **Myocardial Fibrosis:** Late Gadolinium Enhancement (LGE) distribution and burden.
            5. **Demographics:** Age, biological sex.

            **Model Architecture:** Interpretable Scikit-learn estimators (ElasticNet, Logistic Regression, 
            shallow trees) combined with tree/linear SHAP (SHapley Additive exPlanations) to provide 
            transparent clinical reasoning for every prediction.
            """
        )

    with col2:
        st.subheader("📊 Illustrative Trajectory Projection (Architecture)")
        st.caption("Conceptual visualization of 6-month ejection fraction monitoring:")

        months = [0, 1, 2, 3, 4, 5, 6]
        # Illustrative conceptual curves
        favorable_recovery = [28, 30, 34, 39, 44, 49, 52]
        incomplete_recovery = [28, 29, 31, 32, 34, 35, 36]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=favorable_recovery,
            mode="lines+markers",
            name="Complete Recovery Profile (Abstinence + LGE- / TTN-)",
            line=dict(color="#2ca02c", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=months, y=incomplete_recovery,
            mode="lines+markers",
            name="Suboptimal Recovery Profile (LGE+ / TTNtv+)",
            line=dict(color="#d62728", width=3, dash="dash"),
        ))
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Normal Threshold (50%)")
        fig.update_layout(
            title="LVEF Trajectory Over 6 Months (Conceptual Archetypes)",
            xaxis_title="Months From Baseline",
            yaxis_title="LVEF (%)",
            yaxis_range=[20, 60],
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔍 Planned Interpretability Layout (SHAP Integration)")
    st.markdown(
        """
        When predictive models are trained in Milestone 3, each case will display:
        - **Global Feature Importance:** Ranked impact of genetics vs. biomarker levels vs. fibrosis.
        - **Local SHAP Waterfall:** Individual risk factor contributions demonstrating *why* a particular 
          ejection fraction improvement is projected.
        - **Clinical Safety Guardrails:** Clear notation indicating model bounds and uncertainty intervals.
        """
    )


if __name__ == "__main__" or True:
    render_page()
