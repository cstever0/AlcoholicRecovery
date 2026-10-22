"""MRI Analysis page for CRIP."""

import streamlit as st
import plotly.graph_objects as go
from crip.app.components.disclaimer import render_clinical_disclaimer


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - MRI Analysis",
        page_icon="🫀",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("🫀 Cardiac MRI Volumetric Analysis")
    st.markdown(
        """
        Exploration of cardiac magnetic resonance (CMR) imaging pipelines for left ventricular
        segmentation and volumetric assessment.
        """
    )

    st.info(
        "📌 **Milestone 1 Notice:** In accordance with project development constraints, automated "
        "download of external MRI cohorts (e.g. ACDC) and deep segmentation models are scheduled "
        "for Milestone 2. Below is the technical architecture and interactive volumetric engine."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔬 CMR Imaging Architecture")
        st.markdown(
            """
            The CRIP imaging pipeline is engineered to support:
            - **Image Formats:** NIfTI (`.nii`, `.nii.gz`) and DICOM sequences via `NiBabel` and `SimpleITK`.
            - **Sequence Types:** Short-axis Cine bSSFP (balanced steady-state free precession) across cardiac phases.
            - **Segmentation Target:** End-diastolic (ED) and end-systolic (ES) contours of the Left Ventricular blood pool and myocardium.
            - **Deep Learning Framework:** MONAI (Medical Open Network for AI) built on PyTorch.
            - **Derived Biometrics:**
              - End-Diastolic Volume ($EDV$)
              - End-Systolic Volume ($ESV$)
              - Stroke Volume ($SV = EDV - ESV$)
              - Ejection Fraction ($LVEF = \\frac{EDV - ESV}{EDV} \\times 100\\%$)
            """
        )

    with col2:
        st.subheader("📐 Interactive Volumetric Calculator")
        st.caption("Simulate CMR ventricular contour measurements to observe derived LVEF:")

        sim_edv = st.slider("End-Diastolic Volume (EDV, mL)", min_value=80.0, max_value=350.0, value=220.0, step=5.0)
        sim_esv = st.slider("End-Systolic Volume (ESV, mL)", min_value=30.0, max_value=sim_edv - 5.0, value=150.0, step=5.0)

        sv = sim_edv - sim_esv
        lvef = (sv / sim_edv) * 100.0

        st.metric("Computed Stroke Volume", f"{sv:.1f} mL")
        st.metric("Computed LVEF", f"{lvef:.1f}%")

        # Visualization of Volumes
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Ventricular Volumes",
            x=["EDV (Diastole)", "ESV (Systole)", "Stroke Volume (SV)"],
            y=[sim_edv, sim_esv, sv],
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            text=[f"{sim_edv:.1f} mL", f"{sim_esv:.1f} mL", f"{sv:.1f} mL"],
            textposition="auto",
        ))
        fig.update_layout(
            title="Ventricular Phase Volumetrics",
            yaxis_title="Volume (mL)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🧩 Pipeline Roadmap")
    st.markdown(
        """
        | Pipeline Phase | Status | Module | Description |
        |---|---|---|---|
        | I. Foundation & Schemas | **Active (Milestone 1)** | `crip.synthetic.schemas` | Pydantic data schemas, SQLite synthetic EHR |
        | II. Image Ingestion & Contouring | Queued (Milestone 2) | `crip.mri` | NiBabel/SimpleITK ingestion, MONAI U-Net segmentation |
        | III. Predictive Modeling | Queued (Milestone 3) | `crip.modeling` | Interpretable recovery modeling with SHAP explanations |
        | IV. Clinical Validation & Metrics | Queued (Milestone 4) | `crip.evaluation` | Calibration curves, Brier scores, decision curves |
        """
    )


if __name__ == "__main__" or True:
    render_page()
