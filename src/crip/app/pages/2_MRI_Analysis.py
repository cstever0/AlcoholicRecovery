"""MRI Analysis page for CRIP with Real ACDC Cardiac MRI Visualization and Volumetric Quantification."""

from pathlib import Path
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from crip.app.components.disclaimer import render_clinical_disclaimer
from crip.mri.acdc_loader import (
    ACDCDataError,
    ACDCDataLoader,
    ACDCCardiacFrame,
    DEFAULT_ACDC_TRAINING_DIR,
    normalize_for_display,
)
from crip.mri.cardiac_metrics import (
    BiventricularCardiacMeasurements,
    calculate_structure_volume_ml,
    compute_biventricular_measurements,
)

DEFAULT_COHORT_CSV = Path("data/processed/acdc/cardiac_measurements.csv")

# Predefined distinct colors for segmentation labels (RGB float in [0.0, 1.0])
COLOR_MAP = {
    1: np.array([0.15, 0.65, 1.0]),   # RV: Azure / Blue
    2: np.array([0.20, 0.85, 0.35]),  # Myocardium: Green
    3: np.array([1.0, 0.25, 0.25]),   # LV: Crimson / Red
}

COLOR_HEX = {
    1: "#26A6FF",
    2: "#33D959",
    3: "#FF4040",
}


def create_overlay_rgb(
    base_slice: np.ndarray,
    mask_slice: Optional[np.ndarray],
    alpha: float = 0.45,
    show_overlay: bool = True,
) -> np.ndarray:
    """Generate RGB overlay combining grayscale MRI slice and segmentation mask."""
    norm_img = normalize_for_display(base_slice)
    rgb = np.stack([norm_img, norm_img, norm_img], axis=-1)

    if not show_overlay or mask_slice is None:
        return rgb

    overlay = rgb.copy()
    for lbl, color in COLOR_MAP.items():
        region = (mask_slice == lbl)
        if np.any(region):
            overlay[region] = (1.0 - alpha) * rgb[region] + alpha * color

    return np.clip(overlay, 0.0, 1.0)


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - MRI Analysis",
        page_icon="🫀",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("🫀 Cardiac MRI Volumetric Analysis & Visualization")
    st.markdown(
        """
        Interactive exploration of short-axis Cine cardiac magnetic resonance (CMR) imaging
        from the Automated Cardiac Diagnosis Challenge (ACDC) dataset with quantitative
        biventricular functional measurements derived from expert segmentation masks.
        """
    )

    # Sidebar: Dataset configuration and patient selector
    st.sidebar.header("📁 Dataset Configuration")
    dataset_path_input = st.sidebar.text_input(
        "ACDC Dataset Directory",
        value=str(DEFAULT_ACDC_TRAINING_DIR),
        help="Path to ACDC dataset split (e.g., data/raw/acdc/training or data/raw/acdc/testing)",
    )

    loader = ACDCDataLoader(dataset_path_input)

    if not loader.is_available():
        st.error(
            f"⚠️ **ACDC Dataset Not Found at:** `{dataset_path_input}`\n\n"
            "Please check the dataset path or ensure the ACDC dataset is downloaded to `data/raw/acdc/training`.\n"
            "The interactive volumetric simulator below remains available for testing."
        )
        _render_volumetric_simulator()
        return

    patients = loader.list_patients()
    if not patients:
        st.warning("No patient directories discovered in specified dataset path.")
        _render_volumetric_simulator()
        return

    # Patient Selection
    st.sidebar.subheader("👤 Select Patient")
    selected_patient_id = st.sidebar.selectbox(
        "Patient ID",
        patients,
        index=0,
        help="Choose a patient from the ACDC cohort",
    )

    try:
        metadata = loader.get_patient_metadata(selected_patient_id)
        # Load both ED and ES frames for complete biventricular metrics
        ed_frame: ACDCCardiacFrame = loader.load_cardiac_frame(selected_patient_id, phase="ED", load_mask=True)
        es_frame: ACDCCardiacFrame = loader.load_cardiac_frame(selected_patient_id, phase="ES", load_mask=True)
    except ACDCDataError as e:
        st.error(f"Error loading imaging data for {selected_patient_id}: {e}")
        _render_volumetric_simulator()
        return

    # Compute comprehensive measurements
    measurements: Optional[BiventricularCardiacMeasurements] = None
    if ed_frame.mask_data is not None and es_frame.mask_data is not None:
        try:
            measurements = compute_biventricular_measurements(
                ed_mask=ed_frame.mask_data,
                es_mask=es_frame.mask_data,
                voxel_spacing_mm=ed_frame.voxel_spacing,
                patient_id=selected_patient_id,
            )
        except Exception as e:
            st.warning(f"Unable to compute measurements for {selected_patient_id}: {e}")

    # Section 1: Patient Summary & Calculated Functional Parameters
    st.subheader(f"📊 Quantitative Cardiac Measurements — {selected_patient_id} ({metadata.group})")

    st.info(
        "🔬 **Measurement Methodology & Clinical Reference Distinction:**\n\n"
        "The cardiac functional parameters below are calculated directly by summing voxels from expert-provided "
        "ACDC ground-truth masks ($V = N_{\\text{voxels}} \\times \\frac{\\Delta x \\cdot \\Delta y \\cdot \\Delta z}{1000}\\text{ mL}$). "
        "**These values represent experimental geometric integrations from research contours and must not be "
        "interpreted as clinically validated reference standards.** Independent external clinical consensus "
        "measurements are not provided in the challenge distribution."
    )

    if measurements:
        row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(5)
        with row1_col1:
            st.metric("Patient ID", selected_patient_id)
            st.metric("Diagnostic Group", metadata.group)
        with row1_col2:
            st.metric("LV End-Diastolic Vol (LVEDV)", f"{measurements.lvedv_ml:.1f} mL")
            st.metric("RV End-Diastolic Vol (RVEDV)", f"{measurements.rvedv_ml:.1f} mL")
        with row1_col3:
            st.metric("LV End-Systolic Vol (LVESV)", f"{measurements.lvesv_ml:.1f} mL")
            st.metric("RV End-Systolic Vol (RVESV)", f"{measurements.rvesv_ml:.1f} mL")
        with row1_col4:
            st.metric("LV Stroke Volume (LVSV)", f"{measurements.lvsv_ml:.1f} mL")
            st.metric("RV Stroke Volume (RVSV)", f"{measurements.rvsv_ml:.1f} mL")
        with row1_col5:
            st.metric("LV Ejection Fraction (LVEF)", f"{measurements.lvef_percent:.1f}%")
            st.metric("RV Ejection Fraction (RVEF)", f"{measurements.rvef_percent:.1f}%")

        m_mass_col1, m_mass_col2, m_mass_col3 = st.columns(3)
        with m_mass_col1:
            st.metric("Myocardial Volume (ED)", f"{measurements.myocardial_volume_ed_ml:.1f} mL")
        with m_mass_col2:
            st.metric("Estimated Myocardial Mass (ED)", f"{measurements.myocardial_mass_ed_g:.1f} g", help="Assumes standard myocardial density = 1.05 g/mL")
        with m_mass_col3:
            st.metric("Physical Voxel Spacing", f"{ed_frame.voxel_spacing[0]:.2f} × {ed_frame.voxel_spacing[1]:.2f} × {ed_frame.voxel_spacing[2]:.2f} mm")

    st.divider()

    # Section 2: Visual Volume Comparison & Cohort Context
    v_col_left, v_col_right = st.columns([1, 1])

    with v_col_left:
        st.subheader("📈 Ventricular Phase Volume Comparison")
        if measurements:
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Bar(
                name="Left Ventricle (LV)",
                x=["End-Diastole (EDV)", "End-Systole (ESV)", "Stroke Volume (SV)"],
                y=[measurements.lvedv_ml, measurements.lvesv_ml, measurements.lvsv_ml],
                marker_color="#FF4040",
                text=[f"{measurements.lvedv_ml:.1f} mL", f"{measurements.lvesv_ml:.1f} mL", f"{measurements.lvsv_ml:.1f} mL"],
                textposition="auto",
            ))
            vol_fig.add_trace(go.Bar(
                name="Right Ventricle (RV)",
                x=["End-Diastole (EDV)", "End-Systole (ESV)", "Stroke Volume (SV)"],
                y=[measurements.rvedv_ml, measurements.rvesv_ml, measurements.rvsv_ml],
                marker_color="#26A6FF",
                text=[f"{measurements.rvedv_ml:.1f} mL", f"{measurements.rvesv_ml:.1f} mL", f"{measurements.rvsv_ml:.1f} mL"],
                textposition="auto",
            ))
            vol_fig.update_layout(
                barmode="group",
                title="Biventricular Cavity Volumes (ED vs. ES)",
                yaxis_title="Volume (mL)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(vol_fig, use_container_width=True)

    with v_col_right:
        st.subheader("🌐 ACDC Cohort Context")
        cohort_df = None
        if DEFAULT_COHORT_CSV.is_file():
            try:
                cohort_df = pd.read_csv(DEFAULT_COHORT_CSV)
            except Exception:
                cohort_df = None

        if cohort_df is not None and not cohort_df.empty and measurements:
            # Scatter plot of cohort with highlighted patient
            scatter_fig = px.scatter(
                cohort_df,
                x="lvedv_ml",
                y="lvef_percent",
                color="diagnostic_group",
                labels={"lvedv_ml": "LVEDV (mL)", "lvef_percent": "LVEF (%)", "diagnostic_group": "Group"},
                title="Cohort Distribution (LVEDV vs. LVEF)",
                color_discrete_map={"DCM": "#d62728", "HCM": "#9467bd", "MINF": "#ff7f0e", "NOR": "#2ca02c", "RV": "#1f77b4"},
                opacity=0.65,
            )
            # Add patient marker
            scatter_fig.add_trace(go.Scatter(
                x=[measurements.lvedv_ml],
                y=[measurements.lvef_percent],
                mode="markers",
                marker=dict(size=14, color="gold", symbol="star", line=dict(width=2, color="black")),
                name=f"Current: {selected_patient_id}",
            ))
            scatter_fig.add_hline(y=50.0, line_dash="dot", line_color="gray", annotation_text="50% EF Threshold")
            scatter_fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(scatter_fig, use_container_width=True)

            # Quantile position
            lvef_rank = (cohort_df["lvef_percent"] < measurements.lvef_percent).mean() * 100.0
            st.caption(f"📌 **Cohort Position:** {selected_patient_id} has an LVEF of **{measurements.lvef_percent:.1f}%** ({lvef_rank:.0f}th percentile of 100 ACDC cases).")
        else:
            st.info("Cohort summary data not yet generated. Run `python -m crip.mri.pipeline` to generate cohort benchmarks.")

    st.divider()

    # Section 3: MRI Display Controls & Slices
    st.subheader("🖼️ Cine Phase & Slice Navigation")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

    with ctrl_col1:
        phase = st.radio(
            "Cardiac Phase",
            options=["ED", "ES"],
            format_func=lambda x: f"{x} (End-Diastole, Frame {metadata.ed_frame:02d})" if x == "ED" else f"{x} (End-Systole, Frame {metadata.es_frame:02d})",
            horizontal=True,
        )

    with ctrl_col2:
        show_overlay = st.checkbox("Show Expert Segmentation Overlay", value=True)

    with ctrl_col3:
        overlay_alpha = st.slider(
            "Overlay Opacity (Alpha)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            disabled=not show_overlay,
        )

    current_frame = ed_frame if phase == "ED" else es_frame

    # Slice selector
    slice_idx = st.slider(
        "Short-Axis Slice Index (Base ➔ Apex)",
        min_value=0,
        max_value=current_frame.num_slices - 1,
        value=current_frame.num_slices // 2,
        help="Navigate through short-axis slices from ventricular base to apex",
    )

    img_slice = current_frame.get_slice_image(slice_idx, normalized=True)
    mask_slice = current_frame.get_slice_mask(slice_idx)

    # Physical properties caption
    st.caption(
        f"**Stack Dimensions:** {current_frame.num_slices} slices × {current_frame.height} rows × {current_frame.width} cols | "
        f"**Physical Voxel Spacing:** {current_frame.voxel_spacing[0]:.2f} × {current_frame.voxel_spacing[1]:.2f} × {current_frame.voxel_spacing[2]:.2f} mm | "
        f"**Selected Slice:** {slice_idx + 1} of {current_frame.num_slices}"
    )

    # Side-by-Side Comparison (Original vs Overlay)
    overlay_slice = create_overlay_rgb(
        img_slice,
        mask_slice,
        alpha=overlay_alpha,
        show_overlay=show_overlay,
    )

    fig, (ax_orig, ax_overlay) = plt.subplots(1, 2, figsize=(10, 5), facecolor="#0e1117")
    fig.patch.set_facecolor("#0e1117")

    # Left: Original grayscale
    ax_orig.imshow(img_slice, cmap="gray", origin="upper")
    ax_orig.set_title(
        f"Original CMR ({phase} - Slice {slice_idx + 1}/{current_frame.num_slices})",
        color="white",
        fontsize=12,
        pad=8,
    )
    ax_orig.axis("off")

    # Right: Overlay
    ax_overlay.imshow(overlay_slice, origin="upper")
    ax_overlay.set_title(
        f"Expert Ground-Truth Contours ({phase})",
        color="white",
        fontsize=12,
        pad=8,
    )
    ax_overlay.axis("off")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Legend for Segmentation Labels
    st.markdown(
        f"""
        <div style="display: flex; gap: 24px; justify-content: center; margin-top: 10px; margin-bottom: 20px;">
            <span><span style="display:inline-block;width:14px;height:14px;background-color:{COLOR_HEX[1]};border-radius:2px;margin-right:6px;"></span><b>Right Ventricle (RV) Cavity</b> (Label 1)</span>
            <span><span style="display:inline-block;width:14px;height:14px;background-color:{COLOR_HEX[2]};border-radius:2px;margin-right:6px;"></span><b>Myocardium</b> (Label 2)</span>
            <span><span style="display:inline-block;width:14px;height:14px;background-color:{COLOR_HEX[3]};border-radius:2px;margin-right:6px;"></span><b>Left Ventricle (LV) Cavity</b> (Label 3)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    _render_volumetric_simulator()


def _render_volumetric_simulator() -> None:
    """Render the interactive volumetric calculator and development roadmap."""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔬 CMR Imaging Architecture")
        st.markdown(
            r"""
            The CRIP imaging pipeline is engineered to support:
            - **Image Formats:** NIfTI (`.nii`, `.nii.gz`) sequences via `SimpleITK` and `NiBabel`.
            - **Sequence Types:** Short-axis Cine bSSFP across cardiac phases.
            - **Segmentation Targets:** End-diastolic (ED) and end-systolic (ES) contours of the LV blood pool, myocardium, and RV.
            - **Voxel-to-Volume Principle:**
              $$V_{\text{cavity}} = N_{\text{voxels}} \times \frac{\Delta x \times \Delta y \times \Delta z}{1000}\text{ mL}$$
            - **Derived Functional Biometrics:**
              - End-Diastolic Volume ($EDV$)
              - End-Systolic Volume ($ESV$)
              - Stroke Volume ($SV = EDV - ESV$)
              - Ejection Fraction ($EF = \frac{EDV - ESV}{EDV} \times 100\%$)
              - Estimated Myocardial Mass ($V_{\text{myo}} \times 1.05\text{ g/mL}$)
            """
        )

    with col2:
        st.subheader("📐 Interactive Geometric Volumetric Calculator")
        st.caption("Simulate CMR ventricular contour measurements to observe derived LVEF:")

        sim_edv = st.slider("End-Diastolic Volume (EDV, mL)", min_value=80.0, max_value=350.0, value=220.0, step=5.0, key="sim_edv")
        sim_esv = st.slider("End-Systolic Volume (ESV, mL)", min_value=30.0, max_value=sim_edv - 5.0, value=150.0, step=5.0, key="sim_esv")

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
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🧩 Pipeline Roadmap")
    st.markdown(
        """
        | Pipeline Phase | Status | Module | Description |
        |---|---|---|---|
        | I. Foundation & Schemas | **Completed (Milestone 1)** | `crip.synthetic.schemas` | Pydantic data schemas, SQLite synthetic EHR |
        | II-A. ACDC Ingestion & Visualization | **Completed (Milestone 2A)** | `crip.mri.acdc_loader` | Real ACDC MRI loading, multi-label mask overlay |
        | II-B. Volumetrics & Cohort Measurement | **Active (Milestone 2B)** | `crip.mri.cardiac_metrics`, `pipeline` | Physical volumetric validation & cohort analysis |
        | II-C. Deep Learning Segmentation | Queued (Milestone 2C) | `crip.mri.segmentation` | MONAI/PyTorch U-Net automated contouring |
        | III. Predictive Modeling | Queued (Milestone 3) | `crip.modeling` | Interpretable recovery modeling with SHAP explanations |
        | IV. Clinical Validation & Metrics | Queued (Milestone 4) | `crip.evaluation` | Calibration curves, Brier scores, decision curves |
        """
    )


if __name__ == "__main__" or True:
    render_page()
