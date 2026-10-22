"""MRI Analysis page for CRIP with Real ACDC Cardiac MRI Visualization."""

from pathlib import Path
from typing import Optional, Tuple
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
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
from crip.mri.cardiac_metrics import calculate_structure_volume_ml

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
        from the Automated Cardiac Diagnosis Challenge (ACDC) dataset with expert ground-truth
        anatomical segmentations.
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

    # 5.1 Patient Selection
    st.sidebar.subheader("👤 Select Patient")
    selected_patient_id = st.sidebar.selectbox(
        "Patient ID",
        patients,
        index=0,
        help="Choose a patient from the ACDC cohort",
    )

    try:
        metadata = loader.get_patient_metadata(selected_patient_id)
    except ACDCDataError as e:
        st.error(f"Error loading metadata for {selected_patient_id}: {e}")
        _render_volumetric_simulator()
        return

    # Metadata display banner
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Patient ID", metadata.patient_id)
    with m_col2:
        st.metric("Diagnosis Group", metadata.group)
    with m_col3:
        st.metric("ED Frame Index", f"Frame {metadata.ed_frame:02d}")
    with m_col4:
        st.metric("ES Frame Index", f"Frame {metadata.es_frame:02d}")
    with m_col5:
        bmi_str = (
            f"{metadata.weight_kg / ((metadata.height_cm / 100) ** 2):.1f}"
            if (metadata.height_cm and metadata.weight_kg)
            else "N/A"
        )
        st.metric("Height / Weight", f"{metadata.height_cm or '?'} cm / {metadata.weight_kg or '?'} kg")

    st.divider()

    # 5.2 MRI Display Controls
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

    # Load cardiac frame
    try:
        frame: ACDCCardiacFrame = loader.load_cardiac_frame(
            selected_patient_id,
            phase=phase,
            load_mask=True,
        )
    except ACDCDataError as e:
        st.error(f"Failed to load cardiac frame for {selected_patient_id} ({phase}): {e}")
        _render_volumetric_simulator()
        return

    # Slice selector
    slice_idx = st.slider(
        "Short-Axis Slice Index (Base ➔ Apex)",
        min_value=0,
        max_value=frame.num_slices - 1,
        value=frame.num_slices // 2,
        help="Navigate through short-axis slices from ventricular base to apex",
    )

    img_slice = frame.get_slice_image(slice_idx, normalized=True)
    mask_slice = frame.get_slice_mask(slice_idx)

    # Physical properties caption
    st.caption(
        f"**Stack Dimensions:** {frame.num_slices} slices × {frame.height} rows × {frame.width} cols | "
        f"**Physical Voxel Spacing:** {frame.voxel_spacing[0]:.2f} × {frame.voxel_spacing[1]:.2f} × {frame.voxel_spacing[2]:.2f} mm | "
        f"**Selected Slice:** {slice_idx + 1} of {frame.num_slices}"
    )

    # 5.3 & 5.4 Image Comparison (Side-by-Side: Original vs Overlay)
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
        f"Original CMR ({phase} - Slice {slice_idx + 1}/{frame.num_slices})",
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

    # 5.5 Research Context & Physical Volumetric Preview
    st.info(
        "🔬 **Research Context:** Segmentation contours displayed above represent expert human annotations "
        "provided with the ACDC challenge benchmark. **The application does not perform automated AI "
        "segmentation yet**; automated U-Net contouring models will be implemented in subsequent phases."
    )

    # Display preliminary physical volume for this phase
    if frame.mask_data is not None:
        lv_vol_ml, lv_voxels, vox_vol_mm3 = calculate_structure_volume_ml(
            frame.mask_data, frame.voxel_spacing, target_label=3
        )
        myo_vol_ml, myo_voxels, _ = calculate_structure_volume_ml(
            frame.mask_data, frame.voxel_spacing, target_label=2
        )
        rv_vol_ml, rv_voxels, _ = calculate_structure_volume_ml(
            frame.mask_data, frame.voxel_spacing, target_label=1
        )

        st.subheader(f"📊 Quantitative Structure Volumes ({phase} Phase)")
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.metric(
                label=f"LV Cavity Volume ({phase})",
                value=f"{lv_vol_ml:.1f} mL",
                help=f"{lv_voxels:,} voxels × {vox_vol_mm3:.2f} mm³/voxel",
            )
        with v_col2:
            st.metric(
                label=f"Myocardial Volume ({phase})",
                value=f"{myo_vol_ml:.1f} mL",
                help=f"{myo_voxels:,} voxels × {vox_vol_mm3:.2f} mm³/voxel",
            )
        with v_col3:
            st.metric(
                label=f"RV Cavity Volume ({phase})",
                value=f"{rv_vol_ml:.1f} mL",
                help=f"{rv_voxels:,} voxels × {vox_vol_mm3:.2f} mm³/voxel",
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
            - **Sequence Types:** Short-axis Cine bSSFP (balanced steady-state free precession) across cardiac phases.
            - **Segmentation Target:** End-diastolic (ED) and end-systolic (ES) contours of the LV blood pool, myocardium, and RV.
            - **Voxel-to-Volume Principle:**
              $$V_{\text{cavity}} = N_{\text{voxels}} \times \frac{\Delta x \times \Delta y \times \Delta z}{1000}\text{ mL}$$
            - **Derived Biometrics:**
              - End-Diastolic Volume ($EDV$)
              - End-Systolic Volume ($ESV$)
              - Stroke Volume ($SV = EDV - ESV$)
              - Ejection Fraction ($LVEF = \frac{EDV - ESV}{EDV} \times 100\%$)
            """
        )

    with col2:
        st.subheader("📐 Interactive Volumetric Calculator")
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
        | II-A. ACDC Ingestion & Visualization | **Active (Milestone 2A)** | `crip.mri.acdc_loader` | Real ACDC MRI loading, multi-label mask overlay |
        | II-B. Volumetrics & Automatic Segmentation | Queued (Milestone 2B) | `crip.mri.cardiac_metrics` | Physical volumetric validation & MONAI segmentation |
        | III. Predictive Modeling | Queued (Milestone 3) | `crip.modeling` | Interpretable recovery modeling with SHAP explanations |
        | IV. Clinical Validation & Metrics | Queued (Milestone 4) | `crip.evaluation` | Calibration curves, Brier scores, decision curves |
        """
    )


if __name__ == "__main__" or True:
    render_page()
