"""MRI Analysis page for CRIP with Real ACDC Cardiac MRI Visualization and Volumetric Quantification.

Supports both Expert Ground-Truth annotations and Automated 2D U-Net Model predictions
with real-time multi-structure contouring and ejection-fraction accuracy verification.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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
from crip.evaluation.segmentation_metrics import (
    DEFAULT_EVAL_SUMMARY_PATH,
    calculate_multiclass_dice,
)
from crip.mri.segmentation_inference import PatientInferenceResult, SegmentationInferenceEngine
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH

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


@st.cache_resource
def _get_inference_engine() -> SegmentationInferenceEngine:
    """Cache inference engine instance to reuse loaded PyTorch model."""
    return SegmentationInferenceEngine()


@st.cache_data
def _predict_patient_cached(
    patient_id: str,
    checkpoint_mtime: float,
    dataset_dir: str,
) -> PatientInferenceResult:
    """Cache patient inference results based on checkpoint file modification timestamp."""
    loader = ACDCDataLoader(dataset_dir)
    engine = _get_inference_engine()
    return engine.predict_patient(loader, patient_id)


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
        biventricular functional measurements derived from **expert ground-truth annotations**
        or **automated 2D U-Net deep learning predictions**.
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

    # Segmentation Source Selection
    st.sidebar.subheader("🧠 Segmentation Source")
    seg_source = st.sidebar.radio(
        "Mask Source",
        options=["Expert Ground Truth", "Automated U-Net Model (Milestone 2C)"],
        index=0,
        help="Switch between manual expert ground truth contours and deep-learning 2D U-Net automated predictions.",
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

    # Compute expert ground truth measurements
    expert_measurements: Optional[BiventricularCardiacMeasurements] = None
    if ed_frame.mask_data is not None and es_frame.mask_data is not None:
        try:
            expert_measurements = compute_biventricular_measurements(
                ed_mask=ed_frame.mask_data,
                es_mask=es_frame.mask_data,
                voxel_spacing_mm=ed_frame.voxel_spacing,
                patient_id=selected_patient_id,
            )
        except Exception as e:
            st.warning(f"Unable to compute expert measurements for {selected_patient_id}: {e}")

    # Handle Automated U-Net Model inference
    auto_result: Optional[PatientInferenceResult] = None
    active_ed_mask = ed_frame.mask_data
    active_es_mask = es_frame.mask_data
    is_auto_mode = (seg_source == "Automated U-Net Model (Milestone 2C)")

    if is_auto_mode:
        if DEFAULT_CHECKPOINT_PATH.is_file():
            try:
                ckpt_mtime = DEFAULT_CHECKPOINT_PATH.stat().st_mtime
                auto_result = _predict_patient_cached(
                    patient_id=selected_patient_id,
                    checkpoint_mtime=ckpt_mtime,
                    dataset_dir=str(loader.dataset_root),
                )
                active_ed_mask = auto_result.pred_ed_mask
                active_es_mask = auto_result.pred_es_mask
            except Exception as e:
                st.error(f"Automated segmentation inference failed: {e}")
                is_auto_mode = False
        else:
            st.warning(
                f"⚠️ Automated model checkpoint not found at `{DEFAULT_CHECKPOINT_PATH}`.\n\n"
                "Please train the model first by running:\n"
                "```bash\npython src/crip/mri/train_segmentation.py --epochs 25\n```\n"
                "Displaying expert ground truth contours instead."
            )
            is_auto_mode = False

    # Section 1: Patient Summary & Calculated Functional Parameters
    st.subheader(
        f"📊 Quantitative Cardiac Measurements — {selected_patient_id} ({metadata.group})"
        + (" [Automated U-Net Model]" if is_auto_mode else " [Expert Ground Truth]")
    )

    st.info(
        "🔬 **Measurement Methodology & Clinical Reference Distinction:**\n\n"
        "The cardiac functional parameters below are calculated directly by summing voxels from "
        + ("**automated U-Net deep learning segmentations**" if is_auto_mode else "**expert-provided ACDC ground-truth masks**")
        + r" ($V = N_{\text{voxels}} \times \frac{\Delta x \cdot \Delta y \cdot \Delta z}{1000}\text{ mL}$)."
        + " **These values represent experimental geometric integrations for research and must not be "
        "interpreted as clinically validated reference standards.** Independent external clinical consensus "
        "measurements are not provided in the challenge distribution."
    )

    # Active metrics to display
    display_lvedv = auto_result.pred_metrics.lvedv_ml if is_auto_mode and auto_result else (expert_measurements.lvedv_ml if expert_measurements else 0.0)
    display_lvesv = auto_result.pred_metrics.lvesv_ml if is_auto_mode and auto_result else (expert_measurements.lvesv_ml if expert_measurements else 0.0)
    display_lvsv = auto_result.pred_metrics.lvsv_ml if is_auto_mode and auto_result else (expert_measurements.lvsv_ml if expert_measurements else 0.0)
    display_lvef = auto_result.pred_metrics.lvef_percent if is_auto_mode and auto_result else (expert_measurements.lvef_percent if expert_measurements else 0.0)
    display_rvedv = auto_result.pred_metrics.rvedv_ml if is_auto_mode and auto_result else (expert_measurements.rvedv_ml if expert_measurements else 0.0)
    display_rvesv = auto_result.pred_metrics.rvesv_ml if is_auto_mode and auto_result else (expert_measurements.rvesv_ml if expert_measurements else 0.0)
    display_rvsv = auto_result.pred_metrics.rvsv_ml if is_auto_mode and auto_result else (expert_measurements.rvsv_ml if expert_measurements else 0.0)
    display_rvef = auto_result.pred_metrics.rvef_percent if is_auto_mode and auto_result else (expert_measurements.rvef_percent if expert_measurements else 0.0)
    display_myo_vol = auto_result.pred_metrics.myocardial_volume_ed_ml if is_auto_mode and auto_result else (expert_measurements.myocardial_volume_ed_ml if expert_measurements else 0.0)
    display_myo_mass = auto_result.pred_metrics.myocardial_mass_ed_g if is_auto_mode and auto_result else (expert_measurements.myocardial_mass_ed_g if expert_measurements else 0.0)

    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(5)
    with row1_col1:
        st.metric("Patient ID", selected_patient_id)
        st.metric("Diagnostic Group", metadata.group)
    with row1_col2:
        st.metric("LV End-Diastolic Vol (LVEDV)", f"{display_lvedv:.1f} mL")
        st.metric("RV End-Diastolic Vol (RVEDV)", f"{display_rvedv:.1f} mL")
    with row1_col3:
        st.metric("LV End-Systolic Vol (LVESV)", f"{display_lvesv:.1f} mL")
        st.metric("RV End-Systolic Vol (RVESV)", f"{display_rvesv:.1f} mL")
    with row1_col4:
        st.metric("LV Stroke Volume (LVSV)", f"{display_lvsv:.1f} mL")
        st.metric("RV Stroke Volume (RVSV)", f"{display_rvsv:.1f} mL")
    with row1_col5:
        st.metric("LV Ejection Fraction (LVEF)", f"{display_lvef:.1f}%")
        st.metric("RV Ejection Fraction (RVEF)", f"{display_rvef:.1f}%")

    m_mass_col1, m_mass_col2, m_mass_col3 = st.columns(3)
    with m_mass_col1:
        st.metric("Myocardial Volume (ED)", f"{display_myo_vol:.1f} mL")
    with m_mass_col2:
        st.metric("Estimated Myocardial Mass (ED)", f"{display_myo_mass:.1f} g", help="Assumes standard myocardial density = 1.05 g/mL")
    with m_mass_col3:
        st.metric("Physical Voxel Spacing", f"{ed_frame.voxel_spacing[0]:.2f} × {ed_frame.voxel_spacing[1]:.2f} × {ed_frame.voxel_spacing[2]:.2f} mm")

    # If Automated Mode is Active, show Direct Model vs. Expert Comparison
    if is_auto_mode and auto_result and expert_measurements:
        lvef_error = auto_result.pred_metrics.lvef_percent - expert_measurements.lvef_percent
        lvef_abs_error = abs(lvef_error)
        within_5pct = lvef_abs_error <= 5.0

        st.markdown("---")
        st.subheader("🎯 Automated Model vs. Expert Ground-Truth Validation")

        target_badge = (
            f"🟢 **Target Achieved:** Absolute LVEF Error = **{lvef_abs_error:.2f}%** (≤ 5.0 percentage points target)"
            if within_5pct
            else f"🟡 **Target Exceeded:** Absolute LVEF Error = **{lvef_abs_error:.2f}%** (> 5.0 percentage points target)"
        )
        st.markdown(target_badge)

        comp_col1, comp_col2 = st.columns([3, 2])
        with comp_col1:
            comp_data = {
                "Metric": ["LVEDV (mL)", "LVESV (mL)", "LVSV (mL)", "LVEF (%)", "RVEDV (mL)", "RVESV (mL)", "RVEF (%)"],
                "Automated Model": [
                    f"{auto_result.pred_metrics.lvedv_ml:.1f}",
                    f"{auto_result.pred_metrics.lvesv_ml:.1f}",
                    f"{auto_result.pred_metrics.lvsv_ml:.1f}",
                    f"{auto_result.pred_metrics.lvef_percent:.1f}%",
                    f"{auto_result.pred_metrics.rvedv_ml:.1f}",
                    f"{auto_result.pred_metrics.rvesv_ml:.1f}",
                    f"{auto_result.pred_metrics.rvef_percent:.1f}%",
                ],
                "Expert Ground Truth": [
                    f"{expert_measurements.lvedv_ml:.1f}",
                    f"{expert_measurements.lvesv_ml:.1f}",
                    f"{expert_measurements.lvsv_ml:.1f}",
                    f"{expert_measurements.lvef_percent:.1f}%",
                    f"{expert_measurements.rvedv_ml:.1f}",
                    f"{expert_measurements.rvesv_ml:.1f}",
                    f"{expert_measurements.rvef_percent:.1f}%",
                ],
                "Error (Auto - Expert)": [
                    f"{auto_result.pred_metrics.lvedv_ml - expert_measurements.lvedv_ml:+.1f} mL",
                    f"{auto_result.pred_metrics.lvesv_ml - expert_measurements.lvesv_ml:+.1f} mL",
                    f"{auto_result.pred_metrics.lvsv_ml - expert_measurements.lvsv_ml:+.1f} mL",
                    f"{lvef_error:+.1f}%",
                    f"{auto_result.pred_metrics.rvedv_ml - expert_measurements.rvedv_ml:+.1f} mL",
                    f"{auto_result.pred_metrics.rvesv_ml - expert_measurements.rvesv_ml:+.1f} mL",
                    f"{auto_result.pred_metrics.rvef_percent - expert_measurements.rvef_percent:+.1f}%",
                ],
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        with comp_col2:
            # Overlap Dice for this patient
            if auto_result.gt_ed_mask is not None and auto_result.gt_es_mask is not None:
                d_ed = calculate_multiclass_dice(auto_result.pred_ed_mask, auto_result.gt_ed_mask)
                d_es = calculate_multiclass_dice(auto_result.pred_es_mask, auto_result.gt_es_mask)
                d_lv = (d_ed["lv"] + d_es["lv"]) / 2.0
                d_rv = (d_ed["rv"] + d_es["rv"]) / 2.0
                d_myo = (d_ed["myo"] + d_es["myo"]) / 2.0
                d_mean = (d_ed["mean"] + d_es["mean"]) / 2.0

                st.markdown("**Segmentation Spatial Overlap (Dice)**")
                st.metric("LV Cavity Dice", f"{d_lv:.3f}")
                st.metric("RV Cavity Dice", f"{d_rv:.3f}")
                st.metric("Myocardium Dice", f"{d_myo:.3f}")
                st.metric("Mean Foreground Dice", f"{d_mean:.3f}")

    st.divider()

    # Section 2: Visual Volume Comparison & Cohort Context
    v_col_left, v_col_right = st.columns([1, 1])

    with v_col_left:
        st.subheader("📈 Ventricular Phase Volume Comparison")
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(
            name="Left Ventricle (LV)",
            x=["End-Diastole (EDV)", "End-Systole (ESV)", "Stroke Volume (SV)"],
            y=[display_lvedv, display_lvesv, display_lvsv],
            marker_color="#FF4040",
            text=[f"{display_lvedv:.1f} mL", f"{display_lvesv:.1f} mL", f"{display_lvsv:.1f} mL"],
            textposition="auto",
        ))
        vol_fig.add_trace(go.Bar(
            name="Right Ventricle (RV)",
            x=["End-Diastole (EDV)", "End-Systole (ESV)", "Stroke Volume (SV)"],
            y=[display_rvedv, display_rvesv, display_rvsv],
            marker_color="#26A6FF",
            text=[f"{display_rvedv:.1f} mL", f"{display_rvesv:.1f} mL", f"{display_rvsv:.1f} mL"],
            textposition="auto",
        ))
        vol_fig.update_layout(
            barmode="group",
            title=f"Biventricular Cavity Volumes ({'Automated Model' if is_auto_mode else 'Expert Mask'})",
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

        if cohort_df is not None and not cohort_df.empty:
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
            scatter_fig.add_trace(go.Scatter(
                x=[display_lvedv],
                y=[display_lvef],
                mode="markers",
                marker=dict(size=14, color="gold", symbol="star", line=dict(width=2, color="black")),
                name=f"Current: {selected_patient_id}",
            ))
            scatter_fig.add_hline(y=50.0, line_dash="dot", line_color="gray", annotation_text="50% EF Threshold")
            scatter_fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(scatter_fig, use_container_width=True)

            lvef_rank = (cohort_df["lvef_percent"] < display_lvef).mean() * 100.0
            st.caption(f"📌 **Cohort Position:** {selected_patient_id} has an LVEF of **{display_lvef:.1f}%** ({lvef_rank:.0f}th percentile of 100 ACDC cases).")
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
        show_overlay = st.checkbox("Show Segmentation Overlay", value=True)

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
    current_mask_volume = active_ed_mask if phase == "ED" else active_es_mask

    # Slice selector
    slice_idx = st.slider(
        "Short-Axis Slice Index (Base ➔ Apex)",
        min_value=0,
        max_value=current_frame.num_slices - 1,
        value=current_frame.num_slices // 2,
        help="Navigate through short-axis slices from ventricular base to apex",
    )

    img_slice = current_frame.get_slice_image(slice_idx, normalized=True)
    mask_slice = current_mask_volume[slice_idx] if current_mask_volume is not None else None

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
    overlay_title = (
        f"Automated U-Net Contours ({phase})"
        if is_auto_mode
        else f"Expert Ground-Truth Contours ({phase})"
    )
    ax_overlay.imshow(overlay_slice, origin="upper")
    ax_overlay.set_title(
        overlay_title,
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

    # Section 4: Automated Benchmark Results (Held-out Test Cohort)
    if DEFAULT_EVAL_SUMMARY_PATH.is_file():
        st.divider()
        st.subheader("🔬 Automated Segmentation Benchmark (Held-out Test Cohort)")
        try:
            with open(DEFAULT_EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                st.metric("Test Cohort Size", f"{summary_data.get('n_patients', 20)} patients")
                st.metric("Mean Overall Dice", f"{summary_data.get('geometric_metrics', {}).get('dice_mean_overall', 0.0):.3f}")
            with b_col2:
                lvef_m = summary_data.get("lvef_metrics", {})
                st.metric("Mean Absolute LVEF Error", f"{lvef_m.get('mean_absolute_error_pct', 0.0):.2f}%")
                st.metric("LV Cavity Dice", f"{summary_data.get('geometric_metrics', {}).get('dice_lv', 0.0):.3f}")
            with b_col3:
                st.metric("Median Absolute LVEF Error", f"{lvef_m.get('median_absolute_error_pct', 0.0):.2f}%")
                st.metric("RV Cavity Dice", f"{summary_data.get('geometric_metrics', {}).get('dice_rv', 0.0):.3f}")
            with b_col4:
                st.metric("Cases with Error ≤ 5.0%", f"{lvef_m.get('pct_cases_le_5pct_error', 0.0):.1f}%")
                st.metric("Myocardium Dice", f"{summary_data.get('geometric_metrics', {}).get('dice_myo', 0.0):.3f}")
        except Exception as e:
            st.caption(f"Benchmark summary loading notice: {e}")

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
        | II-B. Volumetrics & Cohort Measurement | **Completed (Milestone 2B)** | `crip.mri.cardiac_metrics`, `pipeline` | Physical volumetric validation & cohort analysis |
        | II-C. Deep Learning Segmentation | **Completed (Milestone 2C)** | `crip.mri.segmentation` | MONAI/PyTorch U-Net automated contouring & LVEF evaluation |
        | III. Predictive Modeling | Queued (Milestone 3) | `crip.modeling` | Interpretable recovery modeling with SHAP explanations |
        | IV. Clinical Validation & Metrics | Queued (Milestone 4) | `crip.evaluation` | Calibration curves, Brier scores, decision curves |
        """
    )


if __name__ == "__main__" or True:
    render_page()
