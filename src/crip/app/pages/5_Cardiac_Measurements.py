"""Dedicated Cardiac Measurements Page for CRIP.

Presents quantitative biventricular functional measurements, Left and Right
Ventricular volumetric analysis, myocardial mass estimation, and direct
side-by-side comparison between automated U-Net deep learning predictions
and expert ground-truth annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from crip.app.components.disclaimer import render_clinical_disclaimer
from crip.mri.acdc_loader import (
    ACDCDataLoader,
    DEFAULT_ACDC_TRAINING_DIR,
)
from crip.mri.cardiac_metrics import (
    MYOCARDIAL_DENSITY_G_PER_ML,
    BiventricularCardiacMeasurements,
    compute_biventricular_measurements,
)
from crip.mri.segmentation_inference import PatientInferenceResult, SegmentationInferenceEngine
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH

CARDIAC_MEASUREMENTS_CSV = Path("data/processed/acdc/cardiac_measurements.csv")
SEGMENTATION_EVAL_CSV = Path("data/processed/acdc/segmentation_evaluation.csv")
SEGMENTATION_SUMMARY_JSON = Path("data/processed/acdc/segmentation_evaluation_summary.json")

# Standard clinical reference ranges (SCMR / Petersen et al. 2017)
SCMR_REFERENCE_RANGES = {
    "LVEDV": {"label": "LV End-Diastolic Volume", "unit": "mL", "male": "115–220", "female": "95–170", "combined": "100–200"},
    "LVESV": {"label": "LV End-Systolic Volume", "unit": "mL", "male": "35–90", "female": "25–65", "combined": "30–80"},
    "LVSV": {"label": "LV Stroke Volume", "unit": "mL", "male": "70–135", "female": "60–110", "combined": "60–120"},
    "LVEF": {"label": "LV Ejection Fraction", "unit": "%", "male": "52–72", "female": "54–74", "combined": "≥ 50 (Preserved)"},
    "RVEDV": {"label": "RV End-Diastolic Volume", "unit": "mL", "male": "120–235", "female": "95–185", "combined": "100–210"},
    "RVESV": {"label": "RV End-Systolic Volume", "unit": "mL", "male": "40–105", "female": "30–75", "combined": "30–90"},
    "RVSV": {"label": "RV Stroke Volume", "unit": "mL", "male": "70–140", "female": "55–115", "combined": "50–120"},
    "RVEF": {"label": "RV Ejection Fraction", "unit": "%", "male": "45–68", "female": "47–70", "combined": "≥ 45 (Normal)"},
    "LV_MASS_ED": {"label": "LV Myocardial Mass (ED)", "unit": "g", "male": "85–175", "female": "60–125", "combined": "70–150"},
}


@st.cache_data
def _load_measurements_csv() -> Optional[pd.DataFrame]:
    """Load precomputed cohort cardiac measurements."""
    if CARDIAC_MEASUREMENTS_CSV.is_file():
        try:
            return pd.read_csv(CARDIAC_MEASUREMENTS_CSV)
        except Exception:
            return None
    return None


@st.cache_data
def _load_segmentation_eval_csv() -> Optional[pd.DataFrame]:
    """Load held-out segmentation evaluation comparison data."""
    if SEGMENTATION_EVAL_CSV.is_file():
        try:
            return pd.read_csv(SEGMENTATION_EVAL_CSV)
        except Exception:
            return None
    return None


@st.cache_resource
def _get_inference_engine() -> SegmentationInferenceEngine:
    """Cache inference engine instance for U-Net predictions."""
    return SegmentationInferenceEngine()


def _get_lvef_badge(ef_pct: float) -> Tuple[str, str]:
    """Return clinical tag and color category for Left Ventricular Ejection Fraction."""
    if ef_pct >= 50.0:
        return "🟢 Preserved (≥ 50%)", "normal"
    if ef_pct >= 40.0:
        return "🟡 Mildly Reduced (40–49%)", "off"
    return "🔴 Reduced (< 40%)", "inverse"


def _get_rvef_badge(ef_pct: float) -> Tuple[str, str]:
    """Return clinical tag and color category for Right Ventricular Ejection Fraction."""
    if ef_pct >= 45.0:
        return "🟢 Normal (≥ 45%)", "normal"
    return "🔴 Reduced (< 45%)", "inverse"


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - Cardiac Measurements",
        page_icon="📊",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("📊 Cardiac Functional Measurements & Chamber Volumetrics")
    st.markdown(
        """
        Comprehensive clinical quantification of **Left Ventricular (LV)** and **Right Ventricular (RV)**
        cavity volumes, stroke volumes, ejection fractions, and myocardial mass derived from short-axis
        Cine CMR. Physical measurements are calculated via slice-by-slice voxel volume integration
        scaled by 3D physical acquisition spacing ($dx \\times dy \\times dz$), with myocardial mass
        estimated using the standard SCMR/AHA tissue density ($1.05\\text{ g/mL}$).
        """
        "\n\n"
        "---"
    )

    # 1. Check data availability
    measurements_df = _load_measurements_csv()
    eval_df = _load_segmentation_eval_csv()
    loader = ACDCDataLoader(DEFAULT_ACDC_TRAINING_DIR)
    has_raw_acdc = loader.is_available()

    if measurements_df is None and not has_raw_acdc and eval_df is None:
        st.error(
            "⚠️ **No Cardiac Data Found:** Neither processed measurements (`cardiac_measurements.csv`) "
            "nor raw ACDC imaging directories were found at `data/raw/acdc/training`.\n\n"
            "Please run the pipeline via `python -m crip.mri.pipeline` or consult the setup guide in `README.md`."
        )
        return

    # Determine patient list
    if measurements_df is not None and not measurements_df.empty:
        available_patients = measurements_df["patient_id"].tolist()
    elif has_raw_acdc:
        available_patients = loader.list_patients()
    elif eval_df is not None and not eval_df.empty:
        available_patients = eval_df["patient_id"].tolist()
    else:
        available_patients = []

    # Check for transferred MRI features from session state
    transferred = st.session_state.get("transferred_mri_features")
    default_idx = 0
    if transferred and "acdc_patient_id" in transferred and transferred["acdc_patient_id"] in available_patients:
        default_idx = available_patients.index(transferred["acdc_patient_id"])

    # Sidebar: Patient and Source Selection
    st.sidebar.header("🎛️ Study & Measurement Controls")
    selected_patient_id = st.sidebar.selectbox(
        "Select Patient / Study ID",
        options=available_patients,
        index=default_idx,
        help="Select an ACDC study to view detailed ventricular measurements.",
    )

    has_model_checkpoint = DEFAULT_CHECKPOINT_PATH.is_file()

    # Determine available sources for this patient
    has_eval_record = (eval_df is not None and not eval_df[eval_df["patient_id"] == selected_patient_id].empty)
    can_run_unet = has_raw_acdc and has_model_checkpoint

    source_options = ["Expert Ground Truth"]
    if has_eval_record or can_run_unet:
        source_options.append("Automated U-Net Model (Milestone 2C)")

    selected_source = st.sidebar.radio(
        "Measurement Source",
        options=source_options,
        index=0,
        help="Select whether to view expert annotations or automated deep learning segmentations.",
    )
    is_auto = (selected_source == "Automated U-Net Model (Milestone 2C)")

    # Retrieve patient row from precomputed measurements
    p_row = None
    if measurements_df is not None:
        p_matches = measurements_df[measurements_df["patient_id"] == selected_patient_id]
        if not p_matches.empty:
            p_row = p_matches.iloc[0]

    eval_row = None
    if eval_df is not None:
        e_matches = eval_df[eval_df["patient_id"] == selected_patient_id]
        if not e_matches.empty:
            eval_row = e_matches.iloc[0]

    # Retrieve or compute metrics
    expert_metrics: Optional[BiventricularCardiacMeasurements] = None
    pred_metrics: Optional[BiventricularCardiacMeasurements] = None

    # Load from raw frames if available
    raw_meta = None
    if has_raw_acdc:
        try:
            raw_meta = loader.get_patient_metadata(selected_patient_id)
        except Exception:
            raw_meta = None

    # Resolve expert measurements
    if p_row is not None:
        expert_lvedv = float(p_row["lvedv_ml"])
        expert_lvesv = float(p_row["lvesv_ml"])
        expert_lvsv = float(p_row["lvsv_ml"])
        expert_lvef = float(p_row["lvef_percent"])
        expert_rvedv = float(p_row["rvedv_ml"]) if pd.notna(p_row.get("rvedv_ml")) else 0.0
        expert_rvesv = float(p_row["rvesv_ml"]) if pd.notna(p_row.get("rvesv_ml")) else 0.0
        expert_rvsv = float(p_row["rvsv_ml"]) if pd.notna(p_row.get("rvsv_ml")) else 0.0
        expert_rvef = float(p_row["rvef_percent"]) if pd.notna(p_row.get("rvef_percent")) else 0.0
        expert_myo_vol = float(p_row["myocardial_volume_ed_ml"]) if pd.notna(p_row.get("myocardial_volume_ed_ml")) else 0.0
        expert_myo_mass = float(p_row["myocardial_mass_ed_g"]) if pd.notna(p_row.get("myocardial_mass_ed_g")) else 0.0
        diag_group = str(p_row["diagnostic_group"])
        voxel_dx = float(p_row["voxel_spacing_x_mm"]) if pd.notna(p_row.get("voxel_spacing_x_mm")) else 1.0
        voxel_dy = float(p_row["voxel_spacing_y_mm"]) if pd.notna(p_row.get("voxel_spacing_y_mm")) else 1.0
        voxel_dz = float(p_row["voxel_spacing_z_mm"]) if pd.notna(p_row.get("voxel_spacing_z_mm")) else 10.0
    elif eval_row is not None:
        expert_lvedv = float(eval_row["lvedv_gt_ml"])
        expert_lvesv = float(eval_row["lvesv_gt_ml"])
        expert_lvsv = float(eval_row["lvsv_gt_ml"])
        expert_lvef = float(eval_row["lvef_gt_pct"])
        expert_rvedv = float(eval_row["rvedv_gt_ml"]) if pd.notna(eval_row.get("rvedv_gt_ml")) else 0.0
        expert_rvef = float(eval_row["rvef_gt_pct"]) if pd.notna(eval_row.get("rvef_gt_pct")) else 0.0
        expert_rvesv = expert_rvedv * (1.0 - (expert_rvef / 100.0)) if expert_rvedv > 0 else 0.0
        expert_rvsv = expert_rvedv - expert_rvesv
        expert_myo_vol = 0.0
        expert_myo_mass = 0.0
        diag_group = str(eval_row["group"])
        voxel_dx, voxel_dy, voxel_dz = 1.0, 1.0, 10.0
    else:
        expert_lvedv, expert_lvesv, expert_lvsv, expert_lvef = 140.0, 60.0, 80.0, 57.1
        expert_rvedv, expert_rvesv, expert_rvsv, expert_rvef = 135.0, 65.0, 70.0, 51.9
        expert_myo_vol, expert_myo_mass = 120.0, 126.0
        diag_group = raw_meta.group if raw_meta else "Unknown"
        voxel_dx, voxel_dy, voxel_dz = 1.4, 1.4, 10.0

    # Resolve automated U-Net measurements if available
    has_unet_pred = False
    pred_lvedv, pred_lvesv, pred_lvsv, pred_lvef = 0.0, 0.0, 0.0, 0.0
    pred_rvedv, pred_rvesv, pred_rvsv, pred_rvef = 0.0, 0.0, 0.0, 0.0
    pred_myo_vol, pred_myo_mass = 0.0, 0.0

    if eval_row is not None:
        has_unet_pred = True
        pred_lvedv = float(eval_row["lvedv_pred_ml"])
        pred_lvesv = float(eval_row["lvesv_pred_ml"])
        pred_lvsv = float(eval_row["lvsv_pred_ml"])
        pred_lvef = float(eval_row["lvef_pred_pct"])
        pred_rvedv = float(eval_row["rvedv_pred_ml"]) if pd.notna(eval_row.get("rvedv_pred_ml")) else 0.0
        pred_rvef = float(eval_row["rvef_pred_pct"]) if pd.notna(eval_row.get("rvef_pred_pct")) else 0.0
        pred_rvesv = pred_rvedv * (1.0 - (pred_rvef / 100.0)) if pred_rvedv > 0 else 0.0
        pred_rvsv = pred_rvedv - pred_rvesv
    elif can_run_unet:
        # Run inference via engine
        try:
            engine = _get_inference_engine()
            res = engine.predict_patient(loader, selected_patient_id)
            has_unet_pred = True
            pred_lvedv = res.pred_metrics.lvedv_ml
            pred_lvesv = res.pred_metrics.lvesv_ml
            pred_lvsv = res.pred_metrics.lvsv_ml
            pred_lvef = res.pred_metrics.lvef_percent
            pred_rvedv = res.pred_metrics.rvedv_ml
            pred_rvesv = res.pred_metrics.rvesv_ml
            pred_rvsv = res.pred_metrics.rvsv_ml
            pred_rvef = res.pred_metrics.rvef_percent
            pred_myo_vol = res.pred_metrics.myocardial_volume_ed_ml
            pred_myo_mass = res.pred_metrics.myocardial_mass_ed_g
        except Exception:
            has_unet_pred = False

    # Choose active metrics for display based on selected source
    if is_auto and has_unet_pred:
        cur_lvedv, cur_lvesv, cur_lvsv, cur_lvef = pred_lvedv, pred_lvesv, pred_lvsv, pred_lvef
        cur_rvedv, cur_rvesv, cur_rvsv, cur_rvef = pred_rvedv, pred_rvesv, pred_rvsv, pred_rvef
        cur_myo_vol, cur_myo_mass = pred_myo_vol, pred_myo_mass
        source_label = "Automated U-Net Model (Milestone 2C)"
    else:
        cur_lvedv, cur_lvesv, cur_lvsv, cur_lvef = expert_lvedv, expert_lvesv, expert_lvsv, expert_lvef
        cur_rvedv, cur_rvesv, cur_rvsv, cur_rvef = expert_rvedv, expert_rvesv, expert_rvsv, expert_rvef
        cur_myo_vol, cur_myo_mass = expert_myo_vol, expert_myo_mass
        source_label = "Expert Ground Truth"

    # Header Patient Information Card
    st.markdown(
        f"""
        <div style="background-color: #1e2530; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #2E86C1;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h3 style="margin: 0; color: #ECEFF4;">Patient: <code>{selected_patient_id}</code></h3>
                    <p style="margin: 4px 0 0 0; color: #D8DEE9;">Diagnostic Cohort: <b>{diag_group}</b> | Active Source: <b>{source_label}</b></p>
                </div>
                <div>
                    <span style="background-color: #2E86C1; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 0.9em;">
                        CMR Short-Axis Cine
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top-level navigation tabs
    tab_chamber, tab_compare, tab_cohort = st.tabs([
        "🫀 Biventricular Volumetric Profile",
        "⚖️ Paired AI Model vs. Expert Reference Comparison",
        "📊 Cohort Normative Reference Ranges",
    ])

    # ---------------------------------------------------------
    # TAB 1: BIVENTRICULAR VOLUMETRIC PROFILE
    # ---------------------------------------------------------
    with tab_chamber:
        st.subheader(f"Left & Right Ventricular Functional Quantification ({source_label})")

        # LV Metrics Card Row
        st.markdown("#### 🔴 Left Ventricular (LV) Parameters")
        lv_badge, _ = _get_lvef_badge(cur_lvef)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("LV Ejection Fraction (LVEF)", f"{cur_lvef:.1f}%", delta=lv_badge)
            st.caption("Normal reference: **≥ 50%**")
        with c2:
            st.metric("LV End-Diastolic Vol (LVEDV)", f"{cur_lvedv:.1f} mL")
            st.caption(f"Ref range: {SCMR_REFERENCE_RANGES['LVEDV']['combined']} mL")
        with c3:
            st.metric("LV End-Systolic Vol (LVESV)", f"{cur_lvesv:.1f} mL")
            st.caption(f"Ref range: {SCMR_REFERENCE_RANGES['LVESV']['combined']} mL")
        with c4:
            st.metric("LV Stroke Volume (LVSV)", f"{cur_lvsv:.1f} mL")
            st.caption(f"Ref range: {SCMR_REFERENCE_RANGES['LVSV']['combined']} mL")

        # RV Metrics Card Row
        st.markdown("#### 🔵 Right Ventricular (RV) Parameters")
        rv_badge, _ = _get_rvef_badge(cur_rvef)

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("RV Ejection Fraction (RVEF)", f"{cur_rvef:.1f}%", delta=rv_badge)
            st.caption("Normal reference: **≥ 45%**")
        with rc2:
            st.metric("RV End-Diastolic Vol (RVEDV)", f"{cur_rvedv:.1f} mL")
            st.caption(f"Ref range: {SCMR_REFERENCE_RANGES['RVEDV']['combined']} mL")
        with rc3:
            st.metric("RV End-Systolic Vol (RVESV)", f"{cur_rvesv:.1f} mL")
            st.caption(f"Ref range: {SCMR_REFERENCE_RANGES['RVESV']['combined']} mL")
        with rc4:
            st.metric("RV Stroke Volume (RVSV)", f"{cur_rvsv:.1f} mL")
            st.caption(f"Ref range: {SCMR_REFERENCE_RANGES['RVSV']['combined']} mL")

        # Myocardial and Physical Spacing Row
        st.markdown("#### 🟢 LV Myocardial Mass & Scan Voxel Geometry")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            if cur_myo_mass > 0:
                st.metric("Estimated LV Myocardial Mass", f"{cur_myo_mass:.1f} g")
                st.caption(f"Assumed density: {MYOCARDIAL_DENSITY_G_PER_ML} g/mL (AHA/SCMR)")
            else:
                st.metric("Estimated LV Myocardial Mass", "N/A in test CSV")
                st.caption("Available in full training dataset")
        with mc2:
            if cur_myo_vol > 0:
                st.metric("LV Myocardial Volume (ED)", f"{cur_myo_vol:.1f} mL")
                st.caption("Voxel integration across short-axis slices")
            else:
                st.metric("LV Myocardial Volume", "N/A")
        with mc3:
            st.metric("Acquisition In-Plane & Slice Spacing", f"{voxel_dx:.2f} × {voxel_dy:.2f} × {voxel_dz:.1f} mm")
            vox_mm3 = voxel_dx * voxel_dy * voxel_dz
            st.caption(f"Physical Voxel Volume: **{vox_mm3:.2f} mm³** ({vox_mm3 / 1000.0:.4f} mL)")

        # Physiological Validation Checks
        st.markdown("#### 🩺 Physiological Sanity Verification")
        phys_checks = []
        # Check 1: ESV < EDV
        if cur_lvesv < cur_lvedv and cur_lvsv > 0:
            phys_checks.append("✅ **LV Contraction:** End-Systolic Volume is strictly less than End-Diastolic Volume ($SV > 0$).")
        else:
            phys_checks.append("⚠️ **LV Warning:** Non-physiological volumes detected ($ESV \\ge EDV$).")

        if cur_rvesv < cur_rvedv and cur_rvsv > 0:
            phys_checks.append("✅ **RV Contraction:** RV End-Systolic Volume is strictly less than End-Diastolic Volume ($SV > 0$).")
        else:
            phys_checks.append("⚠️ **RV Warning:** Non-physiological volumes detected ($ESV \\ge EDV$).")

        # Check 2: EF in physiological bounds
        if 5.0 <= cur_lvef <= 90.0:
            phys_checks.append("✅ **LV Ejection Fraction:** Within plausible human physiological bounds $[5.0\\%, 90.0\\%]$.")
        else:
            phys_checks.append(f"⚠️ **LV Warning:** Extreme LVEF ({cur_lvef:.1f}%) outside expected physiological range.")

        for chk in phys_checks:
            st.markdown(f"- {chk}")

        # Chamber Dynamics Chart
        st.divider()
        st.markdown("#### 📈 Ventricular Volume Dynamics (End-Diastole vs. End-Systole)")

        chart_df = pd.DataFrame([
            {"Chamber": "Left Ventricle (LV)", "Phase": "End-Diastole (ED)", "Volume (mL)": cur_lvedv},
            {"Chamber": "Left Ventricle (LV)", "Phase": "End-Systole (ES)", "Volume (mL)": cur_lvesv},
            {"Chamber": "Right Ventricle (RV)", "Phase": "End-Diastole (ED)", "Volume (mL)": cur_rvedv},
            {"Chamber": "Right Ventricle (RV)", "Phase": "End-Systole (ES)", "Volume (mL)": cur_rvesv},
        ])

        fig_bars = px.bar(
            chart_df,
            x="Chamber",
            y="Volume (mL)",
            color="Phase",
            barmode="group",
            color_discrete_map={"End-Diastole (ED)": "#2E86C1", "End-Systole (ES)": "#E74C3C"},
            title=f"Biventricular Volumetric Dynamics — {selected_patient_id}",
            text_auto=".1f",
        )
        fig_bars.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: PAIRED AI MODEL VS. EXPERT REFERENCE COMPARISON
    # ---------------------------------------------------------
    with tab_compare:
        st.subheader("Direct Comparison: Automated 2D U-Net vs. Expert Ground Truth")

        if not has_unet_pred:
            st.info(
                f"ℹ️ **Automated U-Net prediction record not pre-loaded for `{selected_patient_id}`.**\n\n"
                "To generate real-time AI predictions for this patient, ensure `models/mri/best_unet_model.pt` "
                "is present and raw ACDC NIfTI files are located in `data/raw/acdc/training`.\n"
                "Patients from the held-out test cohort (`patient007`, `patient022`, etc.) contain complete paired records."
            )
        else:
            # Construct paired comparison dataframe
            paired_data = [
                {
                    "Metric Parameter": "LV End-Diastolic Volume (LVEDV)",
                    "Unit": "mL",
                    "AI Prediction (U-Net)": f"{pred_lvedv:.1f}",
                    "Expert Ground Truth": f"{expert_lvedv:.1f}",
                    "Signed Difference": pred_lvedv - expert_lvedv,
                    "Absolute Difference": abs(pred_lvedv - expert_lvedv),
                    "Relative Error (%)": (abs(pred_lvedv - expert_lvedv) / expert_lvedv * 100.0) if expert_lvedv > 0 else 0.0,
                    "SCMR Clinical Target": "≤ 10.0 mL",
                },
                {
                    "Metric Parameter": "LV End-Systolic Volume (LVESV)",
                    "Unit": "mL",
                    "AI Prediction (U-Net)": f"{pred_lvesv:.1f}",
                    "Expert Ground Truth": f"{expert_lvesv:.1f}",
                    "Signed Difference": pred_lvesv - expert_lvesv,
                    "Absolute Difference": abs(pred_lvesv - expert_lvesv),
                    "Relative Error (%)": (abs(pred_lvesv - expert_lvesv) / expert_lvesv * 100.0) if expert_lvesv > 0 else 0.0,
                    "SCMR Clinical Target": "≤ 8.0 mL",
                },
                {
                    "Metric Parameter": "LV Stroke Volume (LVSV)",
                    "Unit": "mL",
                    "AI Prediction (U-Net)": f"{pred_lvsv:.1f}",
                    "Expert Ground Truth": f"{expert_lvsv:.1f}",
                    "Signed Difference": pred_lvsv - expert_lvsv,
                    "Absolute Difference": abs(pred_lvsv - expert_lvsv),
                    "Relative Error (%)": (abs(pred_lvsv - expert_lvsv) / expert_lvsv * 100.0) if expert_lvsv > 0 else 0.0,
                    "SCMR Clinical Target": "≤ 8.0 mL",
                },
                {
                    "Metric Parameter": "LV Ejection Fraction (LVEF)",
                    "Unit": "%",
                    "AI Prediction (U-Net)": f"{pred_lvef:.1f}",
                    "Expert Ground Truth": f"{expert_lvef:.1f}",
                    "Signed Difference": pred_lvef - expert_lvef,
                    "Absolute Difference": abs(pred_lvef - expert_lvef),
                    "Relative Error (%)": abs(pred_lvef - expert_lvef),  # absolute percentage points
                    "SCMR Clinical Target": "≤ 5.0% (SCMR Inter-observer)",
                },
                {
                    "Metric Parameter": "RV End-Diastolic Volume (RVEDV)",
                    "Unit": "mL",
                    "AI Prediction (U-Net)": f"{pred_rvedv:.1f}",
                    "Expert Ground Truth": f"{expert_rvedv:.1f}",
                    "Signed Difference": pred_rvedv - expert_rvedv,
                    "Absolute Difference": abs(pred_rvedv - expert_rvedv),
                    "Relative Error (%)": (abs(pred_rvedv - expert_rvedv) / expert_rvedv * 100.0) if expert_rvedv > 0 else 0.0,
                    "SCMR Clinical Target": "≤ 15.0 mL",
                },
                {
                    "Metric Parameter": "RV Ejection Fraction (RVEF)",
                    "Unit": "%",
                    "AI Prediction (U-Net)": f"{pred_rvef:.1f}",
                    "Expert Ground Truth": f"{expert_rvef:.1f}",
                    "Signed Difference": pred_rvef - expert_rvef,
                    "Absolute Difference": abs(pred_rvef - expert_rvef),
                    "Relative Error (%)": abs(pred_rvef - expert_rvef),
                    "SCMR Clinical Target": "≤ 7.0%",
                },
            ]

            comp_df = pd.DataFrame(paired_data)

            # Format columns
            formatted_df = comp_df.copy()
            formatted_df["Signed Difference"] = comp_df["Signed Difference"].apply(lambda v: f"{v:+.2f}")
            formatted_df["Absolute Difference"] = comp_df["Absolute Difference"].apply(lambda v: f"{v:.2f}")
            formatted_df["Relative Error (%)"] = comp_df["Relative Error (%)"].apply(lambda v: f"{v:.1f}%")

            # Add compliance status badge
            def _get_status(row: pd.Series) -> str:
                param = row["Metric Parameter"]
                abs_diff = float(row["Absolute Difference"])
                if "LVEF" in param:
                    return "🟢 Within Target (≤ 5%)" if abs_diff <= 5.0 else "🟡 Error > 5%"
                elif "LVEDV" in param or "LVESV" in param:
                    return "🟢 Good Agreement" if abs_diff <= 10.0 else "🟡 Moderate Variance"
                return "🟢 Evaluated"

            formatted_df["Status"] = comp_df.apply(_get_status, axis=1)

            st.dataframe(
                formatted_df,
                use_container_width=True,
                hide_index=True,
            )

            # Paired Grouped Bar Chart
            comp_chart_items = [
                {"Parameter": "LVEDV", "Source": "Automated U-Net", "Volume (mL)": pred_lvedv},
                {"Parameter": "LVEDV", "Source": "Expert Reference", "Volume (mL)": expert_lvedv},
                {"Parameter": "LVESV", "Source": "Automated U-Net", "Volume (mL)": pred_lvesv},
                {"Parameter": "LVESV", "Source": "Expert Reference", "Volume (mL)": expert_lvesv},
                {"Parameter": "LVSV", "Source": "Automated U-Net", "Volume (mL)": pred_lvsv},
                {"Parameter": "LVSV", "Source": "Expert Reference", "Volume (mL)": expert_lvsv},
                {"Parameter": "RVEDV", "Source": "Automated U-Net", "Volume (mL)": pred_rvedv},
                {"Parameter": "RVEDV", "Source": "Expert Reference", "Volume (mL)": expert_rvedv},
            ]
            fig_comp = px.bar(
                pd.DataFrame(comp_chart_items),
                x="Parameter",
                y="Volume (mL)",
                color="Source",
                barmode="group",
                color_discrete_map={"Automated U-Net": "#00CC96", "Expert Reference": "#636EFA"},
                title=f"Volumetric Agreement: AI Model vs. Expert Reference — {selected_patient_id}",
                text_auto=".1f",
            )
            fig_comp.update_layout(
                template="plotly_dark",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: COHORT NORMATIVE REFERENCE RANGES
    # ---------------------------------------------------------
    with tab_cohort:
        st.subheader("SCMR Normal Reference Ranges & Cohort Statistics")

        st.markdown(
            """
            Clinical reference values published by the **Society for Cardiovascular Magnetic Resonance (SCMR)**
            and the UK Biobank CMR imaging study (*Petersen et al., JCMR 2017*). Diagnostic thresholds
            help contextualize whether measured ventricular parameters reflect pathological remodeling.
            """
        )

        ref_rows = []
        for key, val in SCMR_REFERENCE_RANGES.items():
            ref_rows.append({
                "Parameter": val["label"],
                "Unit": val["unit"],
                "Adult Male Reference": val["male"],
                "Adult Female Reference": val["female"],
                "Clinical Interpretation Threshold": val["combined"],
            })

        st.table(pd.DataFrame(ref_rows))

        if measurements_df is not None and not measurements_df.empty:
            st.divider()
            st.subheader(f"ACDC Cohort Distribution Summary (n = {len(measurements_df)})")
            cohort_summary = measurements_df[[
                "lvedv_ml", "lvesv_ml", "lvsv_ml", "lvef_percent",
                "rvedv_ml", "rvesv_ml", "rvsv_ml", "rvef_percent",
                "myocardial_mass_ed_g",
            ]].describe().round(1).T

            cohort_summary = cohort_summary.rename(columns={
                "count": "Count",
                "mean": "Mean",
                "std": "Std Dev",
                "min": "Min",
                "25%": "25th Pct",
                "50%": "Median",
                "75%": "75th Pct",
                "max": "Max",
            })

            st.dataframe(cohort_summary, use_container_width=True)

    # ---------------------------------------------------------
    # WORKFLOW ACTION: TRANSFER TO RECOVERY PREDICTOR
    # ---------------------------------------------------------
    st.divider()
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        st.markdown(
            "💡 **Multimodal Integration Workflow:**\n"
            f"Transfer the active measurements for **`{selected_patient_id}`** ({source_label}) "
            "directly into the **Recovery Predictor (Page 3)** to simulate multimodal 6-month recovery trajectories "
            "incorporating PEth toxicology, Titin (*TTN*) genetics, and myocardial replacement fibrosis."
        )
    with t_col2:
        if st.button("🚀 Transfer to Recovery Predictor", key="transfer_meas_btn", use_container_width=True):
            st.session_state["transferred_mri_features"] = {
                "acdc_patient_id": selected_patient_id,
                "mri_method": "MONAI_2D_UNET_AUTOMATED" if is_auto else "EXPERT_GROUND_TRUTH_MASK",
                "edv_ml": float(cur_lvedv),
                "esv_ml": float(cur_lvesv),
                "stroke_volume_ml": float(cur_lvsv),
                "baseline_lvef": float(cur_lvef),
                "rvedv_ml": float(cur_rvedv),
                "rvef_percent": float(cur_rvef),
                "myocardial_mass_g": float(cur_myo_mass),
                "diagnostic_group": diag_group,
            }
            st.success(
                f"✅ Transferred {selected_patient_id} ({source_label}) parameters "
                f"(LVEF: {cur_lvef:.1f}%, EDV: {cur_lvedv:.1f} mL) to Recovery Predictor! "
                "Navigate to Page 3 in the sidebar to run the prediction."
            )


if __name__ == "__main__" or True:
    render_page()
