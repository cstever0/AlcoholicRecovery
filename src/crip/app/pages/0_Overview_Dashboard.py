"""Overview Dashboard for CRIP - Consolidated Platform Status & Key Cardiac Measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from crip.app.components.disclaimer import render_clinical_disclaimer
from crip.mri.acdc_loader import (
    ACDCDataLoader,
    DEFAULT_ACDC_TESTING_DIR,
    DEFAULT_ACDC_TRAINING_DIR,
)
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH

CARDIAC_MEASUREMENTS_CSV = Path("data/processed/acdc/cardiac_measurements.csv")
SYNTHETIC_COHORT_CSV = Path("data/synthetic/synthetic_cardiomyopathy_cohort.csv")
RECOVERY_MODEL_PATH = Path("models/recovery/recovery_pipeline.joblib")


@st.cache_data
def _load_measurements_table() -> Optional[pd.DataFrame]:
    """Load cached cardiac measurements table."""
    if CARDIAC_MEASUREMENTS_CSV.is_file():
        try:
            return pd.read_csv(CARDIAC_MEASUREMENTS_CSV)
        except Exception:
            return None
    return None


@st.cache_data
def _check_system_artifacts() -> Dict[str, Dict[str, Any]]:
    """Inspect availability of data and model artifacts across the platform."""
    # 1. ACDC Training Dataset
    acdc_train_loader = ACDCDataLoader(DEFAULT_ACDC_TRAINING_DIR)
    train_available = acdc_train_loader.is_available()
    train_count = len(acdc_train_loader.list_patients()) if train_available else 0

    # 2. ACDC Testing Dataset
    acdc_test_loader = ACDCDataLoader(DEFAULT_ACDC_TESTING_DIR)
    test_available = acdc_test_loader.is_available()
    test_count = len(acdc_test_loader.list_patients()) if test_available else 0

    # 3. MONAI 2D U-Net Checkpoint
    unet_available = DEFAULT_CHECKPOINT_PATH.is_file()
    unet_size_mb = (DEFAULT_CHECKPOINT_PATH.stat().st_size / (1024 * 1024)) if unet_available else 0.0

    # 4. Synthetic Multimodal Cohort
    synth_available = SYNTHETIC_COHORT_CSV.is_file()
    synth_count = 0
    if synth_available:
        try:
            synth_df = pd.read_csv(SYNTHETIC_COHORT_CSV)
            synth_count = len(synth_df)
        except Exception:
            synth_count = 0

    # 5. Recovery Prediction Model
    recovery_available = RECOVERY_MODEL_PATH.is_file()

    return {
        "acdc_training": {
            "name": "ACDC Cine CMR Training Set",
            "available": train_available,
            "detail": f"{train_count} annotated patients" if train_available else "Not installed at data/raw/acdc/training",
        },
        "acdc_testing": {
            "name": "ACDC Cine CMR Testing Set",
            "available": test_available,
            "detail": f"{test_count} unannotated scans" if test_available else "Not installed at data/raw/acdc/testing",
        },
        "unet_model": {
            "name": "MONAI 2D U-Net Checkpoint",
            "available": unet_available,
            "detail": f"{unet_size_mb:.1f} MB (models/mri/best_unet_model.pt)" if unet_available else "Weights not found (train via train_segmentation.py)",
        },
        "synthetic_cohort": {
            "name": "Multimodal Synthetic Cohort",
            "available": synth_available,
            "detail": f"{synth_count:,} patient records" if synth_available else "Not generated at data/synthetic",
        },
        "recovery_model": {
            "name": "Interpretable Recovery Model",
            "available": recovery_available,
            "detail": "Scikit-Learn Logistic Regression (v1.0-m3b)" if recovery_available else "Artifact missing (models/recovery)",
        },
    }


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - Overview Dashboard",
        page_icon="🏠",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("🏠 Cardiomyopathy Recovery Intelligence Platform (CRIP)")
    st.markdown(
        """
        **Clinical AI Research Proof of Concept** investigating recovery trajectories in cardiomyopathy
        associated with heavy alcohol consumption by coupling **real cardiac magnetic resonance (CMR) imaging**
        with **objective toxicological biomarkers (PEth)**, **Titin (*TTN*) sarcomeric genetics**, and **myocardial replacement fibrosis**.
        """
        "\n\n"
        "---"
    )

    # Section 1: Platform Artifacts & Subsystem Status
    st.subheader("🖥️ Platform Status & Subsystem Health")
    artifacts = _check_system_artifacts()

    stat_cols = st.columns(len(artifacts))
    for col, (key, info) in zip(stat_cols, artifacts.items()):
        with col:
            icon = "🟢" if info["available"] else "🔴"
            st.metric(
                label=info["name"],
                value="Available" if info["available"] else "Unavailable",
                delta="Operational" if info["available"] else "Action Required",
                delta_color="normal" if info["available"] else "inverse",
            )
            st.caption(f"{icon} {info['detail']}")

    st.divider()

    # Section 2: Selected Patient / Study Overview & Key Volumetrics
    st.subheader("🫀 Patient & Cardiac Study Overview")

    measurements_df = _load_measurements_table()
    loader = ACDCDataLoader(DEFAULT_ACDC_TRAINING_DIR)
    has_dataset = loader.is_available()

    if not has_dataset and (measurements_df is None or measurements_df.empty):
        st.warning(
            "⚠️ **No Cardiac Imaging Data Found:** Neither raw ACDC scans nor processed measurements (`cardiac_measurements.csv`) "
            "were detected. Please consult the **Full Installation Guide** in `README.md` to download the public challenge data "
            "or run the batch pipeline via `python -m crip.mri.pipeline`."
        )
        return

    # Determine patient list from dataframe or raw loader
    if measurements_df is not None and not measurements_df.empty:
        patient_list = measurements_df["patient_id"].tolist()
    else:
        patient_list = loader.list_patients()

    # Selector for study
    c_sel, c_desc = st.columns([1, 2])
    with c_sel:
        selected_patient = st.selectbox(
            "Select Clinical Study (ACDC Patient ID)",
            options=patient_list,
            index=0,
            help="Select an ACDC cardiac MRI patient record to inspect metadata and quantitative chamber parameters.",
        )

    # Extract metadata and measurements for selected patient
    patient_row = None
    if measurements_df is not None and not measurements_df.empty:
        rows = measurements_df[measurements_df["patient_id"] == selected_patient]
        if not rows.empty:
            patient_row = rows.iloc[0]

    # Fallback to loader metadata if row is not in CSV
    patient_meta = None
    if has_dataset:
        try:
            patient_meta = loader.get_patient_metadata(selected_patient)
        except Exception:
            patient_meta = None

    group = patient_row["diagnostic_group"] if patient_row is not None else (patient_meta.group if patient_meta else "Unknown")
    height_cm = patient_row["height_cm"] if patient_row is not None else (patient_meta.height_cm if patient_meta else None)
    weight_kg = patient_row["weight_kg"] if patient_row is not None else (patient_meta.weight_kg if patient_meta else None)
    ed_frame = int(patient_row["ed_frame_index"]) if patient_row is not None and pd.notna(patient_row.get("ed_frame_index")) else (patient_meta.ed_frame if patient_meta else 1)
    es_frame = int(patient_row["es_frame_index"]) if patient_row is not None and pd.notna(patient_row.get("es_frame_index")) else (patient_meta.es_frame if patient_meta else 12)

    with c_desc:
        st.markdown(
            f"**Study Identifier:** `{selected_patient}` | **Diagnostic Cohort:** `{group}`\n\n"
            f"**Data Provenance:** *ACDC 2017 Challenge (University of Lyon / CREATIS)* — Real Cine Short-Axis MRI.\n\n"
            f"**Acquisition Timing:** End-Diastole (ED) Frame `{ed_frame:02d}` | End-Systole (ES) Frame `{es_frame:02d}`"
        )

    # Key Cardiac Measurement Metric Cards
    st.markdown("### 📊 Primary Cardiac Functional Parameters")

    if patient_row is not None:
        lvedv = float(patient_row["lvedv_ml"])
        lvesv = float(patient_row["lvesv_ml"])
        lvsv = float(patient_row["lvsv_ml"])
        lvef = float(patient_row["lvef_percent"])
        rvedv = float(patient_row["rvedv_ml"]) if pd.notna(patient_row.get("rvedv_ml")) else None
        rvesv = float(patient_row["rvesv_ml"]) if pd.notna(patient_row.get("rvesv_ml")) else None
        rvsv = float(patient_row["rvsv_ml"]) if pd.notna(patient_row.get("rvsv_ml")) else None
        rvef = float(patient_row["rvef_percent"]) if pd.notna(patient_row.get("rvef_percent")) else None
        myo_mass = float(patient_row["myocardial_mass_ed_g"]) if pd.notna(patient_row.get("myocardial_mass_ed_g")) else None

        # Determine clinical LVEF classification tag
        if lvef >= 50.0:
            ef_badge = "🟢 Preserved (≥ 50%)"
        elif lvef >= 40.0:
            ef_badge = "🟡 Mildly Reduced (40–49%)"
        else:
            ef_badge = "🔴 Reduced (< 40%)"

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("LV Ejection Fraction (LVEF)", f"{lvef:.1f}%", delta=ef_badge)
            st.caption(f"Stroke Vol: **{lvsv:.1f} mL**")
        with m_col2:
            st.metric("LV End-Diastolic Vol (LVEDV)", f"{lvedv:.1f} mL", delta="Ref: 100–200 mL")
            st.caption(f"End-Systolic Vol: **{lvesv:.1f} mL**")
        with m_col3:
            st.metric("LV End-Systolic Vol (LVESV)", f"{lvesv:.1f} mL")
            st.caption(f"Physiology Check: {'ESV < EDV' if lvesv < lvedv else 'INVALID'}")
        with m_col4:
            if rvef is not None and rvedv is not None:
                st.metric("RV Ejection Fraction (RVEF)", f"{rvef:.1f}%", delta="Normal ≥ 45%")
                st.caption(f"RVEDV: **{rvedv:.1f} mL**")
            else:
                st.metric("RV Ejection Fraction", "Unavailable")
        with m_col5:
            if myo_mass is not None:
                st.metric("Myocardial Mass (ED)", f"{myo_mass:.1f} g", delta="Density: 1.05 g/mL")
                st.caption(f"Height/Weight: {height_cm or 'N/A'}cm / {weight_kg or 'N/A'}kg")
            else:
                st.metric("Myocardial Mass", "Unavailable")
    else:
        st.info(
            f"ℹ️ **Calculated measurements not pre-extracted for `{selected_patient}`.**\n"
            "Run batch measurement extraction via `python -m crip.mri.pipeline` or navigate to "
            "**Page 2 (MRI Analysis)** to compute real-time volumetric parameters from NIfTI segmentations."
        )

    st.markdown("---")

    # Section 3: Navigation Guide to Exploration Modules
    st.subheader("🧭 Guided Workflow & Navigation")
    nav_c1, nav_c2, nav_c3, nav_c4 = st.columns(4)

    with nav_c1:
        st.markdown("#### 🫀 1. MRI Analysis")
        st.write("Inspect short-axis cine slices, view side-by-side original scans and AI-generated U-Net contours, and transfer metrics.")

    with nav_c2:
        st.markdown("#### 📊 2. Cardiac Measurements")
        st.write("Deep dive into complete biventricular parameters, normal clinical reference ranges, and direct expert vs. AI model differences.")

    with nav_c3:
        st.markdown("#### 📈 3. Recovery Estimate")
        st.write("Combine imaging metrics with PEth toxicology, Titin genetics, and fibrosis to predict 6-month recovery trajectories.")

    with nav_c4:
        st.markdown("#### 🔬 4. Model Evaluation")
        st.write("Review rigorous empirical evaluation: Dice scores, LVEF error distributions, Bland-Altman agreement, and clinical fairness audits.")


if __name__ == "__main__" or True:
    render_page()
