"""Patient Profile page for CRIP."""

import streamlit as st
import pandas as pd
from crip.app.components.disclaimer import render_clinical_disclaimer
from crip.ehr.db import SyntheticEHRDatabase, seed_reference_synthetic_cohort


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - Patient Profile",
        page_icon="👤",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("👤 Synthetic Patient Profile & Clinical Cohort")
    st.markdown(
        """
        Explore baseline cardiac volumetrics, toxicological biomarkers (PEth),
        genetic risk markers (TTN variants), and tissue characterization (LGE fibrosis)
        for synthetic research subjects.
        """
    )

    # Initialize local synthetic database
    db = SyntheticEHRDatabase()
    seed_reference_synthetic_cohort(db)
    patients = db.list_patients()

    if not patients:
        st.info("No synthetic patient records found in local SQLite database.")
        return

    patient_ids = [p.patient_id for p in patients]
    selected_id = st.sidebar.selectbox("Select Synthetic Patient ID", patient_ids, index=0)

    selected_patient = next((p for p in patients if p.patient_id == selected_id), patients[0])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Patient Identifier", value=selected_patient.patient_id)
        st.metric(label="Age / Sex", value=f"{selected_patient.age} y / {selected_patient.biological_sex.value}")
    with col2:
        st.metric(label="Baseline LVEF", value=f"{selected_patient.baseline_lvef:.1f}%")
        st.metric(label="Calculated LVEF", value=f"{selected_patient.volume_derived_lvef:.1f}%")
    with col3:
        st.metric(label="End-Diastolic Vol (EDV)", value=f"{selected_patient.edv_ml:.1f} mL")
        st.metric(label="End-Systolic Vol (ESV)", value=f"{selected_patient.esv_ml:.1f} mL")
    with col4:
        st.metric(label="PEth Biomarker", value=f"{selected_patient.peth.value_ng_ml:.1f} ng/mL")
        st.metric(label="Exposure Risk Tier", value=selected_patient.peth.risk_tier.value.replace("_", " "))

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("🧬 Genetics & Tissue Characterization")
        st.markdown(f"- **TTN Variant Status:** `{selected_patient.ttn_variant.value}`")
        st.markdown(f"- **Late Gadolinium Enhancement (LGE):** `{selected_patient.lge_status.value}`")
        st.markdown(f"- **Stroke Volume:** `{selected_patient.stroke_volume_ml:.1f} mL`")
        st.markdown(f"- **PEth Test Date:** `{selected_patient.peth.measurement_date.isoformat()}`")

    with right_col:
        st.subheader("📅 6-Month Longitudinal Outcome (Reference)")
        follow_up = selected_patient.follow_up
        if follow_up.follow_up_lvef is not None:
            delta_lvef = follow_up.follow_up_lvef - selected_patient.baseline_lvef
            st.markdown(f"- **Follow-up LVEF:** `{follow_up.follow_up_lvef:.1f}%` (Δ {delta_lvef:+.1f}%)")
            st.markdown(f"- **Recovery Classification:** `{follow_up.recovery_category.value}`")
            st.markdown(f"- **Assessment Date:** `{follow_up.follow_up_date}`")
            st.markdown(f"- **Abstinence Confirmed:** `{'Yes' if follow_up.sustained_abstinence_confirmed else 'No / Unknown'}`")
            events = ", ".join(follow_up.adverse_cardiac_events) if follow_up.adverse_cardiac_events else "None"
            st.markdown(f"- **Adverse Events:** `{events}`")
        else:
            st.info("Follow-up evaluation pending for this synthetic record.")

    st.divider()
    st.subheader("📋 Cohort Overview (Local SQLite Simulation)")
    table_data = []
    for p in patients:
        table_data.append({
            "Patient ID": p.patient_id,
            "Age": p.age,
            "Sex": p.biological_sex.value,
            "Baseline LVEF (%)": p.baseline_lvef,
            "EDV (mL)": p.edv_ml,
            "ESV (mL)": p.esv_ml,
            "PEth (ng/mL)": p.peth.value_ng_ml,
            "TTN Variant": p.ttn_variant.value,
            "LGE Status": p.lge_status.value,
            "Follow-up LVEF (%)": p.follow_up.follow_up_lvef if p.follow_up.follow_up_lvef is not None else "N/A",
            "Outcome": p.follow_up.recovery_category.value,
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)


if __name__ == "__main__" or True:
    render_page()
