"""Multimodal Synthetic Patient Profile & Clinical Cohort page for CRIP."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from crip.app.components.disclaimer import render_clinical_disclaimer
from crip.ehr.db import SyntheticEHRDatabase, seed_reference_synthetic_cohort
from crip.synthetic.schemas import (
    AbstinenceScenario,
    FollowUpOutcomes,
    LGEPattern,
    LGEStatus,
    PatientProfile,
    PEthMeasurement,
    PEthRiskTier,
    ProvenanceType,
    RecoveryCategory,
    Sex,
    TTNClassification,
    TTNVariant,
    map_ttn_variant_to_classification,
)

COHORT_CSV_PATH = Path("data/synthetic/synthetic_cardiomyopathy_cohort.csv")


@st.cache_data
def load_cohort_dataframe(csv_path: str) -> Optional[pd.DataFrame]:
    """Load cached cohort dataframe from CSV if available."""
    path = Path(csv_path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def render_page() -> None:
    st.set_page_config(
        page_title="CRIP - Multimodal Patient Profile",
        page_icon="👤",
        layout="wide",
    )

    render_clinical_disclaimer(location="main")
    render_clinical_disclaimer(location="sidebar")

    st.title("👤 Multimodal Synthetic Patient Profile & Clinical Cohort")
    st.markdown(
        """
        Explore baseline cardiac magnetic resonance (CMR) volumetrics, toxicological biomarkers
        (Phosphatidylethanol / PEth), ACMG-AMP 5-tier Titin (*TTN*) genetics, replacement fibrosis (LGE),
        and simulated 6-month left ventricular recovery trajectories for synthetic cardiomyopathy research subjects.
        """
    )

    # Initialize local synthetic database
    db = SyntheticEHRDatabase()
    seed_reference_synthetic_cohort(db)

    # Data source selection in sidebar
    st.sidebar.header("📁 Cohort Data Source")
    cohort_options = ["Reference Illustrative Cohort (N=3)"]
    has_full_cohort = COHORT_CSV_PATH.is_file()
    if has_full_cohort:
        cohort_options.insert(0, "Full Multimodal Cohort (N=3,000)")

    selected_source = st.sidebar.selectbox("Select Cohort Source", cohort_options, index=0)

    # Load data based on selection
    patients: List[PatientProfile] = []
    df_raw: Optional[pd.DataFrame] = None

    if selected_source == "Full Multimodal Cohort (N=3,000)" and has_full_cohort:
        df_raw = load_cohort_dataframe(str(COHORT_CSV_PATH))

        # Check if full cohort is already synced into SQLite
        if db.count_patients() < 100:
            if st.sidebar.button("📥 Sync 3,000 Cohort to SQLite DB"):
                with st.spinner("Importing 3,000 synthetic records into SQLite database..."):
                    db.import_cohort_from_csv(COHORT_CSV_PATH)
                st.sidebar.success(f"Synced {db.count_patients()} records to SQLite!")

    # Load from DB or CSV
    if selected_source == "Reference Illustrative Cohort (N=3)" or df_raw is None:
        patients = db.list_patients(limit=200)
    else:
        # Load from dataframe for fast, interactive browsing
        patients = []
        for _, row in df_raw.head(3000).iterrows():
            try:
                peth_val = float(row["peth_value_ng_ml"]) if pd.notna(row.get("peth_value_ng_ml")) else 0.0
                peth_dt_str = str(row.get("peth_measurement_date")) if pd.notna(row.get("peth_measurement_date")) else "2025-01-15"
                try:
                    peth_dt = date.fromisoformat(peth_dt_str)
                except ValueError:
                    peth_dt = date(2025, 1, 15)
                peth_obj = PEthMeasurement(value_ng_ml=peth_val, measurement_date=peth_dt)

                ttn_cls_val = str(row.get("ttn_classification", "NOT_TESTED"))
                try:
                    ttn_cls = TTNClassification(ttn_cls_val)
                except ValueError:
                    ttn_cls = TTNClassification.NOT_TESTED

                ttn_leg_val = str(row.get("ttn_variant_legacy", "NOT_DETECTED"))
                try:
                    ttn_leg = TTNVariant(ttn_leg_val)
                except ValueError:
                    ttn_leg = TTNVariant.NOT_DETECTED

                lge_st_val = str(row.get("lge_status", "NEGATIVE"))
                lge_st = LGEStatus(lge_st_val) if lge_st_val in LGEStatus._value2member_map_ else LGEStatus.NEGATIVE

                lge_pat_val = str(row.get("lge_pattern", "NONE"))
                lge_pat = LGEPattern(lge_pat_val) if lge_pat_val in LGEPattern._value2member_map_ else LGEPattern.NONE

                fu_dt_str = str(row.get("follow_up_date")) if pd.notna(row.get("follow_up_date")) else None
                fu_dt = date.fromisoformat(fu_dt_str) if fu_dt_str else None
                fu_lvef = float(row["follow_up_lvef"]) if pd.notna(row.get("follow_up_lvef")) else None
                delta = float(row["delta_lvef"]) if pd.notna(row.get("delta_lvef")) else (fu_lvef - float(row["baseline_lvef"]) if fu_lvef is not None else None)
                rec_bin = int(row["recovery_binary"]) if pd.notna(row.get("recovery_binary")) else (1 if delta is not None and delta >= 10.0 else 0)

                rec_cat_val = str(row.get("recovery_category", "PENDING_EVALUATION"))
                try:
                    rec_cat = RecoveryCategory(rec_cat_val)
                except ValueError:
                    rec_cat = RecoveryCategory.PENDING_EVALUATION

                scen_val = str(row.get("assumed_abstinence_scenario", "SUSTAINED_ABSTINENCE"))
                try:
                    scen = AbstinenceScenario(scen_val)
                except ValueError:
                    scen = AbstinenceScenario.SUSTAINED_ABSTINENCE

                sustained = bool(row["sustained_abstinence_confirmed"]) if pd.notna(row.get("sustained_abstinence_confirmed")) else None

                adv_str = str(row.get("adverse_cardiac_events", "NONE"))
                adv_events = [x.strip() for x in adv_str.split(";") if x.strip() and x.strip() != "NONE"]

                fu_obj = FollowUpOutcomes(
                    follow_up_date=fu_dt,
                    follow_up_lvef=fu_lvef,
                    recovery_category=rec_cat,
                    recovery_binary=rec_bin,
                    delta_lvef=delta,
                    adverse_cardiac_events=adv_events,
                    sustained_abstinence_confirmed=sustained,
                    assumed_abstinence_scenario=scen,
                )

                patients.append(
                    PatientProfile(
                        patient_id=str(row["patient_id"]),
                        age=int(row["age"]),
                        biological_sex=Sex(str(row["biological_sex"])),
                        baseline_lvef=float(row["baseline_lvef"]),
                        edv_ml=float(row["edv_ml"]),
                        esv_ml=float(row["esv_ml"]),
                        peth=peth_obj,
                        ttn_variant=ttn_leg,
                        ttn_classification=ttn_cls,
                        ttn_provenance=str(row.get("ttn_provenance", "ClinVar / ACMG-AMP 2015 Tiered Assessment")),
                        lge_status=lge_st,
                        lge_pattern=lge_pat,
                        lge_extent_percent=float(row["lge_extent_percent"]) if pd.notna(row.get("lge_extent_percent")) else None,
                        lge_available=bool(row.get("lge_available", True)) if pd.notna(row.get("lge_available")) else True,
                        rvedv_ml=float(row["rvedv_ml"]) if pd.notna(row.get("rvedv_ml")) else None,
                        rvesv_ml=float(row["rvesv_ml"]) if pd.notna(row.get("rvesv_ml")) else None,
                        rvef_percent=float(row["rvef_percent"]) if pd.notna(row.get("rvef_percent")) else None,
                        myocardial_mass_g=float(row["myocardial_mass_g"]) if pd.notna(row.get("myocardial_mass_g")) else None,
                        reported_drinks_per_week=float(row["reported_drinks_per_week"]) if pd.notna(row.get("reported_drinks_per_week")) else None,
                        time_since_last_drink_days=int(row["time_since_last_drink_days"]) if pd.notna(row.get("time_since_last_drink_days")) else None,
                        years_heavy_drinking=float(row["years_heavy_drinking"]) if pd.notna(row.get("years_heavy_drinking")) else None,
                        alcohol_history_available=bool(row.get("alcohol_history_available", True)),
                        peth_available=bool(row.get("peth_available", True)),
                        mri_source=str(row.get("mri_source", "SYNTHETIC_SIMULATED")),
                        provenance=ProvenanceType.SYNTHETIC_SIMULATED,
                        generation_version=str(row.get("generation_version", "v1.0-m3a")),
                        generation_seed=int(row["generation_seed"]) if pd.notna(row.get("generation_seed")) else None,
                        follow_up=fu_obj,
                    )
                )
            except Exception:
                continue

    if not patients:
        st.info("No synthetic patient records found.")
        return

    # Filter sidebar controls
    st.sidebar.header("🔍 Cohort Filter Controls")

    filter_sex = st.sidebar.selectbox("Biological Sex", ["All", "MALE", "FEMALE"], index=0)
    filter_ttn = st.sidebar.selectbox(
        "TTN Genetic Classification",
        [
            "All",
            "PATHOGENIC",
            "LIKELY_PATHOGENIC",
            "VARIANT_OF_UNCERTAIN_SIGNIFICANCE",
            "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC",
            "NOT_TESTED",
        ],
        index=0,
    )
    filter_lge = st.sidebar.selectbox(
        "Late Gadolinium Enhancement (LGE)",
        ["All", "NEGATIVE", "POSITIVE (Midwall / Transmural / Subepicardial)", "UNAVAILABLE"],
        index=0,
    )
    filter_peth_tier = st.sidebar.selectbox(
        "PEth Exposure Risk Tier",
        ["All", "ABSTINENT_OR_LIGHT (<20 ng/mL)", "SIGNIFICANT_MODERATE (20-200 ng/mL)", "HEAVY_CHRONIC (>200 ng/mL)", "UNAVAILABLE"],
        index=0,
    )
    filter_recovery = st.sidebar.selectbox(
        "6-Month Recovery Status (ΔLVEF ≥ 10%)",
        ["All", "Recovered (ΔLVEF ≥ 10%)", "Non-Recovered (ΔLVEF < 10%)"],
        index=0,
    )

    # Apply filters
    filtered_patients = patients
    if filter_sex != "All":
        filtered_patients = [p for p in filtered_patients if p.biological_sex.value == filter_sex]

    if filter_ttn != "All":
        filtered_patients = [
            p for p in filtered_patients
            if (p.ttn_classification.value if p.ttn_classification else "NOT_TESTED") == filter_ttn
        ]

    if filter_lge != "All":
        if filter_lge == "NEGATIVE":
            filtered_patients = [p for p in filtered_patients if p.lge_available and p.lge_status == LGEStatus.NEGATIVE]
        elif filter_lge.startswith("POSITIVE"):
            filtered_patients = [p for p in filtered_patients if p.lge_available and p.lge_status != LGEStatus.NEGATIVE]
        elif filter_lge == "UNAVAILABLE":
            filtered_patients = [p for p in filtered_patients if not p.lge_available]

    if filter_peth_tier != "All":
        if "ABSTINENT_OR_LIGHT" in filter_peth_tier:
            filtered_patients = [p for p in filtered_patients if p.peth_available and p.peth.risk_tier == PEthRiskTier.ABSTINENT_OR_LIGHT]
        elif "SIGNIFICANT_MODERATE" in filter_peth_tier:
            filtered_patients = [p for p in filtered_patients if p.peth_available and p.peth.risk_tier == PEthRiskTier.SIGNIFICANT_MODERATE]
        elif "HEAVY_CHRONIC" in filter_peth_tier:
            filtered_patients = [p for p in filtered_patients if p.peth_available and p.peth.risk_tier == PEthRiskTier.HEAVY_CHRONIC]
        elif "UNAVAILABLE" in filter_peth_tier:
            filtered_patients = [p for p in filtered_patients if not p.peth_available]

    if filter_recovery != "All":
        if "Recovered" in filter_recovery and "Non" not in filter_recovery:
            filtered_patients = [p for p in filtered_patients if p.follow_up.recovery_binary == 1]
        elif "Non-Recovered" in filter_recovery:
            filtered_patients = [p for p in filtered_patients if p.follow_up.recovery_binary == 0]

    # Overview KPI Metrics Row
    total_cohort = len(patients)
    n_filtered = len(filtered_patients)
    avg_lvef = sum(p.baseline_lvef for p in filtered_patients) / max(n_filtered, 1)
    rec_count = sum(1 for p in filtered_patients if p.follow_up.recovery_binary == 1)
    rec_rate = (rec_count / max(n_filtered, 1)) * 100.0
    lge_pos = sum(1 for p in filtered_patients if p.lge_available and p.lge_status != LGEStatus.NEGATIVE)
    lge_rate = (lge_pos / max(n_filtered, 1)) * 100.0
    ttn_path = sum(1 for p in filtered_patients if p.ttn_classification in (TTNClassification.PATHOGENIC, TTNClassification.LIKELY_PATHOGENIC))
    ttn_rate = (ttn_path / max(n_filtered, 1)) * 100.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Cohort Size", f"{n_filtered:,}", delta=f"of {total_cohort:,} total" if n_filtered != total_cohort else None)
    with kpi2:
        st.metric("Mean Baseline LVEF", f"{avg_lvef:.1f}%")
    with kpi3:
        st.metric("Recovery Rate (ΔEF ≥ 10%)", f"{rec_rate:.1f}%", delta=f"{rec_count} cases")
    with kpi4:
        st.metric("LGE Fibrosis Rate", f"{lge_rate:.1f}%", delta=f"{lge_pos} positive")
    with kpi5:
        st.metric("TTN Truncating Rate", f"{ttn_rate:.1f}%", delta=f"{ttn_path} pathogenic")

    st.divider()

    # Patient Selector
    if not filtered_patients:
        st.warning("No patients match the active filter criteria. Please broaden your filters in the sidebar.")
        return

    patient_ids = [p.patient_id for p in filtered_patients]
    selected_id = st.sidebar.selectbox("Select Patient to Inspect", patient_ids, index=0)
    selected_patient = next((p for p in filtered_patients if p.patient_id == selected_id), filtered_patients[0])

    # Patient Header Banner
    banner_col1, banner_col2, banner_col3, banner_col4 = st.columns([1.5, 1, 1, 1.2])
    with banner_col1:
        st.subheader(f"Patient `{selected_patient.patient_id}`")
        prov_badge = selected_patient.provenance.value if hasattr(selected_patient.provenance, "value") else str(selected_patient.provenance)
        st.caption(f"Provenance: **{prov_badge}** | Gen Model: **{selected_patient.generation_version}**")
    with banner_col2:
        st.metric("Age / Biological Sex", f"{selected_patient.age}y / {selected_patient.biological_sex.value}")
    with banner_col3:
        st.metric("Baseline LVEF", f"{selected_patient.baseline_lvef:.1f}%")
    with banner_col4:
        delta_str = f"Δ {selected_patient.delta_lvef:+.1f}%" if selected_patient.delta_lvef is not None else "N/A"
        rec_status = "RECOVERED (ΔLVEF ≥ 10%)" if selected_patient.recovery_binary == 1 else "NON-RECOVERED"
        st.metric("6-Month Trajectory", delta_str, delta=rec_status)

    # Detailed Multimodal Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🫀 CMR Volumetrics",
        "🍷 Toxicology & Alcohol",
        "🧬 Titin Genetics",
        "🔬 CMR Tissue Fibrosis",
        "📈 6-Month Trajectory",
        "📋 Full Variable Provenance",
    ])

    with tab1:
        st.markdown("#### Baseline Cardiac Magnetic Resonance Volumetrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Left Ventricular EDV", f"{selected_patient.edv_ml:.1f} mL")
            st.metric("Left Ventricular ESV", f"{selected_patient.esv_ml:.1f} mL")
            st.metric("LV Stroke Volume", f"{selected_patient.stroke_volume_ml:.1f} mL")
        with m_col2:
            st.metric("Calculated LVEF", f"{selected_patient.volume_derived_lvef:.1f}%")
            if selected_patient.rvedv_ml is not None:
                st.metric("Right Ventricular EDV", f"{selected_patient.rvedv_ml:.1f} mL")
                st.metric("Right Ventricular ESV", f"{selected_patient.rvesv_ml:.1f} mL" if selected_patient.rvesv_ml else "N/A")
            else:
                st.info("Right ventricular volumes not acquired.")
        with m_col3:
            if selected_patient.rvef_percent is not None:
                st.metric("Calculated RVEF", f"{selected_patient.rvef_percent:.1f}%")
            if selected_patient.myocardial_mass_g is not None:
                st.metric("Estimated LV Mass (g)", f"{selected_patient.myocardial_mass_g:.1f} g")
            st.caption(f"MRI Prior Source: `{selected_patient.mri_source}`")

        # Mathematical consistency confirmation
        if selected_patient.esv_ml < selected_patient.edv_ml:
            st.success("✅ Physiological Volume Consistency Verified: ESV < EDV and Stroke Volume > 0.")
        else:
            st.error("⚠️ Inconsistent Volumetric Parameters: ESV exceeds or equals EDV.")

    with tab2:
        st.markdown("#### Toxicological Biomarkers & Reported Alcohol History")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.subheader("Objective Toxicological Exposure (PEth)")
            if selected_patient.peth_available and selected_patient.peth:
                st.metric("Phosphatidylethanol (PEth)", f"{selected_patient.peth.value_ng_ml:.1f} ng/mL")
                tier_badge = selected_patient.peth.risk_tier.value.replace("_", " ")
                st.markdown(f"- **Exposure Risk Tier:** `{tier_badge}`")
                st.markdown(f"- **Blood Draw Date:** `{selected_patient.peth.measurement_date.isoformat()}`")
                st.markdown("- **Biomarker Window:** Reflects exposure over the preceding 2–4 weeks.")
            else:
                st.warning("⚠️ Baseline PEth quantification unavailable (Missing clinical toxicological observation).")

        with t_col2:
            st.subheader("Self-Reported Consumption History")
            if selected_patient.alcohol_history_available:
                st.markdown(f"- **Reported Weekly Intake:** `{selected_patient.reported_drinks_per_week or 'Unspecified'} standard drinks/wk`")
                st.markdown(f"- **Time Since Last Drink:** `{selected_patient.time_since_last_drink_days or 'Unspecified'} days`")
                st.markdown(f"- **Cumulative Heavy Drinking:** `{selected_patient.years_heavy_drinking or 'Unspecified'} years (>80 g/day)`")
            else:
                st.info("Self-reported alcohol history missing from intake documentation.")

        st.caption(
            "CLINICAL NOTE: PEth is an objective direct ethanol metabolite reflecting recent 2–4 week exposure. "
            "It does not alone diagnose alcoholic cardiomyopathy or substitute for complete medical history."
        )

    with tab3:
        st.markdown("#### Titin (*TTN*) Genetics & ACMG-AMP Classification")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            ttn_cls_val = selected_patient.ttn_classification.value if selected_patient.ttn_classification else "NOT_TESTED"
            st.markdown(f"**ACMG-AMP 5-Tier Classification:**")
            if ttn_cls_val in ("PATHOGENIC", "LIKELY_PATHOGENIC"):
                st.error(f"🔴 **{ttn_cls_val}** (Pathogenic Truncating Variant / TTNtv)")
            elif ttn_cls_val == "VARIANT_OF_UNCERTAIN_SIGNIFICANCE":
                st.warning(f"🟡 **VARIANT OF UNCERTAIN SIGNIFICANCE (VUS)**")
            elif ttn_cls_val == "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC":
                st.success(f"🟢 **NO PATHOGENIC VARIANT DETECTED**")
            else:
                st.info(f"⚪ **NOT TESTED / MISSING GENETIC DATA**")

            st.markdown(f"- **Legacy Classification:** `{selected_patient.ttn_variant.value}`")
            st.markdown(f"- **Curation Provenance:** `{selected_patient.ttn_provenance}`")

        with g_col2:
            st.info(
                "**Clinical Genetics Safety Notice (ACMG-AMP / ClinGen Guidelines):**\n\n"
                "• A Variant of Uncertain Significance (VUS) must **NEVER** be interpreted as a confirmed "
                "pathogenic etiology or used in predictive algorithms to withhold standard therapies.\n\n"
                "• Pathogenic truncating variants (TTNtv) in the A-band act as significant susceptibility modifiers "
                "in dilated and alcoholic cardiomyopathy (Ware et al., NEJM 2016)."
            )

    with tab4:
        st.markdown("#### Late Gadolinium Enhancement (LGE) Tissue Fibrosis")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            if selected_patient.lge_available:
                st.markdown(f"- **LGE Status:** `{selected_patient.lge_status.value}`")
                st.markdown(f"- **Fibrosis Pattern:** `{selected_patient.lge_pattern.value if selected_patient.lge_pattern else 'NONE'}`")
                extent_str = f"{selected_patient.lge_extent_percent:.1f}%" if selected_patient.lge_extent_percent is not None else "0.0%"
                st.metric("Quantified Fibrosis Extent (% of LV Mass)", extent_str)
            else:
                st.warning("⚠️ LGE tissue characterization not performed (e.g., severe renal impairment eGFR < 30 mL/min).")
        with f_col2:
            st.markdown(
                "**Myocardial Tissue Remodeling Significance:**\n\n"
                "Midwall replacement fibrosis (scar) indicates non-reversible collagen deposition and is a strong "
                "independent negative predictor of functional left ventricular reverse remodeling (Halliday et al., Circulation 2017)."
            )

    with tab5:
        st.markdown("#### 6-Month Longitudinal Recovery Trajectory")
        fu = selected_patient.follow_up
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.metric("Follow-up LVEF", f"{fu.follow_up_lvef:.1f}%" if fu.follow_up_lvef is not None else "Pending")
            st.metric("Absolute Change in LVEF (Δ)", f"{fu.delta_lvef:+.1f}%" if fu.delta_lvef is not None else "Pending")
            st.markdown(f"- **Recovery Classification:** `{fu.recovery_category.value}`")
            st.markdown(f"- **Binary Recovery Status:** `{'RECOVERED (1)' if fu.recovery_binary == 1 else 'NON-RECOVERED (0)'}`")
        with r_col2:
            st.markdown(f"- **Simulated Exposure Scenario:** `{fu.assumed_abstinence_scenario.value}`")
            st.markdown(f"- **Sustained Abstinence Confirmed:** `{'Yes' if fu.sustained_abstinence_confirmed else 'No / Continued Drinking'}`")
            st.markdown(f"- **Follow-up Date:** `{fu.follow_up_date or 'Pending'}`")
            events_str = ", ".join(fu.adverse_cardiac_events) if fu.adverse_cardiac_events else "None documented"
            st.markdown(f"- **Adverse Cardiac Events:** `{events_str}`")

        st.caption(
            "EXPERIMENTAL ENDPOINT DEFINITION: Primary recovery is defined as ΔLVEF ≥ 10.0 percentage points at 6-month follow-up. "
            "This endpoint is synthetic and experimental."
        )

    with tab6:
        st.markdown("#### Variable-Level Provenance & Audit Trail")
        prov_dict = selected_patient.variable_provenance
        if prov_dict:
            prov_rows = [{"Variable": k, "Provenance / Data Origin": v} for k, v in prov_dict.items()]
            st.dataframe(pd.DataFrame(prov_rows), use_container_width=True)
        else:
            st.info("Variable provenance dictionary empty for this record. Default origin: SYNTHETIC_SIMULATED.")

    # Cohort Overview Table
    st.divider()
    st.subheader(f"📋 Cohort Registry ({len(filtered_patients):,} Displayed)")
    summary_records = []
    for p in filtered_patients[:100]:
        summary_records.append({
            "Patient ID": p.patient_id,
            "Age": p.age,
            "Sex": p.biological_sex.value,
            "LVEF Baseline (%)": p.baseline_lvef,
            "LVEDV (mL)": p.edv_ml,
            "PEth (ng/mL)": p.peth.value_ng_ml if p.peth_available else "MISSING",
            "PEth Tier": p.peth.risk_tier.value if p.peth_available else "UNAVAILABLE",
            "TTN Genetics": p.ttn_classification.value if p.ttn_classification else "NOT_TESTED",
            "LGE Fibrosis": p.lge_status.value if p.lge_available else "UNAVAILABLE",
            "LVEF 6m (%)": p.follow_up.follow_up_lvef if p.follow_up.follow_up_lvef is not None else "N/A",
            "Δ LVEF (%)": f"{p.delta_lvef:+.1f}" if p.delta_lvef is not None else "N/A",
            "Recovery (1/0)": p.recovery_binary,
            "Outcome": p.follow_up.recovery_category.value,
        })

    st.dataframe(pd.DataFrame(summary_records), use_container_width=True)
    if len(filtered_patients) > 100:
        st.caption(f"Showing first 100 of {len(filtered_patients):,} patients. Use sidebar filters to narrow selection.")


if __name__ == "__main__" or True:
    render_page()
