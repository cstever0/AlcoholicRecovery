"""Interactive Scientific & Academic Poster Figure Explorer for CRIP.

Provides high-resolution visualization, metadata inspection, and multi-format export
(PNG, vector SVG, and vector publication PDF) for all verified figures across:
1. Academic Poster & Core Benchmarks (8 figures)
2. Multi-Dataset Synthetic Integration Suite (8 figures)
3. M&Ms Multi-Vendor External Evaluation (5 figures)
4. EMIDEC LGE Tissue Characterization (6 figures)
5. EMIDEC Deep Learning LGE Segmentation (6 figures)
6. Joint Cine Multi-Dataset Training (3 figures)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

FIGURE_COLLECTIONS: Dict[str, Dict[str, Any]] = {
    "poster": {
        "title": "🎓 Academic Poster & Core Benchmark Figures (Stage 2)",
        "description": "Primary publication figures validating U-Net segmentation, Bland–Altman agreement, and canonical recovery modeling.",
        "directory": Path("docs/evaluation/poster_figures"),
        "figures": [
            {
                "id": "fig1",
                "title": "Fig 1: U-Net Segmentation Dice Similarity by Structure",
                "caption": "Sørensen-Dice overlap distributions for LV cavity (0.906), RV cavity (0.807), and myocardium (0.804) across 20 held-out ACDC test subjects.",
                "base": "fig1_segmentation_dice_by_structure",
            },
            {
                "id": "fig2",
                "title": "Fig 2: 95th-Percentile Hausdorff Distance by Structure",
                "caption": "Spatial boundary distance errors (HD95 mm) at End-Diastole across LV cavity (2.91 mm), RV cavity (7.16 mm), and myocardium (4.94 mm).",
                "base": "fig2_hd95_by_structure",
            },
            {
                "id": "fig3",
                "title": "Fig 3: Predicted vs Reference LVEF Scatter Plot",
                "caption": "Correlation (r = 0.9738) and agreement between automated 2D U-Net LVEF and expert ground truth with identity line and ±5% error envelope.",
                "base": "fig3_predicted_vs_reference_lvef",
            },
            {
                "id": "fig4",
                "title": "Fig 4: Bland-Altman Agreement: LVEF",
                "caption": "Paired difference agreement plot for Left Ventricular Ejection Fraction (Mean Bias: -0.55%, 95% LoA: [-10.53%, +9.43%]).",
                "base": "fig4_bland_altman_lvef",
            },
            {
                "id": "fig5",
                "title": "Fig 5: Bland-Altman Agreement: LVEDV",
                "caption": "Bland-Altman difference agreement for Left Ventricular End-Diastolic Volume (Mean Bias: -1.04 mL, 95% LoA: [-18.72, +16.63] mL).",
                "base": "fig5_bland_altman_lvedv",
            },
            {
                "id": "fig6",
                "title": "Fig 6: Bland-Altman Agreement: LVESV",
                "caption": "Bland-Altman difference agreement for Left Ventricular End-Systolic Volume (Mean Bias: +0.22 mL, 95% LoA: [-16.48, +16.92] mL).",
                "base": "fig6_bland_altman_lvesv",
            },
            {
                "id": "fig7",
                "title": "Fig 7: Canonical Recovery Model ROC Curves",
                "caption": "Comparative ROC curves on the sustained-abstinence test cohort (N = 419): Logistic Regression (AUC = 0.9565) vs Random Forest (AUC = 0.9310).",
                "base": "fig7_recovery_model_roc_comparison",
            },
            {
                "id": "fig8",
                "title": "Fig 8: Recovery Model Calibration Curves",
                "caption": "10-bin reliability calibration curves demonstrating well-calibrated recovery probabilities across deciles (Brier Score: 0.0694).",
                "base": "fig8_recovery_model_calibration",
            },
        ],
    },
    "synthetic_integration": {
        "title": "🧪 Multi-Dataset Synthetic Integration Suite (Stage 5C)",
        "description": "Scientific figures demonstrating empirical distribution alignment, cross-modality correlations, and empirical recovery gains.",
        "directory": Path("docs/evaluation/multidataset/synthetic_integration"),
        "figures": [
            {
                "id": "syn_fig1",
                "title": "Fig 1: Synthetic Feature Distributions",
                "caption": "Kernel density distributions of synthetic cardiac biomarkers (LVEF, LVEDV, LV mass, LGE scar %, PEth) generated under empirical constraints.",
                "base": "fig1_synthetic_feature_distributions",
            },
            {
                "id": "syn_fig2",
                "title": "Fig 2: Synthetic vs Held-Out Empirical Distributions",
                "caption": "Two-sample Kolmogorov-Smirnov overlay comparing generated synthetic cohort against held-out empirical test distributions.",
                "base": "fig2_synthetic_vs_held_out_empirical",
            },
            {
                "id": "syn_fig3",
                "title": "Fig 3: Within-Source Empirical Correlation Matrices",
                "caption": "Cross-correlation heatmaps comparing real empirical physiological covariance against the synthesized feature bank.",
                "base": "fig3_within_source_correlations",
            },
            {
                "id": "syn_fig4",
                "title": "Fig 4: EMIDEC Scenario Distributions (ACM vs Ischemic)",
                "caption": "Stratified scar burden and microvascular obstruction (MVO) distributions across simulated alcoholic cardiomyopathy vs ischemic scenarios.",
                "base": "fig4_emidec_scenario_distributions",
            },
            {
                "id": "syn_fig5",
                "title": "Fig 5: Recovery Model ROC Comparison (Baseline vs Empirical)",
                "caption": "All-comers test set ROC curves comparing baseline generator models (AUC: 0.7101) against empirically informed models (AUC: 0.9048).",
                "base": "fig5_recovery_model_roc_comparison",
            },
            {
                "id": "syn_fig6",
                "title": "Fig 6: Recovery Model Calibration Curves (All-Comers)",
                "caption": "Probability calibration curves demonstrating marked reduction in over-confidence when trained on empirical joint distributions.",
                "base": "fig6_recovery_model_calibration_comparison",
            },
            {
                "id": "syn_fig7",
                "title": "Fig 7: Feature Importance Comparison (Baseline vs Empirical)",
                "caption": "Comparative standardized regression coefficients highlighting emergence of LGE scar burden and MVO as dominant recovery modulators.",
                "base": "fig7_feature_importance_comparison",
            },
            {
                "id": "syn_fig8",
                "title": "Fig 8: Data Provenance and Multi-Dataset Source Contributions",
                "caption": "Flow diagram quantifying empirical records contributed across ACDC cine (100), M&Ms cine (114), and EMIDEC LGE (100).",
                "base": "fig8_provenance_and_source_contributions",
            },
        ],
    },
    "mnms_external": {
        "title": "🏥 M&Ms Multi-Vendor External Validation (Stage 2)",
        "description": "Zero-shot generalizability evaluation of the ACDC-trained 2D U-Net across 114 multi-vendor cine MRI studies.",
        "directory": Path("docs/evaluation/external_datasets/mnms"),
        "figures": [
            {
                "id": "mnms_fig1",
                "title": "Fig 1: In-Domain ACDC vs External M&Ms Dice Overlap",
                "caption": "Comparison of segmentation overlap showing domain shift impact: LV Cavity (0.906 in-domain vs 0.887 external), RV (0.807 vs 0.778).",
                "base": "figure1_acdc_vs_mnms_dice",
            },
            {
                "id": "mnms_fig2",
                "title": "Fig 2: In-Domain ACDC vs External M&Ms HD95 Distance",
                "caption": "Boundary distance comparison showing spatial boundary error across clinical imaging domains.",
                "base": "figure2_acdc_vs_mnms_hd95",
            },
            {
                "id": "mnms_fig3",
                "title": "Fig 3: Performance Stratification by MRI Vendor",
                "caption": "Disaggregated segmentation Dice scores across Canon Medical (0.882), GE Healthcare (0.881), Philips (0.888), and Siemens Healthineers (0.893).",
                "base": "figure3_mnms_vendor_breakdown",
            },
            {
                "id": "mnms_fig4",
                "title": "Fig 4: M&Ms External LVEF Functional Correlation",
                "caption": "Automated vs reference LVEF agreement across 114 external patients showing preserved clinical utility despite multi-vendor domain shift.",
                "base": "figure4_mnms_lvef_correlation",
            },
            {
                "id": "mnms_fig5",
                "title": "Fig 5: Multi-Vendor Cine Segmentation Overlays",
                "caption": "Qualitative comparison showing ground truth vs automated contouring across representative Canon, GE, Philips, and Siemens short-axis slices.",
                "base": "figure5_mnms_segmentation_overlay",
            },
        ],
    },
    "emidec_tissue": {
        "title": "🔬 EMIDEC LGE Tissue Characterization (Stage 3)",
        "description": "Empirical quantification of myocardial infarction and microvascular obstruction (no-reflow) on 100 delayed-enhancement CMR scans.",
        "directory": Path("docs/evaluation/external_datasets/emidec"),
        "figures": [
            {
                "id": "emidec_t1",
                "title": "Fig 1: Representative EMIDEC Delayed-Enhancement Overlay",
                "caption": "Multi-region visualization identifying LV blood cavity, normal myocardium, hyper-enhanced infarction, and central hypointense no-reflow core.",
                "base": "figure1_emidec_representative_overlay",
            },
            {
                "id": "emidec_t2",
                "title": "Fig 2: Normal vs Pathological Myocardial Volumes",
                "caption": "Comparative volumetric distribution of viable myocardium against hyper-enhanced scar across normal and pathological cohorts.",
                "base": "figure2_normal_vs_pathological_volumes",
            },
            {
                "id": "emidec_t3",
                "title": "Fig 3: Myocardial Infarction Volume Distribution",
                "caption": "Empirical infarction burden (mL) across 67 pathological infarction cases with expert reference annotations.",
                "base": "figure3_infarction_volume_distribution",
            },
            {
                "id": "emidec_t4",
                "title": "Fig 4: Microvascular Obstruction (No-Reflow) Distribution",
                "caption": "Prevalence and volumetric distribution of severe microvascular obstruction within the acute myocardial infarction cohort.",
                "base": "figure4_no_reflow_distribution",
            },
            {
                "id": "emidec_t5",
                "title": "Fig 5: Infarct Burden vs No-Reflow Burden Correlation",
                "caption": "Scatter plot showing association between total hyper-enhanced scar volume and central no-reflow extent.",
                "base": "figure5_infarct_vs_noreflow_burden",
            },
            {
                "id": "emidec_t6",
                "title": "Fig 6: Infarct Burden Stratified by Clinical Severity",
                "caption": "Pathology burden broken down by clinical severity and ejection fraction impairment subgroups.",
                "base": "figure6_infarct_burden_by_clinical_severity",
            },
        ],
    },
    "emidec_training": {
        "title": "🤖 EMIDEC Deep Learning LGE Segmentation (Stage 5B)",
        "description": "Experimental five-class LGE U-Net training curves, per-class overlap, and volume correlation audits.",
        "directory": Path("docs/evaluation/multidataset/emidec_training"),
        "figures": [
            {
                "id": "emidec_dl1",
                "title": "Fig 1: Five-Class Training & Validation Loss Curves",
                "caption": "Focal Tversky loss trajectories over 50 training epochs on 60 training and 20 validation LGE studies.",
                "base": "figure1_emidec_loss_curves",
            },
            {
                "id": "emidec_dl2",
                "title": "Fig 2: Per-Class Dice Similarity Overlap",
                "caption": "Held-out test Dice across LV cavity (0.837), normal myocardium (0.596), infarction (0.181), and MVO (0.018).",
                "base": "figure2_emidec_per_class_dice",
            },
            {
                "id": "emidec_dl3",
                "title": "Fig 3: HD95 Boundary Distance Distributions",
                "caption": "Held-out spatial Hausdorff distance distributions reflecting localized lesion boundary discrepancies.",
                "base": "figure3_emidec_hd95_distributions",
            },
            {
                "id": "emidec_dl4",
                "title": "Fig 4: Qualitative Five-Class Prediction Panels",
                "caption": "Side-by-side comparison of expert manual reference annotations versus experimental AI predictions.",
                "base": "figure4_emidec_prediction_panel",
            },
            {
                "id": "emidec_dl5",
                "title": "Fig 5: Infarct Volume Correlation (Predicted vs Ground Truth)",
                "caption": "Scatter correlation between U-Net predicted scar volume and expert manual volumetrics.",
                "base": "figure5_infarct_volume_correlation",
            },
            {
                "id": "emidec_dl6",
                "title": "Fig 6: MVO Volume Correlation (Predicted vs Ground Truth)",
                "caption": "Evaluation of microvascular obstruction volume correlation illustrating acute false-positive lesion over-segmentation.",
                "base": "figure6_mvo_volume_correlation",
            },
        ],
    },
    "joint_cine": {
        "title": "🔄 Joint Cine Multi-Dataset Training (Stage 5A)",
        "description": "Matched comparison between ACDC-only U-Net and joint ACDC + M&Ms cine U-Net on 54 held-out test subjects.",
        "directory": Path("docs/evaluation/multidataset"),
        "figures": [
            {
                "id": "joint_fig1",
                "title": "Fig 1: Matched In-Domain vs Joint Cine Dice Comparison",
                "caption": "Head-to-head comparison demonstrating joint training eliminates external domain degradation while maintaining in-domain fidelity.",
                "base": "figure1_matched_dice_comparison",
            },
            {
                "id": "joint_fig2",
                "title": "Fig 2: LVEF MAE Comparison Across Cohorts",
                "caption": "Mean absolute LVEF functional error across in-domain ACDC and external M&Ms test subjects.",
                "base": "figure2_lvef_mae_comparison",
            },
            {
                "id": "joint_fig3",
                "title": "Fig 3: Vendor-Stratified Generalization Gain",
                "caption": "Vendor-specific Dice improvements achieved by joint multi-vendor training across Canon, GE, Philips, and Siemens scanners.",
                "base": "figure3_vendor_stratified_gain",
            },
        ],
    },
}


def render_figure_explorer() -> None:
    """Render interactive figure gallery with metadata and multi-format download capabilities."""
    st.subheader("🖼️ Scientific & Academic Poster Figure Explorer")
    st.markdown(
        "Interactive explorer for high-resolution publication and poster figures generated during the "
        "reproducible research stages of CRIP. Every figure includes full clinical captions, provenance data, "
        "and direct export options in **PNG**, vector **SVG**, and vector publication **PDF** formats."
    )

    category_keys = list(FIGURE_COLLECTIONS.keys())
    category_labels = [FIGURE_COLLECTIONS[k]["title"] for k in category_keys]

    selected_label = st.selectbox(
        "Select Figure Collection:",
        options=category_labels,
        index=0,
    )
    selected_key = category_keys[category_labels.index(selected_label)]
    active_collection = FIGURE_COLLECTIONS[selected_key]

    st.caption(active_collection["description"])

    figures_list = active_collection["figures"]
    fig_titles = [f["title"] for f in figures_list]

    selected_fig_title = st.selectbox(
        "Select Figure to Inspect:",
        options=fig_titles,
        index=0,
    )
    selected_fig = figures_list[fig_titles.index(selected_fig_title)]

    fig_dir = active_collection["directory"]
    base_name = selected_fig["base"]

    png_path = fig_dir / f"{base_name}.png"
    svg_path = fig_dir / f"{base_name}.svg"
    pdf_path = fig_dir / f"{base_name}.pdf"

    # Display figure image
    if png_path.is_file():
        st.image(str(png_path), caption=selected_fig["caption"], use_container_width=True)
    else:
        st.warning(f"⚠️ Image preview not found at `{png_path}`.")

    # Export / Download Buttons
    st.markdown("#### 📥 Publication Export Downloads")
    dl_col1, dl_col2, dl_col3, dl_spacer = st.columns([1, 1, 1, 2])

    with dl_col1:
        if png_path.is_file():
            st.download_button(
                label="⬇️ Download PNG",
                data=png_path.read_bytes(),
                file_name=f"{base_name}.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.button("PNG Unavailable", disabled=True, use_container_width=True)

    with dl_col2:
        if svg_path.is_file():
            st.download_button(
                label="⬇️ Download SVG",
                data=svg_path.read_bytes(),
                file_name=f"{base_name}.svg",
                mime="image/svg+xml",
                use_container_width=True,
            )
        else:
            st.button("SVG Unavailable", disabled=True, use_container_width=True)

    with dl_col3:
        if pdf_path.is_file():
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_path.read_bytes(),
                file_name=f"{base_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button("PDF Unavailable", disabled=True, use_container_width=True)

    st.divider()

    # Detailed caption and scientific interpretation
    st.markdown(f"**Description & Scientific Interpretation:**\n\n{selected_fig['caption']}")
    st.caption(f"📁 Asset Source Path: `{png_path}`")
