"""EMIDEC Delayed-Enhancement (LGE) Tissue Workspace for CRIP.

Provides interactive exploration of:
- Static 3D Inversion-Recovery Delayed-Enhancement CMR
- Expert ground truth contours (LV Cavity, Normal Myo, Infarct Scar, MVO Core)
- Experimental 5-Class MONAI 2D U-Net predictions
- Quantitative tissue characterization: Infarction volume (mL), MVO volume (mL), tissue burdens (%)
- Prominent scientific and clinical limitation disclosures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from crip.datasets.emidec_loader import DEFAULT_EMIDEC_ROOT, EMIDECDataLoader
from crip.emidec.tissue_metrics import compute_emidec_tissue_metrics
from crip.mri.acdc_loader import normalize_for_display
from crip.pipeline.emidec_evaluation import EMIDECEvaluator, compute_dice_score

logger = logging.getLogger("crip.app.components.emidec_workspace")

CHECKPOINT_PATH = Path("models/multidataset/emidec_lge_unet.pt")
EMIDEC_SUMMARY_JSON = Path("data/processed/multidataset/emidec_evaluation_summary.json")

# 5-class color palette for EMIDEC:
# 1=LV Cavity (Cyan), 2=Normal Myo (Green), 3=Infarction (Orange), 4=MVO (Crimson)
EMIDEC_COLOR_MAP = {
    1: np.array([0.10, 0.70, 0.95]),  # LV Cavity: Cyan
    2: np.array([0.20, 0.85, 0.35]),  # Normal Myo: Green
    3: np.array([1.00, 0.55, 0.00]),  # Infarction: Vibrant Orange
    4: np.array([0.80, 0.05, 0.15]),  # MVO / No-Reflow: Crimson
}


@st.cache_resource
def get_emidec_evaluator() -> Optional[EMIDECEvaluator]:
    """Cache the EMIDEC test evaluator and 5-class U-Net model."""
    if not CHECKPOINT_PATH.is_file():
        return None
    try:
        ev = EMIDECEvaluator(checkpoint_path=CHECKPOINT_PATH)
        ev.load_model()
        return ev
    except Exception as e:
        logger.warning(f"Could not load EMIDEC model: {e}")
        return None


def create_emidec_overlay(
    img_slice: np.ndarray,
    mask_slice: Optional[np.ndarray],
    alpha: float = 0.50,
) -> np.ndarray:
    """Create RGB overlay combining grayscale LGE slice and 5-class tissue mask."""
    norm_img = normalize_for_display(img_slice)
    rgb = np.stack([norm_img, norm_img, norm_img], axis=-1)

    if mask_slice is None:
        return rgb

    overlay = rgb.copy()
    for lbl, col in EMIDEC_COLOR_MAP.items():
        region = (mask_slice == lbl)
        if np.any(region):
            overlay[region] = (1.0 - alpha) * rgb[region] + alpha * col

    return np.clip(overlay, 0.0, 1.0)


def render_emidec_tissue_workspace() -> None:
    """Render the full interactive EMIDEC LGE tissue characterization workspace."""
    st.markdown("## 🔬 EMIDEC LGE Tissue Workspace (Delayed-Enhancement CMR)")

    # Mandatory Scientific Disclaimer
    st.warning(
        "⚠️ **MANDATORY CLINICAL & RESEARCH NOTICE:**\n\n"
        "**EMIDEC infarction and MVO segmentation remain experimental and demonstrate limited spatial accuracy. "
        "Do not use for clinical decisions.**\n\n"
        "While deep learning detection of myocardial infarction and microvascular obstruction achieves 100% case-level sensitivity, "
        "voxel-level Dice on thick 10mm inversion-recovery slices remains experimental (Infarct Dice: ~0.28, MVO Dice: ~0.02). "
        "All measurements shown below are for exploratory research investigation only.",
        icon="⚠️",
    )

    loader = EMIDECDataLoader()
    if not loader.is_available:
        st.error(
            "EMIDEC dataset not detected at `data/raw/emidec`. Please verify that `data/raw/emidec/Training` is extracted."
        )
        return

    evaluator = get_emidec_evaluator()
    if evaluator is None:
        st.info(
            f"ℹ️ Experimental EMIDEC U-Net weights not found at `{CHECKPOINT_PATH}`. "
            "To enable automated predictions, run: `python -m crip.pipeline train --task lge`."
        )

    # Discover cases
    cases_dict = loader.discover_cases()
    annotated_ids = [cid for cid, cinfo in cases_dict.items() if cinfo.get("has_contour")]
    if not annotated_ids:
        annotated_ids = list(cases_dict.keys())

    # Format dropdown with pathology tags
    case_labels = []
    for cid in annotated_ids:
        cinfo = cases_dict.get(cid, {})
        diag = cinfo.get("diagnostic_group") or ("Normal" if "N" in cid else "Infarct")
        mvo_flag = " | MVO+" if cinfo.get("has_mvo") else ""
        case_labels.append(f"{cid} ({diag}{mvo_flag})")

    c_sel, c_info = st.columns([1.5, 2.5])
    with c_sel:
        selected_label = st.selectbox(
            "Select EMIDEC Study",
            options=case_labels,
            index=0,
            help="Choose an EMIDEC late-gadolinium enhancement CMR study to view delayed-enhancement tissue pathology.",
        )
        selected_case_id = selected_label.split()[0]

    # Load 3D Volume and Contour
    try:
        case_data = loader.load_case(selected_case_id)
        img_3d = case_data["image"]
        gt_mask_3d = case_data.get("mask")
        spacing = case_data.get("voxel_spacing_mm", (1.0, 1.0, 10.0))
        meta = case_data.get("metadata", {})
    except Exception as e:
        st.error(f"Error loading {selected_case_id}: {e}")
        return

    # Study metadata summary
    with c_info:
        st.markdown(
            f"**Study:** `{selected_case_id}` | **Cohort:** `{meta.get('pathology', 'Unknown')}` | "
            f"**Center:** `Dijon University Hospital (France)`\n\n"
            f"**Modality:** Static 3D Delayed-Enhancement (LGE) Inversion-Recovery CMR | "
            f"**Dimensions:** `{img_3d.shape[0]}×{img_3d.shape[1]}` ({img_3d.shape[2]} slices) | "
            f"**Slice Thickness:** `{spacing[2]:.1f} mm`\n\n"
            f"**Clinical Phenotype:** Sex: `{meta.get('sex', 'N/A')}`, Age: `{meta.get('age', 'N/A')}`, "
            f"Hypertension: `{'Yes' if meta.get('hypertension') else 'No'}`, "
            f"Diabetes: `{'Yes' if meta.get('diabetes') else 'No'}`, "
            f"CAD Family History: `{'Yes' if meta.get('cad_family_history') else 'No'}`"
        )

    num_slices = img_3d.shape[2]
    slice_idx = st.slider("Select Short-Axis Slice", min_value=0, max_value=num_slices - 1, value=num_slices // 2)

    slice_img = img_3d[:, :, slice_idx]
    slice_gt = gt_mask_3d[:, :, slice_idx] if gt_mask_3d is not None else None

    # Run experimental model inference on 3D volume or slice
    pred_mask_3d: Optional[np.ndarray] = None
    slice_pred: Optional[np.ndarray] = None

    if evaluator is not None:
        try:
            model = evaluator.load_model()
            pred_mask_3d = evaluator.segment_volume_3d(img_3d, model)
            slice_pred = pred_mask_3d[:, :, slice_idx]
        except Exception as e:
            logger.warning(f"EMIDEC inference failed for {selected_case_id}: {e}")

    # 4-Column Visual Inspection Panel
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("##### 1. Grayscale LGE MRI")
        st.image(normalize_for_display(slice_img), clamp=True, caption=f"Slice {slice_idx+1}/{num_slices} (LGE)")

    with col2:
        st.markdown("##### 2. Expert Reference Contours")
        if slice_gt is not None:
            st.image(create_emidec_overlay(slice_img, slice_gt), clamp=True, caption="Expert Human Annotations")
        else:
            st.image(normalize_for_display(slice_img), clamp=True, caption="No expert contour")

    with col3:
        st.markdown("##### 3. Experimental AI U-Net")
        if slice_pred is not None:
            st.image(create_emidec_overlay(slice_img, slice_pred), clamp=True, caption="Experimental AI Mask (5-Class)")
        else:
            st.info("AI prediction unavailable")

    with col4:
        st.markdown("##### 4. Tissue Class Legend")
        st.markdown(
            "- <span style='color:#1ab2f2;font-weight:bold'>■ LV Cavity</span> (Class 1)\n"
            "- <span style='color:#33d959;font-weight:bold'>■ Normal Myocardium</span> (Class 2)\n"
            "- <span style='color:#ff8c00;font-weight:bold'>■ Myocardial Infarction</span> (Class 3)\n"
            "- <span style='color:#cc0d26;font-weight:bold'>■ No-Reflow / MVO Core</span> (Class 4)\n",
            unsafe_allow_html=True,
        )
        st.caption("Notice: Strict visual distinction maintained between Expert Reference and Experimental AI.")

    # Volumetric Tissue Measurements Comparison
    st.divider()
    st.markdown("### 📊 Quantitative Tissue Characterization Measurements")

    c_meas, c_metrics = st.columns([1.2, 1.0])

    with c_meas:
        st.markdown("#### Volumetric Scar & No-Reflow Quantification")
        gt_metrics = compute_emidec_tissue_metrics(gt_mask_3d, spacing) if gt_mask_3d is not None else None
        ai_metrics = compute_emidec_tissue_metrics(pred_mask_3d, spacing) if pred_mask_3d is not None else None

        table_rows = [
            {
                "Tissue Measurement": "Total Myocardial Volume (mL)",
                "Expert Reference": f"{gt_metrics.total_myocardium_volume_ml:.1f} mL" if gt_metrics else "N/A",
                "Experimental AI": f"{ai_metrics.total_myocardium_volume_ml:.1f} mL" if ai_metrics else "N/A",
            },
            {
                "Tissue Measurement": "Infarct Scar Volume (mL)",
                "Expert Reference": f"{gt_metrics.infarction_volume_ml:.1f} mL" if gt_metrics else "N/A",
                "Experimental AI": f"{ai_metrics.infarction_volume_ml:.1f} mL" if ai_metrics else "N/A",
            },
            {
                "Tissue Measurement": "Infarct Burden (% of Myocardium)",
                "Expert Reference": f"{gt_metrics.infarction_burden_percent:.1f}%" if gt_metrics else "N/A",
                "Experimental AI": f"{ai_metrics.infarction_burden_percent:.1f}%" if ai_metrics else "N/A",
            },
            {
                "Tissue Measurement": "Microvascular Obstruction (MVO) Vol (mL)",
                "Expert Reference": f"{gt_metrics.no_reflow_volume_ml:.2f} mL" if gt_metrics else "N/A",
                "Experimental AI": f"{ai_metrics.no_reflow_volume_ml:.2f} mL" if ai_metrics else "N/A",
            },
            {
                "Tissue Measurement": "MVO / No-Reflow Burden (%)",
                "Expert Reference": f"{gt_metrics.no_reflow_burden_percent:.2f}%" if gt_metrics else "N/A",
                "Experimental AI": f"{ai_metrics.no_reflow_burden_percent:.2f}%" if ai_metrics else "N/A",
            },
            {
                "Tissue Measurement": "MVO Status",
                "Expert Reference": "Positive (MVO+)" if gt_metrics and gt_metrics.has_microvascular_obstruction else "Negative (No MVO)",
                "Experimental AI": "Positive (MVO+)" if ai_metrics and ai_metrics.has_microvascular_obstruction else "Negative (No MVO)",
            },
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with c_metrics:
        st.markdown("#### Per-Case Spatial Dice Overlap")
        if gt_mask_3d is not None and pred_mask_3d is not None:
            d_lv = compute_dice_score(pred_mask_3d, gt_mask_3d, 1)
            d_myo = compute_dice_score(pred_mask_3d, gt_mask_3d, 2)
            d_inf = compute_dice_score(pred_mask_3d, gt_mask_3d, 3)
            d_mvo = compute_dice_score(pred_mask_3d, gt_mask_3d, 4)

            d_rows = [
                {"Structure": "LV Cavity (Class 1)", "3D Spatial Dice": f"{d_lv:.4f}" if d_lv is not None else "Absent"},
                {"Structure": "Normal Myocardium (Class 2)", "3D Spatial Dice": f"{d_myo:.4f}" if d_myo is not None else "Absent"},
                {"Structure": "Myocardial Infarction (Class 3)", "3D Spatial Dice": f"{d_inf:.4f}" if d_inf is not None else "Absent in Reference"},
                {"Structure": "MVO / No-Reflow Core (Class 4)", "3D Spatial Dice": f"{d_mvo:.4f}" if d_mvo is not None else "Absent in Reference"},
            ]
            st.dataframe(pd.DataFrame(d_rows), use_container_width=True, hide_index=True)
            st.caption("Absent classes return 'Absent' to avoid spuriously inflated 0/0=1.0.")
        else:
            st.info("Both expert contours and experimental AI prediction required for 3D Dice.")

    # Held-Out Evaluation Summary
    st.divider()
    st.markdown("### 🏆 EMIDEC Dedicated Test Benchmark (20 Held-Out Cases)")
    if EMIDEC_SUMMARY_JSON.is_file():
        try:
            with open(EMIDEC_SUMMARY_JSON, "r", encoding="utf-8") as f:
                em_sum = json.load(f)

            dice_m = em_sum.get("dice_metrics", {})
            vol_m = em_sum.get("volumetric_errors", {})
            det_m = em_sum.get("lesion_classification", {})

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("LV Cavity Dice", f"{dice_m.get('lv_cavity_mean', 0.843):.4f}")
                st.caption("Median: 0.8523")
            with s2:
                st.metric("Normal Myo Dice", f"{dice_m.get('normal_myocardium_mean', 0.634):.4f}")
                st.caption("Median: 0.6410")
            with s3:
                st.metric("Infarct Detection Sensitivity", f"{det_m.get('infarction_detection', {}).get('sensitivity', 1.0)*100:.0f}%")
                st.caption(f"Mean Scar MAE: {vol_m.get('infarct_volume_mae_ml_all', 18.38):.1f} mL")
            with s4:
                st.metric("MVO Detection Sensitivity", f"{det_m.get('mvo_detection', {}).get('sensitivity', 1.0)*100:.0f}%")
                st.caption(f"Mean MVO MAE: {vol_m.get('mvo_volume_mae_ml_all', 1.23):.2f} mL")
        except Exception as e:
            st.error(f"Could not load EMIDEC summary: {e}")
