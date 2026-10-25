"""Cardiac MRI Segmentation and Ejection Fraction Evaluation Metrics.

Computes Dice Similarity Coefficients, 95th Percentile Hausdorff Distances (HD95),
and clinical volumetric errors (LVEDV, LVESV, LVSV, LVEF) between automated
U-Net predictions and withheld expert ground-truth annotations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_hausdorff_distance

from crip.mri.acdc_loader import ACDCDataLoader, DEFAULT_ACDC_TRAINING_DIR
from crip.mri.dataset_splits import get_or_create_splits
from crip.mri.segmentation_inference import PatientInferenceResult, SegmentationInferenceEngine
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH

logger = logging.getLogger("crip.evaluation.segmentation_metrics")

DEFAULT_EVAL_CSV_PATH = Path("data/processed/acdc/segmentation_evaluation.csv")
DEFAULT_EVAL_SUMMARY_PATH = Path("data/processed/acdc/segmentation_evaluation_summary.json")


def calculate_dice_coefficient(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    label: Optional[int] = None,
) -> float:
    """Calculate Dice Similarity Coefficient between two masks.

    Args:
        mask_pred: Predicted mask array (2D or 3D).
        mask_gt: Ground truth mask array (2D or 3D).
        label: Target label value. If None, treats non-zero voxels as foreground.

    Returns:
        Dice coefficient in range [0.0, 1.0].
    """
    if mask_pred.shape != mask_gt.shape:
        raise ValueError(f"Shape mismatch: pred {mask_pred.shape} vs gt {mask_gt.shape}")

    if label is not None:
        bin_pred = (mask_pred == label)
        bin_gt = (mask_gt == label)
    else:
        bin_pred = (mask_pred > 0)
        bin_gt = (mask_gt > 0)

    sum_pred = int(np.count_nonzero(bin_pred))
    sum_gt = int(np.count_nonzero(bin_gt))

    if sum_pred == 0 and sum_gt == 0:
        return 1.0  # Both empty: perfect agreement on absence
    if sum_pred == 0 or sum_gt == 0:
        return 0.0  # One empty: total mismatch

    intersection = int(np.count_nonzero(bin_pred & bin_gt))
    dice = (2.0 * intersection) / (sum_pred + sum_gt)
    return float(dice)


def calculate_multiclass_dice(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
) -> Dict[str, float]:
    """Calculate multi-class Dice for cardiac structures.

    Labels:
      1: Right Ventricular cavity (RV)
      2: Myocardium (MYO)
      3: Left Ventricular cavity (LV)
    """
    rv_dice = calculate_dice_coefficient(mask_pred, mask_gt, label=1)
    myo_dice = calculate_dice_coefficient(mask_pred, mask_gt, label=2)
    lv_dice = calculate_dice_coefficient(mask_pred, mask_gt, label=3)
    mean_dice = float((rv_dice + myo_dice + lv_dice) / 3.0)

    return {
        "rv": rv_dice,
        "myo": myo_dice,
        "lv": lv_dice,
        "mean": mean_dice,
    }


def calculate_hd95_mm(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    voxel_spacing: Sequence[float],
    label: Optional[int] = None,
) -> Optional[float]:
    """Calculate 95th Percentile Hausdorff Distance (HD95) in physical millimeters.

    Args:
        mask_pred: 2D or 3D predicted mask.
        mask_gt: 2D or 3D ground truth mask.
        voxel_spacing: Voxel dimensions in mm, e.g. (spacing_x, spacing_y, spacing_z) or (sx, sy).
        label: Target label value.

    Returns:
        HD95 in millimeters, or None if a structure is entirely absent in one mask.
    """
    if label is not None:
        bin_pred = (mask_pred == label).astype(np.uint8)
        bin_gt = (mask_gt == label).astype(np.uint8)
    else:
        bin_pred = (mask_pred > 0).astype(np.uint8)
        bin_gt = (mask_gt > 0).astype(np.uint8)

    sum_pred = int(np.count_nonzero(bin_pred))
    sum_gt = int(np.count_nonzero(bin_gt))

    if sum_pred == 0 and sum_gt == 0:
        return 0.0
    if sum_pred == 0 or sum_gt == 0:
        return None

    # Convert to PyTorch tensors shaped [1, 1, ...]
    pred_tensor = torch.from_numpy(bin_pred).unsqueeze(0).unsqueeze(0).float()
    gt_tensor = torch.from_numpy(bin_gt).unsqueeze(0).unsqueeze(0).float()

    # Spacing for MONAI: must match spatial dimension count
    num_spatial = bin_pred.ndim
    spacing = tuple(float(s) for s in voxel_spacing[:num_spatial])

    try:
        hd = compute_hausdorff_distance(
            pred_tensor,
            gt_tensor,
            percentile=95,
            spacing=spacing,
        )
        val = float(hd.item())
        return None if np.isnan(val) or np.isinf(val) else val
    except Exception as e:
        logger.debug(f"Hausdorff computation notice: {e}")
        return None


def calculate_multiclass_hd95(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    voxel_spacing: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Calculate physical HD95 for RV, MYO, and LV structures."""
    return {
        "rv": calculate_hd95_mm(mask_pred, mask_gt, voxel_spacing, label=1),
        "myo": calculate_hd95_mm(mask_pred, mask_gt, voxel_spacing, label=2),
        "lv": calculate_hd95_mm(mask_pred, mask_gt, voxel_spacing, label=3),
    }


def evaluate_patient_segmentation(
    result: PatientInferenceResult,
    group: str = "UNKNOWN",
) -> Dict[str, Any]:
    """Compute complete volumetric and geometric evaluation record for one patient."""
    eval_record: Dict[str, Any] = {
        "patient_id": result.patient_id,
        "group": group,
    }

    if result.gt_ed_mask is None or result.gt_es_mask is None or result.gt_metrics is None:
        eval_record["status"] = "NO_GROUND_TRUTH"
        return eval_record

    # 1. Spatial Overlap (Dice)
    ed_dice = calculate_multiclass_dice(result.pred_ed_mask, result.gt_ed_mask)
    es_dice = calculate_multiclass_dice(result.pred_es_mask, result.gt_es_mask)

    eval_record["dice_ed_lv"] = ed_dice["lv"]
    eval_record["dice_ed_rv"] = ed_dice["rv"]
    eval_record["dice_ed_myo"] = ed_dice["myo"]
    eval_record["dice_ed_mean"] = ed_dice["mean"]

    eval_record["dice_es_lv"] = es_dice["lv"]
    eval_record["dice_es_rv"] = es_dice["rv"]
    eval_record["dice_es_myo"] = es_dice["myo"]
    eval_record["dice_es_mean"] = es_dice["mean"]

    # Average across phases
    eval_record["dice_patient_lv"] = float((ed_dice["lv"] + es_dice["lv"]) / 2.0)
    eval_record["dice_patient_rv"] = float((ed_dice["rv"] + es_dice["rv"]) / 2.0)
    eval_record["dice_patient_myo"] = float((ed_dice["myo"] + es_dice["myo"]) / 2.0)
    eval_record["dice_patient_mean"] = float((ed_dice["mean"] + es_dice["mean"]) / 2.0)

    # 2. Boundary Distance (HD95 mm)
    ed_hd95 = calculate_multiclass_hd95(result.pred_ed_mask, result.gt_ed_mask, result.voxel_spacing)
    es_hd95 = calculate_multiclass_hd95(result.pred_es_mask, result.gt_es_mask, result.voxel_spacing)

    eval_record["hd95_ed_lv_mm"] = ed_hd95["lv"]
    eval_record["hd95_ed_rv_mm"] = ed_hd95["rv"]
    eval_record["hd95_ed_myo_mm"] = ed_hd95["myo"]
    eval_record["hd95_es_lv_mm"] = es_hd95["lv"]
    eval_record["hd95_es_rv_mm"] = es_hd95["rv"]
    eval_record["hd95_es_myo_mm"] = es_hd95["myo"]

    # 3. Clinical Left Ventricular Volumetrics
    pred_m = result.pred_metrics
    gt_m = result.gt_metrics

    eval_record["lvedv_pred_ml"] = pred_m.lvedv_ml
    eval_record["lvedv_gt_ml"] = gt_m.lvedv_ml
    eval_record["lvedv_error_ml"] = pred_m.lvedv_ml - gt_m.lvedv_ml
    eval_record["lvedv_abs_error_ml"] = abs(pred_m.lvedv_ml - gt_m.lvedv_ml)

    eval_record["lvesv_pred_ml"] = pred_m.lvesv_ml
    eval_record["lvesv_gt_ml"] = gt_m.lvesv_ml
    eval_record["lvesv_error_ml"] = pred_m.lvesv_ml - gt_m.lvesv_ml
    eval_record["lvesv_abs_error_ml"] = abs(pred_m.lvesv_ml - gt_m.lvesv_ml)

    eval_record["lvsv_pred_ml"] = pred_m.lvsv_ml
    eval_record["lvsv_gt_ml"] = gt_m.lvsv_ml
    eval_record["lvsv_error_ml"] = pred_m.lvsv_ml - gt_m.lvsv_ml
    eval_record["lvsv_abs_error_ml"] = abs(pred_m.lvsv_ml - gt_m.lvsv_ml)

    eval_record["lvef_pred_pct"] = pred_m.lvef_percent
    eval_record["lvef_gt_pct"] = gt_m.lvef_percent
    eval_record["lvef_error_pct"] = pred_m.lvef_percent - gt_m.lvef_percent
    eval_record["lvef_abs_error_pct"] = abs(pred_m.lvef_percent - gt_m.lvef_percent)
    eval_record["within_5pct_target"] = bool(eval_record["lvef_abs_error_pct"] <= 5.0)

    # 4. Right Ventricle Volumetrics
    eval_record["rvedv_pred_ml"] = pred_m.rvedv_ml
    eval_record["rvedv_gt_ml"] = gt_m.rvedv_ml
    eval_record["rvef_pred_pct"] = pred_m.rvef_percent
    eval_record["rvef_gt_pct"] = gt_m.rvef_percent

    eval_record["status"] = "SUCCESS"
    return eval_record


def run_cohort_segmentation_evaluation(
    data_dir: Path | str = DEFAULT_ACDC_TRAINING_DIR,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    split_name: str = "test",
    output_csv_path: Path | str = DEFAULT_EVAL_CSV_PATH,
    output_summary_path: Path | str = DEFAULT_EVAL_SUMMARY_PATH,
) -> Dict[str, Any]:
    """Run full evaluation across patients in the designated split (default: held-out test)."""
    manifest = get_or_create_splits(data_dir=data_dir)
    target_pids = manifest["splits"].get(split_name, [])
    if not target_pids:
        raise ValueError(f"No patients found for split '{split_name}' in manifest.")

    logger.info(f"Running evaluation on {len(target_pids)} patients from '{split_name}' split...")

    loader = ACDCDataLoader(data_dir)
    engine = SegmentationInferenceEngine(checkpoint_path=checkpoint_path)

    results: List[Dict[str, Any]] = []

    for idx, pid in enumerate(target_pids, 1):
        meta = loader.get_patient_metadata(pid)
        logger.info(f"[{idx}/{len(target_pids)}] Evaluating {pid} ({meta.group})...")
        pred_res = engine.predict_patient(loader, pid)
        rec = evaluate_patient_segmentation(pred_res, group=meta.group)
        results.append(rec)

    df = pd.DataFrame(results)

    # Export patient-level results CSV
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved patient-level evaluation results to {out_csv}")

    # Compute aggregate cohort statistics
    successful_df = df[df["status"] == "SUCCESS"]
    n_eval = len(successful_df)

    if n_eval == 0:
        logger.warning("No successful evaluations to summarize.")
        return {"cohort_size": len(df), "successful_evaluations": 0}

    lvef_mae = float(successful_df["lvef_abs_error_pct"].mean())
    lvef_median_ae = float(successful_df["lvef_abs_error_pct"].median())
    lvef_rmse = float(np.sqrt((successful_df["lvef_error_pct"] ** 2).mean()))
    pct_within_5pct = float((successful_df["within_5pct_target"].sum() / n_eval) * 100.0)

    # Dice aggregates
    mean_dice_lv = float(successful_df["dice_patient_lv"].mean())
    mean_dice_rv = float(successful_df["dice_patient_rv"].mean())
    mean_dice_myo = float(successful_df["dice_patient_myo"].mean())
    overall_mean_dice = float(successful_df["dice_patient_mean"].mean())

    # Subgroup breakdown
    group_stats: Dict[str, Dict[str, float]] = {}
    for grp, grp_df in successful_df.groupby("group"):
        group_stats[grp] = {
            "n": int(len(grp_df)),
            "lvef_mae": float(grp_df["lvef_abs_error_pct"].mean()),
            "lvef_rmse": float(np.sqrt((grp_df["lvef_error_pct"] ** 2).mean())),
            "pct_within_5pct": float((grp_df["within_5pct_target"].sum() / len(grp_df)) * 100.0),
            "mean_dice_lv": float(grp_df["dice_patient_lv"].mean()),
            "mean_dice_rv": float(grp_df["dice_patient_rv"].mean()),
            "mean_dice_myo": float(grp_df["dice_patient_myo"].mean()),
        }

    summary = {
        "split_evaluated": split_name,
        "n_patients": n_eval,
        "target_met_le_5pct": bool(lvef_mae <= 5.0),
        "lvef_metrics": {
            "mean_absolute_error_pct": round(lvef_mae, 2),
            "median_absolute_error_pct": round(lvef_median_ae, 2),
            "rmse_pct": round(lvef_rmse, 2),
            "pct_cases_le_5pct_error": round(pct_within_5pct, 1),
            "target_threshold_pct": 5.0,
        },
        "volumetric_errors": {
            "lvedv_mae_ml": round(float(successful_df["lvedv_abs_error_ml"].mean()), 2),
            "lvesv_mae_ml": round(float(successful_df["lvesv_abs_error_ml"].mean()), 2),
            "lvsv_mae_ml": round(float(successful_df["lvsv_abs_error_ml"].mean()), 2),
        },
        "geometric_metrics": {
            "dice_lv": round(mean_dice_lv, 3),
            "dice_rv": round(mean_dice_rv, 3),
            "dice_myo": round(mean_dice_myo, 3),
            "dice_mean_overall": round(overall_mean_dice, 3),
        },
        "diagnostic_subgroups": group_stats,
    }

    out_sum = Path(output_summary_path)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved cohort evaluation summary to {out_sum}")

    return summary
