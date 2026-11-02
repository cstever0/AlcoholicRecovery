"""External Generalizability Evaluation of ACDC-trained U-Net on the M&Ms Cohort.

Performs frozen-model zero-shot external evaluation on the multi-center, multi-vendor
M&Ms (2020) cine CMR dataset to measure cross-vendor and cross-center domain shift:
- 114 paired cases (80 Training/Labeled, 34 Validation)
- 4 scanner vendors (Siemens, Philips, GE, Canon)
- 5 clinical centers
- Multi-structure segmentation Dice (LV, RV, MYO, Mean)
- Boundary surface distance (HD95 mm)
- Left ventricular functional volumetrics (LVEDV, LVESV, LVSV, LVEF)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from crip.datasets.labels import ACDCLabel
from crip.datasets.mnms_loader import MNMSDataLoader
from crip.evaluation.segmentation_metrics import (
    calculate_multiclass_dice,
    calculate_multiclass_hd95,
)
from crip.mri.cardiac_metrics import (
    BiventricularCardiacMeasurements,
    compute_biventricular_measurements,
)
from crip.mri.segmentation_inference import SegmentationInferenceEngine
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH

logger = logging.getLogger("crip.evaluation.external_mnms")

DEFAULT_MNMS_EVAL_CSV = Path("data/processed/mnms/external_evaluation.csv")
DEFAULT_MNMS_EVAL_SUMMARY = Path("data/processed/mnms/external_evaluation_summary.json")

# Verified ACDC Reference Benchmarks (20 Held-out Test Patients)
ACDC_REFERENCE_BENCHMARKS = {
    "cohort_size": 20,
    "mean_foreground_dice": 0.839,
    "lv_dice": 0.906,
    "rv_dice": 0.807,
    "myo_dice": 0.804,
    "lvef_mae": 3.86,
    "lvef_within_5pct_rate": 0.75,
}


def compute_file_sha256(filepath: Path | str) -> str:
    """Calculate SHA-256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_single_mnms_case(
    loader: MNMSDataLoader,
    engine: SegmentationInferenceEngine,
    case_info: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Run frozen-model segmentation and clinical evaluation for one M&Ms case."""
    cid = case_info["case_id"]
    split = case_info.get("split_category", "Unknown")
    vendor = case_info.get("vendor", "Unknown")
    csv_meta = case_info.get("csv_metadata", {})
    centre = csv_meta.get("Centre", "Unknown")
    diagnosis = case_info.get("diagnosis", "Unknown")

    ed_idx = case_info.get("ed_frame")
    es_idx = case_info.get("es_frame")

    if ed_idx is None or es_idx is None:
        raise ValueError(f"Case {cid} is missing required ED or ES frame annotations.")

    # Load 4D cine and mask with canonical label remapping (1=LV, 3=RV -> 3=LV, 1=RV)
    cine_res = loader.load_case_cine(cid, remap_to_canonical=True)
    img_4d = cine_res["image"]
    mask_4d = cine_res["mask"]
    zooms = tuple(float(z) for z in cine_res["voxel_spacing"][:3])

    if mask_4d is None:
        raise ValueError(f"Case {cid} does not contain ground truth segmentation mask.")

    n_frames = img_4d.shape[3]
    if ed_idx >= n_frames or es_idx >= n_frames:
        raise ValueError(f"Case {cid} frame index out of bounds: ED={ed_idx}, ES={es_idx}, N={n_frames}")

    # Transpose from (X, Y, Z) to canonical short-axis slice stack (num_slices, H, W)
    ed_img = np.transpose(img_4d[:, :, :, ed_idx], (2, 1, 0))
    es_img = np.transpose(img_4d[:, :, :, es_idx], (2, 1, 0))
    ed_gt = np.transpose(mask_4d[:, :, :, ed_idx], (2, 1, 0))
    es_gt = np.transpose(mask_4d[:, :, :, es_idx], (2, 1, 0))

    # Perform slice-by-slice inference using frozen U-Net
    pred_ed = engine.predict_cardiac_frame_volume(ed_img, model=model)
    pred_es = engine.predict_cardiac_frame_volume(es_img, model=model)

    # 1. Spatial Overlap (Dice)
    ed_dice = calculate_multiclass_dice(pred_ed, ed_gt)
    es_dice = calculate_multiclass_dice(pred_es, es_gt)

    dice_patient_lv = float((ed_dice["lv"] + es_dice["lv"]) / 2.0)
    dice_patient_rv = float((ed_dice["rv"] + es_dice["rv"]) / 2.0)
    dice_patient_myo = float((ed_dice["myo"] + es_dice["myo"]) / 2.0)
    dice_patient_mean = float((ed_dice["mean"] + es_dice["mean"]) / 2.0)

    # 2. Boundary Surface Distance (HD95 mm)
    ed_hd95 = calculate_multiclass_hd95(pred_ed, ed_gt, zooms)
    es_hd95 = calculate_multiclass_hd95(pred_es, es_gt, zooms)

    # Compute patient-level HD95 (ignoring None if structure absent)
    def avg_optional(v1: Optional[float], v2: Optional[float]) -> Optional[float]:
        vals = [v for v in (v1, v2) if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else None

    hd95_patient_lv = avg_optional(ed_hd95["lv"], es_hd95["lv"])
    hd95_patient_rv = avg_optional(ed_hd95["rv"], es_hd95["rv"])
    hd95_patient_myo = avg_optional(ed_hd95["myo"], es_hd95["myo"])

    # 3. Clinical Biventricular Volumetrics
    pred_m = compute_biventricular_measurements(pred_ed, pred_es, zooms, patient_id=cid)
    gt_m = compute_biventricular_measurements(ed_gt, es_gt, zooms, patient_id=cid)

    lvef_error = float(pred_m.lvef_percent - gt_m.lvef_percent)
    lvef_abs_error = abs(lvef_error)

    record: Dict[str, Any] = {
        "case_id": cid,
        "dataset": "mnms",
        "split_category": split,
        "vendor": vendor,
        "center": f"Center_{centre}" if centre != "Unknown" else "Unknown",
        "diagnosis": diagnosis,
        "num_slices": int(ed_img.shape[0]),
        "image_height": int(ed_img.shape[1]),
        "image_width": int(ed_img.shape[2]),
        "voxel_spacing_x_mm": zooms[0],
        "voxel_spacing_y_mm": zooms[1],
        "voxel_spacing_z_mm": zooms[2],
        "ed_frame": int(ed_idx),
        "es_frame": int(es_idx),
        # Dice Metrics
        "dice_ed_lv": ed_dice["lv"],
        "dice_ed_rv": ed_dice["rv"],
        "dice_ed_myo": ed_dice["myo"],
        "dice_ed_mean": ed_dice["mean"],
        "dice_es_lv": es_dice["lv"],
        "dice_es_rv": es_dice["rv"],
        "dice_es_myo": es_dice["myo"],
        "dice_es_mean": es_dice["mean"],
        "dice_patient_lv": dice_patient_lv,
        "dice_patient_rv": dice_patient_rv,
        "dice_patient_myo": dice_patient_myo,
        "dice_patient_mean": dice_patient_mean,
        # HD95 Metrics
        "hd95_ed_lv_mm": ed_hd95["lv"],
        "hd95_ed_rv_mm": ed_hd95["rv"],
        "hd95_ed_myo_mm": ed_hd95["myo"],
        "hd95_es_lv_mm": es_hd95["lv"],
        "hd95_es_rv_mm": es_hd95["rv"],
        "hd95_es_myo_mm": es_hd95["myo"],
        "hd95_patient_lv_mm": hd95_patient_lv,
        "hd95_patient_rv_mm": hd95_patient_rv,
        "hd95_patient_myo_mm": hd95_patient_myo,
        # Volumetrics
        "lvedv_pred_ml": float(pred_m.lvedv_ml),
        "lvedv_gt_ml": float(gt_m.lvedv_ml),
        "lvedv_error_ml": float(pred_m.lvedv_ml - gt_m.lvedv_ml),
        "lvedv_abs_error_ml": float(abs(pred_m.lvedv_ml - gt_m.lvedv_ml)),
        "lvesv_pred_ml": float(pred_m.lvesv_ml),
        "lvesv_gt_ml": float(gt_m.lvesv_ml),
        "lvesv_error_ml": float(pred_m.lvesv_ml - gt_m.lvesv_ml),
        "lvesv_abs_error_ml": float(abs(pred_m.lvesv_ml - gt_m.lvesv_ml)),
        "lvsv_pred_ml": float(pred_m.lvsv_ml),
        "lvsv_gt_ml": float(gt_m.lvsv_ml),
        "lvsv_error_ml": float(pred_m.lvsv_ml - gt_m.lvsv_ml),
        "lvsv_abs_error_ml": float(abs(pred_m.lvsv_ml - gt_m.lvsv_ml)),
        "lvef_pred_percent": float(pred_m.lvef_percent),
        "lvef_gt_percent": float(gt_m.lvef_percent),
        "lvef_error_percent": lvef_error,
        "lvef_abs_error_percent": lvef_abs_error,
        "lvef_within_5pct": bool(lvef_abs_error <= 5.0),
        "rvedv_pred_ml": float(pred_m.rvedv_ml),
        "rvedv_gt_ml": float(gt_m.rvedv_ml),
        "rvesv_pred_ml": float(pred_m.rvesv_ml),
        "rvesv_gt_ml": float(gt_m.rvesv_ml),
        "rvsv_pred_ml": float(pred_m.rvsv_ml),
        "rvsv_gt_ml": float(gt_m.rvsv_ml),
        "rvef_pred_percent": float(pred_m.rvef_percent),
        "rvef_gt_percent": float(gt_m.rvef_percent),
        "myo_mass_ed_pred_g": float(pred_m.myocardial_mass_ed_g),
        "myo_mass_ed_gt_g": float(gt_m.myocardial_mass_ed_g),
    }

    return record


class MNMSExternalEvaluator:
    """Orchestrates comprehensive external evaluation over the M&Ms cohort."""

    def __init__(
        self,
        loader: Optional[MNMSDataLoader] = None,
        engine: Optional[SegmentationInferenceEngine] = None,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    ) -> None:
        self.loader = loader or MNMSDataLoader()
        self.checkpoint_path = Path(checkpoint_path)
        self.engine = engine or SegmentationInferenceEngine(checkpoint_path=self.checkpoint_path)

    def run_evaluation(
        self,
        case_ids: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Evaluate all eligible paired cases (or a specified subset)."""
        cases = self.loader.discover_cases()
        eligible = {
            cid: c for cid, c in cases.items()
            if c["has_image"] and c["has_mask"] and c["ed_frame"] is not None and c["es_frame"] is not None
        }

        if case_ids is not None:
            eligible = {cid: c for cid, c in eligible.items() if cid in set(case_ids)}

        logger.info(f"Starting external evaluation on {len(eligible)} eligible M&Ms cases...")
        model = self.engine.load_model()
        records: List[Dict[str, Any]] = []

        for idx, (cid, c_info) in enumerate(sorted(eligible.items()), 1):
            try:
                rec = evaluate_single_mnms_case(self.loader, self.engine, c_info, model=model)
                records.append(rec)
                if idx % 20 == 0 or idx == len(eligible):
                    logger.info(f"Evaluated {idx}/{len(eligible)} M&Ms cases.")
            except Exception as e:
                logger.error(f"Failed evaluating case {cid}: {e}", exc_info=True)

        df = pd.DataFrame(records)
        return df

    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate per-case evaluation dataframe into structured metrics report."""
        if df.empty:
            return {"error": "Evaluation DataFrame is empty."}

        checkpoint_sha = compute_file_sha256(self.checkpoint_path) if self.checkpoint_path.is_file() else "UNKNOWN"

        def col_stats(series: pd.Series) -> Dict[str, float]:
            s_clean = series.dropna()
            if s_clean.empty:
                return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
            return {
                "mean": float(s_clean.mean()),
                "median": float(s_clean.median()),
                "std": float(s_clean.std()),
                "p25": float(s_clean.quantile(0.25)),
                "p75": float(s_clean.quantile(0.75)),
            }

        total_cases = len(df)
        lvef_within_5pct_rate = float(df["lvef_within_5pct"].mean()) if "lvef_within_5pct" in df else 0.0

        # Sub-breakdowns
        by_vendor: Dict[str, Any] = {}
        for vendor, group in df.groupby("vendor"):
            by_vendor[vendor] = {
                "count": len(group),
                "dice_mean": float(group["dice_patient_mean"].mean()),
                "dice_lv": float(group["dice_patient_lv"].mean()),
                "dice_rv": float(group["dice_patient_rv"].mean()),
                "dice_myo": float(group["dice_patient_myo"].mean()),
                "lvef_mae": float(group["lvef_abs_error_percent"].mean()),
                "lvef_within_5pct_rate": float(group["lvef_within_5pct"].mean()),
            }

        by_split: Dict[str, Any] = {}
        for split, group in df.groupby("split_category"):
            by_split[split] = {
                "count": len(group),
                "dice_mean": float(group["dice_patient_mean"].mean()),
                "dice_lv": float(group["dice_patient_lv"].mean()),
                "dice_rv": float(group["dice_patient_rv"].mean()),
                "dice_myo": float(group["dice_patient_myo"].mean()),
                "lvef_mae": float(group["lvef_abs_error_percent"].mean()),
                "lvef_within_5pct_rate": float(group["lvef_within_5pct"].mean()),
            }

        by_center: Dict[str, Any] = {}
        for center, group in df.groupby("center"):
            by_center[center] = {
                "count": len(group),
                "dice_mean": float(group["dice_patient_mean"].mean()),
                "dice_lv": float(group["dice_patient_lv"].mean()),
                "dice_rv": float(group["dice_patient_rv"].mean()),
                "dice_myo": float(group["dice_patient_myo"].mean()),
                "lvef_mae": float(group["lvef_abs_error_percent"].mean()),
            }

        summary = {
            "evaluation_date": "2026-11-02",
            "model_identity": "ACDC 2D U-Net (Frozen)",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "target_dataset": "Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation (M&Ms 2020)",
            "total_evaluated_cases": total_cases,
            "acdc_reference_benchmarks": ACDC_REFERENCE_BENCHMARKS,
            "mnms_aggregate_metrics": {
                "dice_patient_mean": col_stats(df["dice_patient_mean"]),
                "dice_patient_lv": col_stats(df["dice_patient_lv"]),
                "dice_patient_rv": col_stats(df["dice_patient_rv"]),
                "dice_patient_myo": col_stats(df["dice_patient_myo"]),
                "hd95_patient_lv_mm": col_stats(df["hd95_patient_lv_mm"]),
                "hd95_patient_rv_mm": col_stats(df["hd95_patient_rv_mm"]),
                "hd95_patient_myo_mm": col_stats(df["hd95_patient_myo_mm"]),
                "lvef_error_percent": col_stats(df["lvef_error_percent"]),
                "lvef_abs_error_percent": col_stats(df["lvef_abs_error_percent"]),
                "lvef_within_5pct_rate": lvef_within_5pct_rate,
                "lvedv_abs_error_ml": col_stats(df["lvedv_abs_error_ml"]),
                "lvesv_abs_error_ml": col_stats(df["lvesv_abs_error_ml"]),
                "lvsv_abs_error_ml": col_stats(df["lvsv_abs_error_ml"]),
            },
            "domain_comparison": {
                "dice_mean_difference": float(df["dice_patient_mean"].mean() - ACDC_REFERENCE_BENCHMARKS["mean_foreground_dice"]),
                "lv_dice_difference": float(df["dice_patient_lv"].mean() - ACDC_REFERENCE_BENCHMARKS["lv_dice"]),
                "rv_dice_difference": float(df["dice_patient_rv"].mean() - ACDC_REFERENCE_BENCHMARKS["rv_dice"]),
                "myo_dice_difference": float(df["dice_patient_myo"].mean() - ACDC_REFERENCE_BENCHMARKS["myo_dice"]),
                "lvef_mae_difference": float(df["lvef_abs_error_percent"].mean() - ACDC_REFERENCE_BENCHMARKS["lvef_mae"]),
            },
            "by_vendor": by_vendor,
            "by_split": by_split,
            "by_center": by_center,
        }

        return summary

    def run_and_save(
        self,
        output_csv: Path | str = DEFAULT_MNMS_EVAL_CSV,
        output_summary: Path | str = DEFAULT_MNMS_EVAL_SUMMARY,
        case_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run evaluation, export CSV and summary JSON."""
        csv_path = Path(output_csv)
        summary_path = Path(output_summary)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.run_evaluation(case_ids=case_ids)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved M&Ms evaluation CSV to {csv_path}")

        summary = self.generate_summary(df)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved M&Ms evaluation summary JSON to {summary_path}")

        return df, summary


def main() -> None:
    """CLI entry point for M&Ms external evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate frozen ACDC U-Net on M&Ms dataset.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to U-Net checkpoint")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_MNMS_EVAL_CSV), help="Output CSV path")
    parser.add_argument("--output-summary", type=str, default=str(DEFAULT_MNMS_EVAL_SUMMARY), help="Output JSON summary path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases (for testing)")
    args = parser.parse_args()

    evaluator = MNMSExternalEvaluator(checkpoint_path=args.checkpoint)
    cases = evaluator.loader.discover_cases()
    paired_cids = sorted([cid for cid, c in cases.items() if c["has_image"] and c["has_mask"]])
    if args.limit:
        paired_cids = paired_cids[:args.limit]

    df, summary = evaluator.run_and_save(
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        case_ids=paired_cids,
    )

    print("\n================== M&Ms External Evaluation Complete ==================")
    print(f"Total Cases Evaluated: {len(df)}")
    print(f"Mean Foreground Dice: {df['dice_patient_mean'].mean():.4f} (ACDC Test: {ACDC_REFERENCE_BENCHMARKS['mean_foreground_dice']:.4f})")
    print(f"LV Cavity Dice:       {df['dice_patient_lv'].mean():.4f} (ACDC Test: {ACDC_REFERENCE_BENCHMARKS['lv_dice']:.4f})")
    print(f"RV Cavity Dice:       {df['dice_patient_rv'].mean():.4f} (ACDC Test: {ACDC_REFERENCE_BENCHMARKS['rv_dice']:.4f})")
    print(f"Myocardium Dice:      {df['dice_patient_myo'].mean():.4f} (ACDC Test: {ACDC_REFERENCE_BENCHMARKS['myo_dice']:.4f})")
    print(f"LVEF MAE:             {df['lvef_abs_error_percent'].mean():.2f}% (ACDC Test: {ACDC_REFERENCE_BENCHMARKS['lvef_mae']:.2f}%)")
    print(f"LVEF <= 5% Error:     {df['lvef_within_5pct'].mean() * 100.0:.1f}%\n")


if __name__ == "__main__":
    main()
