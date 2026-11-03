"""EMIDEC Held-Out Test Evaluation and Scientific Visualization Engine.

Evaluates the trained 5-class EMIDEC LGE U-Net on the 20 designated held-out test patients:
  - Multi-structure Dice: LV cavity, normal myocardium, infarction, MVO
  - 3D Hausdorff Distance 95th percentile (HD95) in physical millimeters
  - Volumetric quantification: Infarct volume (mL), MVO volume (mL), tissue burdens (%)
  - Explicit absent foreground class handling (distinguishing zero-division from false positive)
  - Exports CSV, JSON summary, and publication-ready multi-format figures (PNG, SVG, PDF).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.metrics import compute_hausdorff_distance

from crip.datasets.emidec_loader import DEFAULT_EMIDEC_ROOT, EMIDECDataLoader
from crip.datasets.labels import EMIDECLabel
from crip.emidec.tissue_metrics import compute_emidec_tissue_metrics
from crip.mri.preprocessing import normalize_intensity, resize_slice, restore_mask_to_original_shape
from crip.mri.segmentation_model import build_unet_model, get_optimal_device
from crip.pipeline.splits import DEFAULT_UNIFIED_SPLITS_PATH

logger = logging.getLogger("crip.pipeline.emidec_evaluation")

DEFAULT_EVAL_DIR = Path("data/processed/multidataset")
DEFAULT_FIGURES_DIR = Path("docs/evaluation/multidataset/emidec_training")
DEFAULT_CHECKPOINT_PATH = Path("models/multidataset/emidec_lge_unet.pt")
DEFAULT_HISTORY_PATH = Path("models/multidataset/emidec_lge_unet_history.json")


def compute_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, class_idx: int) -> Optional[float]:
    """Calculate binary Dice for a specific class label.

    Returns None if class is absent in ground truth (preventing spuriously inflated 0/0=1.0).
    """
    pred_c = (pred_mask == class_idx)
    gt_c = (gt_mask == class_idx)
    gt_sum = int(gt_c.sum())
    pred_sum = int(pred_c.sum())

    if gt_sum == 0:
        return None  # Class absent in reference ground truth

    intersection = int(np.logical_and(pred_c, gt_c).sum())
    return float((2.0 * intersection) / (pred_sum + gt_sum))


def compute_class_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_idx: int,
    voxel_spacing: Sequence[float],
) -> Optional[float]:
    """Calculate 95th percentile Hausdorff Distance in mm for a specific class.

    Returns None if class is absent in prediction or ground truth.
    """
    pred_c = (pred_mask == class_idx).astype(np.float32)
    gt_c = (gt_mask == class_idx).astype(np.float32)

    if pred_c.sum() == 0 or gt_c.sum() == 0:
        return None

    # Convert to 5D tensors [B=1, C=1, H, W, D]
    t_pred = torch.from_numpy(pred_c).unsqueeze(0).unsqueeze(0)
    t_gt = torch.from_numpy(gt_c).unsqueeze(0).unsqueeze(0)

    try:
        hd = compute_hausdorff_distance(
            t_pred,
            t_gt,
            percentile=95,
            spacing=tuple(float(s) for s in voxel_spacing),
        )
        val = float(hd.item())
        return val if not np.isnan(val) else None
    except Exception:
        return None


def compute_precision_recall(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_idx: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate precision and recall for a specific class label.

    Returns (precision, recall).
    If class is absent in both, returns (1.0, None).
    If class is absent in reference ground truth, recall is None.
    """
    pred_c = (pred_mask == class_idx)
    gt_c = (gt_mask == class_idx)
    tp = int(np.logical_and(pred_c, gt_c).sum())
    fp = int(np.logical_and(pred_c, ~gt_c).sum())
    fn = int(np.logical_and(~pred_c, gt_c).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if (tp + fn) == 0 else 0.0)
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    return prec, rec


class EMIDECEvaluator:
    """Evaluates 5-class EMIDEC segmentation checkpoint on the held-out test cohort."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        splits_path: Path | str = DEFAULT_UNIFIED_SPLITS_PATH,
        loader: Optional[EMIDECDataLoader] = None,
        device: Optional[torch.device] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.splits_path = Path(splits_path)
        self.loader = loader or EMIDECDataLoader()
        self.device = device or get_optimal_device()
        self.target_size = target_size
        self._model: Optional[torch.nn.Module] = None

    def load_model(self) -> torch.nn.Module:
        """Load trained 5-class U-Net weights."""
        if self._model is not None:
            return self._model

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"EMIDEC checkpoint not found at {self.checkpoint_path}. "
                "Train the model first using: python -m crip.pipeline train --task lge"
            )

        model = build_unet_model(
            in_channels=1,
            out_channels=5,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.15,
        ).to(self.device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self._model = model
        logger.info(f"Loaded EMIDEC U-Net from {self.checkpoint_path} (epoch {ckpt.get('epoch', 'N/A')})")
        return self._model

    def load_held_out_test_cohort(self) -> List[str]:
        """Retrieve the 20 designated held-out test cases from unified splits manifest."""
        with open(self.splits_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest["datasets"]["emidec"]["splits"]["test"]

    @torch.no_grad()
    def segment_volume_3d(self, img_3d: np.ndarray, model: torch.nn.Module) -> np.ndarray:
        """Run slice-by-slice 2D segmentation with nearest-neighbor spatial restoration."""
        orig_h, orig_w, num_slices = img_3d.shape
        pred_3d = np.zeros((orig_h, orig_w, num_slices), dtype=np.int64)

        for s in range(num_slices):
            sl_img = img_3d[:, :, s].astype(np.float32)

            norm_img = normalize_intensity(sl_img)
            std = float(norm_img.std())
            norm_img = (norm_img - float(norm_img.mean())) / (std + 1e-6)

            resized_img = resize_slice(norm_img, target_size=self.target_size, is_mask=False)
            t_img = torch.from_numpy(resized_img).float().unsqueeze(0).unsqueeze(0).to(self.device)

            logits = model(t_img)
            pred_slice_res = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            # Restore to native short-axis slice dimensions
            pred_slice_orig = restore_mask_to_original_shape(pred_slice_res, original_shape=(orig_h, orig_w))
            pred_3d[:, :, s] = pred_slice_orig

        return pred_3d

    def evaluate_case(self, case_id: str, model: torch.nn.Module) -> Dict[str, Any]:
        """Evaluate single EMIDEC case across spatial overlap, HD95, and tissue volumetrics."""
        case_dict = self.loader.load_case(case_id)
        img_3d = case_dict["image"]
        gt_mask_3d = case_dict["contour"]
        zooms = case_dict["voxel_spacing"]
        pathology = case_dict["pathology"]
        clinical = case_dict.get("clinical_metadata", {})

        pred_mask_3d = self.segment_volume_3d(img_3d, model)

        # 1. Spatial Overlap (Dice) per structure
        dice_lv = compute_dice_score(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.LV_CAVITY))
        dice_myo = compute_dice_score(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.NORMAL_MYOCARDIUM))
        dice_infarct = compute_dice_score(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.MYOCARDIAL_INFARCTION))
        dice_mvo = compute_dice_score(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.NO_REFLOW))

        # 2. HD95 per structure
        hd95_lv = compute_class_hd95(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.LV_CAVITY), zooms)
        hd95_myo = compute_class_hd95(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.NORMAL_MYOCARDIUM), zooms)
        hd95_infarct = compute_class_hd95(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.MYOCARDIAL_INFARCTION), zooms)
        hd95_mvo = compute_class_hd95(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.NO_REFLOW), zooms)

        # 3. Precision & Recall per structure
        prec_lv, rec_lv = compute_precision_recall(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.LV_CAVITY))
        prec_myo, rec_myo = compute_precision_recall(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.NORMAL_MYOCARDIUM))
        prec_infarct, rec_infarct = compute_precision_recall(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.MYOCARDIAL_INFARCTION))
        prec_mvo, rec_mvo = compute_precision_recall(pred_mask_3d, gt_mask_3d, int(EMIDECLabel.NO_REFLOW))

        # 4. Clinical tissue volumetrics & burdens
        pred_metrics = compute_emidec_tissue_metrics(pred_mask_3d, zooms, case_id=case_id, pathology_group=pathology)
        gt_metrics = compute_emidec_tissue_metrics(gt_mask_3d, zooms, case_id=case_id, pathology_group=pathology)

        infarct_vol_error_ml = pred_metrics.infarction_volume_ml - gt_metrics.infarction_volume_ml
        mvo_vol_error_ml = pred_metrics.no_reflow_volume_ml - gt_metrics.no_reflow_volume_ml
        infarct_burden_error_pct = pred_metrics.infarction_burden_percent - gt_metrics.infarction_burden_percent
        mvo_burden_error_pct = pred_metrics.no_reflow_burden_percent - gt_metrics.no_reflow_burden_percent

        return {
            "case_id": case_id,
            "pathology_group": pathology,
            "has_gt_infarct": bool(gt_metrics.infarction_volume_ml > 0.0),
            "has_gt_mvo": bool(gt_metrics.has_microvascular_obstruction),
            "has_pred_infarct": bool(pred_metrics.infarction_volume_ml > 0.0),
            "has_pred_mvo": bool(pred_metrics.has_microvascular_obstruction),
            # Spacing
            "voxel_spacing_x_mm": zooms[0],
            "voxel_spacing_y_mm": zooms[1],
            "voxel_spacing_z_mm": zooms[2],
            "num_slices": img_3d.shape[2],
            # Dice
            "dice_lv": dice_lv,
            "dice_myo": dice_myo,
            "dice_infarct": dice_infarct,
            "dice_mvo": dice_mvo,
            # Precision & Recall
            "prec_lv": prec_lv,
            "rec_lv": rec_lv,
            "prec_myo": prec_myo,
            "rec_myo": rec_myo,
            "prec_infarct": prec_infarct,
            "rec_infarct": rec_infarct,
            "prec_mvo": prec_mvo,
            "rec_mvo": rec_mvo,
            # HD95
            "hd95_lv_mm": hd95_lv,
            "hd95_myo_mm": hd95_myo,
            "hd95_infarct_mm": hd95_infarct,
            "hd95_mvo_mm": hd95_mvo,
            # Volumes (mL)
            "lv_vol_pred_ml": pred_metrics.lv_cavity_volume_ml,
            "lv_vol_gt_ml": gt_metrics.lv_cavity_volume_ml,
            "myo_vol_pred_ml": pred_metrics.normal_myocardium_volume_ml,
            "myo_vol_gt_ml": gt_metrics.normal_myocardium_volume_ml,
            "infarct_vol_pred_ml": pred_metrics.infarction_volume_ml,
            "infarct_vol_gt_ml": gt_metrics.infarction_volume_ml,
            "infarct_vol_error_ml": infarct_vol_error_ml,
            "infarct_vol_abs_error_ml": abs(infarct_vol_error_ml),
            "mvo_vol_pred_ml": pred_metrics.no_reflow_volume_ml,
            "mvo_vol_gt_ml": gt_metrics.no_reflow_volume_ml,
            "mvo_vol_error_ml": mvo_vol_error_ml,
            "mvo_vol_abs_error_ml": abs(mvo_vol_error_ml),
            # Burdens (%)
            "infarct_burden_pred_pct": pred_metrics.infarction_burden_percent,
            "infarct_burden_gt_pct": gt_metrics.infarction_burden_percent,
            "infarct_burden_error_pct": infarct_burden_error_pct,
            "infarct_burden_abs_error_pct": abs(infarct_burden_error_pct),
            "mvo_burden_pred_pct": pred_metrics.no_reflow_burden_percent,
            "mvo_burden_gt_pct": gt_metrics.no_reflow_burden_percent,
            "mvo_burden_error_pct": mvo_burden_error_pct,
            "mvo_burden_abs_error_pct": abs(mvo_burden_error_pct),
            # Clinical
            "sex": clinical.get("sex"),
            "age": clinical.get("age"),
            "troponin": clinical.get("troponin"),
        }

    def run_evaluation(
        self,
        output_dir: Path | str = DEFAULT_EVAL_DIR,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run evaluation across all 20 held-out test patients and export artifacts."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = self.load_model()
        test_ids = self.load_held_out_test_cohort()
        logger.info(f"Evaluating {len(test_ids)} EMIDEC held-out test cases...")

        records: List[Dict[str, Any]] = []
        for cid in test_ids:
            try:
                rec = self.evaluate_case(cid, model)
                records.append(rec)
            except Exception as e:
                logger.error(f"Error evaluating EMIDEC case {cid}: {e}")

        df = pd.DataFrame(records)
        csv_path = out_dir / "emidec_evaluation.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved evaluation records ({len(df)} cases) to {csv_path}")

        summary = self._compute_summary(df)
        summary_path = out_dir / "emidec_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved evaluation summary to {summary_path}")

        return df, summary

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate statistical metrics handling absent classes explicitly."""
        def safe_mean(s: pd.Series) -> Optional[float]:
            clean = s.dropna()
            return float(clean.mean()) if len(clean) > 0 else None

        def safe_median(s: pd.Series) -> Optional[float]:
            clean = s.dropna()
            return float(clean.median()) if len(clean) > 0 else None

        # Infarction present cases
        df_inf = df[df["has_gt_infarct"] == True]
        # MVO present cases
        df_mvo = df[df["has_gt_mvo"] == True]
        # Normal controls
        df_norm = df[df["pathology_group"] == "Normal"]
        # Pathological cases
        df_path = df[df["pathology_group"] != "Normal"]

        # Classification performance (raw zero threshold > 0.0 mL)
        # Normal vs Pathological (infarct detection)
        true_path = (df["pathology_group"] != "Normal")
        pred_path = (df["infarct_vol_pred_ml"] > 0.0)
        tp_inf = int(((pred_path == True) & (true_path == True)).sum())
        fp_inf = int(((pred_path == True) & (true_path == False)).sum())
        tn_inf = int(((pred_path == False) & (true_path == False)).sum())
        fn_inf = int(((pred_path == False) & (true_path == True)).sum())
        sens_inf = float(tp_inf / (tp_inf + fn_inf)) if (tp_inf + fn_inf) > 0 else 0.0
        spec_inf = float(tn_inf / (tn_inf + fp_inf)) if (tn_inf + fp_inf) > 0 else 0.0
        prec_inf = float(tp_inf / (tp_inf + fp_inf)) if (tp_inf + fp_inf) > 0 else 0.0

        # MVO detection (raw zero threshold > 0.0 mL)
        true_mvo = (df["has_gt_mvo"] == True)
        pred_mvo = (df["has_pred_mvo"] == True)
        tp_mvo = int(((pred_mvo == True) & (true_mvo == True)).sum())
        fp_mvo = int(((pred_mvo == True) & (true_mvo == False)).sum())
        tn_mvo = int(((pred_mvo == False) & (true_mvo == False)).sum())
        fn_mvo = int(((pred_mvo == False) & (true_mvo == True)).sum())
        sens_mvo = float(tp_mvo / (tp_mvo + fn_mvo)) if (tp_mvo + fn_mvo) > 0 else 0.0
        spec_mvo = float(tn_mvo / (tn_mvo + fp_mvo)) if (tn_mvo + fp_mvo) > 0 else 0.0
        prec_mvo = float(tp_mvo / (tp_mvo + fp_mvo)) if (tp_mvo + fp_mvo) > 0 else 0.0

        # Calibrated lesion classification (thresholds derived strictly on N=20 validation set)
        cal_inf_th = 18.50
        cal_mvo_th = 0.30

        pred_path_cal = (df["infarct_vol_pred_ml"] >= cal_inf_th)
        tp_inf_c = int(((pred_path_cal == True) & (true_path == True)).sum())
        fp_inf_c = int(((pred_path_cal == True) & (true_path == False)).sum())
        tn_inf_c = int(((pred_path_cal == False) & (true_path == False)).sum())
        fn_inf_c = int(((pred_path_cal == False) & (true_path == True)).sum())
        sens_inf_c = float(tp_inf_c / (tp_inf_c + fn_inf_c)) if (tp_inf_c + fn_inf_c) > 0 else 0.0
        spec_inf_c = float(tn_inf_c / (tn_inf_c + fp_inf_c)) if (tn_inf_c + fp_inf_c) > 0 else 0.0
        prec_inf_c = float(tp_inf_c / (tp_inf_c + fp_inf_c)) if (tp_inf_c + fp_inf_c) > 0 else 0.0

        pred_mvo_cal = (df["mvo_vol_pred_ml"] >= cal_mvo_th)
        tp_mvo_c = int(((pred_mvo_cal == True) & (true_mvo == True)).sum())
        fp_mvo_c = int(((pred_mvo_cal == True) & (true_mvo == False)).sum())
        tn_mvo_c = int(((pred_mvo_cal == False) & (true_mvo == False)).sum())
        fn_mvo_c = int(((pred_mvo_cal == False) & (true_mvo == True)).sum())
        sens_mvo_c = float(tp_mvo_c / (tp_mvo_c + fn_mvo_c)) if (tp_mvo_c + fn_mvo_c) > 0 else 0.0
        spec_mvo_c = float(tn_mvo_c / (tn_mvo_c + fp_mvo_c)) if (tn_mvo_c + fp_mvo_c) > 0 else 0.0
        prec_mvo_c = float(tp_mvo_c / (tp_mvo_c + fp_mvo_c)) if (tp_mvo_c + fp_mvo_c) > 0 else 0.0

        return {
            "evaluation_title": "CRIP Dedicated EMIDEC 5-Class Segmentation Evaluation",
            "cohort_sizes": {
                "total_held_out_test_cases": len(df),
                "normal_controls": len(df_norm),
                "pathological_infarct_cases": len(df_path),
                "pathological_mvo_positive_cases": len(df_mvo),
            },
            "dice_metrics": {
                "lv_cavity_mean": safe_mean(df["dice_lv"]),
                "lv_cavity_median": safe_median(df["dice_lv"]),
                "normal_myocardium_mean": safe_mean(df["dice_myo"]),
                "normal_myocardium_median": safe_median(df["dice_myo"]),
                "infarction_mean_in_infarct_cases": safe_mean(df_inf["dice_infarct"]),
                "infarction_median_in_infarct_cases": safe_median(df_inf["dice_infarct"]),
                "mvo_mean_in_mvo_cases": safe_mean(df_mvo["dice_mvo"]),
                "mvo_median_in_mvo_cases": safe_median(df_mvo["dice_mvo"]),
            },
            "hd95_metrics_mm": {
                "lv_cavity_mean": safe_mean(df["hd95_lv_mm"]),
                "normal_myocardium_mean": safe_mean(df["hd95_myo_mm"]),
                "infarction_mean_in_infarct_cases": safe_mean(df_inf["hd95_infarct_mm"]),
                "mvo_mean_in_mvo_cases": safe_mean(df_mvo["hd95_mvo_mm"]),
            },
            "volumetric_errors": {
                "infarct_volume_mae_ml_all": safe_mean(df["infarct_vol_abs_error_ml"]),
                "infarct_volume_mae_ml_pathological": safe_mean(df_path["infarct_vol_abs_error_ml"]),
                "infarct_burden_mae_pct_pathological": safe_mean(df_path["infarct_burden_abs_error_pct"]),
                "mvo_volume_mae_ml_all": safe_mean(df["mvo_vol_abs_error_ml"]),
                "mvo_volume_mae_ml_mvo_cases": safe_mean(df_mvo["mvo_vol_abs_error_ml"]),
                "mvo_burden_mae_pct_mvo_cases": safe_mean(df_mvo["mvo_burden_abs_error_pct"]),
            },
            "volumetric_signed_errors": {
                "infarct_volume_signed_error_ml_all": safe_mean(df["infarct_vol_error_ml"]),
                "infarct_volume_signed_error_ml_pathological": safe_mean(df_path["infarct_vol_error_ml"]),
                "mvo_volume_signed_error_ml_all": safe_mean(df["mvo_vol_error_ml"]),
                "mvo_volume_signed_error_ml_mvo_cases": safe_mean(df_mvo["mvo_vol_error_ml"]),
            },
            "precision_recall_metrics": {
                "lv_cavity_precision_mean": safe_mean(df["prec_lv"]),
                "lv_cavity_recall_mean": safe_mean(df["rec_lv"]),
                "normal_myocardium_precision_mean": safe_mean(df["prec_myo"]),
                "normal_myocardium_recall_mean": safe_mean(df["rec_myo"]),
                "infarction_precision_mean": safe_mean(df["prec_infarct"]),
                "infarction_recall_mean_in_infarct_cases": safe_mean(df_inf["rec_infarct"]),
                "mvo_precision_mean": safe_mean(df["prec_mvo"]),
                "mvo_recall_mean_in_mvo_cases": safe_mean(df_mvo["rec_mvo"]),
            },
            "lesion_classification": {
                "infarction_detection": {
                    "true_positive": tp_inf,
                    "false_positive": fp_inf,
                    "true_negative": tn_inf,
                    "false_negative": fn_inf,
                    "sensitivity": sens_inf,
                    "specificity": spec_inf,
                    "precision": prec_inf,
                },
                "mvo_detection": {
                    "true_positive": tp_mvo,
                    "false_positive": fp_mvo,
                    "true_negative": tn_mvo,
                    "false_negative": fn_mvo,
                    "sensitivity": sens_mvo,
                    "specificity": spec_mvo,
                    "precision": prec_mvo,
                },
            },
            "lesion_classification_calibrated": {
                "thresholds_ml": {
                    "infarction_threshold_ml": cal_inf_th,
                    "mvo_threshold_ml": cal_mvo_th,
                    "calibration_cohort": "Validation (N=20)",
                },
                "infarction_detection": {
                    "true_positive": tp_inf_c,
                    "false_positive": fp_inf_c,
                    "true_negative": tn_inf_c,
                    "false_negative": fn_inf_c,
                    "sensitivity": sens_inf_c,
                    "specificity": spec_inf_c,
                    "precision": prec_inf_c,
                },
                "mvo_detection": {
                    "true_positive": tp_mvo_c,
                    "false_positive": fp_mvo_c,
                    "true_negative": tn_mvo_c,
                    "false_negative": fn_mvo_c,
                    "sensitivity": sens_mvo_c,
                    "specificity": spec_mvo_c,
                    "precision": prec_mvo_c,
                },
            },
        }

    def generate_figures(
        self,
        df: pd.DataFrame,
        output_dir: Path | str = DEFAULT_FIGURES_DIR,
        history_path: Path | str = DEFAULT_HISTORY_PATH,
    ) -> List[Path]:
        """Generate all 6 required scientific figures in PNG, SVG, and PDF."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        created_files: List[Path] = []

        def save_fig(fig: plt.Figure, name_stem: str) -> None:
            for ext in ["png", "svg", "pdf"]:
                p = out_dir / f"{name_stem}.{ext}"
                fig.savefig(p, dpi=300, bbox_inches="tight")
                created_files.append(p)
            plt.close(fig)

        # Style defaults
        plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

        # -------------------------------------------------------------
        # Figure 1: Training and Validation Loss Curves
        # -------------------------------------------------------------
        h_path = Path(history_path)
        if h_path.is_file():
            with open(h_path, "r", encoding="utf-8") as f:
                history = json.load(f)

            epochs = [h["epoch"] for h in history]
            tr_loss = [h["train_loss"] for h in history]
            va_loss = [h["val_loss"] for h in history]
            comb_dice = [h["dice_mean"] for h in history]

            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()

            l1 = ax1.plot(epochs, tr_loss, "b-o", markersize=4, label="Train Loss (DiceFocal)")
            l2 = ax1.plot(epochs, va_loss, "m-s", markersize=4, label="Val Loss (DiceFocal)")
            l3 = ax2.plot(epochs, comb_dice, "g--^", markersize=4, label="Val Combined Dice")

            ax1.set_xlabel("Training Epoch")
            ax1.set_ylabel("Loss (DiceFocal)")
            ax2.set_ylabel("Combined Foreground Dice")
            ax1.set_title("Figure 1: EMIDEC 5-Class U-Net Convergence Dynamics\n(Task: LGE Tissue Characterization, N=60 Train, N=20 Val)")
            ax1.grid(True, linestyle=":", alpha=0.6)

            lines = l1 + l2 + l3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper right")
            save_fig(fig, "figure1_emidec_loss_curves")

        # -------------------------------------------------------------
        # Figure 2: Per-Class Foreground Dice Scores
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        d_lv = df["dice_lv"].dropna()
        d_myo = df["dice_myo"].dropna()
        d_inf = df[df["has_gt_infarct"] == True]["dice_infarct"].dropna()
        d_mvo = df[df["has_gt_mvo"] == True]["dice_mvo"].dropna()

        data_dice = [d_lv, d_myo, d_inf, d_mvo]
        labels = [f"LV Cavity\n(N={len(d_lv)})", f"Normal Myo\n(N={len(d_myo)})", f"Infarction\n(N={len(d_inf)})", f"MVO / No-Reflow\n(N={len(d_mvo)})"]
        colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

        bplot = ax.boxplot(data_dice, tick_labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Plot means
        means = [d.mean() if len(d) > 0 else 0 for d in data_dice]
        ax.plot(range(1, 5), means, "r*", markersize=10, label="Class Mean")

        ax.set_ylabel("Dice Similarity Coefficient")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Figure 2: Held-Out Test Per-Class Spatial Overlap (Dice)\n(EMIDEC Test Cohort N=20, Evaluated Where Present)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
        save_fig(fig, "figure2_emidec_per_class_dice")

        # -------------------------------------------------------------
        # Figure 3: Infarct and MVO HD95 Distributions
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        h_inf = df[df["has_gt_infarct"] == True]["hd95_infarct_mm"].dropna()
        h_mvo = df[df["has_gt_mvo"] == True]["hd95_mvo_mm"].dropna()

        data_hd = [h_inf, h_mvo]
        hd_labels = [f"Infarction HD95\n(N={len(h_inf)})", f"MVO HD95\n(N={len(h_mvo)})"]
        bplot2 = ax.boxplot(data_hd, tick_labels=hd_labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        bplot2["boxes"][0].set_facecolor("#d62728")
        bplot2["boxes"][0].set_alpha(0.7)
        if len(bplot2["boxes"]) > 1:
            bplot2["boxes"][1].set_facecolor("#9467bd")
            bplot2["boxes"][1].set_alpha(0.7)

        ax.set_ylabel("95th Percentile Hausdorff Distance (mm)")
        ax.set_title("Figure 3: Lesion Boundary Delineation Accuracy (HD95 mm)\n(EMIDEC Test Cohort, Lower is Better)")
        ax.grid(True, linestyle=":", alpha=0.6)
        save_fig(fig, "figure3_emidec_hd95_distributions")

        # -------------------------------------------------------------
        # Figure 4: Multi-Panel Slice Visualizations (LGE vs GT vs Pred)
        # -------------------------------------------------------------
        # Select representative pathological case with MVO
        mvo_cases = df[df["has_gt_mvo"] == True]["case_id"].tolist()
        vis_case = mvo_cases[0] if mvo_cases else df["case_id"].iloc[0]

        model = self.load_model()
        c_dict = self.loader.load_case(vis_case)
        img_3d = c_dict["image"]
        gt_3d = c_dict["contour"]
        pred_3d = self.segment_volume_3d(img_3d, model)

        # Pick mid-ventricular slice with infarct/MVO
        mid_s = img_3d.shape[2] // 2
        for s in range(img_3d.shape[2]):
            if 4 in gt_3d[:, :, s]:
                mid_s = s
                break

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # Panel 1: Original LGE
        axes[0].imshow(img_3d[:, :, mid_s], cmap="gray")
        axes[0].set_title(f"Native LGE CMR\n({vis_case}, Slice {mid_s + 1})")
        axes[0].axis("off")

        # Colormap for tissue classes: 0=black, 1=blue, 2=green, 3=red, 4=cyan
        from matplotlib.colors import ListedColormap
        cmap_tissue = ListedColormap(["black", "#1f77b4", "#2ca02c", "#d62728", "#00ffff"])

        # Panel 2: Ground Truth
        axes[1].imshow(gt_3d[:, :, mid_s], cmap=cmap_tissue, vmin=0, vmax=4)
        axes[1].set_title("Expert Reference Annotation\n(1:LV, 2:Myo, 3:Scar, 4:MVO)")
        axes[1].axis("off")

        # Panel 3: Model Prediction
        axes[2].imshow(pred_3d[:, :, mid_s], cmap=cmap_tissue, vmin=0, vmax=4)
        axes[2].set_title("EMIDEC U-Net Prediction\n(5-Class Automated Inference)")
        axes[2].axis("off")

        fig.suptitle("Figure 4: Qualitative Tissue Characterization Comparison (Held-Out Test)", fontsize=13)
        save_fig(fig, "figure4_emidec_prediction_panel")

        # -------------------------------------------------------------
        # Figure 5: Predicted vs Reference Infarct Volume Correlation
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        gt_inf = df["infarct_vol_gt_ml"].values
        pr_inf = df["infarct_vol_pred_ml"].values

        ax.scatter(gt_inf, pr_inf, c="#d62728", s=50, alpha=0.8, edgecolors="k", label="Test Patients (N=20)")
        max_val = max(float(gt_inf.max()), float(pr_inf.max()), 10.0) + 5.0
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7, label="Line of Identity (y = x)")

        # Compute correlation
        if len(gt_inf) > 1 and np.std(gt_inf) > 0 and np.std(pr_inf) > 0:
            r = float(np.corrcoef(gt_inf, pr_inf)[0, 1])
            ax.text(0.05, 0.90, f"Pearson r = {r:.3f}\nMAE = {df['infarct_vol_abs_error_ml'].mean():.2f} mL", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

        ax.set_xlabel("Expert Reference Infarct Volume (mL)")
        ax.set_ylabel("U-Net Predicted Infarct Volume (mL)")
        ax.set_xlim(-1, max_val)
        ax.set_ylim(-1, max_val)
        ax.set_title("Figure 5: Infarct Volume Agreement\n(EMIDEC Held-Out Test Cohort N=20)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
        save_fig(fig, "figure5_infarct_volume_correlation")

        # -------------------------------------------------------------
        # Figure 6: Predicted vs Reference MVO Volume Correlation
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        gt_mvo = df["mvo_vol_gt_ml"].values
        pr_mvo = df["mvo_vol_pred_ml"].values

        ax.scatter(gt_mvo, pr_mvo, c="#9467bd", s=50, alpha=0.8, edgecolors="k", label="Test Patients (N=20)")
        max_mvo = max(float(gt_mvo.max()), float(pr_mvo.max()), 5.0) + 2.0
        ax.plot([0, max_mvo], [0, max_mvo], "k--", alpha=0.7, label="Line of Identity (y = x)")

        if len(gt_mvo) > 1 and np.std(gt_mvo) > 0 and np.std(pr_mvo) > 0:
            r_m = float(np.corrcoef(gt_mvo, pr_mvo)[0, 1])
            ax.text(0.05, 0.90, f"Pearson r = {r_m:.3f}\nMAE = {df['mvo_vol_abs_error_ml'].mean():.2f} mL", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

        ax.set_xlabel("Expert Reference MVO Volume (mL)")
        ax.set_ylabel("U-Net Predicted MVO Volume (mL)")
        ax.set_xlim(-0.5, max_mvo)
        ax.set_ylim(-0.5, max_mvo)
        ax.set_title("Figure 6: MVO / No-Reflow Core Volume Agreement\n(EMIDEC Held-Out Test Cohort N=20)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
        save_fig(fig, "figure6_mvo_volume_correlation")

        logger.info(f"Generated 6 figures (PNG, SVG, PDF) in {out_dir}")
        return created_files
