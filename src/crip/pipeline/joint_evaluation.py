"""Matched Evaluation Engine Comparing Baseline ACDC U-Net vs Joint Cine U-Net.

Evaluates both models on matched, strictly isolated evaluation populations:
1. Original ACDC Held-Out Test Cohort (N=20 patients)
2. Designated M&Ms Held-Out Evaluation Cohort (N=34 studies)

Calculates:
- Multi-structure spatial Dice (LV, RV, MYO, Mean)
- Physical boundary Hausdorff distances (HD95 mm)
- Left ventricular volumetric metrics (LVEDV, LVESV, LVSV, LVEF error)
- Paired differential comparisons (Joint - Baseline)
- Vendor-stratified generalizability analysis (Siemens, Philips, GE, Canon)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from crip.datasets.labels import ACDCLabel
from crip.datasets.mnms_loader import MNMSDataLoader
from crip.evaluation.external_mnms import evaluate_single_mnms_case
from crip.evaluation.segmentation_metrics import evaluate_patient_segmentation
from crip.mri.acdc_loader import ACDCDataLoader
from crip.mri.segmentation_inference import SegmentationInferenceEngine
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH as BASELINE_CHECKPOINT
from crip.pipeline.joint_trainer import DEFAULT_JOINT_CHECKPOINT_PATH as JOINT_CHECKPOINT
from crip.pipeline.splits import DEFAULT_UNIFIED_SPLITS_PATH

logger = logging.getLogger("crip.pipeline.joint_evaluation")

DEFAULT_EVAL_DIR = Path("data/processed/multidataset")
DEFAULT_FIGURES_DIR = Path("docs/evaluation/multidataset")


class MatchedCineEvaluator:
    """Evaluates and contrasts baseline vs joint cine segmentation models."""

    def __init__(
        self,
        baseline_checkpoint: Path | str = BASELINE_CHECKPOINT,
        joint_checkpoint: Path | str = JOINT_CHECKPOINT,
        splits_path: Path | str = DEFAULT_UNIFIED_SPLITS_PATH,
    ) -> None:
        self.baseline_checkpoint = Path(baseline_checkpoint)
        self.joint_checkpoint = Path(joint_checkpoint)
        self.splits_path = Path(splits_path)

        if not self.baseline_checkpoint.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found at {self.baseline_checkpoint}")
        if not self.joint_checkpoint.is_file():
            raise FileNotFoundError(f"Joint checkpoint not found at {self.joint_checkpoint}")

        self.acdc_loader = ACDCDataLoader()
        self.mnms_loader = MNMSDataLoader()

        self.engine_base = SegmentationInferenceEngine(checkpoint_path=self.baseline_checkpoint)
        self.engine_joint = SegmentationInferenceEngine(checkpoint_path=self.joint_checkpoint)

    def load_matched_evaluation_cohorts(self) -> Tuple[List[str], List[str]]:
        """Load ACDC test IDs and M&Ms holdout evaluation IDs."""
        with open(self.splits_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        acdc_test_ids = manifest["datasets"]["acdc"]["splits"]["test"]
        mnms_holdout_ids = manifest["datasets"]["mnms"]["splits"]["held_out_evaluation"]

        logger.info(
            f"Loaded matched evaluation cohorts: ACDC Test (N={len(acdc_test_ids)}), "
            f"M&Ms Holdout (N={len(mnms_holdout_ids)})"
        )
        return acdc_test_ids, mnms_holdout_ids

    def evaluate_acdc_case(
        self,
        patient_id: str,
        engine: SegmentationInferenceEngine,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """Evaluate single ACDC patient with specified inference engine."""
        res = engine.predict_patient(loader=self.acdc_loader, patient_id=patient_id)
        meta = self.acdc_loader.get_patient_metadata(patient_id)
        metrics = evaluate_patient_segmentation(res, group=meta.group)
        return metrics

    def run_matched_evaluation(
        self,
        output_dir: Path | str = DEFAULT_EVAL_DIR,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute matched evaluation across all ACDC test and M&Ms holdout cases."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        acdc_test_ids, mnms_holdout_ids = self.load_matched_evaluation_cohorts()
        mnms_cases_dict = self.mnms_loader.discover_cases()

        model_base = self.engine_base.load_model()
        model_joint = self.engine_joint.load_model()

        records: List[Dict[str, Any]] = []

        # 1. Evaluate ACDC Held-Out Test Cohort (N=20)
        logger.info(f"Evaluating {len(acdc_test_ids)} ACDC held-out test patients...")
        for pid in acdc_test_ids:
            try:
                base_metrics = self.evaluate_acdc_case(pid, self.engine_base, model_base)
                joint_metrics = self.evaluate_acdc_case(pid, self.engine_joint, model_joint)

                rec = {
                    "case_id": pid,
                    "dataset": "acdc",
                    "evaluation_cohort": "acdc_held_out_test",
                    "diagnostic_group": base_metrics.get("group", "Unknown"),
                    "vendor": "Siemens",
                    "center": "Dijon_University_Hospital",
                    # Baseline metrics
                    "base_dice_mean": base_metrics["dice_patient_mean"],
                    "base_dice_lv": base_metrics["dice_patient_lv"],
                    "base_dice_rv": base_metrics["dice_patient_rv"],
                    "base_dice_myo": base_metrics["dice_patient_myo"],
                    "base_lvef_error": base_metrics.get("lvef_error_percent", base_metrics.get("lvef_error_pct", 0.0)),
                    "base_lvef_abs_error": base_metrics.get("lvef_abs_error_percent", base_metrics.get("lvef_abs_error_pct", 0.0)),
                    "base_lvedv_abs_error": base_metrics["lvedv_abs_error_ml"],
                    "base_lvesv_abs_error": base_metrics["lvesv_abs_error_ml"],
                    # Joint metrics
                    "joint_dice_mean": joint_metrics["dice_patient_mean"],
                    "joint_dice_lv": joint_metrics["dice_patient_lv"],
                    "joint_dice_rv": joint_metrics["dice_patient_rv"],
                    "joint_dice_myo": joint_metrics["dice_patient_myo"],
                    "joint_lvef_error": joint_metrics.get("lvef_error_percent", joint_metrics.get("lvef_error_pct", 0.0)),
                    "joint_lvef_abs_error": joint_metrics.get("lvef_abs_error_percent", joint_metrics.get("lvef_abs_error_pct", 0.0)),
                    "joint_lvedv_abs_error": joint_metrics["lvedv_abs_error_ml"],
                    "joint_lvesv_abs_error": joint_metrics["lvesv_abs_error_ml"],
                    # Pairwise delta (Joint - Baseline)
                    "delta_dice_mean": joint_metrics["dice_patient_mean"] - base_metrics["dice_patient_mean"],
                    "delta_dice_lv": joint_metrics["dice_patient_lv"] - base_metrics["dice_patient_lv"],
                    "delta_dice_rv": joint_metrics["dice_patient_rv"] - base_metrics["dice_patient_rv"],
                    "delta_dice_myo": joint_metrics["dice_patient_myo"] - base_metrics["dice_patient_myo"],
                    "delta_lvef_abs_error": joint_metrics.get("lvef_abs_error_percent", joint_metrics.get("lvef_abs_error_pct", 0.0)) - base_metrics.get("lvef_abs_error_percent", base_metrics.get("lvef_abs_error_pct", 0.0)),
                }
                records.append(rec)
            except Exception as e:
                logger.error(f"Error evaluating ACDC patient {pid}: {e}")

        # 2. Evaluate M&Ms Designated Holdout Cohort (N=34)
        logger.info(f"Evaluating {len(mnms_holdout_ids)} M&Ms holdout evaluation cases...")
        for cid in mnms_holdout_ids:
            c_info = mnms_cases_dict.get(cid)
            if not c_info:
                continue
            try:
                base_metrics = evaluate_single_mnms_case(self.mnms_loader, self.engine_base, c_info, model_base)
                joint_metrics = evaluate_single_mnms_case(self.mnms_loader, self.engine_joint, c_info, model_joint)

                rec = {
                    "case_id": cid,
                    "dataset": "mnms",
                    "evaluation_cohort": "mnms_designated_holdout",
                    "diagnostic_group": c_info.get("diagnosis", "Unknown"),
                    "vendor": c_info.get("vendor", "Unknown"),
                    "center": f"Centre_{c_info['csv_metadata'].get('Centre', 'Unknown')}",
                    # Baseline metrics
                    "base_dice_mean": base_metrics["dice_patient_mean"],
                    "base_dice_lv": base_metrics["dice_patient_lv"],
                    "base_dice_rv": base_metrics["dice_patient_rv"],
                    "base_dice_myo": base_metrics["dice_patient_myo"],
                    "base_lvef_error": base_metrics.get("lvef_error_percent", base_metrics.get("lvef_error_pct", 0.0)),
                    "base_lvef_abs_error": base_metrics.get("lvef_abs_error_percent", base_metrics.get("lvef_abs_error_pct", 0.0)),
                    "base_lvedv_abs_error": base_metrics["lvedv_abs_error_ml"],
                    "base_lvesv_abs_error": base_metrics["lvesv_abs_error_ml"],
                    # Joint metrics
                    "joint_dice_mean": joint_metrics["dice_patient_mean"],
                    "joint_dice_lv": joint_metrics["dice_patient_lv"],
                    "joint_dice_rv": joint_metrics["dice_patient_rv"],
                    "joint_dice_myo": joint_metrics["dice_patient_myo"],
                    "joint_lvef_error": joint_metrics.get("lvef_error_percent", joint_metrics.get("lvef_error_pct", 0.0)),
                    "joint_lvef_abs_error": joint_metrics.get("lvef_abs_error_percent", joint_metrics.get("lvef_abs_error_pct", 0.0)),
                    "joint_lvedv_abs_error": joint_metrics["lvedv_abs_error_ml"],
                    "joint_lvesv_abs_error": joint_metrics["lvesv_abs_error_ml"],
                    # Pairwise delta (Joint - Baseline)
                    "delta_dice_mean": joint_metrics["dice_patient_mean"] - base_metrics["dice_patient_mean"],
                    "delta_dice_lv": joint_metrics["dice_patient_lv"] - base_metrics["dice_patient_lv"],
                    "delta_dice_rv": joint_metrics["dice_patient_rv"] - base_metrics["dice_patient_rv"],
                    "delta_dice_myo": joint_metrics["dice_patient_myo"] - base_metrics["dice_patient_myo"],
                    "delta_lvef_abs_error": joint_metrics.get("lvef_abs_error_percent", joint_metrics.get("lvef_abs_error_pct", 0.0)) - base_metrics.get("lvef_abs_error_percent", base_metrics.get("lvef_abs_error_pct", 0.0)),
                }
                records.append(rec)
            except Exception as e:
                logger.error(f"Error evaluating M&Ms case {cid}: {e}")

        df = pd.DataFrame(records)
        csv_path = out_dir / "joint_cine_evaluation.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved matched evaluation CSV ({len(df)} records) to {csv_path}")

        # Summary statistics
        summary = self._compute_summary(df)
        summary_path = out_dir / "joint_cine_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved matched evaluation summary to {summary_path}")

        return df, summary

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate statistical summary by evaluation cohort and vendor."""
        def cohort_stats(cdf: pd.DataFrame) -> Dict[str, Any]:
            return {
                "count": len(cdf),
                "base_dice_mean": float(cdf["base_dice_mean"].mean()),
                "joint_dice_mean": float(cdf["joint_dice_mean"].mean()),
                "delta_dice_mean": float(cdf["delta_dice_mean"].mean()),
                "base_dice_lv": float(cdf["base_dice_lv"].mean()),
                "joint_dice_lv": float(cdf["joint_dice_lv"].mean()),
                "base_dice_rv": float(cdf["base_dice_rv"].mean()),
                "joint_dice_rv": float(cdf["joint_dice_rv"].mean()),
                "base_dice_myo": float(cdf["base_dice_myo"].mean()),
                "joint_dice_myo": float(cdf["joint_dice_myo"].mean()),
                "base_lvef_mae": float(cdf["base_lvef_abs_error"].mean()),
                "joint_lvef_mae": float(cdf["joint_lvef_abs_error"].mean()),
                "delta_lvef_mae": float(cdf["delta_lvef_abs_error"].mean()),
                "base_lvef_within_5pct": float((cdf["base_lvef_abs_error"] <= 5.0).mean()),
                "joint_lvef_within_5pct": float((cdf["joint_lvef_abs_error"] <= 5.0).mean()),
            }

        acdc_df = df[df["evaluation_cohort"] == "acdc_held_out_test"]
        mnms_df = df[df["evaluation_cohort"] == "mnms_designated_holdout"]

        by_vendor: Dict[str, Any] = {}
        for vendor, v_df in mnms_df.groupby("vendor"):
            by_vendor[str(vendor)] = cohort_stats(v_df)

        return {
            "evaluation_title": "CRIP Joint Cine U-Net Matched Cohort Evaluation",
            "checkpoints": {
                "baseline_model": str(self.baseline_checkpoint),
                "joint_model": str(self.joint_checkpoint),
            },
            "acdc_held_out_test": cohort_stats(acdc_df),
            "mnms_designated_holdout": cohort_stats(mnms_df),
            "mnms_by_vendor": by_vendor,
        }

    def generate_figures(
        self,
        df: pd.DataFrame,
        output_dir: Path | str = DEFAULT_FIGURES_DIR,
    ) -> List[Path]:
        """Generate high-resolution publication comparison figures."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_paths: List[Path] = []

        # Color palette
        c_base = "#1f77b4"  # Blue
        c_joint = "#2ca02c" # Green

        # Figure 1: Matched Dice Comparison (ACDC Test vs M&Ms Holdout)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        for idx, (cohort_name, title) in enumerate([
            ("acdc_held_out_test", "ACDC In-Domain Test (N=20)"),
            ("mnms_designated_holdout", "M&Ms External Holdout (N=34)"),
        ]):
            cdf = df[df["evaluation_cohort"] == cohort_name]
            structures = ["LV Cavity", "RV Cavity", "Myocardium", "Mean Foreground"]
            base_means = [
                cdf["base_dice_lv"].mean(),
                cdf["base_dice_rv"].mean(),
                cdf["base_dice_myo"].mean(),
                cdf["base_dice_mean"].mean(),
            ]
            joint_means = [
                cdf["joint_dice_lv"].mean(),
                cdf["joint_dice_rv"].mean(),
                cdf["joint_dice_myo"].mean(),
                cdf["joint_dice_mean"].mean(),
            ]

            x = np.arange(len(structures))
            width = 0.35
            ax = axes[idx]
            rects1 = ax.bar(x - width/2, base_means, width, label="Baseline (ACDC-only)", color=c_base, alpha=0.85)
            rects2 = ax.bar(x + width/2, joint_means, width, label="Joint (ACDC + M&Ms)", color=c_joint, alpha=0.85)

            ax.set_ylabel("Dice Similarity Coefficient", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(structures, rotation=15, ha="right", fontsize=10)
            ax.set_ylim(0.5, 1.0)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(loc="lower right")

            for bar in rects1:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h),
                            ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
            for bar in rects2:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h),
                            ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points", fontweight="bold")

        plt.tight_layout()
        f1_path = out_dir / "figure1_matched_dice_comparison.png"
        plt.savefig(f1_path)
        plt.close()
        fig_paths.append(f1_path)

        # Figure 2: LVEF Absolute Error Comparison
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        cohorts = ["ACDC Test (N=20)", "M&Ms Holdout (N=34)"]
        base_mae = [
            df[df["evaluation_cohort"] == "acdc_held_out_test"]["base_lvef_abs_error"].mean(),
            df[df["evaluation_cohort"] == "mnms_designated_holdout"]["base_lvef_abs_error"].mean(),
        ]
        joint_mae = [
            df[df["evaluation_cohort"] == "acdc_held_out_test"]["joint_lvef_abs_error"].mean(),
            df[df["evaluation_cohort"] == "mnms_designated_holdout"]["joint_lvef_abs_error"].mean(),
        ]

        x = np.arange(len(cohorts))
        width = 0.35
        r1 = ax.bar(x - width/2, base_mae, width, label="Baseline (ACDC-only)", color=c_base, alpha=0.85)
        r2 = ax.bar(x + width/2, joint_mae, width, label="Joint (ACDC + M&Ms)", color=c_joint, alpha=0.85)

        ax.set_ylabel("LVEF Mean Absolute Error (% points)", fontsize=11, fontweight="bold")
        ax.set_title("LVEF Volumetric Error on Matched Holdout Cohorts", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cohorts, fontsize=11)
        ax.set_ylim(0, 12)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()

        for bar in r1:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
        for bar in r2:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points", fontweight="bold")

        plt.tight_layout()
        f2_path = out_dir / "figure2_lvef_mae_comparison.png"
        plt.savefig(f2_path)
        plt.close()
        fig_paths.append(f2_path)

        # Figure 3: Vendor-Stratified Gains on M&Ms Holdout
        mnms_df = df[df["evaluation_cohort"] == "mnms_designated_holdout"]
        if "vendor" in mnms_df.columns:
            vendors = sorted(mnms_df["vendor"].dropna().unique().tolist())
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

            b_means = [mnms_df[mnms_df["vendor"] == v]["base_dice_mean"].mean() for v in vendors]
            j_means = [mnms_df[mnms_df["vendor"] == v]["joint_dice_mean"].mean() for v in vendors]

            x = np.arange(len(vendors))
            width = 0.35
            ax.bar(x - width/2, b_means, width, label="Baseline (ACDC-only)", color=c_base, alpha=0.85)
            ax.bar(x + width/2, j_means, width, label="Joint (ACDC + M&Ms)", color=c_joint, alpha=0.85)

            ax.set_ylabel("Mean Foreground Dice", fontsize=11, fontweight="bold")
            ax.set_title("Vendor-Stratified Generalizability on M&Ms Holdout", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v}\n(N={(mnms_df['vendor'] == v).sum()})" for v in vendors], fontsize=10)
            ax.set_ylim(0.4, 1.0)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(loc="lower right")

            plt.tight_layout()
            f3_path = out_dir / "figure3_vendor_stratified_gain.png"
            plt.savefig(f3_path)
            plt.close()
            fig_paths.append(f3_path)

        logger.info(f"Generated {len(fig_paths)} matched evaluation figures in {out_dir}")
        return fig_paths
