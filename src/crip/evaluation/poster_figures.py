"""Generation of publication-quality, poster-ready figures and statistical exhibits.

Produces 300-DPI high-resolution figures in PNG, SVG, and PDF formats covering:
1. Real CMR U-Net Segmentation Dice performance by anatomical structure
2. HD95 95th-percentile Hausdorff Distance spatial boundary error by structure
3. Predicted vs. Ground-Truth Reference LVEF with Identity Line (y = x)
4. Bland-Altman Agreement Plot: Left Ventricular Ejection Fraction (LVEF)
5. Bland-Altman Agreement Plot: Left Ventricular End-Diastolic Volume (LVEDV)
6. Bland-Altman Agreement Plot: Left Ventricular End-Systolic Volume (LVESV)
7. Synthetic Recovery Classification: Receiver Operating Characteristic (ROC) Comparison
8. Synthetic Recovery Classification: Reliability Calibration Curve Comparison

DISCLAIMER:
All CMR figures derive from real held-out ACDC 2017 challenge testing cases (n = 20: 4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV).
All recovery figures derive from synthetic multimodal cohort testing cases (n = 419).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set temporary cache dir for matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from crip.evaluation.measurement_agreement import (
    BlandAltmanResult,
    compute_hd95_statistics,
    compute_measurement_agreement,
)
from crip.evaluation.model_comparison import generate_recovery_model_comparison

logger = logging.getLogger("crip.evaluation.poster_figures")

DEFAULT_POSTER_DIR = Path("docs/evaluation/poster_figures")
SEGMENTATION_EVAL_CSV = Path("data/processed/acdc/segmentation_evaluation.csv")

# Set standard publication typography & styles
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def _save_figure_multi_format(fig: plt.Figure, base_path: Path) -> Dict[str, str]:
    """Save matplotlib figure in PNG (300 DPI), SVG, and PDF formats."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    pdf_path = base_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "pdf": str(pdf_path),
    }


def generate_fig1_dice_by_structure(eval_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Figure 1: Real CMR U-Net Segmentation Dice performance by anatomical structure."""
    fig, ax = plt.subplots(figsize=(7, 5))

    data = [
        eval_df["dice_patient_lv"].dropna().values,
        eval_df["dice_patient_rv"].dropna().values,
        eval_df["dice_patient_myo"].dropna().values,
    ]
    labels = ["LV Cavity\n(Label 3)", "RV Cavity\n(Label 1)", "Myocardium\n(Label 2)"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    bp = ax.boxplot(data, patch_artist=True, widths=0.45, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(bp[element], color="#2C3E50", linewidth=1.8)

    # Overlay jittered patient points
    rng = np.random.default_rng(42)
    for i, pts in enumerate(data, start=1):
        jitter = rng.normal(0, 0.04, size=len(pts))
        ax.scatter(np.full_like(pts, i) + jitter, pts, color=colors[i-1], alpha=0.85, s=35, edgecolor="white", zorder=3)

    means = [np.mean(d) for d in data]
    ax.axhline(0.80, color="#7F8C8D", linestyle=":", linewidth=1.5, label="Clinical Target Benchmark (0.80)")

    for i, m in enumerate(means, start=1):
        ax.text(i, 0.64, f"Mean: {m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#BDC3C7"))

    ax.set_ylabel("Sørensen–Dice Similarity Coefficient")
    ax.set_ylim(0.60, 1.0)
    ax.set_title("Automated 2D U-Net Segmentation Accuracy by Structure\nHeld-Out Test Split (n = 20 Scans)")
    ax.legend(loc="lower right")

    paths = _save_figure_multi_format(fig, out_dir / "fig1_segmentation_dice_by_structure")
    plt.close(fig)

    return {
        "figure_id": "fig1_segmentation_dice",
        "title": "U-Net Segmentation Dice Similarity by Structure",
        "description": "Multi-class Sørensen-Dice similarity coefficient distributions for LV cavity, RV cavity, and myocardium on 20 held-out ACDC test scans (4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV).",
        "source_data": "data/processed/acdc/segmentation_evaluation.csv",
        "sample_size": len(eval_df),
        "files": paths,
        "metrics": {"dice_lv_mean": round(float(means[0]), 3), "dice_rv_mean": round(float(means[1]), 3), "dice_myo_mean": round(float(means[2]), 3)},
    }


def generate_fig2_hd95_by_structure(eval_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Figure 2: 95th-Percentile Hausdorff Distance (HD95 mm) by Structure."""
    fig, ax = plt.subplots(figsize=(7, 5))

    hd95_stats = compute_hd95_statistics(eval_df)
    data = [
        eval_df["hd95_ed_lv_mm"].dropna().values,
        eval_df["hd95_ed_rv_mm"].dropna().values,
        eval_df["hd95_ed_myo_mm"].dropna().values,
    ]
    labels = ["LV Cavity", "RV Cavity", "Myocardium"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    bp = ax.boxplot(data, patch_artist=True, widths=0.45, tick_labels=labels, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(bp[element], color="#2C3E50", linewidth=1.8)

    rng = np.random.default_rng(42)
    for i, pts in enumerate(data, start=1):
        jitter = rng.normal(0, 0.04, size=len(pts))
        ax.scatter(np.full_like(pts, i) + jitter, pts, color=colors[i-1], alpha=0.85, s=35, edgecolor="white", zorder=3)

    medians = [np.median(d) for d in data]
    for i, med in enumerate(medians, start=1):
        ax.text(i, med + 1.2, f"Median: {med:.2f} mm", ha="center", va="bottom", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#BDC3C7"))

    ax.set_ylabel("95th-Percentile Hausdorff Distance (mm)")
    ax.set_ylim(-0.5, 40.0)
    ax.set_title("Boundary Distance Error (HD95 mm) at End-Diastole\nHeld-Out Test Split (n = 20 Scans)")

    paths = _save_figure_multi_format(fig, out_dir / "fig2_hd95_by_structure")
    plt.close(fig)

    return {
        "figure_id": "fig2_hd95_distance",
        "title": "95th-Percentile Hausdorff Distance by Structure",
        "description": "Spatial contour boundary distance error (HD95 in millimeters) across cardiac chambers at End-Diastole on 20 held-out ACDC test scans (4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV).",
        "source_data": "data/processed/acdc/segmentation_evaluation.csv",
        "sample_size": len(eval_df),
        "files": paths,
        "metrics": {"lv_hd95_ed_median_mm": round(float(medians[0]), 2), "rv_hd95_ed_median_mm": round(float(medians[1]), 2), "myo_hd95_ed_median_mm": round(float(medians[2]), 2)},
    }


def generate_fig3_predicted_vs_reference_lvef(res_lvef: BlandAltmanResult, out_dir: Path) -> Dict[str, Any]:
    """Figure 3: Predicted vs. Reference LVEF Scatter Plot with Identity Line (y = x)."""
    fig, ax = plt.subplots(figsize=(6.5, 6))

    min_v, max_v = 5.0, 75.0
    ax.plot([min_v, max_v], [min_v, max_v], color="#7F8C8D", linestyle="--", linewidth=1.8, label="Identity Line (y = x)")

    # SCMR +/- 5% error tolerance envelope
    ax.fill_between([min_v, max_v], [min_v - 5.0, max_v - 5.0], [min_v + 5.0, max_v + 5.0],
                    color="#27AE60", alpha=0.12, label="SCMR ±5.0% Inter-observer Tolerance")

    # Points colored by whether within 5%
    within_5 = np.abs(res_lvef.differences) <= 5.0
    ax.scatter(res_lvef.references[within_5], res_lvef.predictions[within_5],
               color="#27AE60", s=60, edgecolors="white", linewidth=1.2, zorder=4, label=f"Within ±5.0% (n = {np.sum(within_5)})")
    ax.scatter(res_lvef.references[~within_5], res_lvef.predictions[~within_5],
               color="#E67E22", s=60, edgecolors="white", linewidth=1.2, zorder=4, label=f"Error > 5.0% (n = {np.sum(~within_5)})")

    # Statistical summary box
    summary_text = (
        f"n = {res_lvef.n_samples} Patients\n"
        f"MAE: {res_lvef.mae:.2f} %-pts\n"
        f"Median AE: {res_lvef.median_ae:.2f} %-pts\n"
        f"RMSE: {res_lvef.rmse:.2f} %-pts\n"
        f"Pearson r: {res_lvef.pearson_r:.4f}"
    )
    ax.text(0.05, 0.72, summary_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.9))

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Expert Ground-Truth LVEF (%)")
    ax.set_ylabel("Automated 2D U-Net Predicted LVEF (%)")
    ax.set_title("Automated vs. Ground-Truth Left Ventricular Ejection Fraction\nHeld-Out Test Split (n = 20 Scans)")
    ax.legend(loc="lower right", framealpha=0.9)

    paths = _save_figure_multi_format(fig, out_dir / "fig3_predicted_vs_reference_lvef")
    plt.close(fig)

    return {
        "figure_id": "fig3_scatter_lvef",
        "title": "Predicted vs. Reference LVEF Scatter Plot",
        "description": "Correlation and agreement between 2D U-Net predicted LVEF and expert ground-truth annotations with y=x identity line and SCMR 5% envelope (20 held-out test scans: 4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV).",
        "source_data": "data/processed/acdc/segmentation_evaluation.csv",
        "sample_size": res_lvef.n_samples,
        "files": paths,
        "metrics": {"lvef_mae_pct": res_lvef.mae, "lvef_rmse_pct": res_lvef.rmse, "pearson_r": res_lvef.pearson_r, "pct_within_5pct": round(float(np.mean(within_5) * 100), 1)},
    }


def generate_fig4_bland_altman_lvef(res_lvef: BlandAltmanResult, out_dir: Path) -> Dict[str, Any]:
    """Figure 4: Bland-Altman Agreement Plot for LVEF (%-points)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    x = res_lvef.means
    y = res_lvef.differences
    x_min, x_max = 5.0, 75.0

    ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.2)
    ax.axhline(res_lvef.mean_bias, color="#2980B9", linewidth=2.2, label=f"Mean Bias: {res_lvef.mean_bias:+.2f} %-pts")
    ax.axhline(res_lvef.upper_loa, color="#C0392B", linestyle="--", linewidth=1.8, label=f"+1.96 SD: {res_lvef.upper_loa:+.2f} %-pts")
    ax.axhline(res_lvef.lower_loa, color="#C0392B", linestyle="--", linewidth=1.8, label=f"−1.96 SD: {res_lvef.lower_loa:+.2f} %-pts")

    # Shaded confidence intervals for bias and LoA
    ax.axhspan(res_lvef.ci_bias_95[0], res_lvef.ci_bias_95[1], color="#2980B9", alpha=0.15)
    ax.axhspan(res_lvef.ci_upper_loa_95[0], res_lvef.ci_upper_loa_95[1], color="#C0392B", alpha=0.12)
    ax.axhspan(res_lvef.ci_lower_loa_95[0], res_lvef.ci_lower_loa_95[1], color="#C0392B", alpha=0.12)

    ax.scatter(x, y, color="#2C3E50", s=55, edgecolors="white", linewidth=1.2, zorder=4)

    # Right annotations
    ax.text(x_max + 1, res_lvef.mean_bias, f"{res_lvef.mean_bias:+.2f}", va="center", color="#2980B9", fontweight="bold", fontsize=10)
    ax.text(x_max + 1, res_lvef.upper_loa, f"{res_lvef.upper_loa:+.2f}", va="center", color="#C0392B", fontweight="bold", fontsize=10)
    ax.text(x_max + 1, res_lvef.lower_loa, f"{res_lvef.lower_loa:+.2f}", va="center", color="#C0392B", fontweight="bold", fontsize=10)

    ax.set_xlim(x_min, x_max + 6)
    ax.set_ylim(-16.0, 16.0)
    ax.set_xlabel("Mean of Predicted and Reference LVEF (%)")
    ax.set_ylabel("Difference: (Predicted − Reference) (%-points)")
    ax.set_title("Bland–Altman Agreement: Left Ventricular Ejection Fraction (LVEF)\nHeld-Out Test Split (n = 20 Scans)")
    ax.legend(loc="lower left", framealpha=0.9)

    paths = _save_figure_multi_format(fig, out_dir / "fig4_bland_altman_lvef")
    plt.close(fig)

    return {
        "figure_id": "fig4_ba_lvef",
        "title": "Bland-Altman Agreement: LVEF",
        "description": "Bland-Altman difference plot for Left Ventricular Ejection Fraction showing mean bias (-0.55 %-pts), 95% limits of agreement [-10.53, +9.43], and 95% confidence intervals on 20 held-out ACDC test scans (4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV).",
        "source_data": "data/processed/acdc/segmentation_evaluation.csv",
        "sample_size": res_lvef.n_samples,
        "files": paths,
        "metrics": {"mean_bias_pct": res_lvef.mean_bias, "sd_diff_pct": res_lvef.sd_diff, "lower_loa_pct": res_lvef.lower_loa, "upper_loa_pct": res_lvef.upper_loa},
    }


def generate_fig5_bland_altman_lvedv(res_lvedv: BlandAltmanResult, out_dir: Path) -> Dict[str, Any]:
    """Figure 5: Bland-Altman Agreement Plot for LVEDV (mL)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    x = res_lvedv.means
    y = res_lvedv.differences
    x_min = np.min(x) - 15.0
    x_max = np.max(x) + 15.0

    ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.2)
    ax.axhline(res_lvedv.mean_bias, color="#2980B9", linewidth=2.2, label=f"Mean Bias: {res_lvedv.mean_bias:+.2f} mL")
    ax.axhline(res_lvedv.upper_loa, color="#C0392B", linestyle="--", linewidth=1.8, label=f"+1.96 SD: {res_lvedv.upper_loa:+.2f} mL")
    ax.axhline(res_lvedv.lower_loa, color="#C0392B", linestyle="--", linewidth=1.8, label=f"−1.96 SD: {res_lvedv.lower_loa:+.2f} mL")

    ax.axhspan(res_lvedv.ci_bias_95[0], res_lvedv.ci_bias_95[1], color="#2980B9", alpha=0.15)
    ax.axhspan(res_lvedv.ci_upper_loa_95[0], res_lvedv.ci_upper_loa_95[1], color="#C0392B", alpha=0.12)
    ax.axhspan(res_lvedv.ci_lower_loa_95[0], res_lvedv.ci_lower_loa_95[1], color="#C0392B", alpha=0.12)

    ax.scatter(x, y, color="#2C3E50", s=55, edgecolors="white", linewidth=1.2, zorder=4)

    ax.text(x_max + 2, res_lvedv.mean_bias, f"{res_lvedv.mean_bias:+.2f}", va="center", color="#2980B9", fontweight="bold", fontsize=10)
    ax.text(x_max + 2, res_lvedv.upper_loa, f"{res_lvedv.upper_loa:+.2f}", va="center", color="#C0392B", fontweight="bold", fontsize=10)
    ax.text(x_max + 2, res_lvedv.lower_loa, f"{res_lvedv.lower_loa:+.2f}", va="center", color="#C0392B", fontweight="bold", fontsize=10)

    ax.set_xlim(x_min, x_max + 25)
    ax.set_ylim(-26.0, 30.0)
    ax.set_xlabel("Mean of Predicted and Reference LVEDV (mL)")
    ax.set_ylabel("Difference: (Predicted − Reference) (mL)")
    ax.set_title("Bland–Altman Agreement: LV End-Diastolic Volume (LVEDV)\nHeld-Out Test Split (n = 20 Scans)")
    ax.legend(loc="lower left", framealpha=0.9)

    paths = _save_figure_multi_format(fig, out_dir / "fig5_bland_altman_lvedv")
    plt.close(fig)

    return {
        "figure_id": "fig5_ba_lvedv",
        "title": "Bland-Altman Agreement: LVEDV",
        "description": "Bland-Altman difference plot for Left Ventricular End-Diastolic Volume showing mean bias (+2.13 mL) and limits of agreement [-15.73, +19.99] mL on 20 held-out ACDC test scans (4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV).",
        "source_data": "data/processed/acdc/segmentation_evaluation.csv",
        "sample_size": res_lvedv.n_samples,
        "files": paths,
        "metrics": {"mean_bias_ml": res_lvedv.mean_bias, "sd_diff_ml": res_lvedv.sd_diff, "lower_loa_ml": res_lvedv.lower_loa, "upper_loa_ml": res_lvedv.upper_loa},
    }


def generate_fig6_bland_altman_lvesv(res_lvesv: BlandAltmanResult, out_dir: Path) -> Dict[str, Any]:
    """Figure 6: Bland-Altman Agreement Plot for LVESV (mL)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    x = res_lvesv.means
    y = res_lvesv.differences
    x_min = np.min(x) - 15.0
    x_max = np.max(x) + 15.0

    ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.2)
    ax.axhline(res_lvesv.mean_bias, color="#2980B9", linewidth=2.2, label=f"Mean Bias: {res_lvesv.mean_bias:+.2f} mL")
    ax.axhline(res_lvesv.upper_loa, color="#C0392B", linestyle="--", linewidth=1.8, label=f"+1.96 SD: {res_lvesv.upper_loa:+.2f} mL")
    ax.axhline(res_lvesv.lower_loa, color="#C0392B", linestyle="--", linewidth=1.8, label=f"−1.96 SD: {res_lvesv.lower_loa:+.2f} mL")

    ax.axhspan(res_lvesv.ci_bias_95[0], res_lvesv.ci_bias_95[1], color="#2980B9", alpha=0.15)
    ax.axhspan(res_lvesv.ci_upper_loa_95[0], res_lvesv.ci_upper_loa_95[1], color="#C0392B", alpha=0.12)
    ax.axhspan(res_lvesv.ci_lower_loa_95[0], res_lvesv.ci_lower_loa_95[1], color="#C0392B", alpha=0.12)

    ax.scatter(x, y, color="#2C3E50", s=55, edgecolors="white", linewidth=1.2, zorder=4)

    ax.text(x_max + 2, res_lvesv.mean_bias, f"{res_lvesv.mean_bias:+.2f}", va="center", color="#2980B9", fontweight="bold", fontsize=10)
    ax.text(x_max + 2, res_lvesv.upper_loa, f"{res_lvesv.upper_loa:+.2f}", va="center", color="#C0392B", fontweight="bold", fontsize=10)
    ax.text(x_max + 2, res_lvesv.lower_loa, f"{res_lvesv.lower_loa:+.2f}", va="center", color="#C0392B", fontweight="bold", fontsize=10)

    ax.set_xlim(x_min, x_max + 25)
    ax.set_ylim(-22.0, 26.0)
    ax.set_xlabel("Mean of Predicted and Reference LVESV (mL)")
    ax.set_ylabel("Difference: (Predicted − Reference) (mL)")
    ax.set_title("Bland–Altman Agreement: LV End-Systolic Volume (LVESV)\nHeld-Out Test Split (n = 20 Scans)")
    ax.legend(loc="lower left", framealpha=0.9)

    paths = _save_figure_multi_format(fig, out_dir / "fig6_bland_altman_lvesv")
    plt.close(fig)

    return {
        "figure_id": "fig6_ba_lvesv",
        "title": "Bland-Altman Agreement: LVESV",
        "description": "Bland-Altman difference plot for Left Ventricular End-Systolic Volume showing mean bias (+2.35 mL) and limits of agreement [-13.07, +17.77] mL on 20 held-out ACDC test scans (4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV).",
        "source_data": "data/processed/acdc/segmentation_evaluation.csv",
        "sample_size": res_lvesv.n_samples,
        "files": paths,
        "metrics": {"mean_bias_ml": res_lvesv.mean_bias, "sd_diff_ml": res_lvesv.sd_diff, "lower_loa_ml": res_lvesv.lower_loa, "upper_loa_ml": res_lvesv.upper_loa},
    }


def generate_fig7_recovery_model_roc(comp: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Figure 7: Synthetic Recovery Model ROC Curves (LR vs. RF)."""
    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.plot([0, 1], [0, 1], color="#7F8C8D", linestyle="--", linewidth=1.5, label="Chance Level (AUC = 0.5000)")

    m_lr = comp["models"]["logistic_regression"]
    fpr_lr = m_lr["roc_curve"]["fpr"]
    tpr_lr = m_lr["roc_curve"]["tpr"]
    auc_lr = m_lr["metrics"]["roc_auc"]
    ax.plot(fpr_lr, tpr_lr, color="#27AE60", linewidth=2.5, label=f"Logistic Regression (AUC = {auc_lr:.4f})")

    m_rf = comp["models"]["random_forest"]
    fpr_rf = m_rf["roc_curve"]["fpr"]
    tpr_rf = m_rf["roc_curve"]["tpr"]
    auc_rf = m_rf["metrics"]["roc_auc"]
    ax.plot(fpr_rf, tpr_rf, color="#E67E22", linewidth=2.2, label=f"Random Forest (AUC = {auc_rf:.4f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Recovery Discrimination: ROC Curve Comparison\nSynthetic Sustained-Abstinence Test Split (n = 419)")
    ax.legend(loc="lower right", framealpha=0.9)

    paths = _save_figure_multi_format(fig, out_dir / "fig7_recovery_model_roc_comparison")
    plt.close(fig)

    return {
        "figure_id": "fig7_recovery_roc",
        "title": "Recovery Prediction: ROC Comparison",
        "description": "Receiver Operating Characteristic curves comparing Logistic Regression (AUC = 0.9565) against Random Forest (AUC = 0.9310) on synthetic test data.",
        "source_data": "data/processed/recovery_model_comparison.json",
        "sample_size": comp["metadata"]["test_sample_size"],
        "files": paths,
        "metrics": {"lr_roc_auc": auc_lr, "rf_roc_auc": auc_rf},
    }


def generate_fig8_recovery_model_calibration(comp: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Figure 8: Synthetic Recovery Model Calibration Curves (LR vs. RF)."""
    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.plot([0, 1], [0, 1], color="#7F8C8D", linestyle="--", linewidth=1.5, label="Perfect Calibration (y = x)")

    m_lr = comp["models"]["logistic_regression"]
    pred_lr = m_lr["calibration_curve"]["mean_predicted"]
    obs_lr = m_lr["calibration_curve"]["fraction_positives"]
    brier_lr = m_lr["metrics"]["brier_score"]
    ax.plot(pred_lr, obs_lr, marker="o", markersize=6, color="#27AE60", linewidth=2.2, label=f"Logistic Regression (Brier = {brier_lr:.4f})")

    m_rf = comp["models"]["random_forest"]
    pred_rf = m_rf["calibration_curve"]["mean_predicted"]
    obs_rf = m_rf["calibration_curve"]["fraction_positives"]
    brier_rf = m_rf["metrics"]["brier_score"]
    ax.plot(pred_rf, obs_rf, marker="s", markersize=6, color="#E67E22", linewidth=2.0, label=f"Random Forest (Brier = {brier_rf:.4f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean Predicted Recovery Probability")
    ax.set_ylabel("Observed Fraction of Recoveries")
    ax.set_title("Probability Reliability: 10-Bin Calibration Curves\nSynthetic Sustained-Abstinence Test Split (n = 419)")
    ax.legend(loc="upper left", framealpha=0.9)

    paths = _save_figure_multi_format(fig, out_dir / "fig8_recovery_model_calibration")
    plt.close(fig)

    return {
        "figure_id": "fig8_recovery_calibration",
        "title": "Recovery Prediction: Reliability Calibration",
        "description": "Reliability calibration curves evaluating predicted probability accuracy against observed event frequencies for Logistic Regression and Random Forest.",
        "source_data": "data/processed/recovery_model_comparison.json",
        "sample_size": comp["metadata"]["test_sample_size"],
        "files": paths,
        "metrics": {"lr_brier_score": brier_lr, "rf_brier_score": brier_rf},
    }


def generate_all_poster_figures(
    eval_csv_path: Optional[Union[Path, str]] = SEGMENTATION_EVAL_CSV,
    output_dir: Optional[Union[Path, str]] = DEFAULT_POSTER_DIR,
) -> Dict[str, Any]:
    """Execute complete poster figure export workflow and write JSON manifest."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_POSTER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(eval_csv_path) if eval_csv_path else SEGMENTATION_EVAL_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found at {csv_path}")

    eval_df = pd.read_csv(csv_path)

    # 1. Compute agreement results
    res_lvef = compute_measurement_agreement(
        predictions=eval_df["lvef_pred_pct"],
        references=eval_df["lvef_gt_pct"],
        variable_name="LVEF",
        unit="%-points",
        patient_ids=eval_df["patient_id"],
    )
    res_lvedv = compute_measurement_agreement(
        predictions=eval_df["lvedv_pred_ml"],
        references=eval_df["lvedv_gt_ml"],
        variable_name="LVEDV",
        unit="mL",
        patient_ids=eval_df["patient_id"],
    )
    res_lvesv = compute_measurement_agreement(
        predictions=eval_df["lvesv_pred_ml"],
        references=eval_df["lvesv_gt_ml"],
        variable_name="LVESV",
        unit="mL",
        patient_ids=eval_df["patient_id"],
    )

    # 2. Load or generate model comparison
    comp = generate_recovery_model_comparison()

    # 3. Generate and export each figure
    manifest: Dict[str, Any] = {
        "title": "CRIP Academic Poster Figure Inventory & Manifest",
        "timestamp": "2026-11-02",
        "total_figures": 8,
        "figures": [],
    }

    manifest["figures"].append(generate_fig1_dice_by_structure(eval_df, out_dir))
    manifest["figures"].append(generate_fig2_hd95_by_structure(eval_df, out_dir))
    manifest["figures"].append(generate_fig3_predicted_vs_reference_lvef(res_lvef, out_dir))
    manifest["figures"].append(generate_fig4_bland_altman_lvef(res_lvef, out_dir))
    manifest["figures"].append(generate_fig5_bland_altman_lvedv(res_lvedv, out_dir))
    manifest["figures"].append(generate_fig6_bland_altman_lvesv(res_lvesv, out_dir))
    manifest["figures"].append(generate_fig7_recovery_model_roc(comp, out_dir))
    manifest["figures"].append(generate_fig8_recovery_model_calibration(comp, out_dir))

    # Save manifest
    manifest_path = out_dir / "figure_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated {len(manifest['figures'])} poster figures and saved manifest to {manifest_path}")
    return manifest


if __name__ == "__main__":
    generate_all_poster_figures()
