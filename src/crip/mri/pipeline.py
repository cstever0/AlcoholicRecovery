"""Patient and Cohort-Level Cardiac Measurement Pipeline for ACDC.

This module automates the extraction, physical volumetric calculation,
physiological validation, and structured export (CSV, JSON) of biventricular
and myocardial parameters across the ACDC cohort.

DISCLAIMER:
All measurements are derived directly from expert-provided ACDC segmentation masks.
This pipeline is an investigative research tool and does not provide clinical diagnosis.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crip.mri.acdc_loader import (
    ACDCDataError,
    ACDCDataLoader,
    DEFAULT_ACDC_TRAINING_DIR,
)
from crip.mri.cardiac_metrics import (
    BiventricularCardiacMeasurements,
    compute_biventricular_measurements,
)

logger = logging.getLogger("crip.mri.pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_PROCESSED_DIR = Path("data/processed/acdc")


def process_single_patient(
    loader: ACDCDataLoader,
    patient_id: str,
) -> Dict[str, Any]:
    """Process a single ACDC patient, computing biventricular metrics and linking source files."""
    meta = loader.get_patient_metadata(patient_id)
    ed_frame = loader.load_cardiac_frame(patient_id, phase="ED", load_mask=True)
    es_frame = loader.load_cardiac_frame(patient_id, phase="ES", load_mask=True)

    if ed_frame.mask_data is None or es_frame.mask_data is None:
        raise ACDCDataError(f"Missing segmentation mask for patient {patient_id}")

    measurements = compute_biventricular_measurements(
        ed_mask=ed_frame.mask_data,
        es_mask=es_frame.mask_data,
        voxel_spacing_mm=ed_frame.voxel_spacing,
        patient_id=patient_id,
    )

    p_dir = meta.patient_dir
    ed_img_file = p_dir / f"{patient_id}_frame{meta.ed_frame:02d}.nii.gz"
    ed_mask_file = p_dir / f"{patient_id}_frame{meta.ed_frame:02d}_gt.nii.gz"
    es_img_file = p_dir / f"{patient_id}_frame{meta.es_frame:02d}.nii.gz"
    es_mask_file = p_dir / f"{patient_id}_frame{meta.es_frame:02d}_gt.nii.gz"

    return {
        "patient_id": patient_id,
        "diagnostic_group": meta.group,
        "height_cm": meta.height_cm,
        "weight_kg": meta.weight_kg,
        "ed_frame_index": meta.ed_frame,
        "es_frame_index": meta.es_frame,
        "lvedv_ml": round(measurements.lvedv_ml, 2),
        "lvesv_ml": round(measurements.lvesv_ml, 2),
        "lvsv_ml": round(measurements.lvsv_ml, 2),
        "lvef_percent": round(measurements.lvef_percent, 2),
        "rvedv_ml": round(measurements.rvedv_ml, 2),
        "rvesv_ml": round(measurements.rvesv_ml, 2),
        "rvsv_ml": round(measurements.rvsv_ml, 2),
        "rvef_percent": round(measurements.rvef_percent, 2),
        "myocardial_volume_ed_ml": round(measurements.myocardial_volume_ed_ml, 2),
        "myocardial_volume_es_ml": round(measurements.myocardial_volume_es_ml, 2),
        "myocardial_mass_ed_g": round(measurements.myocardial_mass_ed_g, 2),
        "myocardial_mass_es_g": round(measurements.myocardial_mass_es_g, 2),
        "voxel_spacing_x_mm": round(ed_frame.voxel_spacing[0], 4),
        "voxel_spacing_y_mm": round(ed_frame.voxel_spacing[1], 4),
        "voxel_spacing_z_mm": round(ed_frame.voxel_spacing[2], 4),
        "num_slices": ed_frame.num_slices,
        "image_height": ed_frame.height,
        "image_width": ed_frame.width,
        "ed_image_source_file": str(ed_img_file),
        "ed_mask_source_file": str(ed_mask_file),
        "es_image_source_file": str(es_img_file),
        "es_mask_source_file": str(es_mask_file),
        "is_physiologically_valid": measurements.is_physiologically_valid,
        "validation_warnings": "; ".join(measurements.validation_warnings) if measurements.validation_warnings else "None",
    }


def compute_cohort_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute parametric and non-parametric summary metrics across the cohort."""
    if df.empty:
        return {}

    summary = {
        "cohort_size": int(len(df)),
        "lvedv_ml": {
            "mean": round(float(df["lvedv_ml"].mean()), 2),
            "std": round(float(df["lvedv_ml"].std()), 2),
            "median": round(float(df["lvedv_ml"].median()), 2),
            "iqr": round(float(df["lvedv_ml"].quantile(0.75) - df["lvedv_ml"].quantile(0.25)), 2),
            "min": round(float(df["lvedv_ml"].min()), 2),
            "max": round(float(df["lvedv_ml"].max()), 2),
        },
        "lvesv_ml": {
            "mean": round(float(df["lvesv_ml"].mean()), 2),
            "std": round(float(df["lvesv_ml"].std()), 2),
            "median": round(float(df["lvesv_ml"].median()), 2),
            "iqr": round(float(df["lvesv_ml"].quantile(0.75) - df["lvesv_ml"].quantile(0.25)), 2),
            "min": round(float(df["lvesv_ml"].min()), 2),
            "max": round(float(df["lvesv_ml"].max()), 2),
        },
        "lvef_percent": {
            "mean": round(float(df["lvef_percent"].mean()), 2),
            "std": round(float(df["lvef_percent"].std()), 2),
            "median": round(float(df["lvef_percent"].median()), 2),
            "iqr": round(float(df["lvef_percent"].quantile(0.75) - df["lvef_percent"].quantile(0.25)), 2),
            "min": round(float(df["lvef_percent"].min()), 2),
            "max": round(float(df["lvef_percent"].max()), 2),
        },
        "rvedv_ml": {
            "mean": round(float(df["rvedv_ml"].mean()), 2),
            "std": round(float(df["rvedv_ml"].std()), 2),
            "median": round(float(df["rvedv_ml"].median()), 2),
            "min": round(float(df["rvedv_ml"].min()), 2),
            "max": round(float(df["rvedv_ml"].max()), 2),
        },
        "rvef_percent": {
            "mean": round(float(df["rvef_percent"].mean()), 2),
            "std": round(float(df["rvef_percent"].std()), 2),
            "median": round(float(df["rvef_percent"].median()), 2),
            "min": round(float(df["rvef_percent"].min()), 2),
            "max": round(float(df["rvef_percent"].max()), 2),
        },
        "myocardial_mass_ed_g": {
            "mean": round(float(df["myocardial_mass_ed_g"].mean()), 2),
            "std": round(float(df["myocardial_mass_ed_g"].std()), 2),
            "median": round(float(df["myocardial_mass_ed_g"].median()), 2),
            "min": round(float(df["myocardial_mass_ed_g"].min()), 2),
            "max": round(float(df["myocardial_mass_ed_g"].max()), 2),
        },
        "by_diagnosis_group": {},
    }

    for grp, grp_df in df.groupby("diagnostic_group"):
        summary["by_diagnosis_group"][str(grp)] = {
            "count": int(len(grp_df)),
            "lvedv_ml_mean": round(float(grp_df["lvedv_ml"].mean()), 2),
            "lvedv_ml_std": round(float(grp_df["lvedv_ml"].std()), 2),
            "lvesv_ml_mean": round(float(grp_df["lvesv_ml"].mean()), 2),
            "lvesv_ml_std": round(float(grp_df["lvesv_ml"].std()), 2),
            "lvef_percent_mean": round(float(grp_df["lvef_percent"].mean()), 2),
            "lvef_percent_std": round(float(grp_df["lvef_percent"].std()), 2),
            "lvef_percent_median": round(float(grp_df["lvef_percent"].median()), 2),
            "lvef_percent_iqr": round(float(grp_df["lvef_percent"].quantile(0.75) - grp_df["lvef_percent"].quantile(0.25)), 2),
            "rvedv_ml_mean": round(float(grp_df["rvedv_ml"].mean()), 2),
            "rvef_percent_mean": round(float(grp_df["rvef_percent"].mean()), 2),
            "myo_mass_ed_g_mean": round(float(grp_df["myocardial_mass_ed_g"].mean()), 2),
        }

    return summary


def generate_cohort_plots(
    df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Generate reproducible descriptive plots for the ACDC cohort."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # 1. Distribution of LVEF
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["lvef_percent"], bins=15, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax.axvline(50.0, color="crimson", linestyle="--", linewidth=1.5, label="Normal EF Threshold (50%)")
    ax.set_title("Distribution of Left Ventricular Ejection Fraction (LVEF)", fontsize=12)
    ax.set_xlabel("LVEF (%)", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p1 = output_dir / "distribution_lvef.png"
    plt.savefig(p1, dpi=150)
    plt.close(fig)
    generated_files.append(p1)

    # 2. Distribution of LVEDV
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["lvedv_ml"], bins=15, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax.set_title("Distribution of End-Diastolic Volume (LVEDV)", fontsize=12)
    ax.set_xlabel("LVEDV (mL)", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p2 = output_dir / "distribution_lvedv.png"
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    generated_files.append(p2)

    # 3. LVEF by Diagnostic Category
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = sorted(df["diagnostic_group"].unique())
    group_data = [df[df["diagnostic_group"] == g]["lvef_percent"].values for g in groups]
    box = ax.boxplot(group_data, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1, label="50% Threshold")
    ax.set_title("LVEF Stratified by ACDC Diagnostic Category", fontsize=12)
    ax.set_xlabel("Diagnostic Group", fontsize=11)
    ax.set_ylabel("LVEF (%)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p3 = output_dir / "lvef_by_diagnostic_group.png"
    plt.savefig(p3, dpi=150)
    plt.close(fig)
    generated_files.append(p3)

    # 4. LVEDV versus LVEF Scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_colors = {"DCM": "#d62728", "HCM": "#9467bd", "MINF": "#ff7f0e", "NOR": "#2ca02c", "RV": "#1f77b4"}
    for g in groups:
        subset = df[df["diagnostic_group"] == g]
        ax.scatter(
            subset["lvedv_ml"],
            subset["lvef_percent"],
            label=f"{g} (n={len(subset)})",
            color=scatter_colors.get(g, "gray"),
            alpha=0.8,
            s=50,
            edgecolors="none",
        )
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("End-Diastolic Volume (LVEDV) vs. Ejection Fraction (LVEF)", fontsize=12)
    ax.set_xlabel("LVEDV (mL)", fontsize=11)
    ax.set_ylabel("LVEF (%)", fontsize=11)
    ax.legend(title="ACDC Group", loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p4 = output_dir / "lvedv_vs_lvef.png"
    plt.savefig(p4, dpi=150)
    plt.close(fig)
    generated_files.append(p4)

    return generated_files


def run_pipeline(
    data_dir: Path | str = DEFAULT_ACDC_TRAINING_DIR,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    generate_figures: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete patient and cohort measurement extraction."""
    root = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = ACDCDataLoader(root)
    patient_ids = loader.list_patients()
    logger.info(f"Discovered {len(patient_ids)} patients in {root}")

    successful_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []

    for pid in patient_ids:
        try:
            rec = process_single_patient(loader, pid)
            successful_records.append(rec)
        except Exception as e:
            logger.error(f"Failed processing {pid}: {e}")
            failed_records.append({"patient_id": pid, "error": str(e)})

    df = pd.DataFrame(successful_records)

    # Export CSV and JSON
    csv_path = out_dir / "cardiac_measurements.csv"
    json_path = out_dir / "cardiac_measurements.json"
    report_path = out_dir / "measurement_validation_report.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(successful_records, f, indent=2)

    cohort_stats = compute_cohort_descriptive_stats(df)

    validation_report = {
        "pipeline_status": "SUCCESS" if not failed_records else "PARTIAL_SUCCESS",
        "dataset_path": str(root.resolve()),
        "total_discovered": len(patient_ids),
        "total_processed_successfully": len(successful_records),
        "total_failed": len(failed_records),
        "failed_patients": failed_records,
        "cohort_statistics": cohort_stats,
        "files_generated": [
            str(csv_path),
            str(json_path),
            str(report_path),
        ],
        "notes": (
            "All measurements computed strictly from expert-provided ACDC ground-truth masks. "
            "Absolute EF accuracy against external clinical consensus cannot be established "
            "independently as external scalar reference measurements are not included in the raw ACDC files."
        ),
    }

    if generate_figures and not df.empty:
        figs_dir = out_dir / "figures"
        docs_figs_dir = Path("docs/acdc/figures")
        gen_figs = generate_cohort_plots(df, figs_dir)
        generate_cohort_plots(df, docs_figs_dir)
        validation_report["figures_generated"] = [str(p) for p in gen_figs]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info(f"Results exported to {csv_path} and {json_path}")
    logger.info(f"Validation report saved to {report_path}")

    return df, validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="ACDC Cardiac Measurement and Cohort Pipeline.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_ACDC_TRAINING_DIR),
        help="Path to ACDC training dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Path to output processed directory",
    )
    args = parser.parse_args()

    df, report = run_pipeline(args.data_dir, args.output_dir)
    print("\n" + "=" * 70)
    print("      ACDC COHORT MEASUREMENT EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Discovered Patients:  {report['total_discovered']}")
    print(f"Successfully Measured:      {report['total_processed_successfully']}")
    print(f"Failed Cases:               {report['total_failed']}")
    if not df.empty:
        stats = report["cohort_statistics"]
        print("-" * 70)
        print(f"LVEDV (mL): Mean = {stats['lvedv_ml']['mean']} ± {stats['lvedv_ml']['std']} | Range = [{stats['lvedv_ml']['min']}, {stats['lvedv_ml']['max']}]")
        print(f"LVESV (mL): Mean = {stats['lvesv_ml']['mean']} ± {stats['lvesv_ml']['std']} | Range = [{stats['lvesv_ml']['min']}, {stats['lvesv_ml']['max']}]")
        print(f"LVEF (%):   Mean = {stats['lvef_percent']['mean']} ± {stats['lvef_percent']['std']} | Median (IQR) = {stats['lvef_percent']['median']} ({stats['lvef_percent']['iqr']})")
        print(f"RVEDV (mL): Mean = {stats['rvedv_ml']['mean']} ± {stats['rvedv_ml']['std']}")
        print(f"RVEF (%):   Mean = {stats['rvef_percent']['mean']} ± {stats['rvef_percent']['std']}")
        print(f"Myo Mass:   Mean = {stats['myocardial_mass_ed_g']['mean']} ± {stats['myocardial_mass_ed_g']['std']} g")
        print("-" * 70)
        print("Breakdown by Diagnostic Group:")
        for grp, gstats in stats["by_diagnosis_group"].items():
            print(f"  [{grp:<4} - {gstats['count']} pts] LVEF: {gstats['lvef_percent_mean']}±{gstats['lvef_percent_std']}% | LVEDV: {gstats['lvedv_ml_mean']}±{gstats['lvedv_ml_std']} mL | Myo Mass: {gstats['myo_mass_ed_g_mean']} g")
    print("=" * 70)


if __name__ == "__main__":
    main()
