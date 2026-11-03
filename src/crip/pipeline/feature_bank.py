"""Unified Empirical Feature Bank for Multi-Dataset Cardiac CMR.

Aggregates, standardizes, and stores expert-derived empirical features across:
- ACDC (100 cases, Cine SSFP, biventricular volumetrics & function)
- M&Ms (114 paired cases, multi-vendor Cine SSFP, biventricular function & demographics)
- EMIDEC (100 cases, static LGE CMR, myocardial viability, scar & MVO burdens)

Strict Invariants:
1. Zero synthetic blending: retains strict patient provenance; never merges across datasets.
2. Ground-truth fidelity: uses expert-annotated masks exclusively, separating empirical priors
   from model inference errors.
3. Modality-specific validity: marks static LGE scans as lacking dynamic LVEF, and cine scans
   as lacking delayed-enhancement tissue viability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from crip.datasets.mnms_loader import MNMSDataLoader
from crip.pipeline.schemas import EmpiricalFeatureRecord, Modality

logger = logging.getLogger("crip.pipeline.feature_bank")

DEFAULT_ACDC_CSV = Path("data/processed/acdc/cardiac_measurements.csv")
DEFAULT_MNMS_CSV = Path("data/processed/mnms/external_evaluation.csv")
DEFAULT_EMIDEC_CSV = Path("data/processed/emidec/tissue_measurements.csv")

DEFAULT_OUTPUT_CSV = Path("data/processed/unified/empirical_feature_bank.csv")
DEFAULT_OUTPUT_SUMMARY = Path("data/processed/unified/empirical_feature_bank_summary.json")


def extract_acdc_features(acdc_csv: Path = DEFAULT_ACDC_CSV) -> List[EmpiricalFeatureRecord]:
    """Extract standardized empirical records from ACDC expert measurements."""
    if not acdc_csv.is_file():
        raise FileNotFoundError(f"ACDC measurements CSV not found: {acdc_csv}")

    df = pd.read_csv(acdc_csv)
    records: List[EmpiricalFeatureRecord] = []

    for _, row in df.iterrows():
        rec = EmpiricalFeatureRecord(
            case_id=str(row["patient_id"]),
            dataset_id="acdc",
            modality=Modality.CINE_CMR,
            mask_provenance="expert_ground_truth",
            measurement_method="Simpson_rule_voxel_integration",
            diagnostic_group=str(row.get("diagnostic_group")),
            scanner_vendor="Siemens",
            center="Dijon_University_Hospital",
            magnetic_field_t=None,  # 1.5T or 3.0T Siemens Magnetom
            lvedv_ml=float(row["lvedv_ml"]) if pd.notna(row.get("lvedv_ml")) else None,
            lvesv_ml=float(row["lvesv_ml"]) if pd.notna(row.get("lvesv_ml")) else None,
            lvsv_ml=float(row["lvsv_ml"]) if pd.notna(row.get("lvsv_ml")) else None,
            lvef_percent=float(row["lvef_percent"]) if pd.notna(row.get("lvef_percent")) else None,
            rvedv_ml=float(row["rvedv_ml"]) if pd.notna(row.get("rvedv_ml")) else None,
            rvesv_ml=float(row["rvesv_ml"]) if pd.notna(row.get("rvesv_ml")) else None,
            rvsv_ml=float(row["rvsv_ml"]) if pd.notna(row.get("rvsv_ml")) else None,
            rvef_percent=float(row["rvef_percent"]) if pd.notna(row.get("rvef_percent")) else None,
            myocardial_mass_g=float(row["myocardial_mass_ed_g"]) if pd.notna(row.get("myocardial_mass_ed_g")) else None,
            normal_myo_vol_ml=float(row["myocardial_volume_ed_ml"]) if pd.notna(row.get("myocardial_volume_ed_ml")) else None,
            total_myo_vol_ml=float(row["myocardial_volume_ed_ml"]) if pd.notna(row.get("myocardial_volume_ed_ml")) else None,
            height_cm=float(row["height_cm"]) if pd.notna(row.get("height_cm")) else None,
            weight_kg=float(row["weight_kg"]) if pd.notna(row.get("weight_kg")) else None,
            is_cine_complete=True,
            is_viability_complete=False,
            is_clinical_complete=pd.notna(row.get("height_cm")) and pd.notna(row.get("weight_kg")),
        )
        records.append(rec)

    return records


def extract_mnms_features(
    mnms_csv: Path = DEFAULT_MNMS_CSV,
    mnms_loader: Optional[MNMSDataLoader] = None,
) -> List[EmpiricalFeatureRecord]:
    """Extract standardized empirical records from M&Ms ground-truth measurements."""
    if not mnms_csv.is_file():
        raise FileNotFoundError(f"M&Ms evaluation CSV not found: {mnms_csv}")

    loader = mnms_loader or MNMSDataLoader()
    cases = loader.discover_cases()
    df = pd.read_csv(mnms_csv)
    records: List[EmpiricalFeatureRecord] = []

    for _, row in df.iterrows():
        cid = str(row["case_id"])
        c_info = cases.get(cid, {})
        csv_meta = c_info.get("csv_metadata", {})

        age = float(csv_meta.get("Age")) if csv_meta.get("Age") is not None and not pd.isna(csv_meta.get("Age")) else None
        sex = str(csv_meta.get("Sex")) if csv_meta.get("Sex") is not None and not pd.isna(csv_meta.get("Sex")) else None
        height = float(csv_meta.get("Height")) if csv_meta.get("Height") is not None and not pd.isna(csv_meta.get("Height")) else None
        weight = float(csv_meta.get("Weight")) if csv_meta.get("Weight") is not None and not pd.isna(csv_meta.get("Weight")) else None

        rec = EmpiricalFeatureRecord(
            case_id=cid,
            dataset_id="mnms",
            modality=Modality.CINE_CMR,
            mask_provenance="expert_ground_truth",
            measurement_method="Simpson_rule_voxel_integration",
            diagnostic_group=str(row.get("diagnosis", "")),
            scanner_vendor=str(row.get("vendor", "")),
            center=str(row.get("center", "")),
            magnetic_field_t=None,
            lvedv_ml=float(row["lvedv_gt_ml"]) if pd.notna(row.get("lvedv_gt_ml")) else None,
            lvesv_ml=float(row["lvesv_gt_ml"]) if pd.notna(row.get("lvesv_gt_ml")) else None,
            lvsv_ml=float(row["lvsv_gt_ml"]) if pd.notna(row.get("lvsv_gt_ml")) else None,
            lvef_percent=float(row["lvef_gt_percent"]) if pd.notna(row.get("lvef_gt_percent")) else None,
            rvedv_ml=float(row["rvedv_gt_ml"]) if pd.notna(row.get("rvedv_gt_ml")) else None,
            rvesv_ml=float(row["rvesv_gt_ml"]) if pd.notna(row.get("rvesv_gt_ml")) else None,
            rvsv_ml=float(row["rvsv_gt_ml"]) if pd.notna(row.get("rvsv_gt_ml")) else None,
            rvef_percent=float(row["rvef_gt_percent"]) if pd.notna(row.get("rvef_gt_percent")) else None,
            myocardial_mass_g=float(row["myo_mass_ed_gt_g"]) if pd.notna(row.get("myo_mass_ed_gt_g")) else None,
            age=age,
            sex=sex,
            height_cm=height,
            weight_kg=weight,
            is_cine_complete=True,
            is_viability_complete=False,
            is_clinical_complete=age is not None and sex is not None,
        )
        records.append(rec)

    return records


def extract_emidec_features(emidec_csv: Path = DEFAULT_EMIDEC_CSV) -> List[EmpiricalFeatureRecord]:
    """Extract standardized empirical records from EMIDEC expert viability measurements."""
    if not emidec_csv.is_file():
        raise FileNotFoundError(f"EMIDEC measurements CSV not found: {emidec_csv}")

    df = pd.read_csv(emidec_csv)
    records: List[EmpiricalFeatureRecord] = []

    for _, row in df.iterrows():
        age = float(row["age"]) if pd.notna(row.get("age")) else None
        sex = str(row["sex"]) if pd.notna(row.get("sex")) else None
        htn = bool(row["hypertension"]) if pd.notna(row.get("hypertension")) else None
        dm = bool(row["diabetes"]) if pd.notna(row.get("diabetes")) else None
        fam_cad = bool(row["cad_family_history"]) if pd.notna(row.get("cad_family_history")) else None

        rec = EmpiricalFeatureRecord(
            case_id=str(row["case_id"]),
            dataset_id="emidec",
            modality=Modality.LGE_CMR,
            mask_provenance="expert_ground_truth",
            measurement_method="Voxel_integration",
            diagnostic_group=str(row.get("pathology_group")),
            scanner_vendor="Siemens",
            center="Dijon_University_Hospital",
            magnetic_field_t=None,
            normal_myo_vol_ml=float(row["normal_myocardium_volume_ml"]) if pd.notna(row.get("normal_myocardium_volume_ml")) else None,
            infarct_vol_ml=float(row["infarction_volume_ml"]) if pd.notna(row.get("infarction_volume_ml")) else None,
            no_reflow_vol_ml=float(row["no_reflow_volume_ml"]) if pd.notna(row.get("no_reflow_volume_ml")) else None,
            total_myo_vol_ml=float(row["total_myocardium_volume_ml"]) if pd.notna(row.get("total_myocardium_volume_ml")) else None,
            infarct_burden_percent=float(row["infarction_burden_percent"]) if pd.notna(row.get("infarction_burden_percent")) else None,
            no_reflow_burden_percent=float(row["no_reflow_burden_percent"]) if pd.notna(row.get("no_reflow_burden_percent")) else None,
            has_microvascular_obstruction=bool(row["has_microvascular_obstruction"]) if pd.notna(row.get("has_microvascular_obstruction")) else None,
            myocardial_mass_g=float(row["estimated_myocardial_mass_g"]) if pd.notna(row.get("estimated_myocardial_mass_g")) else None,
            age=age,
            sex=sex,
            hypertension=htn,
            diabetes=dm,
            cad_family_history=fam_cad,
            is_cine_complete=False,
            is_viability_complete=True,
            is_clinical_complete=age is not None and sex is not None,
        )
        records.append(rec)

    return records


def build_unified_feature_bank(
    acdc_csv: Path = DEFAULT_ACDC_CSV,
    mnms_csv: Path = DEFAULT_MNMS_CSV,
    emidec_csv: Path = DEFAULT_EMIDEC_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assemble the unified empirical feature bank across all three datasets."""
    acdc_records = extract_acdc_features(acdc_csv)
    mnms_records = extract_mnms_features(mnms_csv)
    emidec_records = extract_emidec_features(emidec_csv)

    all_records = acdc_records + mnms_records + emidec_records
    dicts = [r.to_dict() for r in all_records]
    df = pd.DataFrame(dicts)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved unified empirical feature bank ({len(df)} records) to {output_csv}")

    # Build cohort summary
    summary = {
        "total_empirical_records": len(df),
        "cohort_breakdown": {
            "acdc_cases": len(acdc_records),
            "mnms_cases": len(mnms_records),
            "emidec_cases": len(emidec_records),
        },
        "modalities": {
            "cine_cmr_cases": int((df["modality"] == Modality.CINE_CMR.value).sum()),
            "lge_cmr_cases": int((df["modality"] == Modality.LGE_CMR.value).sum()),
        },
        "empirical_priors": {
            "cine": {
                "all_cine_cases": int((df["is_cine_complete"] == True).sum()),
                "lvedv_ml_mean": float(df["lvedv_ml"].dropna().mean()),
                "lvedv_ml_std": float(df["lvedv_ml"].dropna().std()),
                "lvef_percent_mean": float(df["lvef_percent"].dropna().mean()),
                "lvef_percent_std": float(df["lvef_percent"].dropna().std()),
                "myo_mass_g_mean": float(df["myocardial_mass_g"].dropna().mean()),
                "myo_mass_g_std": float(df["myocardial_mass_g"].dropna().std()),
            },
            "lge_viability": {
                "all_viability_cases": int((df["is_viability_complete"] == True).sum()),
                "pathological_cases": int(((df["dataset_id"] == "emidec") & (df["diagnostic_group"] != "Normal")).sum()),
                "infarct_volume_ml_mean_pathological": float(
                    df[(df["dataset_id"] == "emidec") & (df["diagnostic_group"] != "Normal")]["infarct_vol_ml"].dropna().mean()
                ),
                "infarct_burden_percent_mean_pathological": float(
                    df[(df["dataset_id"] == "emidec") & (df["diagnostic_group"] != "Normal")]["infarct_burden_percent"].dropna().mean()
                ),
                "mvo_prevalence_rate_pathological": float(
                    (df[(df["dataset_id"] == "emidec") & (df["diagnostic_group"] != "Normal")]["has_microvascular_obstruction"] == True).mean()
                ),
            },
        },
        "provenance": {
            "mask_provenance": "expert_ground_truth_only",
            "leakage_protection": "zero_cross_dataset_merging",
        },
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved feature bank summary to {output_summary}")
    return df, summary
