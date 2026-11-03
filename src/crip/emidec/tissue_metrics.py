"""Delayed-Enhancement Cardiac MRI Tissue Characterization Metrics.

Quantifies Left Ventricular (LV) blood pool, normal myocardium, myocardial infarction
(scar hyperenhancement), and microvascular obstruction (no-reflow core) from Late
Gadolinium Enhancement (LGE) CMR in the EMIDEC (2020) cohort.

Mathematical Definitions:
-------------------------
1. Physical Voxel Volume:
     V_voxel = (spacing_x * spacing_y * spacing_z) / 1000.0  [mL, where 1000 mm^3 = 1 mL]

2. Segmented Structure Volumes:
     V_structure = N_voxels * V_voxel  [mL]

3. Total Myocardial Volume:
     V_total_myo = V_normal_myo + V_infarct + V_no_reflow  [mL]
   Note: Left ventricular cavity blood pool (Label 1) is cavity, NOT myocardium.

4. Tissue Burden Percentages (Explicit Denominator: Total Annotated Myocardium):
     Infarction Burden (%) = (V_infarct / V_total_myo) * 100.0
     No-Reflow Burden (%)   = (V_no_reflow / V_total_myo) * 100.0
     Combined Abnormal (%)  = ((V_infarct + V_no_reflow) / V_total_myo) * 100.0

5. Estimated Myocardial Mass:
     Mass_myo = V_total_myo * 1.05  [g, AHA/SCMR density consensus 1.05 g/mL]

CLINICAL GOVERNANCE WARNING:
----------------------------
1. EMIDEC acquisitions are static 3D delayed-enhancement volumes without cine frames.
   Computing dynamic Left Ventricular Ejection Fraction (LVEF) from EMIDEC is medically
   invalid and strictly disallowed.
2. Ischemic myocardial infarction (focal subendocardial/transmural hyperenhancement
   following an epicardial coronary artery territory) MUST NOT be equated with or
   confused with alcohol-related non-ischemic cardiomyopathy (ACM) fibrosis, which
   exhibits non-ischemic mid-wall, diffuse, or patchy replacement scar patterns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from crip.datasets.emidec_loader import EMIDECDataLoader
from crip.datasets.labels import EMIDECLabel

logger = logging.getLogger("crip.emidec.tissue_metrics")

# Standard clinical tissue density constant for myocardium (g/mL or g/cm³)
MYOCARDIAL_DENSITY_G_PER_ML: float = 1.05

DEFAULT_MEASUREMENTS_CSV = Path("data/processed/emidec/tissue_measurements.csv")
DEFAULT_MEASUREMENTS_SUMMARY = Path("data/processed/emidec/tissue_measurements_summary.json")


@dataclass
class EMIDECTissueMeasurements:
    """Strongly-typed quantitative tissue characterization record for an EMIDEC case."""
    case_id: str
    pathology_group: str  # "Normal" or "Myocardial Infarction"
    voxel_spacing_mm: Tuple[float, float, float]
    voxel_volume_mm3: float
    # Volumes in milliliters (mL)
    lv_cavity_volume_ml: float
    normal_myocardium_volume_ml: float
    infarction_volume_ml: float
    no_reflow_volume_ml: float
    total_myocardium_volume_ml: float
    estimated_myocardial_mass_g: float
    # Tissue Burdens (% of Total Myocardium)
    infarction_burden_percent: float
    no_reflow_burden_percent: float
    combined_abnormal_burden_percent: float
    # Diagnostic flags
    has_microvascular_obstruction: bool
    voxel_counts: Dict[str, int] = field(default_factory=dict)
    is_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert measurements to serializable dictionary."""
        d = asdict(self)
        d["voxel_spacing_mm"] = list(self.voxel_spacing_mm)
        return d


def validate_physical_spacing(spacing: Sequence[float]) -> Tuple[float, float, float]:
    """Validate that voxel spacing sequence contains 3 positive finite numbers."""
    if len(spacing) < 3:
        raise ValueError(f"Voxel spacing must have at least 3 dimensions, got {spacing}")
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    for dim_name, val in [("spacing_x", sx), ("spacing_y", sy), ("spacing_z", sz)]:
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"Non-finite {dim_name}: {val}")
        if val <= 0.0:
            raise ValueError(f"Non-positive {dim_name}: {val}. Voxel dimensions must be > 0.")
    return sx, sy, sz


def compute_emidec_tissue_metrics(
    mask_3d: np.ndarray,
    voxel_spacing_mm: Sequence[float],
    case_id: str = "UNKNOWN",
    pathology_group: str = "UNKNOWN",
) -> EMIDECTissueMeasurements:
    """Calculate multi-component tissue volumes and relative burdens from an EMIDEC contour mask.

    Labels:
      - 0: Background
      - 1: Left Ventricular Cavity
      - 2: Normal Myocardium
      - 3: Myocardial Infarction (Scar / Hyperenhancement)
      - 4: Microvascular Obstruction (No-Reflow Core)

    Args:
        mask_3d: 3D integer array of segmentation labels.
        voxel_spacing_mm: Voxel spacing (dx, dy, dz) in millimeters.
        case_id: Unique case identifier.
        pathology_group: Clinical cohort ('Normal' or 'Myocardial Infarction').

    Returns:
        EMIDECTissueMeasurements object with exact volumes, burdens, and validations.
    """
    if not isinstance(mask_3d, np.ndarray) or mask_3d.ndim != 3:
        raise ValueError(f"mask_3d must be a 3D NumPy array, got shape {getattr(mask_3d, 'shape', None)}")

    sx, sy, sz = validate_physical_spacing(voxel_spacing_mm)
    voxel_volume_mm3 = sx * sy * sz
    voxel_volume_ml = voxel_volume_mm3 / 1000.0

    warnings: List[str] = []

    # Count voxels per label
    count_bg = int(np.count_nonzero(mask_3d == int(EMIDECLabel.BACKGROUND)))
    count_lv = int(np.count_nonzero(mask_3d == int(EMIDECLabel.LV_CAVITY)))
    count_norm_myo = int(np.count_nonzero(mask_3d == int(EMIDECLabel.NORMAL_MYOCARDIUM)))
    count_infarct = int(np.count_nonzero(mask_3d == int(EMIDECLabel.MYOCARDIAL_INFARCTION)))
    count_no_reflow = int(np.count_nonzero(mask_3d == int(EMIDECLabel.NO_REFLOW)))

    voxel_counts = {
        "background": count_bg,
        "lv_cavity": count_lv,
        "normal_myocardium": count_norm_myo,
        "myocardial_infarction": count_infarct,
        "no_reflow": count_no_reflow,
    }

    # Check for unrecognized labels
    valid_labels = {0, 1, 2, 3, 4}
    present_labels = set(np.unique(mask_3d).astype(int).tolist())
    extra_labels = present_labels - valid_labels
    if extra_labels:
        warnings.append(f"Unrecognized segmentation labels present: {sorted(list(extra_labels))}")

    # Calculate volumes in mL
    lv_vol_ml = count_lv * voxel_volume_ml
    norm_myo_vol_ml = count_norm_myo * voxel_volume_ml
    infarct_vol_ml = count_infarct * voxel_volume_ml
    no_reflow_vol_ml = count_no_reflow * voxel_volume_ml

    # Total annotated myocardium = Normal Myo + Infarct + No-Reflow
    # Excludes LV Cavity (blood pool)
    total_myo_vol_ml = norm_myo_vol_ml + infarct_vol_ml + no_reflow_vol_ml
    est_myo_mass_g = total_myo_vol_ml * MYOCARDIAL_DENSITY_G_PER_ML

    # Burdens as percentage of Total Myocardium
    if total_myo_vol_ml > 0.0:
        infarct_burden = (infarct_vol_ml / total_myo_vol_ml) * 100.0
        no_reflow_burden = (no_reflow_vol_ml / total_myo_vol_ml) * 100.0
        combined_abnormal_burden = ((infarct_vol_ml + no_reflow_vol_ml) / total_myo_vol_ml) * 100.0
    else:
        infarct_burden = 0.0
        no_reflow_burden = 0.0
        combined_abnormal_burden = 0.0
        warnings.append("Total annotated myocardial volume is zero; tissue burdens set to 0.0%")

    has_mvo = count_no_reflow > 0

    # Sanity checks on pathological vs normal expectations
    if pathology_group == "Normal" and (count_infarct > 0 or count_no_reflow > 0):
        warnings.append(f"Normal control case contains non-zero abnormal tissue (infarct={count_infarct}, MVO={count_no_reflow})")

    is_valid = len([w for w in warnings if "zero" in w.lower() or "unrecognized" in w.lower()]) == 0

    return EMIDECTissueMeasurements(
        case_id=case_id,
        pathology_group=pathology_group,
        voxel_spacing_mm=(sx, sy, sz),
        voxel_volume_mm3=voxel_volume_mm3,
        lv_cavity_volume_ml=lv_vol_ml,
        normal_myocardium_volume_ml=norm_myo_vol_ml,
        infarction_volume_ml=infarct_vol_ml,
        no_reflow_volume_ml=no_reflow_vol_ml,
        total_myocardium_volume_ml=total_myo_vol_ml,
        estimated_myocardial_mass_g=est_myo_mass_g,
        infarction_burden_percent=infarct_burden,
        no_reflow_burden_percent=no_reflow_burden,
        combined_abnormal_burden_percent=combined_abnormal_burden,
        has_microvascular_obstruction=has_mvo,
        voxel_counts=voxel_counts,
        is_valid=is_valid,
        validation_warnings=warnings,
    )


class EMIDECTissueAnalyzer:
    """Orchestrates comprehensive cohort tissue characterization across EMIDEC."""

    def __init__(self, loader: Optional[EMIDECDataLoader] = None) -> None:
        self.loader = loader or EMIDECDataLoader()

    def analyze_cohort(self) -> pd.DataFrame:
        """Process all 100 annotated training cases and compile into a DataFrame."""
        cases = self.loader.discover_cases()
        annotated_cases = {
            cid: c for cid, c in cases.items()
            if c["split"] == "train" and c["has_contour"] and c["contour_path"] is not None
        }

        logger.info(f"Computing tissue metrics for {len(annotated_cases)} annotated EMIDEC cases...")
        records: List[Dict[str, Any]] = []

        for cid, c_info in sorted(annotated_cases.items()):
            case_data = self.loader.load_case(cid)
            mask_3d = case_data["contour"]
            zooms = case_data["voxel_spacing"]
            pathology = case_data["pathology"]
            clin = case_data.get("clinical_metadata", {})

            metrics = compute_emidec_tissue_metrics(
                mask_3d=mask_3d,
                voxel_spacing_mm=zooms,
                case_id=cid,
                pathology_group=pathology,
            )

            rec = metrics.to_dict()
            # Flatten clinical metadata into record
            rec["sex"] = clin.get("sex")
            rec["age"] = clin.get("age")
            rec["tobacco_tier"] = clin.get("tobacco_tier")
            rec["overweight"] = clin.get("overweight")
            rec["hypertension"] = clin.get("hypertension")
            rec["diabetes"] = clin.get("diabetes")
            rec["cad_family_history"] = clin.get("cad_family_history")
            rec["ecg_st_elevation"] = clin.get("ecg_st_elevation")
            rec["troponin"] = clin.get("troponin")
            rec["killip_max"] = clin.get("killip_max")
            rec["lvef_reference_percent"] = clin.get("lvef_reference_percent")
            rec["nt_probnp"] = clin.get("nt_probnp")
            rec["slice_gap_reported"] = clin.get("slice_gap_reported")

            records.append(rec)

        df = pd.DataFrame(records)
        return df

    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary separating Normal controls from Pathological infarct cases."""
        def stats(s: pd.Series) -> Dict[str, float]:
            s_clean = s.dropna()
            if s_clean.empty:
                return {"mean": 0.0, "median": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0}
            return {
                "mean": float(s_clean.mean()),
                "median": float(s_clean.median()),
                "std": float(s_clean.std()),
                "p25": float(s_clean.quantile(0.25)),
                "p75": float(s_clean.quantile(0.75)),
                "min": float(s_clean.min()),
                "max": float(s_clean.max()),
            }

        total_cases = len(df)
        normal_df = df[df["pathology_group"] == "Normal"]
        path_df = df[df["pathology_group"] == "Myocardial Infarction"]

        summary = {
            "analysis_date": "2026-11-02",
            "dataset_id": "emidec",
            "cohort_total": total_cases,
            "normal_controls_count": len(normal_df),
            "pathological_infarct_count": len(path_df),
            "mvo_prevalence_in_infarct": {
                "cases_with_mvo": int(path_df["has_microvascular_obstruction"].sum()),
                "mvo_rate_percent": float(path_df["has_microvascular_obstruction"].mean() * 100.0),
            },
            "overall_cohort": {
                "total_myocardium_volume_ml": stats(df["total_myocardium_volume_ml"]),
                "lv_cavity_volume_ml": stats(df["lv_cavity_volume_ml"]),
                "infarction_volume_ml": stats(df["infarction_volume_ml"]),
                "no_reflow_volume_ml": stats(df["no_reflow_volume_ml"]),
                "infarction_burden_percent": stats(df["infarction_burden_percent"]),
                "no_reflow_burden_percent": stats(df["no_reflow_burden_percent"]),
                "combined_abnormal_burden_percent": stats(df["combined_abnormal_burden_percent"]),
            },
            "normal_controls": {
                "total_myocardium_volume_ml": stats(normal_df["total_myocardium_volume_ml"]),
                "lv_cavity_volume_ml": stats(normal_df["lv_cavity_volume_ml"]),
                "normal_myocardium_volume_ml": stats(normal_df["normal_myocardium_volume_ml"]),
                "infarction_volume_ml": stats(normal_df["infarction_volume_ml"]),
                "no_reflow_volume_ml": stats(normal_df["no_reflow_volume_ml"]),
                "infarction_burden_percent": stats(normal_df["infarction_burden_percent"]),
            },
            "pathological_cohort": {
                "total_myocardium_volume_ml": stats(path_df["total_myocardium_volume_ml"]),
                "lv_cavity_volume_ml": stats(path_df["lv_cavity_volume_ml"]),
                "normal_myocardium_volume_ml": stats(path_df["normal_myocardium_volume_ml"]),
                "infarction_volume_ml": stats(path_df["infarction_volume_ml"]),
                "no_reflow_volume_ml": stats(path_df["no_reflow_volume_ml"]),
                "infarction_burden_percent": stats(path_df["infarction_burden_percent"]),
                "no_reflow_burden_percent": stats(path_df["no_reflow_burden_percent"]),
                "combined_abnormal_burden_percent": stats(path_df["combined_abnormal_burden_percent"]),
            },
        }

        return summary

    def run_and_save(
        self,
        output_csv: Path | str = DEFAULT_MEASUREMENTS_CSV,
        output_summary: Path | str = DEFAULT_MEASUREMENTS_SUMMARY,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute metrics across cohort, save CSV and JSON summary."""
        csv_path = Path(output_csv)
        summary_path = Path(output_summary)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.analyze_cohort()
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved EMIDEC tissue measurements to {csv_path}")

        summary = self.generate_summary(df)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved EMIDEC summary JSON to {summary_path}")

        return df, summary


def main() -> None:
    """CLI entry point for EMIDEC tissue characterization."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    analyzer = EMIDECTissueAnalyzer()
    df, summary = analyzer.run_and_save()

    print("\n================== EMIDEC Tissue Characterization Complete ==================")
    print(f"Total Annotated Cases: {len(df)}")
    print(f"Normal Controls:       {summary['normal_controls_count']}")
    print(f"Pathological Infarct:  {summary['pathological_infarct_count']}")
    print(f"MVO Prevalence:        {summary['mvo_prevalence_in_infarct']['cases_with_mvo']} / {summary['pathological_infarct_count']} ({summary['mvo_prevalence_in_infarct']['mvo_rate_percent']:.1f}%)")
    
    path_stats = summary["pathological_cohort"]
    print("\nPathological Cohort Tissue Metrics:")
    print(f"  Total Myocardium Volume: {path_stats['total_myocardium_volume_ml']['mean']:.2f} ± {path_stats['total_myocardium_volume_ml']['std']:.2f} mL")
    print(f"  Infarction Volume:       {path_stats['infarction_volume_ml']['mean']:.2f} ± {path_stats['infarction_volume_ml']['std']:.2f} mL (Median: {path_stats['infarction_volume_ml']['median']:.2f} mL)")
    print(f"  No-Reflow (MVO) Volume:  {path_stats['no_reflow_volume_ml']['mean']:.2f} ± {path_stats['no_reflow_volume_ml']['std']:.2f} mL (Max: {path_stats['no_reflow_volume_ml']['max']:.2f} mL)")
    print(f"  Infarct Burden (% Myo):  {path_stats['infarction_burden_percent']['mean']:.2f}% ± {path_stats['infarction_burden_percent']['std']:.2f}% (Median: {path_stats['infarction_burden_percent']['median']:.2f}%)")
    print(f"  Combined Abnormal (%):   {path_stats['combined_abnormal_burden_percent']['mean']:.2f}% ± {path_stats['combined_abnormal_burden_percent']['std']:.2f}%\n")


if __name__ == "__main__":
    main()
