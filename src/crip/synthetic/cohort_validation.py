"""Synthetic Cohort Quality Evaluation and Statistical Validation Engine.

Conducts rigorous automated audits of generated synthetic cardiomyopathy cohorts:
  - Completeness, schema conformity, duplicate detection, and boundary checks
  - Univariate distribution summaries (demographics, CMR volumetrics, PEth, TTN, LGE)
  - Subgroup recovery stratification (by abstinence, PEth tier, LGE status, TTN status)
  - Non-trivial separability verification (univariate ROC-AUC checks to prevent trivial single-feature rules)
  - Sensitivity analysis across alternate clinical assumptions
  - High-resolution validation visualization exports
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crip.synthetic.cohort_generator import SyntheticCohortGenerator
from crip.synthetic.schemas import (
    PatientProfile,
    PEthRiskTier,
    TTNClassification,
    LGEStatus,
)

logger = logging.getLogger("crip.synthetic.cohort_validation")

DEFAULT_COHORT_CSV = Path("data/synthetic/synthetic_cardiomyopathy_cohort.csv")
DEFAULT_SUMMARY_JSON = Path("data/synthetic/cohort_validation_summary.json")
DOCS_SUMMARY_JSON = Path("docs/research/cohort_validation_summary.json")
FIGURES_DIR = Path("docs/research/figures")


def audit_cohort_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify absence of duplicates, schema violations, and impossible physiological values."""
    n_total = len(df)
    n_unique_ids = df["patient_id"].nunique()
    has_duplicates = (n_unique_ids != n_total)

    # Biological and physical sanity checks
    invalid_checks = {
        "esv_ge_edv": int((df["esv_ml"] >= df["edv_ml"]).sum()),
        "negative_edv": int((df["edv_ml"] <= 0).sum()),
        "negative_esv": int((df["esv_ml"] <= 0).sum()),
        "lvef_out_of_bounds": int(((df["baseline_lvef"] < 5.0) | (df["baseline_lvef"] > 75.0)).sum()),
        "age_out_of_bounds": int(((df["age"] < 18) | (df["age"] > 105)).sum()),
        "negative_peth": int((df["peth_value_ng_ml"].dropna() < 0).sum()),
        "lvef_math_mismatch": int((abs(((df["edv_ml"] - df["esv_ml"]) / df["edv_ml"] * 100.0) - df["baseline_lvef"]) > 5.0).sum()),
        "six_month_lvef_out_of_bounds": int(((df["follow_up_lvef"] < 5.0) | (df["follow_up_lvef"] > 85.0)).sum()),
        "recovery_label_inconsistent": int((df["recovery_binary"] != (df["delta_lvef"] >= 10.0).astype(int)).sum()),
    }

    all_passed = (not has_duplicates) and all(count == 0 for count in invalid_checks.values())

    return {
        "total_records": n_total,
        "unique_patient_ids": n_unique_ids,
        "has_duplicate_ids": has_duplicates,
        "integrity_violations": invalid_checks,
        "all_integrity_checks_passed": all_passed,
    }


def compute_distribution_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute parametric and non-parametric statistics across modeled variables."""
    # PEth statistics among available tests
    peth_valid = df["peth_value_ng_ml"].dropna()
    peth_stats = {
        "n_available": int(len(peth_valid)),
        "mean_ng_ml": round(float(peth_valid.mean()), 2),
        "median_ng_ml": round(float(peth_valid.median()), 2),
        "std_ng_ml": round(float(peth_valid.std()), 2),
        "iqr_25_75": [round(float(peth_valid.quantile(0.25)), 2), round(float(peth_valid.quantile(0.75)), 2)],
        "pct_abstinent_light": round(float((peth_valid < 20.0).mean() * 100.0), 2),
        "pct_moderate": round(float(((peth_valid >= 20.0) & (peth_valid <= 200.0)).mean() * 100.0), 2),
        "pct_heavy": round(float((peth_valid > 200.0).mean() * 100.0), 2),
    }

    # TTN breakdown
    ttn_counts = df["ttn_classification"].value_counts(normalize=True) * 100.0
    ttn_stats = {k: round(float(v), 2) for k, v in ttn_counts.to_dict().items()}

    # LGE breakdown
    lge_stat_counts = df["lge_status"].value_counts(normalize=True) * 100.0
    lge_pat_counts = df[df["lge_status"] == "POSITIVE"]["lge_pattern"].value_counts(normalize=True) * 100.0
    lge_extent_valid = df["lge_extent_percent"].dropna()

    lge_stats = {
        "status_distribution_pct": {k: round(float(v), 2) for k, v in lge_stat_counts.to_dict().items()},
        "pattern_distribution_positive_pct": {k: round(float(v), 2) for k, v in lge_pat_counts.to_dict().items()},
        "extent_mean_pct_lv_mass": round(float(lge_extent_valid.mean()), 2) if len(lge_extent_valid) > 0 else 0.0,
        "extent_median_pct": round(float(lge_extent_valid.median()), 2) if len(lge_extent_valid) > 0 else 0.0,
    }

    # Cardiac function statistics
    cardiac_stats = {
        "baseline_lvef": {
            "mean": round(float(df["baseline_lvef"].mean()), 2),
            "std": round(float(df["baseline_lvef"].std()), 2),
            "median": round(float(df["baseline_lvef"].median()), 2),
            "min": round(float(df["baseline_lvef"].min()), 2),
            "max": round(float(df["baseline_lvef"].max()), 2),
        },
        "lvedv_ml": {
            "mean": round(float(df["edv_ml"].mean()), 2),
            "std": round(float(df["edv_ml"].std()), 2),
            "median": round(float(df["edv_ml"].median()), 2),
        },
        "lvesv_ml": {
            "mean": round(float(df["esv_ml"].mean()), 2),
            "std": round(float(df["esv_ml"].std()), 2),
        },
        "follow_up_lvef": {
            "mean": round(float(df["follow_up_lvef"].mean()), 2),
            "std": round(float(df["follow_up_lvef"].std()), 2),
            "median": round(float(df["follow_up_lvef"].median()), 2),
        },
        "delta_lvef": {
            "mean": round(float(df["delta_lvef"].mean()), 2),
            "std": round(float(df["delta_lvef"].std()), 2),
            "median": round(float(df["delta_lvef"].median()), 2),
            "min": round(float(df["delta_lvef"].min()), 2),
            "max": round(float(df["delta_lvef"].max()), 2),
        },
        "overall_recovery_prevalence_pct": round(float((df["recovery_binary"] == 1).mean() * 100.0), 2),
    }

    return {
        "peth_biomarker": peth_stats,
        "ttn_genetics": ttn_stats,
        "lge_tissue_fibrosis": lge_stats,
        "cardiac_function_and_recovery": cardiac_stats,
    }


def compute_subgroup_recovery_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate recovery endpoint prevalence across key clinical subgroups."""
    subgroups: Dict[str, Any] = {}

    # By Sustained Abstinence
    abs_grp = df.groupby("sustained_abstinence_confirmed")["recovery_binary"].agg(["count", "mean"])
    subgroups["by_sustained_abstinence"] = {
        "sustained_abstinence": {
            "n": int(abs_grp.loc[True, "count"]) if True in abs_grp.index else 0,
            "recovery_rate_pct": round(float(abs_grp.loc[True, "mean"] * 100.0), 2) if True in abs_grp.index else 0.0,
        },
        "continued_drinking": {
            "n": int(abs_grp.loc[False, "count"]) if False in abs_grp.index else 0,
            "recovery_rate_pct": round(float(abs_grp.loc[False, "mean"] * 100.0), 2) if False in abs_grp.index else 0.0,
        },
    }

    # By PEth Exposure Risk Tier
    peth_grp = df.groupby("peth_risk_tier")["recovery_binary"].agg(["count", "mean"])
    subgroups["by_peth_tier"] = {
        tier: {
            "n": int(peth_grp.loc[tier, "count"]),
            "recovery_rate_pct": round(float(peth_grp.loc[tier, "mean"] * 100.0), 2),
        }
        for tier in peth_grp.index
    }

    # By LGE Status
    lge_grp = df.groupby("lge_status")["recovery_binary"].agg(["count", "mean"])
    subgroups["by_lge_status"] = {
        status: {
            "n": int(lge_grp.loc[status, "count"]),
            "recovery_rate_pct": round(float(lge_grp.loc[status, "mean"] * 100.0), 2),
        }
        for status in lge_grp.index
    }

    # By TTN Genetic Classification
    ttn_grp = df.groupby("ttn_classification")["recovery_binary"].agg(["count", "mean"])
    subgroups["by_ttn_classification"] = {
        cat: {
            "n": int(ttn_grp.loc[cat, "count"]),
            "recovery_rate_pct": round(float(ttn_grp.loc[cat, "mean"] * 100.0), 2),
        }
        for cat in ttn_grp.index
    }

    return subgroups


def test_single_feature_separability(df: pd.DataFrame) -> Dict[str, Any]:
    """Test whether any single baseline feature trivially separates the recovery endpoint.

    Clinical research validation criterion:
    The recovery endpoint must reflect stochastic, multi-factor biological reverse remodeling.
    No single baseline feature should yield a univariate ROC-AUC > 0.82.
    """
    y = df["recovery_binary"].values
    results: Dict[str, float] = {}

    candidate_features = {
        "age": df["age"].values,
        "baseline_lvef": df["baseline_lvef"].values,
        "edv_ml": df["edv_ml"].values,
        "esv_ml": df["esv_ml"].values,
        "peth_imputed": df["peth_value_ng_ml"].fillna(df["peth_value_ng_ml"].median()).values,
        "reported_drinks": df["reported_drinks_per_week"].fillna(0.0).values,
        "lge_extent": df["lge_extent_percent"].fillna(0.0).values,
        "is_male": (df["biological_sex"] == "MALE").astype(float).values,
        "is_lge_positive": (df["lge_status"] == "POSITIVE").astype(float).values,
        "is_ttn_pathogenic": (df["ttn_classification"] == "PATHOGENIC").astype(float).values,
    }

    for name, vals in candidate_features.items():
        try:
            auc = float(roc_auc_score(y, vals))
            # Invert AUC if feature is negatively associated
            auc_effective = max(auc, 1.0 - auc)
            results[name] = round(auc_effective, 3)
        except Exception as e:
            logger.warning(f"Could not compute AUC for {name}: {e}")

    max_auc_feature = max(results.keys(), key=lambda k: results[k])
    max_auc_value = results[max_auc_feature]
    is_non_trivial = max_auc_value < 0.82

    return {
        "univariate_roc_aucs": results,
        "highest_single_feature": max_auc_feature,
        "highest_single_feature_auc": max_auc_value,
        "passed_non_trivial_separability": is_non_trivial,
        "threshold_ceiling": 0.82,
        "verdict": (
            "PASS: No single feature artificially separates the simulated recovery outcome."
            if is_non_trivial
            else "FAIL: Single feature excessively predicts recovery outcome."
        ),
    }


def run_sensitivity_analysis(
    seed: int = 42,
    base_n: int = 1000,
) -> Dict[str, Any]:
    """Conduct sensitivity analyses under alternate plausible clinical assumptions."""
    logger.info("Executing sensitivity analyses across alternate clinical scenarios...")

    scenarios = [
        {"name": "Standard (70% Abstinence)", "abstinence_prob": 0.70},
        {"name": "Low Abstinence (40% Abstinence)", "abstinence_prob": 0.40},
        {"name": "High Abstinence (90% Abstinence)", "abstinence_prob": 0.90},
    ]

    sensitivity_results: List[Dict[str, Any]] = []

    for sc in scenarios:
        gen = SyntheticCohortGenerator(seed=seed)
        profiles, _, meta = gen.generate_cohort(
            n_patients=base_n,
            abstinence_prob=sc["abstinence_prob"],
        )
        deltas = [p.delta_lvef for p in profiles if p.delta_lvef is not None]
        recs = [p.recovery_binary for p in profiles if p.recovery_binary is not None]

        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        rec_rate = float(np.mean(recs) * 100.0) if recs else 0.0

        sensitivity_results.append({
            "scenario": sc["name"],
            "abstinence_probability": sc["abstinence_prob"],
            "sample_size": len(profiles),
            "mean_delta_lvef": round(mean_delta, 2),
            "recovery_prevalence_pct": round(rec_rate, 2),
        })

    return {"sensitivity_scenarios": sensitivity_results}


def generate_validation_figures(
    df: pd.DataFrame,
    output_dir: Path | str = FIGURES_DIR,
) -> List[Path]:
    """Generate and save publication-grade cohort validation figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_figs: List[Path] = []

    # Figure 1: 4-Panel Core Variable Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("white")

    # Panel A: Age Distribution
    axes[0, 0].hist(df["age"], bins=20, color="#1f77b4", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title("A: Patient Age Distribution (Years)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Patient Count")

    # Panel B: Baseline LVEF
    axes[0, 1].hist(df["baseline_lvef"], bins=20, color="#d62728", edgecolor="black", alpha=0.8)
    axes[0, 1].axvline(df["baseline_lvef"].mean(), color="black", linestyle="--", label=f"Mean: {df['baseline_lvef'].mean():.1f}%")
    axes[0, 1].set_title("B: Baseline Left Ventricular Ejection Fraction (%)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Baseline LVEF (%)")
    axes[0, 1].legend()

    # Panel C: PEth Biomarker (Log Scale)
    peth_valid = df["peth_value_ng_ml"].dropna()
    peth_positive = peth_valid[peth_valid > 0]
    axes[1, 0].hist(np.log10(peth_positive + 1), bins=25, color="#ff7f0e", edgecolor="black", alpha=0.8)
    axes[1, 0].axvline(np.log10(20 + 1), color="blue", linestyle=":", label="20 ng/mL Cutoff")
    axes[1, 0].axvline(np.log10(200 + 1), color="red", linestyle=":", label="200 ng/mL Cutoff")
    axes[1, 0].set_title("C: Blood PEth Distribution [log10(ng/mL + 1)]", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("log10(PEth + 1)")
    axes[1, 0].set_ylabel("Patient Count")
    axes[1, 0].legend()

    # Panel D: Six-Month Delta LVEF
    axes[1, 1].hist(df["delta_lvef"], bins=25, color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[1, 1].axvline(10.0, color="purple", linestyle="--", linewidth=2, label="Recovery Threshold (+10%)")
    axes[1, 1].set_title("D: Six-Month Delta LVEF Trajectory (%)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Delta LVEF (percentage points)")
    axes[1, 1].legend()

    plt.tight_layout()
    f1_path = out_dir / "synthetic_cohort_distributions.png"
    plt.savefig(f1_path, dpi=300)
    plt.close(fig)
    saved_figs.append(f1_path)

    # Figure 2: Subgroup Recovery Prevalence
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    subgroup_labels = [
        "Sustained Abstinence",
        "Continued Drinking",
        "PEth: Heavy (>200)",
        "PEth: Light (<20)",
        "LGE: Negative",
        "LGE: Positive",
        "TTN: Negative",
        "TTN: Pathogenic",
    ]

    peth_heavy = (df["peth_value_ng_ml"] > 200.0)
    peth_light = (df["peth_value_ng_ml"] < 20.0)

    rates = [
        float(df[df["sustained_abstinence_confirmed"] == True]["recovery_binary"].mean() * 100.0),
        float(df[df["sustained_abstinence_confirmed"] == False]["recovery_binary"].mean() * 100.0),
        float(df[peth_heavy]["recovery_binary"].mean() * 100.0),
        float(df[peth_light]["recovery_binary"].mean() * 100.0),
        float(df[df["lge_status"] == "NEGATIVE"]["recovery_binary"].mean() * 100.0),
        float(df[df["lge_status"] == "POSITIVE"]["recovery_binary"].mean() * 100.0),
        float(df[df["ttn_classification"] == "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"]["recovery_binary"].mean() * 100.0),
        float(df[df["ttn_classification"] == "PATHOGENIC"]["recovery_binary"].mean() * 100.0),
    ]

    colors = ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f", "#2ca02c", "#d62728", "#1f77b4", "#e377c2"]
    bars = ax2.barh(subgroup_labels, rates, color=colors, edgecolor="black", alpha=0.85)
    ax2.axvline(float((df["recovery_binary"] == 1).mean() * 100.0), color="black", linestyle="--", label=f"Cohort Mean ({float((df['recovery_binary'] == 1).mean() * 100.0):.1f}%)")
    ax2.set_xlabel("Recovery Rate (% with Delta LVEF >= 10 percentage points)")
    ax2.set_title("Six-Month LVEF Recovery Rate Stratified by Clinical Subgroups", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.legend(loc="lower right")

    for bar, rate in zip(bars, rates):
        ax2.text(rate + 1.5, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    f2_path = out_dir / "recovery_by_subgroups.png"
    plt.savefig(f2_path, dpi=300)
    plt.close(fig2)
    saved_figs.append(f2_path)

    # Also copy to data/synthetic/figures
    data_fig_dir = Path("data/synthetic/figures")
    data_fig_dir.mkdir(parents=True, exist_ok=True)
    for f in saved_figs:
        dest = data_fig_dir / f.name
        dest.write_bytes(f.read_bytes())

    logger.info(f"Generated {len(saved_figs)} validation figures in {out_dir}")
    return saved_figs


def run_full_cohort_evaluation(
    csv_path: Path | str = DEFAULT_COHORT_CSV,
    summary_path: Path | str = DEFAULT_SUMMARY_JSON,
) -> Dict[str, Any]:
    """Execute complete validation suite and export summary JSON."""
    src = Path(csv_path)
    if not src.is_file():
        raise FileNotFoundError(f"Cohort CSV not found: {src}")

    df = pd.read_csv(src)
    logger.info(f"Loaded {len(df)} records from {src} for comprehensive quality audit...")

    integrity = audit_cohort_integrity(df)
    distributions = compute_distribution_metrics(df)
    subgroups = compute_subgroup_recovery_rates(df)
    separability = test_single_feature_separability(df)
    sensitivity = run_sensitivity_analysis(seed=42, base_n=1000)
    figures = generate_validation_figures(df)

    summary = {
        "evaluation_timestamp": "2026-10-25T19:30:00Z",
        "cohort_source": str(src),
        "integrity_audit": integrity,
        "univariate_distributions": distributions,
        "subgroup_recovery_rates": subgroups,
        "separability_test": separability,
        "sensitivity_analysis": sensitivity,
        "generated_figure_paths": [str(f) for f in figures],
    }

    dest = Path(summary_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save copy to docs
    DOCS_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved cohort validation summary to {dest} and {DOCS_SUMMARY_JSON}")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Audit and validate synthetic cardiomyopathy cohort")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_COHORT_CSV), help="Path to cohort CSV")
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_JSON), help="Output summary path")
    args = parser.parse_args()

    run_full_cohort_evaluation(csv_path=args.csv_path, summary_path=args.summary_path)


if __name__ == "__main__":
    main()
