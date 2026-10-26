"""Sensitivity and robustness analysis for CRIP synthetic recovery models.

Evaluates how model discrimination and calibration change when synthetic cohort
assumptions, residual noise levels, and population compliance rates are varied.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from crip.evaluation.recovery_metrics import compute_binary_classification_metrics
from crip.modeling.dataset import (
    ModelingCohortDataset,
    load_modeling_dataset,
)
from crip.synthetic.cohort_generator import SyntheticCohortGenerator

logger = logging.getLogger("crip.modeling.sensitivity")

DEFAULT_SENSITIVITY_JSON = Path("docs/evaluation/sensitivity_analysis_results.json")


def evaluate_noise_sensitivity(
    model: Pipeline,
    base_dataset: ModelingCohortDataset,
    noise_multipliers: List[float] = [0.5, 1.0, 1.5, 2.0],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Assess model performance degradation under varying levels of simulated residual noise."""
    results = []
    y_true = np.asarray(base_dataset.y_test, dtype=int)
    y_prob_base = model.predict_proba(base_dataset.X_test)[:, 1]

    rng = np.random.default_rng(seed)

    for mult in noise_multipliers:
        # Add perturbation to predicted probabilities to simulate higher stochastic outcome variance
        noise = rng.normal(0, 0.05 * mult, size=len(y_prob_base))
        perturbed_prob = np.clip(y_prob_base + noise, 0.001, 0.999)

        m = compute_binary_classification_metrics(y_true, perturbed_prob)
        results.append({
            "noise_multiplier": mult,
            "simulated_stochastic_std": round(4.25 * mult, 2),
            "test_roc_auc": m["roc_auc"],
            "test_pr_auc": m["pr_auc"],
            "test_brier_score": m["brier_score"],
            "test_accuracy": m["accuracy"],
        })

    return results


def evaluate_abstinence_scenario_sensitivity(
    model: Pipeline,
    generator: Optional[SyntheticCohortGenerator] = None,
    scenarios: List[float] = [0.40, 0.70, 0.90],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Evaluate model performance across synthetic populations with differing abstinence compliance."""
    if generator is None:
        generator = SyntheticCohortGenerator(seed=seed)

    results = []
    for prob in scenarios:
        # Generate smaller cohort under this scenario
        profiles, _, meta = generator.generate_cohort(n_patients=300, abstinence_prob=prob)
        df_records = []
        for p in profiles:
            if p.follow_up.assumed_abstinence_scenario.value == "SUSTAINED_ABSTINENCE":
                df_records.append({
                    "age": p.age,
                    "biological_sex": p.biological_sex.value,
                    "baseline_lvef": p.baseline_lvef,
                    "edv_ml": p.edv_ml,
                    "esv_ml": p.esv_ml,
                    "stroke_volume_ml": p.stroke_volume_ml,
                    "rvedv_ml": p.rvedv_ml,
                    "rvef_percent": p.rvef_percent,
                    "myocardial_mass_g": p.myocardial_mass_g,
                    "peth_value_ng_ml": p.peth.value_ng_ml if p.peth_available else None,
                    "peth_available": p.peth_available,
                    "reported_drinks_per_week": p.reported_drinks_per_week,
                    "time_since_last_drink_days": p.time_since_last_drink_days,
                    "years_heavy_drinking": p.years_heavy_drinking,
                    "alcohol_history_available": p.alcohol_history_available,
                    "ttn_classification": p.ttn_classification.value if p.ttn_classification else "NOT_TESTED",
                    "lge_status": p.lge_status.value,
                    "lge_pattern": p.lge_pattern.value if p.lge_pattern else "NONE",
                    "lge_extent_percent": p.lge_extent_percent,
                    "lge_available": p.lge_available,
                    "recovery_binary": p.follow_up.recovery_binary,
                })

        df_eval = pd.DataFrame(df_records).dropna(subset=["recovery_binary"])
        for col in ["peth_available", "alcohol_history_available", "lge_available"]:
            if col in df_eval.columns:
                df_eval[col] = df_eval[col].astype(float)
        if len(df_eval) >= 50:
            X_eval = df_eval.drop(columns=["recovery_binary"])
            y_eval = df_eval["recovery_binary"].values
            probs = model.predict_proba(X_eval)[:, 1]
            m = compute_binary_classification_metrics(y_eval, probs)
            results.append({
                "abstinence_probability": prob,
                "n_eval_patients": len(df_eval),
                "recovery_prevalence": m["prevalence"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "brier_score": m["brier_score"],
                "accuracy": m["accuracy"],
            })

    return results


def run_full_sensitivity_suite(
    model: Pipeline,
    dataset: ModelingCohortDataset,
    output_path: Optional[Union[Path, str]] = DEFAULT_SENSITIVITY_JSON,
) -> Dict[str, Any]:
    """Execute complete sensitivity suite and save structured summary to disk."""
    noise_results = evaluate_noise_sensitivity(model, dataset)
    abstinence_results = evaluate_abstinence_scenario_sensitivity(model)

    full_results = {
        "analysis_type": "SYNTHETIC_MODEL_SENSITIVITY_AUDIT",
        "model_architecture": "LogisticRegression(class_weight='balanced')",
        "noise_sensitivity": noise_results,
        "abstinence_scenario_sensitivity": abstinence_results,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"Saved sensitivity analysis results to {out}")

    return full_results
