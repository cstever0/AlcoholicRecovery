"""Quantitative evaluation and calibration metrics for CRIP synthetic recovery prediction.

DISCLAIMER:
Evaluates machine learning models on synthetic validation and held-out test partitions.
Simulated benchmark metrics must never be represented as clinical validation on real patients.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline

from crip.modeling.dataset import ModelingCohortDataset
from crip.modeling.train import compute_binary_classification_metrics

logger = logging.getLogger("crip.evaluation.recovery_metrics")

DEFAULT_EVAL_JSON_PATH = Path("data/processed/recovery_test_metrics.json")
DOCS_EVAL_JSON_PATH = Path("docs/evaluation/recovery_test_metrics.json")


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute empirical 95% bootstrap confidence intervals for primary discrimination metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_samples = []
    brier_samples = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        # Skip resamples with only one class
        if len(np.unique(y_b)) < 2:
            continue
        m = compute_binary_classification_metrics(y_b, p_b)
        auc_samples.append(m["roc_auc"])
        brier_samples.append(m["brier_score"])

    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    auc_ci = (
        round(float(np.percentile(auc_samples, lower_p)), 4),
        round(float(np.percentile(auc_samples, upper_p)), 4),
    ) if auc_samples else (float("nan"), float("nan"))

    brier_ci = (
        round(float(np.percentile(brier_samples, lower_p)), 4),
        round(float(np.percentile(brier_samples, upper_p)), 4),
    ) if brier_samples else (float("nan"), float("nan"))

    return {
        "roc_auc_ci_95": auc_ci,
        "brier_score_ci_95": brier_ci,
    }


def evaluate_final_model_on_test(
    model: Pipeline,
    dataset: ModelingCohortDataset,
    target_auc: float = 0.80,
    output_path: Optional[Union[Path, str]] = DEFAULT_EVAL_JSON_PATH,
) -> Dict[str, Any]:
    """Execute rigorous one-time evaluation of the finalized model on held-out test data.

    Calculates:
      - ROC-AUC, PR-AUC, Brier score, Log loss
      - Accuracy, Sensitivity, Specificity, Precision, F1
      - Confusion matrix
      - Binned calibration curve
      - 95% Bootstrap confidence intervals
      - Target threshold flag (ROC-AUC >= 0.80)
    """
    y_test = np.asarray(dataset.y_test, dtype=int)
    y_prob = model.predict_proba(dataset.X_test)[:, 1]

    # Standard metrics
    metrics = compute_binary_classification_metrics(y_test, y_prob)

    # Bootstrap CIs
    ci = compute_bootstrap_confidence_intervals(y_test, y_prob, n_bootstraps=1000, seed=42)

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    calibration_points = [
        {"bin_idx": i, "mean_predicted_prob": round(float(p_pred), 4), "observed_frequency": round(float(p_true), 4)}
        for i, (p_true, p_pred) in enumerate(zip(prob_true, prob_pred))
    ]

    target_achieved = bool(metrics["roc_auc"] >= target_auc)

    evaluation_report = {
        "evaluation_split": "HELD_OUT_TEST",
        "scenario": dataset.scenario,
        "n_test_samples": len(y_test),
        "target_prevalence": metrics["prevalence"],
        "target_auc_benchmark": target_auc,
        "target_achieved": target_achieved,
        "metrics": metrics,
        "confidence_intervals": ci,
        "calibration_curve": calibration_points,
    }

    # Save to disk
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, indent=2)
        logger.info(f"Saved test evaluation report to {out_path}")

        # Also write copy to docs/evaluation/
        try:
            DOCS_EVAL_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DOCS_EVAL_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(evaluation_report, f, indent=2)
        except Exception:
            pass

    return evaluation_report
