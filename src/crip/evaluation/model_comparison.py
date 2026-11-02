"""Comparative evaluation and visualization for Logistic Regression and Random Forest recovery models.

DISCLAIMER:
All evaluations are conducted on synthetic sustained-abstinence cohort partitions (n = 419 held-out test).
Simulated recovery predictions must NEVER be interpreted as clinical validation on human patients.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from crip.modeling.dataset import load_modeling_dataset
from crip.modeling.train import compute_binary_classification_metrics, train_baseline_models

logger = logging.getLogger("crip.evaluation.model_comparison")

DEFAULT_COMPARISON_JSON = Path("data/processed/recovery_model_comparison.json")


def generate_recovery_model_comparison(
    output_path: Optional[Union[Path, str]] = DEFAULT_COMPARISON_JSON,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate baseline Logistic Regression and Random Forest models on identical test partition.

    Saves and returns machine-readable comparison dictionary containing:
      - Comparative metrics table (ROC-AUC, PR-AUC, Brier, Acc, Sens, Spec, Prec, F1, Log loss)
      - ROC curve coordinates (FPR, TPR)
      - Precision-Recall curve coordinates (Precision, Recall, baseline prevalence)
      - Binned calibration curve coordinates
      - Confusion matrices
    """
    out_file = Path(output_path) if output_path else None
    if out_file and out_file.is_file() and not force_regenerate:
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cached comparison: {e}. Recomputing.")

    dataset = load_modeling_dataset()
    models = train_baseline_models(dataset, seed=42)

    y_test = np.asarray(dataset.y_test, dtype=int)
    prevalence = float(np.mean(y_test))

    target_models = ["logistic_regression", "random_forest"]
    comparison_results: Dict[str, Any] = {
        "metadata": {
            "evaluation_cohort": "SYNTHETIC_SUSTAINED_ABSTINENCE",
            "test_sample_size": len(y_test),
            "target_prevalence": round(prevalence, 4),
            "decision_threshold": 0.50,
            "disclaimer": "Simulated recovery models trained on synthetic clinical data only.",
        },
        "models": {},
        "comparison_table": [],
    }

    for name in target_models:
        pipe = models[name]
        y_prob = pipe.predict_proba(dataset.X_test)[:, 1]

        # Standard summary metrics
        m = compute_binary_classification_metrics(y_test, y_prob, threshold=0.50)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Precision-Recall Curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

        # Calibration Curve (uniform 10 bins)
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

        # Downsample curve points for compact JSON storage if necessary
        comparison_results["models"][name] = {
            "metrics": m,
            "roc_curve": {
                "fpr": [round(float(v), 5) for v in fpr],
                "tpr": [round(float(v), 5) for v in tpr],
            },
            "pr_curve": {
                "precision": [round(float(v), 5) for v in prec_curve],
                "recall": [round(float(v), 5) for v in rec_curve],
            },
            "calibration_curve": {
                "mean_predicted": [round(float(v), 5) for v in prob_pred],
                "fraction_positives": [round(float(v), 5) for v in prob_true],
            },
            "confusion_matrix": m["confusion_matrix"],
        }

        display_name = "Logistic Regression (Primary)" if name == "logistic_regression" else "Random Forest (Nonlinear Ensemble)"
        comparison_results["comparison_table"].append({
            "Model Name": display_name,
            "Test Sample Size": len(y_test),
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Accuracy": f"{m['accuracy'] * 100:.1f}%",
            "Sensitivity": f"{m['sensitivity'] * 100:.1f}%",
            "Specificity": f"{m['specificity'] * 100:.1f}%",
            "Precision": f"{m['precision'] * 100:.1f}%",
            "F1 Score": m["f1_score"],
            "Brier Score": m["brier_score"],
            "Log Loss": m["log_loss"],
        })

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"Saved recovery model comparison to {out_file}")

    return comparison_results


def create_roc_comparison_plot(
    comp: Dict[str, Any],
    height: int = 420,
    dark_mode: bool = True,
) -> go.Figure:
    """Create interactive Plotly ROC curves comparing Logistic Regression and Random Forest."""
    fig = go.Figure()

    # Diagonal no-discrimination reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance Baseline (AUC = 0.50)",
            line=dict(color="#8892B0", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    colors = {"logistic_regression": "#00CC96", "random_forest": "#FFA15A"}
    names = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}

    for key, color in colors.items():
        if key in comp["models"]:
            m_data = comp["models"][key]
            auc_val = m_data["metrics"]["roc_auc"]
            fpr = m_data["roc_curve"]["fpr"]
            tpr = m_data["roc_curve"]["tpr"]

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{names[key]} (ROC-AUC = {auc_val:.4f})",
                    line=dict(color=color, width=2.5),
                )
            )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Comparison — Synthetic Test Cohort (n = 419)",
        xaxis_title="False Positive Rate (1 − Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=50, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_pr_comparison_plot(
    comp: Dict[str, Any],
    height: int = 420,
    dark_mode: bool = True,
) -> go.Figure:
    """Create interactive Plotly Precision-Recall curves comparing Logistic Regression and Random Forest."""
    fig = go.Figure()

    prevalence = comp.get("metadata", {}).get("target_prevalence", 0.7828)

    # Horizontal prevalence baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence, prevalence],
            mode="lines",
            name=f"Prevalence Baseline ({prevalence*100:.1f}%)",
            line=dict(color="#8892B0", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    colors = {"logistic_regression": "#00CC96", "random_forest": "#FFA15A"}
    names = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}

    for key, color in colors.items():
        if key in comp["models"]:
            m_data = comp["models"][key]
            pr_auc = m_data["metrics"]["pr_auc"]
            rec = m_data["pr_curve"]["recall"]
            prec = m_data["pr_curve"]["precision"]

            fig.add_trace(
                go.Scatter(
                    x=rec,
                    y=prec,
                    mode="lines",
                    name=f"{names[key]} (PR-AUC = {pr_auc:.4f})",
                    line=dict(color=color, width=2.5),
                )
            )

    fig.update_layout(
        title="Precision-Recall (PR) Curve Comparison — Synthetic Test Cohort (n = 419)",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=50, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_calibration_comparison_plot(
    comp: Dict[str, Any],
    height: int = 420,
    dark_mode: bool = True,
) -> go.Figure:
    """Create interactive Plotly calibration curves comparing predicted probabilities to empirical recovery frequencies."""
    fig = go.Figure()

    # Perfectly calibrated diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration (y = x)",
            line=dict(color="#8892B0", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    colors = {"logistic_regression": "#00CC96", "random_forest": "#FFA15A"}
    names = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}

    for key, color in colors.items():
        if key in comp["models"]:
            m_data = comp["models"][key]
            brier = m_data["metrics"]["brier_score"]
            x_pred = m_data["calibration_curve"]["mean_predicted"]
            y_obs = m_data["calibration_curve"]["fraction_positives"]

            fig.add_trace(
                go.Scatter(
                    x=x_pred,
                    y=y_obs,
                    mode="lines+markers",
                    name=f"{names[key]} (Brier = {brier:.4f})",
                    line=dict(color=color, width=2),
                    marker=dict(size=7),
                )
            )

    fig.update_layout(
        title="Reliability Calibration Curves (10 Bins) — Synthetic Test Cohort (n = 419)",
        xaxis_title="Mean Predicted Recovery Probability",
        yaxis_title="Observed Fraction of Recoveries",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=50, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_confusion_matrix_figure(
    cm_dict: Dict[str, int],
    model_name: str,
    height: int = 340,
    dark_mode: bool = True,
) -> go.Figure:
    """Create 2x2 confusion matrix heatmap with counts and percentages."""
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    total = tn + fp + fn + tp
    matrix = np.array([[tn, fp], [fn, tp]])
    text = [
        [f"True Neg (TN)<br><b>{tn}</b> ({tn/total*100:.1f}%)", f"False Pos (FP)<br><b>{fp}</b> ({fp/total*100:.1f}%)"],
        [f"False Neg (FN)<br><b>{fn}</b> ({fn/total*100:.1f}%)", f"True Pos (TP)<br><b>{tp}</b> ({tp/total*100:.1f}%)"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Non-Recovery", "Predicted Recovery"],
            y=["Actual Non-Recovery", "Actual Recovery"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorscale="Teal" if dark_mode else "Blues",
            showscale=False,
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix: {model_name} (Threshold = 0.50)",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=50, b=30),
    )

    return fig
