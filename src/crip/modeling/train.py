"""Model training, baseline benchmarking, and feature ablation experiments for CRIP recovery modeling.

DISCLAIMER:
Trains interpretable machine learning models on synthetic clinical data only.
Simulated recovery predictions must never be represented as real-patient probabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from crip.modeling.dataset import (
    BINARY_INDICATOR_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ModelingCohortDataset,
    build_feature_preprocessor,
)

logger = logging.getLogger("crip.modeling.train")

# ==============================================================================
# 1. Feature Ablation Group Definitions
# ==============================================================================

CMR_FEATURES: List[str] = [
    "baseline_lvef",
    "edv_ml",
    "esv_ml",
    "stroke_volume_ml",
    "rvedv_ml",
    "rvef_percent",
    "myocardial_mass_g",
]

PETH_TOXICOLOGY_FEATURES: List[str] = [
    "peth_value_ng_ml",
    "peth_available",
    "reported_drinks_per_week",
    "time_since_last_drink_days",
    "years_heavy_drinking",
    "alcohol_history_available",
]

TTN_GENETICS_FEATURES: List[str] = [
    "ttn_classification",
]

LGE_FIBROSIS_FEATURES: List[str] = [
    "lge_status",
    "lge_pattern",
    "lge_extent_percent",
    "lge_available",
]

DEMOGRAPHIC_FEATURES: List[str] = [
    "age",
    "biological_sex",
]

ABLATION_GROUPS: Dict[str, List[str]] = {
    "1_CMR_Volumetrics_Only": DEMOGRAPHIC_FEATURES + CMR_FEATURES,
    "2_CMR_Plus_PEth": DEMOGRAPHIC_FEATURES + CMR_FEATURES + PETH_TOXICOLOGY_FEATURES,
    "3_CMR_Plus_TTN": DEMOGRAPHIC_FEATURES + CMR_FEATURES + TTN_GENETICS_FEATURES,
    "4_CMR_Plus_LGE": DEMOGRAPHIC_FEATURES + CMR_FEATURES + LGE_FIBROSIS_FEATURES,
    "5_PEth_TTN_LGE_No_CMR": DEMOGRAPHIC_FEATURES + PETH_TOXICOLOGY_FEATURES + TTN_GENETICS_FEATURES + LGE_FIBROSIS_FEATURES,
    "6_Full_Multimodal": DEMOGRAPHIC_FEATURES + CMR_FEATURES + PETH_TOXICOLOGY_FEATURES + TTN_GENETICS_FEATURES + LGE_FIBROSIS_FEATURES,
}


# ==============================================================================
# 2. Evaluation Helper
# ==============================================================================

def compute_binary_classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics for binary recovery classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)

    # Discrimination
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")

    # Calibration & Error
    brier = float(brier_score_loss(y_true, y_prob))
    try:
        ce_loss = float(log_loss(y_true, y_prob))
    except ValueError:
        ce_loss = float("nan")

    # Classification matrix metrics
    acc = float(accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        "n_samples": len(y_true),
        "prevalence": float(np.mean(y_true)),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ce_loss, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "npv": round(npv, 4),
        "f1_score": round(f1, 4),
        "decision_threshold": threshold,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


# ==============================================================================
# 3. Model Training & Baseline Benchmarking
# ==============================================================================

def train_baseline_models(
    dataset: ModelingCohortDataset,
    seed: int = 42,
) -> Dict[str, Pipeline]:
    """Train baseline models on the training split using identical feature preprocessing.

    Models:
      1. DummyClassifier (prevalence baseline)
      2. LogisticRegression (primary interpretable model)
      3. RandomForestClassifier (non-linear ensemble benchmark)

    Returns:
        Dictionary mapping model names to fitted Scikit-learn Pipelines.
    """
    preprocessor = build_feature_preprocessor(
        numerical_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
        binary_features=dataset.binary_features,
    )

    models = {
        "dummy_prevalence": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=seed,
            class_weight="balanced",
        ),
    }

    fitted_pipelines: Dict[str, Pipeline] = {}
    for name, clf in models.items():
        logger.info(f"Training model: {name}")
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", clf),
        ])
        pipeline.fit(dataset.X_train, dataset.y_train)
        fitted_pipelines[name] = pipeline

    return fitted_pipelines


def evaluate_models_on_validation(
    models: Dict[str, Pipeline],
    dataset: ModelingCohortDataset,
) -> pd.DataFrame:
    """Evaluate trained pipelines on the validation split."""
    records = []
    for name, pipeline in models.items():
        probs = pipeline.predict_proba(dataset.X_val)[:, 1]
        metrics = compute_binary_classification_metrics(dataset.y_val, probs)
        records.append({
            "model": name,
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
            "val_brier_score": metrics["brier_score"],
            "val_accuracy": metrics["accuracy"],
            "val_sensitivity": metrics["sensitivity"],
            "val_specificity": metrics["specificity"],
            "val_f1": metrics["f1_score"],
        })

    df_results = pd.DataFrame(records).sort_values(by="val_roc_auc", ascending=False).reset_index(drop=True)
    return df_results


# ==============================================================================
# 4. Feature Ablation Experiments (Phase 6)
# ==============================================================================

def run_feature_ablation_study(
    dataset: ModelingCohortDataset,
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Evaluate performance contributions of distinct clinical modalities on the validation set.

    Configurations:
      1. Baseline CMR only (LVEF, EDV, ESV, RVEDV, RVEF, Mass)
      2. Baseline CMR + PEth toxicology
      3. Baseline CMR + TTN genetics
      4. Baseline CMR + LGE tissue fibrosis
      5. PEth + TTN + LGE (without CMR volumetrics)
      6. Full multimodal set (all eligible baseline variables)

    Returns:
        DataFrame detailing validation discrimination and calibration across configurations.
    """
    records = []

    for group_name, features in ABLATION_GROUPS.items():
        # Filter features present in dataset
        active_features = [f for f in features if f in dataset.feature_names]
        active_num = [f for f in active_features if f in NUMERICAL_FEATURES]
        active_cat = [f for f in active_features if f in CATEGORICAL_FEATURES]
        active_bin = [f for f in active_features if f in BINARY_INDICATOR_FEATURES]

        prep = build_feature_preprocessor(active_num, active_cat, active_bin)
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, class_weight="balanced")
        pipe = Pipeline([("prep", prep), ("clf", model)])

        # Train on X_train[active_features]
        X_tr = dataset.X_train[active_features]
        X_va = dataset.X_val[active_features]

        pipe.fit(X_tr, dataset.y_train)
        probs = pipe.predict_proba(X_va)[:, 1]

        m = compute_binary_classification_metrics(dataset.y_val, probs)

        records.append({
            "ablation_group": group_name,
            "feature_count": len(active_features),
            "features_included": ", ".join(active_features),
            "val_roc_auc": m["roc_auc"],
            "val_pr_auc": m["pr_auc"],
            "val_brier_score": m["brier_score"],
            "val_accuracy": m["accuracy"],
            "val_sensitivity": m["sensitivity"],
            "val_specificity": m["specificity"],
            "val_f1": m["f1_score"],
        })

    df_ablation = pd.DataFrame(records).sort_values(by="val_roc_auc", ascending=False).reset_index(drop=True)

    if output_dir:
        out_p = Path(output_dir) / "ablation_study_results.json"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(df_ablation.to_dict(orient="records"), f, indent=2)
        logger.info(f"Saved feature ablation study results to {out_p}")

    return df_ablation
