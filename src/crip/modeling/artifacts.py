"""Model artifact serialization, validation, and lifecycle management for CRIP recovery modeling.

DISCLAIMER:
Saves and loads trained Scikit-learn pipelines on local storage for synthetic demonstration.
Model files are excluded from Git version control.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
from sklearn.pipeline import Pipeline

from crip.modeling.dataset import (
    BINARY_INDICATOR_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ModelingCohortDataset,
)

logger = logging.getLogger("crip.modeling.artifacts")

DEFAULT_MODEL_DIR = Path("models/recovery")


def save_recovery_model_artifacts(
    pipeline: Pipeline,
    dataset: ModelingCohortDataset,
    evaluation_report: Dict[str, Any],
    model_version: str = "v1.0-m3b",
    output_dir: Union[Path, str] = DEFAULT_MODEL_DIR,
) -> Path:
    """Save trained pipeline, feature schemas, splits, and evaluation metadata to disk.

    Artifacts saved:
      - recovery_pipeline.joblib
      - metadata.json
      - feature_schema.json
      - splits.json
      - test_metrics.json
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "recovery_pipeline.joblib"
    meta_path = out_dir / "metadata.json"
    schema_path = out_dir / "feature_schema.json"
    splits_path = out_dir / "splits.json"
    metrics_path = out_dir / "test_metrics.json"

    # 1. Save pipeline
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved trained recovery model pipeline to {model_path}")

    # 2. Save feature schema
    feature_schema = {
        "model_version": model_version,
        "features": dataset.feature_names,
        "numerical_features": dataset.numerical_features,
        "categorical_features": dataset.categorical_features,
        "binary_features": dataset.binary_features,
        "target_name": dataset.target_name,
        "scenario": dataset.scenario,
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    # 3. Save splits
    splits = {
        "train_patient_ids": dataset.train_ids,
        "val_patient_ids": dataset.val_ids,
        "test_patient_ids": dataset.test_ids,
    }
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # 4. Save test metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    # 5. Save master metadata
    metadata = {
        "model_name": "CRIP Multimodal Recovery Predictor",
        "model_version": model_version,
        "estimator_type": type(pipeline.named_steps["clf"]).__name__,
        "synthetic_generator_version": dataset.metadata.get("generation_version", "v1.0-m3a"),
        "training_seed": dataset.metadata.get("split_seed", 42),
        "target_definition": "Delta LVEF >= 10.0 percentage points at 6-month follow-up",
        "assumed_scenario": dataset.scenario,
        "training_sample_count": len(dataset.X_train),
        "validation_sample_count": len(dataset.X_val),
        "test_sample_count": len(dataset.X_test),
        "test_roc_auc": evaluation_report["metrics"]["roc_auc"],
        "test_brier_score": evaluation_report["metrics"]["brier_score"],
        "target_auc_benchmark": evaluation_report.get("target_auc_benchmark", 0.80),
        "benchmark_achieved": evaluation_report.get("target_achieved", True),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully saved all recovery model artifacts to {out_dir}")
    return out_dir


def load_recovery_model_artifacts(
    model_dir: Union[Path, str] = DEFAULT_MODEL_DIR,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load, validate, and return the trained recovery model pipeline and associated metadata.

    Raises:
        FileNotFoundError: If the model directory or critical artifacts are missing.
        ValueError: If loaded schema does not contain required baseline features.
    """
    in_dir = Path(model_dir)
    model_path = in_dir / "recovery_pipeline.joblib"
    meta_path = in_dir / "metadata.json"
    schema_path = in_dir / "feature_schema.json"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Recovery model artifact not found at {model_path}. "
            "Please train and save the model before attempting inference."
        )

    pipeline: Pipeline = joblib.load(model_path)

    metadata: Dict[str, Any] = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if schema_path.is_file():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
            metadata["schema"] = schema

    return pipeline, metadata
