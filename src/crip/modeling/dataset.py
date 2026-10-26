"""Dataset preparation, feature whitelisting, and reproducible partitioning for CRIP recovery modeling.

DISCLAIMER:
Provides dataset loading and splitting for synthetic patient records only.
Enforces strict prevention of data leakage: no post-baseline or outcome variables
may be included as predictor features.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("crip.modeling.dataset")

DEFAULT_COHORT_CSV = Path("data/synthetic/synthetic_cardiomyopathy_cohort.csv")
DEFAULT_SPLIT_MANIFEST_PATH = Path("data/processed/recovery_modeling_splits.json")

# ==============================================================================
# 1. Feature Whitelist & Leakage Prevention Rules
# ==============================================================================

# Strictly allowed baseline clinical predictors
NUMERICAL_FEATURES: List[str] = [
    "age",
    "baseline_lvef",
    "edv_ml",
    "esv_ml",
    "stroke_volume_ml",
    "rvedv_ml",
    "rvef_percent",
    "myocardial_mass_g",
    "peth_value_ng_ml",
    "reported_drinks_per_week",
    "time_since_last_drink_days",
    "years_heavy_drinking",
    "lge_extent_percent",
]

CATEGORICAL_FEATURES: List[str] = [
    "biological_sex",
    "ttn_classification",
    "lge_status",
    "lge_pattern",
]

BINARY_INDICATOR_FEATURES: List[str] = [
    "peth_available",
    "alcohol_history_available",
    "lge_available",
]

FEATURE_WHITELIST: List[str] = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_INDICATOR_FEATURES
)

# Strictly forbidden variables: target definitions, follow-up outcomes, future measurements
FORBIDDEN_LEAKAGE_COLUMNS: List[str] = [
    "recovery_binary",
    "recovery_category",
    "delta_lvef",
    "follow_up_lvef",
    "follow_up_date",
    "adverse_cardiac_events",
    "sustained_abstinence_confirmed",
    "six_month_lvef",
]

TARGET_COLUMN: str = "recovery_binary"


# ==============================================================================
# 2. Data Structure
# ==============================================================================

@dataclass
class ModelingCohortDataset:
    """Structured container for partitioned modeling data and metadata."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    binary_features: List[str]
    target_name: str
    scenario: str
    metadata: Dict[str, Any]

    @property
    def feature_names(self) -> List[str]:
        return list(self.X_train.columns)

    @property
    def total_eligible_patients(self) -> int:
        return len(self.X_train) + len(self.X_val) + len(self.X_test)


# ==============================================================================
# 3. Loading, Whitelisting, and Partitioning
# ==============================================================================

def validate_feature_whitelist(columns: List[str]) -> Tuple[List[str], List[str]]:
    """Verify that columns conform strictly to the feature whitelist and contain no leakage."""
    allowed = []
    forbidden_found = []
    unrecognized = []

    for col in columns:
        if col in FORBIDDEN_LEAKAGE_COLUMNS:
            forbidden_found.append(col)
        elif col in FEATURE_WHITELIST:
            allowed.append(col)
        else:
            unrecognized.append(col)

    if forbidden_found:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Predictor set contains forbidden target/future variables: {forbidden_found}"
        )

    return allowed, unrecognized


def load_modeling_dataset(
    csv_path: Union[Path, str] = DEFAULT_COHORT_CSV,
    scenario: str = "SUSTAINED_ABSTINENCE",
    test_size: float = 0.20,
    val_size: float = 0.20,
    seed: int = 42,
    custom_features: Optional[List[str]] = None,
) -> ModelingCohortDataset:
    """Load, validate, filter, and partition the synthetic cohort for predictive modeling.

    Args:
        csv_path: Path to the synthetic cardiomyopathy cohort CSV.
        scenario: Clinical scenario filter ('SUSTAINED_ABSTINENCE' or 'ALL').
        test_size: Proportion of eligible cases allocated to held-out testing.
        val_size: Proportion of remaining non-test cases allocated to validation.
        seed: Random seed for deterministic stratified partitioning.
        custom_features: Optional subset of whitelisted features for ablation studies.

    Returns:
        ModelingCohortDataset with partitioned training, validation, and test splits.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic cohort CSV file not found: {path}")

    raw_df = pd.read_csv(path)
    total_raw_n = len(raw_df)

    # 1. Target column verification
    if TARGET_COLUMN not in raw_df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not present in cohort CSV.")

    # 2. Scenario eligibility filtering
    if scenario == "SUSTAINED_ABSTINENCE":
        if "assumed_abstinence_scenario" not in raw_df.columns:
            raise KeyError("Column 'assumed_abstinence_scenario' missing; cannot filter cohort.")
        df = raw_df[raw_df["assumed_abstinence_scenario"] == "SUSTAINED_ABSTINENCE"].copy()
    elif scenario == "ALL":
        df = raw_df.copy()
    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Must be 'SUSTAINED_ABSTINENCE' or 'ALL'.")

    # Drop any rows missing the target
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    eligible_n = len(df)

    if eligible_n < 50:
        raise ValueError(f"Eligible modeling cohort too small ({eligible_n} patients).")

    # 3. Select predictor features
    if custom_features is not None:
        validate_feature_whitelist(custom_features)
        feature_cols = [c for c in custom_features if c in df.columns]
    else:
        feature_cols = [c for c in FEATURE_WHITELIST if c in df.columns]

    validate_feature_whitelist(feature_cols)

    # Partition numerical, categorical, and binary features
    num_cols = [c for c in feature_cols if c in NUMERICAL_FEATURES]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    bin_cols = [c for c in feature_cols if c in BINARY_INDICATOR_FEATURES]

    # Convert binary flags to float/int
    for col in bin_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"]).astype(float)
        else:
            df[col] = df[col].astype(float)

    # Clean categorical columns
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("MISSING")

    # Clean numerical columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Stratified train/val/test splitting
    patient_ids = np.array(df["patient_id"].astype(str).tolist()) if "patient_id" in df.columns else np.arange(eligible_n).astype(str)
    y = np.array(df[TARGET_COLUMN].tolist(), dtype=int)
    X = df[feature_cols].copy().reset_index(drop=True)

    # Split into train_val (1 - test_size) and test (test_size)
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X,
        y,
        patient_ids,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Split train_val into train and val
    val_rel_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=val_rel_size,
        random_state=seed,
        stratify=y_train_val,
    )

    # Reset indices
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(y_train, name=TARGET_COLUMN, index=X_train.index)
    X_val = X_val.reset_index(drop=True)
    y_val = pd.Series(y_val, name=TARGET_COLUMN, index=X_val.index)
    X_test = X_test.reset_index(drop=True)
    y_test = pd.Series(y_test, name=TARGET_COLUMN, index=X_test.index)

    # Compile metadata
    rec_train_prev = float(y_train.mean())
    rec_val_prev = float(y_val.mean())
    rec_test_prev = float(y_test.mean())

    metadata = {
        "csv_path": str(path),
        "total_raw_patients": total_raw_n,
        "eligible_patients": eligible_n,
        "scenario": scenario,
        "split_seed": seed,
        "train_n": len(X_train),
        "val_n": len(X_val),
        "test_n": len(X_test),
        "train_recovery_prevalence": round(rec_train_prev, 4),
        "val_recovery_prevalence": round(rec_val_prev, 4),
        "test_recovery_prevalence": round(rec_test_prev, 4),
        "feature_count": len(feature_cols),
        "numerical_feature_count": len(num_cols),
        "categorical_feature_count": len(cat_cols),
        "binary_feature_count": len(bin_cols),
    }

    logger.info(
        f"Modeling dataset prepared: {eligible_n} eligible cases "
        f"(Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}). "
        f"Target prevalence: {rec_train_prev:.1%}"
    )

    return ModelingCohortDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_ids=list(ids_train),
        val_ids=list(ids_val),
        test_ids=list(ids_test),
        numerical_features=num_cols,
        categorical_features=cat_cols,
        binary_features=bin_cols,
        target_name=TARGET_COLUMN,
        scenario=scenario,
        metadata=metadata,
    )


# ==============================================================================
# 4. Preprocessing Pipeline Construction
# ==============================================================================

def build_feature_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    binary_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build an isolated Scikit-learn ColumnTransformer for baseline features.

    Ensures missing values are imputed and categorical features are one-hot encoded
    without data leakage from validation or test splits.
    """
    transformers = []

    if numerical_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numerical_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    if binary_features:
        bin_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("bin", bin_pipeline, binary_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def save_split_manifest(
    dataset: ModelingCohortDataset,
    output_path: Union[Path, str] = DEFAULT_SPLIT_MANIFEST_PATH,
) -> Path:
    """Save the train/val/test patient ID manifest to disk for reproducibility."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "metadata": dataset.metadata,
        "train_patient_ids": dataset.train_ids,
        "val_patient_ids": dataset.val_ids,
        "test_patient_ids": dataset.test_ids,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved modeling split manifest to {out}")
    return out
