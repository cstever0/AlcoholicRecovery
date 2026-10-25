"""Reproducible Patient-Level Dataset Splitting for ACDC Cardiac MRI.

Partitions the 100 annotated ACDC training patients into:
  - 60 Training patients (60%)
  - 20 Validation patients (20%)
  - 20 Held-out Test patients (20%)
Stratified equally across all 5 diagnostic categories (DCM, HCM, MINF, NOR, RV),
ensuring exactly 12 train, 4 val, and 4 test patients per diagnosis group.

CRITICAL CLINICAL & METHODOLOGICAL SAFETY:
------------------------------------------
1. Splitting is strictly performed at the PATIENT level. All cine phases (ED, ES)
   and all short-axis slices belonging to a given patient reside in the same split.
2. The held-out test split is strictly reserved for final unbiased evaluation;
   it is never used during model training, hyperparameter tuning, or checkpoint selection.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crip.mri.acdc_loader import ACDCDataLoader, DEFAULT_ACDC_TRAINING_DIR

logger = logging.getLogger("crip.mri.dataset_splits")

DEFAULT_SPLIT_MANIFEST_PATH = Path("data/processed/acdc/dataset_split.json")
DOCS_SPLIT_MANIFEST_PATH = Path("docs/acdc/dataset_split.json")


def create_stratified_acdc_splits(
    data_dir: Path | str = DEFAULT_ACDC_TRAINING_DIR,
    seed: int = 42,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Generate deterministic, stratified patient-level dataset splits.

    Args:
        data_dir: Path to ACDC training dataset directory.
        seed: Random seed for deterministic reproducibility.
        train_ratio: Target proportion for training split (default: 0.60).
        val_ratio: Target proportion for validation split (default: 0.20).
        test_ratio: Target proportion for held-out test split (default: 0.20).

    Returns:
        Structured dictionary containing split assignments, diagnostic distributions,
        and patient metadata.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0: {train_ratio + val_ratio + test_ratio}")

    loader = ACDCDataLoader(data_dir)
    patient_ids = loader.list_patients()
    if not patient_ids:
        raise ValueError(f"No patients found in {data_dir}")

    # Group patients by diagnostic category
    patients_by_group: Dict[str, List[str]] = defaultdict(list)
    patient_meta_map: Dict[str, Dict[str, Any]] = {}

    for pid in patient_ids:
        meta = loader.get_patient_metadata(pid)
        patients_by_group[meta.group].append(pid)
        patient_meta_map[pid] = {
            "patient_id": pid,
            "group": meta.group,
            "ed_frame": meta.ed_frame,
            "es_frame": meta.es_frame,
            "height_cm": meta.height_cm,
            "weight_kg": meta.weight_kg,
        }

    rng = np.random.RandomState(seed)

    train_pids: List[str] = []
    val_pids: List[str] = []
    test_pids: List[str] = []

    group_distribution: Dict[str, Dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    # Stratified patient selection across each diagnostic group
    for group_name in sorted(patients_by_group.keys()):
        group_pids = sorted(patients_by_group[group_name])
        n_group = len(group_pids)

        # Shuffle deterministically
        shuffled = list(rng.permutation(group_pids))

        n_train = int(round(n_group * train_ratio))
        n_val = int(round(n_group * val_ratio))
        # Ensure remaining go to test
        n_test = n_group - n_train - n_val

        grp_train = shuffled[:n_train]
        grp_val = shuffled[n_train : n_train + n_val]
        grp_test = shuffled[n_train + n_val :]

        train_pids.extend(grp_train)
        val_pids.extend(grp_val)
        test_pids.extend(grp_test)

        group_distribution["train"][group_name] = len(grp_train)
        group_distribution["val"][group_name] = len(grp_val)
        group_distribution["test"][group_name] = len(grp_test)

    # Sort final patient ID lists for clean determinism
    train_pids = sorted(train_pids)
    val_pids = sorted(val_pids)
    test_pids = sorted(test_pids)

    # Integrity verification: mutual exclusivity and collective exhaustiveness
    all_assigned = set(train_pids) | set(val_pids) | set(test_pids)
    assert len(all_assigned) == len(patient_ids), "Missing or duplicate patient assignments"
    assert len(set(train_pids) & set(val_pids)) == 0, "Train and Val overlap detected"
    assert len(set(train_pids) & set(test_pids)) == 0, "Train and Test overlap detected"
    assert len(set(val_pids) & set(test_pids)) == 0, "Val and Test overlap detected"

    manifest = {
        "seed": seed,
        "stratification_key": "group",
        "total_patients": len(patient_ids),
        "counts": {
            "train": len(train_pids),
            "val": len(val_pids),
            "test": len(test_pids),
        },
        "group_distribution": {
            k: dict(v) for k, v in group_distribution.items()
        },
        "splits": {
            "train": train_pids,
            "val": val_pids,
            "test": test_pids,
        },
        "patient_metadata": patient_meta_map,
    }

    return manifest


def save_dataset_splits(
    manifest: Dict[str, Any],
    destination_path: Path | str = DEFAULT_SPLIT_MANIFEST_PATH,
) -> Path:
    """Save dataset split manifest to JSON file."""
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return dest


def load_dataset_splits(
    manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Load dataset split manifest from JSON file."""
    src = Path(manifest_path)
    if not src.is_file():
        raise FileNotFoundError(f"Dataset split manifest not found: {src}")
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_create_splits(
    manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST_PATH,
    data_dir: Path | str = DEFAULT_ACDC_TRAINING_DIR,
    seed: int = 42,
) -> Dict[str, Any]:
    """Load existing split manifest if present, or generate and save a new one."""
    path = Path(manifest_path)
    if path.is_file():
        try:
            return load_dataset_splits(path)
        except Exception as e:
            logger.warning(f"Could not load existing manifest at {path}: {e}. Regenerating...")

    manifest = create_stratified_acdc_splits(data_dir=data_dir, seed=seed)
    save_dataset_splits(manifest, path)
    # Also save a copy to docs for persistent version tracking
    save_dataset_splits(manifest, DOCS_SPLIT_MANIFEST_PATH)
    return manifest
