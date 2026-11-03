"""Regression and Provenance Integrity Tests for Stage 7.1.

Guarantees:
1. Strict separation of ACDC training vs. held-out test patient identifiers.
2. Prevention of training subjects (e.g. patient001) from being labeled as held-out evaluation cases.
3. Verification that actual held-out test subjects (e.g. patient007) match persisted split manifests.
4. Prevention of unannotated EMIDEC challenge cases (Case_101-Case_150) from being labeled as expert-mask evaluation cases.
5. Verification that annotated EMIDEC cases (Case_001-Case_100 / Case_N001-P100) have authentic ground-truth contours.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from crip.datasets.emidec_loader import EMIDECDataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
ACDC_SPLITS_PATH = ROOT_DIR / "docs/acdc/dataset_split.json"
UNIFIED_SPLITS_PATH = ROOT_DIR / "data/processed/unified/cohort_splits.json"


class TestACDCPatientSplitIntegrity:
    """Ensure patient assignments strictly respect training vs held-out test partitions."""

    def test_acdc_splits_file_exists_and_valid(self):
        assert ACDC_SPLITS_PATH.is_file(), f"Missing {ACDC_SPLITS_PATH}"
        with open(ACDC_SPLITS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        splits = data.get("splits", {})
        assert len(splits.get("train", [])) == 60
        assert len(splits.get("val", [])) == 20
        assert len(splits.get("test", [])) == 20

    def test_patient001_is_strictly_training_not_test(self):
        with open(ACDC_SPLITS_PATH, "r", encoding="utf-8") as f:
            acdc_data = json.load(f)
        acdc_splits = acdc_data.get("splits", {})
        assert "patient001" in acdc_splits["train"], "patient001 must be in ACDC train split"
        assert "patient001" not in acdc_splits["test"], "patient001 must NOT be in ACDC test split"

        with open(UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as f:
            u_data = json.load(f)
        acdc_u = u_data["datasets"]["acdc"]["splits"]
        assert "patient001" in acdc_u["train"], "patient001 must be in unified ACDC train split"
        assert "patient001" not in acdc_u["test"], "patient001 must NOT be in unified ACDC test split"

    def test_patient007_is_confirmed_held_out_test(self):
        with open(ACDC_SPLITS_PATH, "r", encoding="utf-8") as f:
            acdc_data = json.load(f)
        acdc_splits = acdc_data.get("splits", {})
        assert "patient007" in acdc_splits["test"], "patient007 must be in ACDC held-out test split"
        assert "patient007" not in acdc_splits["train"], "patient007 must NOT be in train split"

        with open(UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as f:
            u_data = json.load(f)
        acdc_u = u_data["datasets"]["acdc"]["splits"]
        assert "patient007" in acdc_u["test"], "patient007 must be in unified ACDC test split"


class TestEMIDECMaskProvenanceIntegrity:
    """Ensure unannotated challenge cases cannot be misrepresented as expert-mask ground truth."""

    def test_case_101_is_unannotated_challenge_case_without_mask(self):
        loader = EMIDECDataLoader()
        meta = loader.get_case_metadata("Case_101")
        assert meta.mask_path is None, "Case_101 must not have a mask path"
        assert meta.has_mask is False, "Case_101 must have has_mask == False"
        assert meta.raw_labels_present == [], "Case_101 must have no raw labels"

    def test_unannotated_cases_excluded_from_unified_splits(self):
        with open(UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as f:
            u_data = json.load(f)
        emidec_splits = u_data["datasets"]["emidec"]["splits"]

        # Ensure no challenge cases (101-150) are in the training/val/test splits
        for split_name, cases in emidec_splits.items():
            for c in cases:
                assert not (c.startswith("Case_1") and len(c) == 8 and c[5:].isdigit() and int(c[5:]) >= 101), (
                    f"Unannotated challenge case {c} found in {split_name} split!"
                )

    def test_case_p003_has_verified_expert_mask_in_test_partition(self):
        with open(UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as f:
            u_data = json.load(f)
        emidec_splits = u_data["datasets"]["emidec"]["splits"]
        assert "Case_P003" in emidec_splits["test"], "Case_P003 must be in EMIDEC test split"

        loader = EMIDECDataLoader()
        meta = loader.get_case_metadata("Case_P003")
        assert meta.has_mask is True, "Case_P003 must have verified ground truth mask"
        assert meta.mask_path is not None and meta.mask_path.is_file()
        assert 3 in meta.raw_labels_present, "Case_P003 must contain infarction label (3)"
