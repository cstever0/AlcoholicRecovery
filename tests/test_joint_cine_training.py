"""Deterministic Unit and Integration Tests for Joint Cine Training and Evaluation.

Verifies:
1. Joint dataset indexing across ACDC and M&Ms cohorts.
2. Balanced dataset-aware batch sampling.
3. Label translation invariance (M&Ms 1=LV, 3=RV -> Canonical 1=RV, 3=LV).
4. Strict checkpoint isolation (baseline ACDC checkpoint is never overwritten).
5. Matched evaluation cohort isolation (zero overlap with training/validation).
6. Missing dataset and checkpoint error handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from crip.datasets.labels import ACDCLabel, MNMSLabel, map_mnms_to_canonical
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH as BASELINE_CHECKPOINT
from crip.pipeline.joint_dataset import (
    BalancedDatasetBatchSampler,
    JointCineSegmentationDataset,
)
from crip.pipeline.joint_trainer import (
    DEFAULT_JOINT_CHECKPOINT_PATH,
    compute_file_sha256,
)
from crip.pipeline.models_config import CineSegmentationConfig
from crip.pipeline.splits import DEFAULT_UNIFIED_SPLITS_PATH


class TestJointDatasetAndSampling:
    """Validates joint multi-dataset data pipeline and balanced sampling."""

    def test_joint_dataset_indexing(self) -> None:
        ds = JointCineSegmentationDataset(
            acdc_patient_ids=["patient001"],
            mnms_case_ids=["A0S9V9"],
            target_size=(224, 224),
            augment=False,
            cache_data=True,
        )
        assert len(ds) > 0
        datasets = {s["dataset"] for s in ds.slices}
        assert "acdc" in datasets
        assert "mnms" in datasets

        sample = ds[0]
        assert sample["image"].shape == (1, 224, 224)
        assert sample["mask"].shape == (1, 224, 224)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.int64

    def test_balanced_batch_sampler(self) -> None:
        ds = JointCineSegmentationDataset(
            acdc_patient_ids=["patient001"],
            mnms_case_ids=["A0S9V9"],
            target_size=(224, 224),
            augment=False,
            cache_data=True,
        )
        batch_size = 4
        sampler = BalancedDatasetBatchSampler(ds, batch_size=batch_size, seed=42)
        loader = DataLoader(ds, batch_sampler=sampler)

        for batch in loader:
            datasets_in_batch = batch["dataset"]
            assert len(datasets_in_batch) == batch_size
            acdc_count = sum(1 for d in datasets_in_batch if d == "acdc")
            mnms_count = sum(1 for d in datasets_in_batch if d == "mnms")
            assert acdc_count == batch_size // 2
            assert mnms_count == batch_size // 2
            break

    def test_label_translation_invariance(self) -> None:
        # Create synthetic M&Ms mask with raw labels
        raw_mnms = np.array([
            [int(MNMSLabel.BACKGROUND), int(MNMSLabel.LV_CAVITY)],
            [int(MNMSLabel.MYOCARDIUM), int(MNMSLabel.RV_CAVITY)],
        ], dtype=np.int64)

        canonical = map_mnms_to_canonical(raw_mnms)

        # In canonical:
        # Background remains 0
        assert canonical[0, 0] == int(ACDCLabel.BACKGROUND)
        # M&Ms LV (1) -> Canonical LV (3)
        assert canonical[0, 1] == int(ACDCLabel.LV_CAVITY)
        # M&Ms Myo (2) -> Canonical Myo (2)
        assert canonical[1, 0] == int(ACDCLabel.MYOCARDIUM)
        # M&Ms RV (3) -> Canonical RV (1)
        assert canonical[1, 1] == int(ACDCLabel.RV_CAVITY)


class TestCheckpointAndSplitIsolation:
    """Validates patient isolation and checkpoint protection."""

    def test_baseline_checkpoint_protected(self) -> None:
        assert BASELINE_CHECKPOINT.is_file()
        assert DEFAULT_JOINT_CHECKPOINT_PATH != BASELINE_CHECKPOINT

    def test_matched_cohort_isolation(self) -> None:
        with open(DEFAULT_UNIFIED_SPLITS_PATH, "r") as f:
            manifest = json.load(f)

        acdc = manifest["datasets"]["acdc"]["splits"]
        mnms = manifest["datasets"]["mnms"]["splits"]

        # ACDC test set (20 patients) must have zero overlap with training
        assert set(acdc["test"]).isdisjoint(set(acdc["train"]))
        assert set(acdc["test"]).isdisjoint(set(acdc["val"]))
        assert len(acdc["test"]) == 20

        # M&Ms held-out evaluation cohort (34 studies) must have zero overlap with training pool
        assert set(mnms["held_out_evaluation"]).isdisjoint(set(mnms["train"]))
        assert set(mnms["held_out_evaluation"]).isdisjoint(set(mnms["val"]))
        assert len(mnms["held_out_evaluation"]) == 34
        assert len(mnms["train"]) == 60
        assert len(mnms["val"]) == 20

    def test_missing_checkpoint_handling(self, tmp_path: Path) -> None:
        from crip.pipeline.joint_evaluation import MatchedCineEvaluator
        non_existent = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError):
            MatchedCineEvaluator(joint_checkpoint=non_existent)


class TestWSLSafeguardsAndPreflight:
    """Validates preflight checks, missing dataset detection, and sampler protections."""

    def test_sampler_empty_mnms_raises_value_error(self) -> None:
        class MockDataset:
            def __init__(self):
                self.slices = [{"dataset": "acdc"} for _ in range(10)]

        ds = MockDataset()
        with pytest.raises(ValueError, match="Joint cine training requires annotated M&Ms"):
            BalancedDatasetBatchSampler(ds, batch_size=4, seed=42)

    def test_sampler_empty_acdc_raises_value_error(self) -> None:
        class MockDataset:
            def __init__(self):
                self.slices = [{"dataset": "mnms"} for _ in range(10)]

        ds = MockDataset()
        with pytest.raises(ValueError, match="ACDC pool is empty"):
            BalancedDatasetBatchSampler(ds, batch_size=4, seed=42)

    def test_sampler_both_empty_raises_value_error(self) -> None:
        class MockDataset:
            def __init__(self):
                self.slices = []

        ds = MockDataset()
        with pytest.raises(ValueError, match="completely empty"):
            BalancedDatasetBatchSampler(ds, batch_size=4, seed=42)

    def test_preflight_audit_missing_mnms(self, tmp_path: Path) -> None:
        from crip.pipeline.joint_trainer import JointCineTrainer

        cfg = CineSegmentationConfig(mnms_data_dir=str(tmp_path / "empty_mnms"))
        trainer = JointCineTrainer(config=cfg)
        report = trainer.validate_preflight(strict=False)

        assert report["is_valid"] is False
        assert report["mnms_train_available"] == 0
        assert any("Joint cine training requires annotated M&Ms" in e for e in report["errors"])

        with pytest.raises(RuntimeError, match="Joint cine training requires annotated M&Ms"):
            trainer.validate_preflight(strict=True)

    def test_preflight_audit_missing_acdc(self, tmp_path: Path) -> None:
        from crip.pipeline.joint_trainer import JointCineTrainer

        cfg = CineSegmentationConfig(acdc_data_dir=str(tmp_path / "empty_acdc"))
        trainer = JointCineTrainer(config=cfg)
        report = trainer.validate_preflight(strict=False)

        assert report["is_valid"] is False
        assert report["acdc_train_available"] == 0
        assert any("ACDC raw training dataset not found" in e for e in report["errors"])

    def test_preflight_audit_passes_with_real_data(self) -> None:
        from crip.pipeline.joint_trainer import JointCineTrainer

        trainer = JointCineTrainer()
        report = trainer.validate_preflight(strict=False)

        assert report["is_valid"] is True
        assert report["acdc_train_available"] == 60
        assert report["acdc_val_available"] == 20
        assert report["mnms_train_available"] == 60
        assert report["mnms_val_available"] == 20
        assert report["holdout_quarantine_verified"] is True
        assert len(report["errors"]) == 0

    def test_cli_dry_run_success(self) -> None:
        from crip.pipeline.cli import main

        code = main(["train", "--task", "cine", "--dry-run"])
        assert code == 0

    def test_cli_dry_run_fails_with_missing_mnms(self, tmp_path: Path) -> None:
        from crip.pipeline.cli import main

        code = main(["train", "--task", "cine", "--dry-run", "--mnms-dir", str(tmp_path / "missing")])
        assert code == 1

