"""Deterministic Regression Tests for EMIDEC Cohort Manifest Repair and LGE Training Preflight.

Validates:
1. Canonical EMIDEC manifest remains unchanged when already valid.
2. Empty EMIDEC manifest is safely repaired from authoritative canonical assignments.
3. ACDC and M&Ms splits remain strictly untouched during EMIDEC repair.
4. Missing physical cases on disk safely aborts repair with descriptive error.
5. Non-empty conflicting splits prevent silent overwriting and raise ValueError.
6. Manifest repair is fully idempotent.
7. LGE preflight rejects empty splits with clear, actionable remediation instructions.
8. LGE preflight validates full 60/20/20 cohort, masks, slices, and zero patient overlap.
9. EMIDEC trainer rejects invalid manifests before constructing DataLoaders.
10. Missing manifest file initializes cleanly from canonical manifest.
11. CLI prepare --repair-missing-splits executes successfully.
"""

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crip.pipeline.cli import main
from crip.pipeline.emidec_trainer import EMIDECTrainer, LGESegmentationConfig
from crip.pipeline.splits import (
    CANONICAL_UNIFIED_SPLITS_PATH,
    DEFAULT_UNIFIED_SPLITS_PATH,
    repair_unified_cohort_manifest,
)


@pytest.fixture
def canonical_splits_data():
    """Load reference canonical cohort splits."""
    with open(CANONICAL_UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestEMIDECManifestRepair:
    """Test suite for manifest repair logic and safeguards."""

    def test_valid_manifest_remains_unchanged(self, tmp_path, canonical_splits_data):
        """A complete and valid EMIDEC manifest must not be modified."""
        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(canonical_splits_data, f, indent=2)

        res = repair_unified_cohort_manifest(manifest_path=test_file)
        assert res["repaired"] is False
        assert res["status"] == "ALREADY_VALID"
        assert res["emidec_counts"]["train"] == 60
        assert res["emidec_counts"]["val"] == 20
        assert res["emidec_counts"]["test"] == 20

        # No backup should be created when no repair is performed
        assert not (tmp_path / "cohort_splits.json.bak").exists()

    def test_empty_emidec_manifest_is_safely_repaired(self, tmp_path, canonical_splits_data):
        """An empty EMIDEC split section is populated from canonical splits while creating a backup."""
        corrupted_data = copy.deepcopy(canonical_splits_data)
        corrupted_data["datasets"]["emidec"]["splits"] = {"train": [], "val": [], "test": []}
        corrupted_data["datasets"]["emidec"]["total_cases"] = 0
        corrupted_data["datasets"]["emidec"]["split_counts"] = {"train": 0, "val": 0, "test": 0}

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=2)

        res = repair_unified_cohort_manifest(manifest_path=test_file)
        assert res["repaired"] is True
        assert res["status"] in ("REPAIRED", "REPAIRED_FROM_CANONICAL")
        assert res["emidec_counts"]["train"] == 60
        assert res["emidec_counts"]["val"] == 20
        assert res["emidec_counts"]["test"] == 20

        # Backup file must exist and contain the corrupted data
        bak_file = tmp_path / "cohort_splits.json.bak"
        assert bak_file.is_file()
        with open(bak_file, "r", encoding="utf-8") as f:
            bak_data = json.load(f)
        assert len(bak_data["datasets"]["emidec"]["splits"]["train"]) == 0

        # Repaired file must be valid
        with open(test_file, "r", encoding="utf-8") as f:
            repaired_data = json.load(f)
        assert len(repaired_data["datasets"]["emidec"]["splits"]["train"]) == 60
        assert len(repaired_data["datasets"]["emidec"]["splits"]["val"]) == 20
        assert len(repaired_data["datasets"]["emidec"]["splits"]["test"]) == 20

    def test_acdc_and_mnms_splits_strictly_preserved(self, tmp_path, canonical_splits_data):
        """Repairing EMIDEC splits must not alter ACDC or M&Ms splits in any way."""
        corrupted_data = copy.deepcopy(canonical_splits_data)
        corrupted_data["datasets"]["emidec"]["splits"] = {"train": [], "val": [], "test": []}

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=2)

        repair_unified_cohort_manifest(manifest_path=test_file)

        with open(test_file, "r", encoding="utf-8") as f:
            repaired_data = json.load(f)

        assert repaired_data["datasets"]["acdc"] == canonical_splits_data["datasets"]["acdc"]
        assert repaired_data["datasets"]["mnms"] == canonical_splits_data["datasets"]["mnms"]

    def test_missing_cases_on_disk_aborts_repair(self, tmp_path, canonical_splits_data):
        """Repair must fail with RuntimeError if expected annotated cases are not on disk."""
        corrupted_data = copy.deepcopy(canonical_splits_data)
        corrupted_data["datasets"]["emidec"]["splits"] = {"train": [], "val": [], "test": []}

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=2)

        # Mock EMIDEC loader discovering only 50 cases
        mock_cases = {f"Case_{i:03d}": {"has_contour": True} for i in range(1, 51)}

        with patch("crip.datasets.emidec_loader.EMIDECDataLoader.discover_cases", return_value=mock_cases):
            with pytest.raises(RuntimeError) as exc_info:
                repair_unified_cohort_manifest(manifest_path=test_file)
            assert "Cannot repair EMIDEC manifest: Found 50 / 100 annotated cases on disk" in str(exc_info.value)

    def test_conflicting_splits_prevents_silent_overwrite(self, tmp_path, canonical_splits_data):
        """Non-empty conflicting splits must raise ValueError instead of overwriting."""
        conflicting_data = copy.deepcopy(canonical_splits_data)
        # Put an arbitrary incomplete subset in train
        conflicting_data["datasets"]["emidec"]["splits"]["train"] = ["Case_001", "Case_002"]

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(conflicting_data, f, indent=2)

        with pytest.raises(ValueError) as exc_info:
            repair_unified_cohort_manifest(manifest_path=test_file)
        assert "Conflicting, non-empty EMIDEC splits detected" in str(exc_info.value)

    def test_repair_is_idempotent(self, tmp_path, canonical_splits_data):
        """Subsequent repair executions on an already repaired manifest must be no-ops."""
        corrupted_data = copy.deepcopy(canonical_splits_data)
        corrupted_data["datasets"]["emidec"]["splits"] = {"train": [], "val": [], "test": []}

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=2)

        # First run: repairs
        res1 = repair_unified_cohort_manifest(manifest_path=test_file)
        assert res1["repaired"] is True
        assert res1["status"] in ("REPAIRED", "REPAIRED_FROM_CANONICAL")

        # Second run: already valid
        res2 = repair_unified_cohort_manifest(manifest_path=test_file)
        assert res2["repaired"] is False
        assert res2["status"] == "ALREADY_VALID"

    def test_missing_manifest_initializes_from_canonical(self, tmp_path):
        """If manifest file does not exist, repair initializes it from canonical manifest."""
        test_file = tmp_path / "new_dir" / "cohort_splits.json"
        assert not test_file.exists()

        res = repair_unified_cohort_manifest(manifest_path=test_file)
        assert res["repaired"] is True
        assert res["status"] == "INITIALIZED_FROM_CANONICAL"
        assert test_file.is_file()

        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["datasets"]["emidec"]["splits"]["train"]) == 60


class TestEMIDECLGEPreflight:
    """Test suite for EMIDEC LGE trainer preflight and training guards."""

    def test_preflight_rejects_empty_manifest(self, tmp_path, canonical_splits_data):
        """Preflight must reject an empty EMIDEC manifest and provide actionable instructions."""
        corrupted_data = copy.deepcopy(canonical_splits_data)
        corrupted_data["datasets"]["emidec"]["splits"] = {"train": [], "val": [], "test": []}

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=2)

        config = LGESegmentationConfig()
        trainer = EMIDECTrainer(config=config, splits_path=test_file)

        report = trainer.validate_preflight(strict=False)
        assert report["is_valid"] is False
        assert any("prepare --repair-missing-splits" in err for err in report["errors"])

        with pytest.raises(RuntimeError) as exc_info:
            trainer.validate_preflight(strict=True)
        assert "prepare --repair-missing-splits" in str(exc_info.value)

    def test_preflight_validates_complete_dataset(self):
        """Preflight must pass and verify all 60/20/20 cases, 432/136 slices, and zero overlap."""
        config = LGESegmentationConfig()
        trainer = EMIDECTrainer(config=config)
        report = trainer.validate_preflight(strict=False)

        assert report["is_valid"] is True
        assert report["manifest_present"] is True
        assert report["train_cases_expected"] == 60
        assert report["val_cases_expected"] == 20
        assert report["test_cases_expected"] == 20
        assert report["train_available"] == 60
        assert report["val_available"] == 20
        assert report["test_available"] == 20
        assert report["patient_overlap"] == 0
        assert report["expert_masks_verified"] is True
        assert report["train_slices"] == 432
        assert report["val_slices"] == 136
        assert report["infarct_slices"] == 326
        assert report["mvo_slices"] == 112

    def test_train_aborts_on_invalid_preflight_before_dataloaders(self, tmp_path, canonical_splits_data):
        """EMIDECTrainer.train() must abort at preflight before creating DataLoaders with 0 samples."""
        corrupted_data = copy.deepcopy(canonical_splits_data)
        corrupted_data["datasets"]["emidec"]["splits"] = {"train": [], "val": [], "test": []}

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=2)

        config = LGESegmentationConfig(epochs=1)
        trainer = EMIDECTrainer(
            config=config,
            splits_path=test_file,
            checkpoint_path=tmp_path / "test_model.pt",
        )

        with pytest.raises(RuntimeError) as exc_info:
            trainer.train(epochs=1)
        # Must fail with clear preflight message, not PyTorch ValueError
        assert "EMIDEC preflight validation failed" in str(exc_info.value)
        assert "prepare --repair-missing-splits" in str(exc_info.value)

    def test_cli_prepare_repair_missing_splits(self, tmp_path, canonical_splits_data):
        """CLI prepare --repair-missing-splits runs successfully and repairs empty splits."""
        corrupted_data = copy.deepcopy(canonical_splits_data)
        corrupted_data["datasets"]["emidec"]["splits"] = {"train": [], "val": [], "test": []}

        test_file = tmp_path / "cohort_splits.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=2)

        exit_code = main(["prepare", "--output", str(test_file), "--repair-missing-splits"])
        assert exit_code == 0

        with open(test_file, "r", encoding="utf-8") as f:
            repaired_data = json.load(f)
        assert len(repaired_data["datasets"]["emidec"]["splits"]["train"]) == 60

    def test_cli_train_dry_run(self):
        """CLI train --task lge --dry-run returns exit code 0."""
        exit_code = main(["train", "--task", "lge", "--dry-run"])
        assert exit_code == 0
