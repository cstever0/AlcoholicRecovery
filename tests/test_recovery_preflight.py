"""Deterministic unit tests for Recovery Pipeline Preflight and Dependency Validation.

Validates:
1. Preflight audit passes when all source measurement CSVs and feature bank exist.
2. Missing ACDC measurements triggers preflight failure with remediation command.
3. Missing M&Ms external evaluation triggers preflight failure with remediation command.
4. Missing EMIDEC tissue measurements triggers preflight failure with remediation command.
5. Missing unified feature bank triggers preflight failure with remediation command.
6. Strict mode raises RuntimeError with actionable instructions.
7. CLI dry run reflects validation status accurately without executing training.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from crip.pipeline.cli import main
from crip.pipeline.recovery_trainer import validate_recovery_preflight


class TestRecoveryPreflightAudit:
    """Test suite for recovery training preflight prerequisites."""

    def test_preflight_passes_with_existing_artifacts(self):
        """Preflight must succeed when all source measurement files and feature bank are present."""
        report = validate_recovery_preflight(strict=False)
        assert report["is_valid"] is True
        assert report["acdc_csv_present"] is True
        assert report["mnms_csv_present"] is True
        assert report["emidec_csv_present"] is True
        assert report["feature_bank_present"] is True
        assert report["acdc_records"] >= 100
        assert report["mnms_records"] >= 114
        assert report["emidec_records"] >= 100
        assert report["feature_bank_records"] >= 314
        assert len(report["errors"]) == 0

    def test_missing_acdc_measurements_triggers_error_and_remediation(self, tmp_path):
        """Missing ACDC measurements must fail preflight and instruct user to run crip.mri.pipeline."""
        with patch("crip.pipeline.recovery_trainer.DEFAULT_ACDC_CSV", tmp_path / "missing_acdc.csv"):
            report = validate_recovery_preflight(strict=False)
            assert report["is_valid"] is False
            assert any("Missing ACDC measurements" in err for err in report["errors"])
            assert any("crip.mri.pipeline" in rem for rem in report["remediation"])

    def test_missing_mnms_evaluation_triggers_error_and_remediation(self, tmp_path):
        """Missing M&Ms evaluation must fail preflight and instruct user to run crip.evaluation.external_mnms."""
        with patch("crip.pipeline.recovery_trainer.DEFAULT_MNMS_CSV", tmp_path / "missing_mnms.csv"):
            report = validate_recovery_preflight(strict=False)
            assert report["is_valid"] is False
            assert any("Missing M&Ms external evaluation" in err for err in report["errors"])
            assert any("crip.evaluation.external_mnms" in rem for rem in report["remediation"])

    def test_missing_emidec_measurements_triggers_error_and_remediation(self, tmp_path):
        """Missing EMIDEC measurements must fail preflight and instruct user to run crip.emidec.tissue_metrics."""
        with patch("crip.pipeline.recovery_trainer.DEFAULT_EMIDEC_CSV", tmp_path / "missing_emidec.csv"):
            report = validate_recovery_preflight(strict=False)
            assert report["is_valid"] is False
            assert any("Missing EMIDEC tissue measurements" in err for err in report["errors"])
            assert any("crip.emidec.tissue_metrics" in rem for rem in report["remediation"])

    def test_missing_feature_bank_triggers_error_and_remediation(self, tmp_path):
        """Missing feature bank must fail preflight and instruct user to run crip.pipeline features."""
        with patch("crip.pipeline.recovery_trainer.DEFAULT_FEATURE_BANK_CSV", tmp_path / "missing_fb.csv"):
            report = validate_recovery_preflight(strict=False)
            assert report["is_valid"] is False
            assert any("Missing unified empirical feature bank" in err for err in report["errors"])
            assert any("crip.pipeline features" in rem for rem in report["remediation"])

    def test_strict_mode_raises_runtime_error(self, tmp_path):
        """In strict mode, failing preflight must raise RuntimeError with remediation commands."""
        with patch("crip.pipeline.recovery_trainer.DEFAULT_FEATURE_BANK_CSV", tmp_path / "missing_fb.csv"):
            with pytest.raises(RuntimeError) as exc_info:
                validate_recovery_preflight(strict=True)
            assert "Recovery Pipeline Preflight Validation Failed" in str(exc_info.value)
            assert "crip.pipeline features" in str(exc_info.value)

    def test_cli_recovery_dry_run_success(self):
        """CLI train --task recovery --dry-run returns 0 when valid."""
        exit_code = main(["train", "--task", "recovery", "--dry-run"])
        assert exit_code == 0

    def test_cli_recovery_dry_run_failure_when_missing_feature_bank(self, tmp_path):
        """CLI train --task recovery --dry-run returns 1 when feature bank is missing."""
        with patch("crip.pipeline.recovery_trainer.DEFAULT_FEATURE_BANK_CSV", tmp_path / "missing_fb.csv"):
            exit_code = main(["train", "--task", "recovery", "--dry-run"])
            assert exit_code == 1
