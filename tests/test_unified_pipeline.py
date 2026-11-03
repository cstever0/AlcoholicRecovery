"""Deterministic Unit and Integration Tests for CRIP Unified Pipeline Architecture.

Verifies:
1. Unified task and data schemas (Task enums, case records, feature bank records).
2. Cohort split generation and strict zero-leakage invariant.
3. Invariant preservation of the 20-patient held-out ACDC benchmark test set.
4. M&Ms partition isolation (80 training pool vs 34 untouched held-out evaluation).
5. EMIDEC balanced stratified partitioning (60/20/20 and 5-fold CV).
6. Empirical feature bank extraction across 314 cases without synthetic conflation.
7. Empirically grounded synthetic cohort simulation and baseline backward compatibility.
8. CLI subcommands (audit, prepare, features, synthesize, evaluate, train dry-run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crip.pipeline.cli import main as cli_main
from crip.pipeline.feature_bank import (
    DEFAULT_OUTPUT_CSV as FEATURE_BANK_CSV,
    DEFAULT_OUTPUT_SUMMARY as FEATURE_BANK_SUMMARY,
    build_unified_feature_bank,
)
from crip.pipeline.models_config import CineSegmentationConfig, LGESegmentationConfig
from crip.pipeline.schemas import (
    EmpiricalFeatureRecord,
    Modality,
    PartitionSplit,
    PipelineTask,
    UnifiedCaseRecord,
)
from crip.pipeline.splits import (
    DEFAULT_ACDC_SPLIT_JSON,
    DEFAULT_UNIFIED_SPLITS_PATH,
    build_unified_cohort_manifest,
    create_emidec_splits,
    create_mnms_splits,
    load_acdc_splits,
)
from crip.pipeline.synthetic_bridge import (
    EmpiricallyGroundedCohortGenerator,
    load_unified_empirical_priors,
)


class TestUnifiedPipelineSchemas:
    """Validates unified task definitions and schema models."""

    def test_pipeline_task_identifiers(self) -> None:
        tasks = [t.value for t in PipelineTask]
        assert "CINE_VENTRICULAR_SEGMENTATION" in tasks
        assert "LGE_TISSUE_SEGMENTATION" in tasks
        assert "EMPIRICAL_FEATURE_EXTRACTION" in tasks
        assert "SYNTHETIC_COHORT_GENERATION" in tasks
        assert "SYNTHETIC_RECOVERY_MODELING" in tasks

    def test_unified_case_record_roundtrip(self) -> None:
        rec = UnifiedCaseRecord(
            dataset_id="acdc",
            patient_id="patient001",
            modality=Modality.CINE_CMR,
            image_path="data/raw/acdc/training/patient001/patient001_4d.nii.gz",
            source_partition="training",
            assigned_split=PartitionSplit.TRAIN,
            has_mask=True,
            supported_tasks=[
                PipelineTask.CINE_VENTRICULAR_SEGMENTATION,
                PipelineTask.EMPIRICAL_FEATURE_EXTRACTION,
            ],
            clinical_variables={"diagnostic_group": "DCM", "height_cm": 184.0},
        )
        d = rec.to_dict()
        assert d["modality"] == "CINE_CMR"
        assert d["assigned_split"] == "train"
        assert len(d["supported_tasks"]) == 2

        restored = UnifiedCaseRecord.from_dict(d)
        assert restored.patient_id == "patient001"
        assert restored.modality == Modality.CINE_CMR
        assert restored.assigned_split == PartitionSplit.TRAIN
        assert PipelineTask.CINE_VENTRICULAR_SEGMENTATION in restored.supported_tasks

    def test_empirical_feature_record_serialization(self) -> None:
        feat = EmpiricalFeatureRecord(
            case_id="patient001",
            dataset_id="acdc",
            modality=Modality.CINE_CMR,
            mask_provenance="expert_ground_truth",
            measurement_method="Simpson_rule_voxel_integration",
            diagnostic_group="DCM",
            lvedv_ml=220.5,
            lvef_percent=26.4,
            is_cine_complete=True,
            is_viability_complete=False,
        )
        d = feat.to_dict()
        assert d["case_id"] == "patient001"
        assert d["lvedv_ml"] == 220.5
        assert d["modality"] == "CINE_CMR"


class TestCohortPartitionsAndLeakage:
    """Validates patient-level split generation and strict zero-leakage protection."""

    def test_acdc_test_partition_preservation(self) -> None:
        splits = load_acdc_splits()
        assert len(splits["test"]) == 20
        assert len(splits["train"]) == 60
        assert len(splits["val"]) == 20

        # Verify against saved reference JSON
        with open(DEFAULT_ACDC_SPLIT_JSON, "r") as f:
            ref = json.load(f)["splits"]
        assert sorted(splits["test"]) == sorted(ref["test"])

    def test_zero_leakage_within_all_datasets(self) -> None:
        manifest = build_unified_cohort_manifest(seed=42)

        # Check ACDC
        acdc_s = manifest["datasets"]["acdc"]["splits"]
        assert set(acdc_s["train"]).isdisjoint(set(acdc_s["val"]))
        assert set(acdc_s["train"]).isdisjoint(set(acdc_s["test"]))
        assert set(acdc_s["val"]).isdisjoint(set(acdc_s["test"]))

        # Check EMIDEC
        emidec_s = manifest["datasets"]["emidec"]["splits"]
        assert set(emidec_s["train"]).isdisjoint(set(emidec_s["val"]))
        assert set(emidec_s["train"]).isdisjoint(set(emidec_s["test"]))
        assert set(emidec_s["val"]).isdisjoint(set(emidec_s["test"]))
        assert len(emidec_s["train"]) + len(emidec_s["val"]) + len(emidec_s["test"]) == 100

        # Check M&Ms
        mnms_s = manifest["datasets"]["mnms"]["splits"]
        assert set(mnms_s["train_pool"]).isdisjoint(set(mnms_s["held_out_external"]))
        assert len(mnms_s["train_pool"]) == 80
        assert len(mnms_s["held_out_external"]) == 34

    def test_emidec_cross_validation_folds(self) -> None:
        emidec_splits = create_emidec_splits(seed=42)
        folds = emidec_splits["five_fold_cv"]
        assert len(folds) == 5

        all_cases = set()
        for f_name, fold in folds.items():
            assert fold["val_count"] == 20
            assert fold["train_count"] == 80
            assert set(fold["train"]).isdisjoint(set(fold["val"]))
            all_cases.update(fold["val"])

        # All 100 cases must appear in validation across the 5 folds
        assert len(all_cases) == 100


class TestEmpiricalFeatureBank:
    """Validates empirical feature bank compilation and statistical priors."""

    def test_feature_bank_generation(self) -> None:
        df, summary = build_unified_feature_bank()
        assert len(df) == 314
        assert summary["cohort_breakdown"]["acdc_cases"] == 100
        assert summary["cohort_breakdown"]["mnms_cases"] == 114
        assert summary["cohort_breakdown"]["emidec_cases"] == 100

        # Verify modality separation
        assert (df["modality"] == Modality.CINE_CMR.value).sum() == 214
        assert (df["modality"] == Modality.LGE_CMR.value).sum() == 100

        # Verify EMIDEC viability priors
        lge_priors = summary["empirical_priors"]["lge_viability"]
        assert lge_priors["pathological_cases"] == 67
        assert 18.0 <= lge_priors["infarct_burden_percent_mean_pathological"] <= 19.5
        assert 0.58 <= lge_priors["mvo_prevalence_rate_pathological"] <= 0.62

    def test_feature_bank_schema_columns(self) -> None:
        df = pd.read_csv(FEATURE_BANK_CSV)
        required_cols = [
            "case_id", "dataset_id", "modality", "mask_provenance",
            "measurement_method", "lvedv_ml", "lvef_percent",
            "infarct_vol_ml", "has_microvascular_obstruction",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column {col} in feature bank"


class TestSyntheticBridge:
    """Validates baseline reproducibility and empirical synthesis."""

    def test_baseline_preservation(self) -> None:
        gen = EmpiricallyGroundedCohortGenerator(seed=42, empirical_mode=False)
        profiles, outcomes, summary = gen.generate_cohort(n_patients=5)
        assert len(profiles) == 5
        assert summary["empirical_mode"] is False
        assert summary["scenario_provenance"] == "canonical_baseline_acdc_dcm"

    def test_empirically_grounded_mode(self) -> None:
        gen = EmpiricallyGroundedCohortGenerator(seed=42, empirical_mode=True)
        profiles, outcomes, summary = gen.generate_cohort(n_patients=5)
        assert len(profiles) == 5
        assert summary["empirical_mode"] is True
        assert summary["scenario_provenance"] == "empirically_grounded_multidataset_v1"
        assert "empirical_priors" in summary


class TestPipelineCLI:
    """Validates pipeline CLI subcommands and safeguards."""

    def test_cli_audit(self) -> None:
        assert cli_main(["audit"]) == 0

    def test_cli_prepare(self, tmp_path: Path) -> None:
        out = tmp_path / "test_splits.json"
        assert cli_main(["prepare", "--output", str(out)]) == 0
        assert out.is_file()

    def test_cli_features(self, tmp_path: Path) -> None:
        out_csv = tmp_path / "test_fb.csv"
        out_sum = tmp_path / "test_sum.json"
        assert cli_main(["features", "--output-csv", str(out_csv), "--output-summary", str(out_sum)]) == 0
        assert out_csv.is_file()
        assert out_sum.is_file()

    def test_cli_synthesize(self) -> None:
        assert cli_main(["synthesize", "--n-patients", "10"]) == 0
        assert cli_main(["synthesize", "--n-patients", "10", "--empirical"]) == 0

    def test_cli_evaluate(self) -> None:
        assert cli_main(["evaluate", "--summary-only"]) == 0

    def test_cli_train_safeguards(self) -> None:
        # Dry-run training configuration check must succeed and enforce safety notice
        assert cli_main(["train", "--task", "cine", "--dry-run"]) == 0
        assert cli_main(["train", "--task", "lge", "--dry-run"]) == 0
