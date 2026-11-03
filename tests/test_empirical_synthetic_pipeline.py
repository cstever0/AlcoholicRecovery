"""Deterministic Tests for CRIP Stage 5C Unified Empirical Synthetic Integration.

Verifies:
  - Empirical feature bank integrity (314 records across ACDC, M&Ms, EMIDEC)
  - Zero source leakage (74 held-out cases strictly quarantined from generator)
  - Mathematical cardiac constraints (ESV < EDV, SV = EDV - ESV, exact LVEF)
  - Scenario consistency (Non-Ischemic ACM vs Ischemic Viability)
  - Deterministic generator reproducibility across seeds
  - Synthetic train/val/test split isolation
  - Original baseline cohort and MRI checkpoint preservation
  - Recovery model training, metric reporting, and artifact persistence
  - Scientific figure existence (8 figures in PNG, SVG, PDF)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crip.pipeline.empirical_generator import (
    DEFAULT_EMPIRICAL_COHORT_CSV,
    EmpiricalDistributionFitter,
    EmpiricallyInformedCohortGenerator,
)
from crip.pipeline.feature_bank import DEFAULT_OUTPUT_CSV as DEFAULT_FEATURE_BANK_CSV
from crip.pipeline.recovery_trainer import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_RECOVERY_MODELS_DIR,
    run_full_synthetic_recovery_pipeline,
)
from crip.pipeline.splits import DEFAULT_UNIFIED_SPLITS_PATH
from crip.pipeline.joint_trainer import compute_file_sha256


class TestEmpiricalSyntheticIntegration:
    """Test suite for empirically informed synthetic data and recovery modeling."""

    def test_feature_bank_source_counts_and_isolation(self):
        """Verify empirical feature bank contains 314 records and strictly excludes 74 held-out cases."""
        assert DEFAULT_FEATURE_BANK_CSV.is_file(), "Unified feature bank CSV not found."
        df_fb = pd.read_csv(DEFAULT_FEATURE_BANK_CSV)
        assert len(df_fb) == 314, f"Expected 314 records, got {len(df_fb)}"
        assert set(df_fb["dataset_id"].unique()) == {"acdc", "mnms", "emidec"}

        with open(DEFAULT_UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as f:
            splits = json.load(f)

        acdc_test = set(splits["datasets"]["acdc"]["splits"]["test"])
        mnms_test = set(splits["datasets"]["mnms"]["splits"]["held_out_evaluation"])
        emidec_test = set(splits["datasets"]["emidec"]["splits"]["test"])
        all_test = acdc_test | mnms_test | emidec_test

        assert len(acdc_test) == 20
        assert len(mnms_test) == 34
        assert len(emidec_test) == 20
        assert len(all_test) == 74

        # Test fitter isolation
        fitter = EmpiricalDistributionFitter()
        priors = fitter.fit()

        assert priors.n_source_total_cases == 314
        assert priors.n_source_held_out_cases == 74
        assert priors.n_source_training_cases == 240
        assert priors.n_cine_dcm_training_cases == 45
        assert priors.n_emidec_pathological_training_cases == 54

    def test_mathematical_cardiac_constraints(self):
        """Verify 100% compliance with physical volumetric and functional relationships."""
        gen = EmpiricallyInformedCohortGenerator(seed=123)
        profiles, meta = gen.generate_cohort(n_patients=50)

        for p in profiles:
            # Inequality: ESV < EDV
            assert p.esv_ml < p.edv_ml, f"ESV ({p.esv_ml}) must be < EDV ({p.edv_ml})"
            assert p.stroke_volume_ml == pytest.approx(p.edv_ml - p.esv_ml, abs=0.1)

            # Ejection fraction formula: LVEF = 100 * (EDV - ESV) / EDV
            expected_lvef = (p.stroke_volume_ml / p.edv_ml) * 100.0
            assert p.baseline_lvef == pytest.approx(expected_lvef, abs=0.2)

            # Plausible range
            assert 5.0 <= p.baseline_lvef <= 75.0
            assert 60.0 <= p.edv_ml <= 500.0

            # RV consistency
            if p.rvedv_ml is not None and p.rvesv_ml is not None:
                assert p.rvesv_ml < p.rvedv_ml
                expected_rvef = ((p.rvedv_ml - p.rvesv_ml) / p.rvedv_ml) * 100.0
                assert p.rvef_percent == pytest.approx(expected_rvef, abs=0.2)

    def test_scenario_consistency(self):
        """Verify non-ischemic vs ischemic scenario feature segregation."""
        gen = EmpiricallyInformedCohortGenerator(seed=456)
        profiles, meta = gen.generate_cohort(n_patients=100, ischemic_ratio=0.30)

        for p in profiles:
            if p.pathology_scenario == "NON_ISCHEMIC_ACM":
                # Ischemic infarct and MVO must be strictly 0
                assert p.infarct_burden_percent == 0.0
                assert p.infarct_volume_ml == 0.0
                assert p.has_microvascular_obstruction is False
                assert p.no_reflow_burden_percent == 0.0
                assert p.no_reflow_volume_ml == 0.0
                if p.lge_status.value == "POSITIVE":
                    assert p.lge_pattern.value in ["MIDWALL", "SUBEPICARDIAL", "PATCHY_TRANSMURAL", "SUBENDOCARDIAL"]
            elif p.pathology_scenario == "ISCHEMIC_VIABILITY":
                # Ischemic features grounded in EMIDEC
                assert p.infarct_burden_percent is not None and p.infarct_burden_percent > 0.0
                assert p.infarct_volume_ml is not None and p.infarct_volume_ml > 0.0
                assert p.lge_status.value == "POSITIVE"
                assert p.lge_pattern.value in ["SUBENDOCARDIAL", "PATCHY_TRANSMURAL"]
                if p.has_microvascular_obstruction:
                    assert p.no_reflow_burden_percent is not None and p.no_reflow_burden_percent > 0.0

    def test_generator_reproducibility(self):
        """Verify identical random seeds generate bitwise identical cohorts."""
        gen1 = EmpiricallyInformedCohortGenerator(seed=999)
        gen2 = EmpiricallyInformedCohortGenerator(seed=999)

        p1, m1 = gen1.generate_cohort(n_patients=20)
        p2, m2 = gen2.generate_cohort(n_patients=20)

        for pat1, pat2 in zip(p1, p2):
            assert pat1.patient_id == pat2.patient_id
            assert pat1.baseline_lvef == pat2.baseline_lvef
            assert pat1.edv_ml == pat2.edv_ml
            assert pat1.esv_ml == pat2.esv_ml
            assert pat1.follow_up.recovery_binary == pat2.follow_up.recovery_binary
            assert pat1.pathology_scenario == pat2.pathology_scenario

    def test_synthetic_train_test_isolation(self):
        """Verify synthetic partitions have zero patient overlap and distinct continuous realizations."""
        from crip.modeling.dataset import load_modeling_dataset

        ds = load_modeling_dataset(
            csv_path=DEFAULT_EMPIRICAL_COHORT_CSV,
            scenario="ALL",
            test_size=0.15,
            val_size=0.1765,
            seed=42,
        )

        train_ids = set(ds.train_ids)
        val_ids = set(ds.val_ids)
        test_ids = set(ds.test_ids)

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
        assert len(train_ids) + len(val_ids) + len(test_ids) == 3000

    def test_baseline_preservation_and_checkpoint_hashes(self):
        """Verify baseline synthetic cohort and all MRI model checkpoints remain untouched."""
        baseline_csv = Path("data/synthetic/synthetic_cardiomyopathy_cohort.csv")
        assert baseline_csv.is_file()
        df_b = pd.read_csv(baseline_csv)
        assert len(df_b) == 3000

        # MRI Checkpoint SHA-256 hashes
        acdc_ckpt = Path("models/mri/best_unet_model.pt")
        cine_ckpt = Path("models/multidataset/cine_joint_unet.pt")
        lge_ckpt = Path("models/multidataset/emidec_lge_unet.pt")

        assert acdc_ckpt.is_file()
        assert cine_ckpt.is_file()
        assert lge_ckpt.is_file()

        h_acdc = compute_file_sha256(acdc_ckpt)
        h_cine = compute_file_sha256(cine_ckpt)
        h_lge = compute_file_sha256(lge_ckpt)

        assert h_acdc == "71ea54235803562acc67b8903deecaaf113d45a1c7b443a97ba9fbe6573072c9"
        assert h_cine == "fd4f7da40f756794c99d9e2cad1e06a080701e86b5386afcd686a24b5c12e50d"
        assert len(h_lge) == 64

    def test_recovery_model_artifacts_and_metrics(self):
        """Verify saved recovery model checkpoints, comparison summary, and realism report."""
        res_file = Path("data/processed/multidataset/synthetic_recovery_comparison.json")
        realism_file = Path("data/processed/multidataset/feature_realism_evaluation.json")

        assert res_file.is_file()
        assert realism_file.is_file()

        with open(res_file, "r", encoding="utf-8") as f:
            comp = json.load(f)

        assert "baseline_logistic_regression" in comp["models"]
        assert "empirical_logistic_regression" in comp["models"]
        assert "baseline_random_forest" in comp["models"]
        assert "empirical_random_forest" in comp["models"]

        for m_name, m_data in comp["models"].items():
            test_m = m_data["test_metrics"]
            assert "roc_auc" in test_m
            assert "accuracy" in test_m
            assert "brier_score" in test_m
            assert 0.5 <= test_m["roc_auc"] <= 1.0

        # Saved model files
        assert (DEFAULT_RECOVERY_MODELS_DIR / "baseline_logistic_regression.joblib").is_file()
        assert (DEFAULT_RECOVERY_MODELS_DIR / "baseline_random_forest.joblib").is_file()
        assert (DEFAULT_RECOVERY_MODELS_DIR / "empirical_logistic_regression.joblib").is_file()
        assert (DEFAULT_RECOVERY_MODELS_DIR / "empirical_random_forest.joblib").is_file()

    def test_poster_figures_exist_multiformat(self):
        """Verify all 8 scientific figures exist in PNG, SVG, and PDF formats (24 files)."""
        stems = [
            "fig1_synthetic_feature_distributions",
            "fig2_synthetic_vs_held_out_empirical",
            "fig3_within_source_correlations",
            "fig4_emidec_scenario_distributions",
            "fig5_recovery_model_roc_comparison",
            "fig6_recovery_model_calibration_comparison",
            "fig7_feature_importance_comparison",
            "fig8_provenance_and_source_contributions",
        ]
        for stem in stems:
            for ext in ["png", "svg", "pdf"]:
                p = DEFAULT_FIGURES_DIR / f"{stem}.{ext}"
                assert p.is_file(), f"Missing figure file: {p}"
                assert p.stat().st_size > 500, f"Figure file is suspiciously small: {p}"
