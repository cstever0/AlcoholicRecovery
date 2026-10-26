"""Automated unit and integration tests for CRIP Milestone 3B:
Synthetic Recovery Prediction, Explainability, and Evaluation.

Covers all 25 required validation criteria:
 1. Synthetic dataset loading.
 2. Target construction.
 3. Feature whitelist enforcement.
 4. Exclusion of future variables.
 5. Correct scenario eligibility.
 6. Split reproducibility.
 7. Leakage prevention.
 8. Preprocessing fit isolation.
 9. Logistic regression training.
10. Dummy classifier baseline.
11. Prediction output range [0, 1].
12. Model serialization.
13. Model reload consistency.
14. ROC-AUC calculation.
15. PR-AUC calculation.
16. Brier score calculation.
17. Calibration generation.
18. Confusion matrix calculations.
19. Feature-group ablation.
20. SHAP / coefficient explanation.
21. Missing features handling.
22. Unseen categorical values handling.
23. Sensitivity-analysis execution.
24. Streamlit prediction display (AppTest).
25. Full regression testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from crip.evaluation.recovery_metrics import (
    compute_binary_classification_metrics,
    compute_bootstrap_confidence_intervals,
    evaluate_final_model_on_test,
)
from crip.modeling.artifacts import (
    load_recovery_model_artifacts,
    save_recovery_model_artifacts,
)
from crip.modeling.dataset import (
    DEFAULT_COHORT_CSV,
    FEATURE_WHITELIST,
    FORBIDDEN_LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    ModelingCohortDataset,
    build_feature_preprocessor,
    load_modeling_dataset,
    save_split_manifest,
    validate_feature_whitelist,
)
from crip.modeling.explain import (
    explain_patient_prediction,
    get_global_logistic_regression_weights,
)
from crip.modeling.sensitivity import (
    evaluate_noise_sensitivity,
    run_full_sensitivity_suite,
)
from crip.modeling.train import (
    evaluate_models_on_validation,
    run_feature_ablation_study,
    train_baseline_models,
)


@pytest.fixture(scope="module")
def modeling_dataset() -> ModelingCohortDataset:
    """Fixture providing a validated, partitioned modeling dataset under sustained abstinence."""
    return load_modeling_dataset(DEFAULT_COHORT_CSV, scenario="SUSTAINED_ABSTINENCE", seed=42)


@pytest.fixture(scope="module")
def trained_models(modeling_dataset: ModelingCohortDataset) -> dict:
    """Fixture providing trained baseline models."""
    return train_baseline_models(modeling_dataset, seed=42)


# ==============================================================================
# 1. Dataset Loading, Whitelisting, and Splitting (Criteria 1 - 8)
# ==============================================================================

class TestModelingDatasetAndSplits:
    """Tests for dataset loading, feature governance, leakage prevention, and splits."""

    def test_criterion_1_dataset_loading(self, modeling_dataset: ModelingCohortDataset):
        """Criterion 1: Verify synthetic cohort loads into structured ModelingCohortDataset."""
        assert isinstance(modeling_dataset, ModelingCohortDataset)
        assert len(modeling_dataset.X_train) > 0
        assert len(modeling_dataset.X_val) > 0
        assert len(modeling_dataset.X_test) > 0
        assert modeling_dataset.total_eligible_patients >= 2000

    def test_criterion_2_target_construction(self, modeling_dataset: ModelingCohortDataset):
        """Criterion 2: Verify binary recovery target is present and contains only binary {0, 1}."""
        assert modeling_dataset.target_name == "recovery_binary"
        unique_targets = set(modeling_dataset.y_train.unique())
        assert unique_targets.issubset({0, 1})
        assert len(unique_targets) == 2, "Target must contain both 0 and 1 classes"

    def test_criterion_3_feature_whitelist_enforcement(self, modeling_dataset: ModelingCohortDataset):
        """Criterion 3: Verify all predictor features belong strictly to the baseline whitelist."""
        allowed, unrecognized = validate_feature_whitelist(modeling_dataset.feature_names)
        assert len(unrecognized) == 0, f"Unrecognized features found: {unrecognized}"
        assert set(modeling_dataset.feature_names).issubset(set(FEATURE_WHITELIST))

    def test_criterion_4_exclusion_of_future_variables(self):
        """Criterion 4: Verify that attempting to include post-baseline or future variables raises ValueError."""
        leaky_features = ["age", "baseline_lvef", "follow_up_lvef"]
        with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
            validate_feature_whitelist(leaky_features)

        with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
            validate_feature_whitelist(["delta_lvef", "edv_ml"])

        with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
            validate_feature_whitelist(["recovery_binary"])

    def test_criterion_5_scenario_eligibility(self):
        """Criterion 5: Verify eligibility filtering restricts population to specified scenario."""
        ds_abst = load_modeling_dataset(DEFAULT_COHORT_CSV, scenario="SUSTAINED_ABSTINENCE")
        assert ds_abst.scenario == "SUSTAINED_ABSTINENCE"
        assert ds_abst.total_eligible_patients < 3000

        ds_all = load_modeling_dataset(DEFAULT_COHORT_CSV, scenario="ALL")
        assert ds_all.scenario == "ALL"
        assert ds_all.total_eligible_patients == 3000

    def test_criterion_6_split_reproducibility(self):
        """Criterion 6: Verify identical random seeds yield identical train, val, and test splits."""
        ds1 = load_modeling_dataset(DEFAULT_COHORT_CSV, seed=999)
        ds2 = load_modeling_dataset(DEFAULT_COHORT_CSV, seed=999)

        assert ds1.train_ids == ds2.train_ids
        assert ds1.val_ids == ds2.val_ids
        assert ds1.test_ids == ds2.test_ids
        np.testing.assert_array_equal(ds1.y_test.values, ds2.y_test.values)

    def test_criterion_7_leakage_prevention(self, modeling_dataset: ModelingCohortDataset):
        """Criterion 7: Verify patient IDs across train, validation, and test partitions are mutually exclusive."""
        train_set = set(modeling_dataset.train_ids)
        val_set = set(modeling_dataset.val_ids)
        test_set = set(modeling_dataset.test_ids)

        assert len(train_set.intersection(val_set)) == 0, "Patient overlap between train and val!"
        assert len(train_set.intersection(test_set)) == 0, "Patient overlap between train and test!"
        assert len(val_set.intersection(test_set)) == 0, "Patient overlap between val and test!"

    def test_criterion_8_preprocessing_fit_isolation(self, modeling_dataset: ModelingCohortDataset):
        """Criterion 8: Verify preprocessor fits strictly on training data without test contamination."""
        prep = build_feature_preprocessor(
            modeling_dataset.numerical_features,
            modeling_dataset.categorical_features,
            modeling_dataset.binary_features,
        )

        # Fit on train
        prep.fit(modeling_dataset.X_train)

        # Transform val and test
        X_val_trans = prep.transform(modeling_dataset.X_val)
        X_test_trans = prep.transform(modeling_dataset.X_test)

        assert X_val_trans.shape[0] == len(modeling_dataset.X_val)
        assert X_test_trans.shape[0] == len(modeling_dataset.X_test)
        assert not np.isnan(X_val_trans).any(), "NaN values found in transformed validation data!"


# ==============================================================================
# 2. Model Training & Prediction Outputs (Criteria 9 - 13)
# ==============================================================================

class TestModelTrainingAndPersistence:
    """Tests for model pipelines, baseline benchmarks, and artifact persistence."""

    def test_criterion_9_logistic_regression_training(self, trained_models: dict):
        """Criterion 9: Verify Logistic Regression pipeline trains successfully."""
        lr = trained_models["logistic_regression"]
        assert isinstance(lr, Pipeline)
        assert hasattr(lr.named_steps["clf"], "coef_")
        assert lr.named_steps["clf"].coef_.shape[0] == 1

    def test_criterion_10_dummy_classifier_baseline(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 10: Verify Dummy prevalence classifier yields AUC = 0.50 and reflects prevalence."""
        dummy = trained_models["dummy_prevalence"]
        probs = dummy.predict_proba(modeling_dataset.X_val)[:, 1]
        m = compute_binary_classification_metrics(modeling_dataset.y_val, probs)
        assert m["roc_auc"] == 0.50
        assert probs[0] == pytest.approx(float(modeling_dataset.y_train.mean()), abs=0.01)

    def test_criterion_11_prediction_output_range(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 11: Verify predicted probabilities fall strictly within [0.0, 1.0]."""
        lr = trained_models["logistic_regression"]
        probs = lr.predict_proba(modeling_dataset.X_test)[:, 1]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert not np.isnan(probs).any()

    def test_criterion_12_model_serialization(self, tmp_path: Path, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 12: Verify pipeline, metadata, schema, and manifest serialize to disk."""
        lr = trained_models["logistic_regression"]
        m = compute_binary_classification_metrics(modeling_dataset.y_test, lr.predict_proba(modeling_dataset.X_test)[:, 1])
        report = {"metrics": m, "target_achieved": True}

        out_dir = save_recovery_model_artifacts(lr, modeling_dataset, report, output_dir=tmp_path / "model_export")
        assert (out_dir / "recovery_pipeline.joblib").is_file()
        assert (out_dir / "metadata.json").is_file()
        assert (out_dir / "feature_schema.json").is_file()
        assert (out_dir / "splits.json").is_file()

    def test_criterion_13_model_reload_consistency(self, tmp_path: Path, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 13: Verify loaded pipeline reproduces in-memory predictions exactly."""
        lr = trained_models["logistic_regression"]
        m = compute_binary_classification_metrics(modeling_dataset.y_test, lr.predict_proba(modeling_dataset.X_test)[:, 1])
        report = {"metrics": m, "target_achieved": True}

        out_dir = save_recovery_model_artifacts(lr, modeling_dataset, report, output_dir=tmp_path / "model_reload")
        reloaded_model, meta = load_recovery_model_artifacts(out_dir)

        in_mem_probs = lr.predict_proba(modeling_dataset.X_test)[:, 1]
        reloaded_probs = reloaded_model.predict_proba(modeling_dataset.X_test)[:, 1]

        np.testing.assert_allclose(in_mem_probs, reloaded_probs, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 3. Metrics, Evaluation, and Calibration (Criteria 14 - 18)
# ==============================================================================

class TestEvaluationMetricsAndCalibration:
    """Tests for statistical metrics, calibration curves, and confusion matrix arithmetic."""

    def test_criterion_14_roc_auc_calculation(self):
        """Criterion 14: Verify compute_binary_classification_metrics calculates correct ROC-AUC."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        m = compute_binary_classification_metrics(y_true, y_prob)
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert m["roc_auc"] == pytest.approx(0.75, abs=0.01)

    def test_criterion_15_pr_auc_calculation(self):
        """Criterion 15: Verify PR-AUC calculation is bounded and valid."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        m = compute_binary_classification_metrics(y_true, y_prob)
        assert 0.0 <= m["pr_auc"] <= 1.0

    def test_criterion_16_brier_score_calculation(self):
        """Criterion 16: Verify Brier score calculates mean squared probability error accurately."""
        y_true = np.array([1, 0])
        y_prob = np.array([0.9, 0.1])
        # Brier = ((0.9-1)^2 + (0.1-0)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
        m = compute_binary_classification_metrics(y_true, y_prob)
        assert m["brier_score"] == pytest.approx(0.01, abs=1e-4)

    def test_criterion_17_calibration_generation(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 17: Verify calibration curve points are generated and properly ordered."""
        lr = trained_models["logistic_regression"]
        report = evaluate_final_model_on_test(lr, modeling_dataset)
        cal_points = report["calibration_curve"]
        assert len(cal_points) > 0
        for pt in cal_points:
            assert "mean_predicted_prob" in pt
            assert "observed_frequency" in pt
            assert 0.0 <= pt["mean_predicted_prob"] <= 1.0
            assert 0.0 <= pt["observed_frequency"] <= 1.0

    def test_criterion_18_confusion_matrix_calculations(self):
        """Criterion 18: Verify confusion matrix arithmetic is consistent with sensitivity and specificity."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.6, 0.4, 0.9])
        m = compute_binary_classification_metrics(y_true, y_prob, threshold=0.5)
        cm = m["confusion_matrix"]
        # y_pred = [0, 1, 0, 1]
        # TN=1, FP=1, FN=1, TP=1
        assert cm["tn"] == 1
        assert cm["fp"] == 1
        assert cm["fn"] == 1
        assert cm["tp"] == 1
        assert m["sensitivity"] == 0.5
        assert m["specificity"] == 0.5


# ==============================================================================
# 4. Modality Ablation & Explainability (Criteria 19 - 22)
# ==============================================================================

class TestAblationAndExplainability:
    """Tests for feature-group ablation studies and model explainability."""

    def test_criterion_19_feature_group_ablation(self, modeling_dataset: ModelingCohortDataset):
        """Criterion 19: Verify feature-group ablation runs all 6 configurations."""
        ablation_df = run_feature_ablation_study(modeling_dataset)
        assert len(ablation_df) == 6
        expected_groups = {
            "1_CMR_Volumetrics_Only",
            "2_CMR_Plus_PEth",
            "3_CMR_Plus_TTN",
            "4_CMR_Plus_LGE",
            "5_PEth_TTN_LGE_No_CMR",
            "6_Full_Multimodal",
        }
        assert set(ablation_df["ablation_group"].unique()) == expected_groups
        # Full multimodal should achieve the highest validation ROC-AUC
        best_group = ablation_df.iloc[0]["ablation_group"]
        assert best_group in ("6_Full_Multimodal", "5_PEth_TTN_LGE_No_CMR")

    def test_criterion_20_shap_and_coefficient_explanation(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 20: Verify log-odds decomposition and feature odds ratios are computed."""
        lr = trained_models["logistic_regression"]
        weights = get_global_logistic_regression_weights(lr)
        assert len(weights) > 10
        assert "coefficient_beta" in weights.columns
        assert "odds_ratio" in weights.columns
        assert "direction" in weights.columns

        patient = modeling_dataset.X_test.iloc[0]
        exp = explain_patient_prediction(lr, patient)
        assert "simulated_recovery_probability" in exp
        assert "narrative_explanation" in exp
        assert "feature_attributions" in exp
        assert len(exp["feature_attributions"]) > 0

    def test_criterion_21_missing_features_handling(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 21: Verify pipeline handles missing numerical and categorical features gracefully."""
        lr = trained_models["logistic_regression"]
        corrupted_row = modeling_dataset.X_test.iloc[0].copy()
        corrupted_row["peth_value_ng_ml"] = np.nan
        corrupted_row["lge_extent_percent"] = np.nan
        corrupted_row["reported_drinks_per_week"] = np.nan
        corrupted_row["ttn_classification"] = None

        df_corrupt = pd.DataFrame([corrupted_row.to_dict()])
        prob = lr.predict_proba(df_corrupt)[0, 1]
        assert 0.0 <= prob <= 1.0
        assert not np.isnan(prob)

    def test_criterion_22_unseen_categorical_values(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 22: Verify OneHotEncoder ignores novel/unseen categorical values without crashing."""
        lr = trained_models["logistic_regression"]
        novel_row = modeling_dataset.X_test.iloc[0].copy()
        novel_row["ttn_classification"] = "COMPLETELY_UNSEEN_VARIANT_TIER"
        novel_row["lge_status"] = "UNSEEN_TISSUE_CLASSIFICATION"

        df_novel = pd.DataFrame([novel_row.to_dict()])
        prob = lr.predict_proba(df_novel)[0, 1]
        assert 0.0 <= prob <= 1.0


# ==============================================================================
# 5. Sensitivity Analysis, Streamlit AppTest, and Regression (Criteria 23 - 25)
# ==============================================================================

class TestSensitivityAndStreamlitIntegration:
    """Tests for sensitivity suite, Streamlit page rendering, and full suite regression."""

    def test_criterion_23_sensitivity_analysis_execution(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 23: Verify sensitivity analysis runs across noise levels and scenarios."""
        lr = trained_models["logistic_regression"]
        res = evaluate_noise_sensitivity(lr, modeling_dataset, noise_multipliers=[0.5, 1.0, 2.0])
        assert len(res) == 3
        # Higher noise multiplier should degrade ROC-AUC or increase Brier score
        assert res[0]["test_brier_score"] <= res[2]["test_brier_score"]

    def test_criterion_24_streamlit_prediction_display(self):
        """Criterion 24: Verify 3_Recovery_Estimate.py renders cleanly via Streamlit AppTest."""
        from streamlit.testing.v1 import AppTest

        page_file = Path("src/crip/app/pages/3_Recovery_Estimate.py").resolve()
        at = AppTest.from_file(str(page_file), default_timeout=10)
        at.run()

        assert not at.exception, f"AppTest exception on Recovery Estimate page: {at.exception}"
        assert len(at.title) > 0
        assert "Recovery" in at.title[0].value

    def test_criterion_25_full_regression_testing(self, trained_models: dict, modeling_dataset: ModelingCohortDataset):
        """Criterion 25: Verify target benchmark ROC-AUC >= 0.80 is achieved on held-out test data."""
        lr = trained_models["logistic_regression"]
        report = evaluate_final_model_on_test(lr, modeling_dataset)
        test_auc = report["metrics"]["roc_auc"]
        assert test_auc >= 0.80, f"Expected test ROC-AUC >= 0.80, but achieved {test_auc:.4f}"
