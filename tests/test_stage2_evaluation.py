"""Unit and integration tests for Assignment Enhancement Stage 2.

Covers:
- Bland-Altman bias, limits of agreement, ddof=1, standard errors, confidence intervals
- Array validations, unequal lengths, missing data / NaNs
- Scatter plots with identity line (y = x)
- HD95 summary calculations, spatial units (mm)
- Recovery model comparison schema, metrics (ROC-AUC, PR-AUC, Brier, Sens, Spec)
- Poster figure generation and file manifest validation
- Streamlit Evaluation page rendering with AppTest
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from crip.evaluation.measurement_agreement import (
    BlandAltmanResult,
    compute_hd95_statistics,
    compute_measurement_agreement,
    create_bland_altman_plot,
    create_hd95_box_plot,
    create_identity_scatter_plot,
)
from crip.evaluation.model_comparison import (
    create_calibration_comparison_plot,
    create_confusion_matrix_figure,
    create_pr_comparison_plot,
    create_roc_comparison_plot,
    generate_recovery_model_comparison,
)
from crip.evaluation.poster_figures import generate_all_poster_figures


@pytest.fixture
def synthetic_evaluation_df() -> pd.DataFrame:
    """Deterministic synthetic evaluation DataFrame with realistic schema for portable unit testing."""
    n = 20
    groups = ["DCM"] * 4 + ["HCM"] * 4 + ["MINF"] * 4 + ["NOR"] * 4 + ["RV"] * 4
    return pd.DataFrame({
        "patient_id": [f"synth{i:03d}" for i in range(1, n + 1)],
        "group": groups,
        "dice_patient_lv": [0.90 + 0.01 * (i % 3) for i in range(n)],
        "dice_patient_rv": [0.80 + 0.01 * (i % 3) for i in range(n)],
        "dice_patient_myo": [0.80 + 0.01 * (i % 3) for i in range(n)],
        "dice_patient_mean": [0.83 + 0.01 * (i % 3) for i in range(n)],
        "hd95_ed_lv_mm": [2.5 + 0.1 * i for i in range(n)],
        "hd95_ed_rv_mm": [6.5 + 0.1 * i for i in range(n)],
        "hd95_ed_myo_mm": [4.5 + 0.1 * i for i in range(n)],
        "hd95_es_lv_mm": [3.0 + 0.1 * i for i in range(n)],
        "hd95_es_rv_mm": [7.0 + 0.1 * i for i in range(n)],
        "hd95_es_myo_mm": [5.0 + 0.1 * i for i in range(n)],
        "lvedv_pred_ml": [120.0 + 2.0 * i for i in range(n)],
        "lvedv_gt_ml": [118.0 + 2.0 * i for i in range(n)],
        "lvedv_error_ml": [2.0] * n,
        "lvedv_abs_error_ml": [2.0] * n,
        "lvesv_pred_ml": [60.0 + 1.0 * i for i in range(n)],
        "lvesv_gt_ml": [58.0 + 1.0 * i for i in range(n)],
        "lvesv_error_ml": [2.0] * n,
        "lvesv_abs_error_ml": [2.0] * n,
        "lvef_pred_pct": [50.0 + 0.5 * (i % 5) for i in range(n)],
        "lvef_gt_pct": [50.5 + 0.5 * (i % 5) for i in range(n)],
        "lvef_error_pct": [-0.5] * n,
        "lvef_abs_error_pct": [0.5] * n,
        "within_5pct_target": [True] * n,
        "status": ["SUCCESS"] * n,
    })


class TestBlandAltmanAndMeasurementAgreement:
    """Test Bland-Altman statistics, error metrics, and input validations."""

    def test_bland_altman_synthetic_known_values(self):
        """Verify bias, SD (ddof=1), and LoA on known analytical vectors."""
        # preds = [12, 18, 25, 33]
        # refs  = [10, 20, 20, 30]
        # diff  = [+2, -2, +5, +3] -> sum = 8, mean = 2.0
        preds = np.array([12.0, 18.0, 25.0, 33.0])
        refs = np.array([10.0, 20.0, 20.0, 30.0])

        res = compute_measurement_agreement(preds, refs, variable_name="TestVar", unit="mL")

        assert res.n_samples == 4
        assert pytest.approx(res.mean_bias, rel=1e-4) == 2.0

        # Differences: [2, -2, 5, 3]. Mean = 2.0
        # Deviations: [0, -4, 3, 1]. Squared: [0, 16, 9, 1] = 26.
        # Sample variance (ddof=1): 26 / (4 - 1) = 26 / 3 = 8.6667
        # Sample SD: sqrt(26 / 3) = 2.9439
        expected_sd = float(np.sqrt(26.0 / 3.0))
        assert pytest.approx(res.sd_diff, rel=1e-4) == expected_sd

        # 95% LoA: bias +/- 1.96 * SD
        expected_lower = 2.0 - 1.96 * expected_sd
        expected_upper = 2.0 + 1.96 * expected_sd
        assert pytest.approx(res.lower_loa, rel=1e-4) == expected_lower
        assert pytest.approx(res.upper_loa, rel=1e-4) == expected_upper

        # MAE: (2 + 2 + 5 + 3) / 4 = 12 / 4 = 3.0
        assert pytest.approx(res.mae, rel=1e-4) == 3.0
        # RMSE: sqrt((4 + 4 + 25 + 9) / 4) = sqrt(42 / 4) = sqrt(10.5) = 3.2404
        assert pytest.approx(res.rmse, rel=1e-4) == np.sqrt(10.5)

    def test_bland_altman_confidence_intervals_distinguished_from_loa(self):
        """Verify confidence intervals for bias and LoA are computed and distinct from LoA."""
        preds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        refs = np.array([9.0, 21.0, 29.0, 42.0, 48.0])

        res = compute_measurement_agreement(preds, refs, "LVEF", "%-points")

        # CIs should bracket the estimated points
        assert res.ci_bias_95[0] < res.mean_bias < res.ci_bias_95[1]
        assert res.ci_lower_loa_95[0] < res.lower_loa < res.ci_lower_loa_95[1]
        assert res.ci_upper_loa_95[0] < res.upper_loa < res.ci_upper_loa_95[1]

        # LoA span should be strictly wider than the bias CI
        loa_span = res.upper_loa - res.lower_loa
        bias_ci_span = res.ci_bias_95[1] - res.ci_bias_95[0]
        assert loa_span > bias_ci_span

    def test_positive_and_negative_signed_differences(self):
        """Verify signed differences correctly preserve directionality (over vs under estimation)."""
        preds = np.array([105.0, 90.0])
        refs = np.array([100.0, 100.0])
        res = compute_measurement_agreement(preds, refs, "Vol", "mL")

        assert res.differences[0] == +5.0  # Overestimation by model
        assert res.differences[1] == -10.0  # Underestimation by model
        assert res.mean_bias == -2.5

    def test_mismatched_array_lengths_raises_valueerror(self):
        """Verify ValueError when predictions and references have differing lengths."""
        with pytest.raises(ValueError, match="Mismatched array lengths"):
            compute_measurement_agreement([1.0, 2.0], [1.0, 2.0, 3.0], "Var", "mL")

    def test_empty_or_single_element_array_raises_valueerror(self):
        """Verify ValueError when array has fewer than 2 elements."""
        with pytest.raises(ValueError, match="At least 2 samples required"):
            compute_measurement_agreement([1.0], [1.0], "Var", "mL")

    def test_handling_of_nans(self):
        """Verify NaNs are filtered cleanly when sufficient valid pairs exist."""
        preds = np.array([10.0, np.nan, 30.0, 40.0])
        refs = np.array([12.0, 20.0, np.nan, 38.0])

        # Valid pairs: (10, 12) and (40, 38)
        res = compute_measurement_agreement(preds, refs, "Var", "mL")
        assert res.n_samples == 2

    def test_real_acdc_20_test_cases_reproducibility(self):
        """Verify exact agreement results against empirical 20-test-case benchmarks."""
        eval_csv = Path("data/processed/acdc/segmentation_evaluation.csv")
        if not eval_csv.is_file():
            pytest.skip(
                "Real ACDC 20-case evaluation artifact ('data/processed/acdc/segmentation_evaluation.csv') not found. "
                "Run 'python -m crip.evaluation.segmentation_metrics' with raw ACDC data and checkpoint to generate."
            )
        df = pd.read_csv(eval_csv)
        assert len(df) == 20

        # LVEDV
        res_edv = compute_measurement_agreement(df["lvedv_pred_ml"], df["lvedv_gt_ml"], "LVEDV", "mL")
        assert pytest.approx(res_edv.mean_bias, abs=0.05) == +2.13
        assert pytest.approx(res_edv.lower_loa, abs=0.05) == -15.73
        assert pytest.approx(res_edv.upper_loa, abs=0.05) == +19.99
        assert pytest.approx(res_edv.mae, abs=0.05) == 6.29

        # LVESV
        res_esv = compute_measurement_agreement(df["lvesv_pred_ml"], df["lvesv_gt_ml"], "LVESV", "mL")
        assert pytest.approx(res_esv.mean_bias, abs=0.05) == +2.35
        assert pytest.approx(res_esv.lower_loa, abs=0.05) == -13.07
        assert pytest.approx(res_esv.upper_loa, abs=0.05) == +17.77
        assert pytest.approx(res_esv.mae, abs=0.05) == 5.86

        # LVEF
        res_ef = compute_measurement_agreement(df["lvef_pred_pct"], df["lvef_gt_pct"], "LVEF", "%-points")
        assert pytest.approx(res_ef.mean_bias, abs=0.05) == -0.55
        assert pytest.approx(res_ef.lower_loa, abs=0.05) == -10.53
        assert pytest.approx(res_ef.upper_loa, abs=0.05) == +9.43
        assert pytest.approx(res_ef.mae, abs=0.05) == 3.86


class TestPlotlyFigureGeneration:
    """Verify Plotly figure constructors create valid interactive figures."""

    def test_identity_scatter_plot_traces_and_layout(self):
        """Verify identity scatter plot includes identity line and data points."""
        preds = np.array([20.0, 35.0, 50.0])
        refs = np.array([22.0, 34.0, 48.0])
        res = compute_measurement_agreement(preds, refs, "LVEF", "%-points")

        fig = create_identity_scatter_plot(res)
        assert len(fig.data) == 2  # Identity line + scatter points
        assert fig.data[0].name == "Identity Line (y = x)"
        assert fig.data[1].name == "Test Cases"
        assert len(fig.layout.annotations) >= 1

    def test_bland_altman_plot_traces_and_limits(self):
        """Verify Bland-Altman plot includes zero, bias, and LoA lines."""
        preds = np.array([20.0, 35.0, 50.0])
        refs = np.array([22.0, 34.0, 48.0])
        res = compute_measurement_agreement(preds, refs, "LVEDV", "mL")

        fig = create_bland_altman_plot(res)
        assert len(fig.data) == 5  # Zero line, bias line, upper LoA, lower LoA, points
        trace_names = [t.name for t in fig.data]
        assert any("Mean Bias" in name for name in trace_names)
        assert any("Upper 95% LoA" in name for name in trace_names)
        assert any("Lower 95% LoA" in name for name in trace_names)


class TestHausdorffDistanceEvaluation:
    """Verify HD95 metrics extraction, spatial millimeter units, and box plots."""

    def test_hd95_summary_statistics_structure(self, synthetic_evaluation_df):
        """Verify HD95 statistics output dictionary schema using synthetic fixture."""
        stats_dict = compute_hd95_statistics(synthetic_evaluation_df)

        assert set(stats_dict.keys()) == {"LV", "RV", "MYO"}
        for k, info in stats_dict.items():
            assert info["unit"] == "mm"
            assert "ed" in info
            assert "es" in info
            assert "pooled" in info
            ed_s = info["ed"]
            assert ed_s["median"] > 0
            assert ed_s["mean"] > 0
            assert ed_s["iqr_low"] <= ed_s["median"] <= ed_s["iqr_high"]

    def test_real_acdc_hd95_reproducibility(self):
        """Integration test: Verify empirical HD95 benchmark values on real ACDC 20 test cases."""
        eval_csv = Path("data/processed/acdc/segmentation_evaluation.csv")
        if not eval_csv.is_file():
            pytest.skip(
                "Real ACDC 20-case evaluation artifact ('data/processed/acdc/segmentation_evaluation.csv') not found. "
                "Run 'python -m crip.evaluation.segmentation_metrics' with raw ACDC data and checkpoint to generate."
            )
        df = pd.read_csv(eval_csv)
        stats_dict = compute_hd95_statistics(df)

        # Confirm verified End-Diastole medians
        assert pytest.approx(stats_dict["LV"]["ed"]["median"], abs=0.1) == 2.91
        assert pytest.approx(stats_dict["RV"]["ed"]["median"], abs=0.1) == 7.16
        assert pytest.approx(stats_dict["MYO"]["ed"]["median"], abs=0.1) == 4.94

    def test_hd95_box_plot_creation(self, synthetic_evaluation_df):
        """Verify HD95 box plot creates traces for LV, RV, and Myocardium using synthetic fixture."""
        fig = create_hd95_box_plot(synthetic_evaluation_df)
        assert len(fig.data) == 3


class TestRecoveryModelComparison:
    """Verify Logistic Regression vs. Random Forest evaluation and persistence."""

    def test_recovery_model_comparison_generation_and_schema(self, tmp_path):
        """Verify model comparison generates correct schema and metrics."""
        test_json = tmp_path / "test_comparison.json"
        comp = generate_recovery_model_comparison(output_path=test_json, force_regenerate=True)

        assert test_json.is_file()
        assert "metadata" in comp
        assert "models" in comp
        assert "comparison_table" in comp

        meta = comp["metadata"]
        assert meta["test_sample_size"] == 419
        assert meta["decision_threshold"] == 0.50

        # Check Logistic Regression metrics
        lr_m = comp["models"]["logistic_regression"]["metrics"]
        assert pytest.approx(lr_m["roc_auc"], abs=0.005) == 0.9565
        assert pytest.approx(lr_m["pr_auc"], abs=0.005) == 0.9835
        assert pytest.approx(lr_m["brier_score"], abs=0.005) == 0.0694
        assert pytest.approx(lr_m["accuracy"], abs=0.005) == 0.8926

        # Check Random Forest metrics
        rf_m = comp["models"]["random_forest"]["metrics"]
        assert pytest.approx(rf_m["roc_auc"], abs=0.005) == 0.9310
        assert pytest.approx(rf_m["brier_score"], abs=0.005) == 0.0930
        assert pytest.approx(rf_m["accuracy"], abs=0.005) == 0.8711

        # Check comparison table rows
        assert len(comp["comparison_table"]) == 2

    def test_model_comparison_plotly_figures(self):
        """Verify ROC, PR, calibration, and confusion matrix figure generators."""
        comp = generate_recovery_model_comparison()

        fig_roc = create_roc_comparison_plot(comp)
        assert len(fig_roc.data) >= 3  # Diagonal + 2 models

        fig_pr = create_pr_comparison_plot(comp)
        assert len(fig_pr.data) >= 3  # Baseline + 2 models

        fig_cal = create_calibration_comparison_plot(comp)
        assert len(fig_cal.data) >= 3  # Diagonal + 2 models

        cm = comp["models"]["logistic_regression"]["confusion_matrix"]
        fig_cm = create_confusion_matrix_figure(cm, "Logistic Regression")
        assert len(fig_cm.data) == 1


class TestPosterFiguresAndStreamlitIntegration:
    """Verify poster-ready figure exports and Streamlit evaluation page rendering."""

    def test_poster_figures_export_and_manifest(self, tmp_path, synthetic_evaluation_df):
        """Unit test: Verify all 8 figures are exported in PNG, SVG, and PDF with valid manifest using synthetic fixture."""
        csv_path = tmp_path / "synthetic_eval.csv"
        synthetic_evaluation_df.to_csv(csv_path, index=False)

        out_dir = tmp_path / "poster_figures"
        manifest = generate_all_poster_figures(eval_csv_path=csv_path, output_dir=out_dir)

        assert manifest["total_figures"] == 8
        assert len(manifest["figures"]) == 8

        manifest_file = out_dir / "figure_manifest.json"
        assert manifest_file.is_file()

        for fig_info in manifest["figures"]:
            assert "figure_id" in fig_info
            assert "files" in fig_info
            f_paths = fig_info["files"]
            assert Path(f_paths["png"]).is_file()
            assert Path(f_paths["svg"]).is_file()
            assert Path(f_paths["pdf"]).is_file()
            assert Path(f_paths["png"]).stat().st_size > 10000  # Non-empty high-res PNG

    def test_real_acdc_poster_figures_export(self, tmp_path):
        """Integration test: Verify poster figure generation against real ACDC evaluation CSV."""
        eval_csv = Path("data/processed/acdc/segmentation_evaluation.csv")
        if not eval_csv.is_file():
            pytest.skip(
                "Real ACDC 20-case evaluation artifact ('data/processed/acdc/segmentation_evaluation.csv') not found. "
                "Run 'python -m crip.evaluation.segmentation_metrics' with raw ACDC data and checkpoint to generate."
            )
        out_dir = tmp_path / "real_poster_figures"
        manifest = generate_all_poster_figures(eval_csv_path=eval_csv, output_dir=out_dir)
        assert manifest["total_figures"] == 8
        assert len(manifest["figures"]) == 8

    def test_evaluation_page_renders_with_apptest(self):
        """Verify 4_Evaluation.py renders all 4 tabs cleanly via Streamlit AppTest."""
        root_dir = Path(__file__).resolve().parent.parent
        eval_script = root_dir / "src/crip/app/pages/4_Evaluation.py"

        at = AppTest.from_file(str(eval_script), default_timeout=10)
        at.run()

        assert not at.exception
        assert len(at.tabs) >= 4
        assert len(at.metric) >= 4
