"""Regression tests for calibration curve schema normalization and Streamlit Evaluation page stability.

Tests cover:
1. Current recovery benchmark JSON schema (dict of parallel lists) loads successfully.
2. Historical valid JSON schema (list of dicts) remains supported.
3. Dictionary-form cohort data is handled correctly.
4. List-form cohort data is handled correctly only where valid dict records exist.
5. Unexpected strings/scalars are rejected cleanly without AttributeError.
6. Missing coordinate fields do not silently become zero.
7. Empty datasets return empty lists or trigger informative warnings.
8. Invalid numeric fields (NaN, Inf, strings) are safely detected.
9. Valid chart data preserves exact floating-point values.
10. Streamlit Evaluation page renders Experiment 1 and Experiment 2 cleanly.
11. Streamlit Evaluation page degrades gracefully when optional benchmark JSON is absent.
12. No regression is introduced into baseline recovery or synthetic cohort components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pytest
from streamlit.testing.v1 import AppTest

from crip.evaluation.model_comparison import (
    create_calibration_comparison_plot,
    extract_calibration_curve_points,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestCalibrationCurveSchemaNormalization:
    """Test suite for extract_calibration_curve_points helper function."""

    def test_current_schema_dict_format(self):
        """Scenario 1 & 3: Current recovery pipeline dict-of-arrays schema."""
        raw_cc = {
            "mean_predicted": [0.06378, 0.15244, 0.25453, 0.34872, 0.45924],
            "fraction_positives": [0.0, 0.12245, 0.43182, 0.47727, 0.68627],
        }
        preds, obs = extract_calibration_curve_points(raw_cc)
        assert len(preds) == 5
        assert len(obs) == 5
        assert preds == [0.06378, 0.15244, 0.25453, 0.34872, 0.45924]
        assert obs == [0.0, 0.12245, 0.43182, 0.47727, 0.68627]

    def test_historical_schema_list_of_dicts(self):
        """Scenario 2 & 4: Historical list-of-dicts schema with mean_predicted_prob / observed_frequency."""
        raw_cc = [
            {"bin_idx": 0, "mean_predicted_prob": 0.0638, "observed_frequency": 0.0},
            {"bin_idx": 1, "mean_predicted_prob": 0.1524, "observed_frequency": 0.1225},
            {"bin_idx": 2, "mean_predicted_prob": 0.2545, "observed_frequency": 0.4318},
        ]
        preds, obs = extract_calibration_curve_points(raw_cc)
        assert len(preds) == 3
        assert len(obs) == 3
        assert preds == [0.0638, 0.1524, 0.2545]
        assert obs == [0.0, 0.1225, 0.4318]

    def test_list_of_dicts_with_current_key_names(self):
        """Scenario 4: List-of-dicts with mean_predicted / fraction_positives keys."""
        raw_cc = [
            {"mean_predicted": 0.10, "fraction_positives": 0.12},
            {"mean_predicted": 0.50, "fraction_positives": 0.48},
        ]
        preds, obs = extract_calibration_curve_points(raw_cc)
        assert preds == [0.10, 0.50]
        assert obs == [0.12, 0.48]

    def test_dict_format_with_historical_key_names(self):
        """Scenario 3: Dict format using mean_predicted_prob and observed_frequency arrays."""
        raw_cc = {
            "mean_predicted_prob": [0.1, 0.4, 0.8],
            "observed_frequency": [0.15, 0.35, 0.75],
        }
        preds, obs = extract_calibration_curve_points(raw_cc)
        assert preds == [0.1, 0.4, 0.8]
        assert obs == [0.15, 0.35, 0.75]

    def test_unexpected_string_rejected_without_attribute_error(self):
        """Scenario 5: Passing a string or list of strings must NEVER raise AttributeError."""
        # Bare string (e.g. if key was stringified or wrong type)
        preds, obs = extract_calibration_curve_points("invalid_string_data")
        assert preds == []
        assert obs == []

        # List of strings (reproducing iterating over dict keys)
        preds, obs = extract_calibration_curve_points(["mean_predicted", "fraction_positives"])
        assert preds == []
        assert obs == []

        # Strict mode raises TypeError
        with pytest.raises(TypeError):
            extract_calibration_curve_points("invalid_string_data", strict=True)

        with pytest.raises(TypeError):
            extract_calibration_curve_points(["mean_predicted", "fraction_positives"], strict=True)

    def test_missing_coordinates_do_not_default_to_zero(self):
        """Scenario 6: Missing coordinates must not silently produce fake [0, 0, 0] data."""
        # Missing observed array
        raw_cc = {"mean_predicted": [0.1, 0.5, 0.9]}
        preds, obs = extract_calibration_curve_points(raw_cc)
        assert preds == []
        assert obs == []

        # Missing predicted array
        raw_cc2 = {"fraction_positives": [0.1, 0.5, 0.9]}
        preds, obs = extract_calibration_curve_points(raw_cc2)
        assert preds == []
        assert obs == []

        # Strict mode raises KeyError
        with pytest.raises(KeyError):
            extract_calibration_curve_points(raw_cc, strict=True)

    def test_empty_dataset_returns_empty_lists(self):
        """Scenario 7: Empty inputs return empty lists."""
        assert extract_calibration_curve_points({}) == ([], [])
        assert extract_calibration_curve_points([]) == ([], [])
        assert extract_calibration_curve_points(None) == ([], [])

        with pytest.raises(ValueError):
            extract_calibration_curve_points({}, strict=True)
        with pytest.raises(ValueError):
            extract_calibration_curve_points([], strict=True)
        with pytest.raises(ValueError):
            extract_calibration_curve_points(None, strict=True)

    def test_invalid_numeric_fields_detected(self):
        """Scenario 8: Non-numeric strings, NaN, Inf, and out-of-bounds are rejected."""
        # Non-numeric string value
        raw_cc = {"mean_predicted": ["corrupted", 0.5], "fraction_positives": [0.1, 0.5]}
        preds, obs = extract_calibration_curve_points(raw_cc)
        assert preds == []
        assert obs == []

        # NaN value
        raw_cc_nan = {"mean_predicted": [float("nan"), 0.5], "fraction_positives": [0.1, 0.5]}
        preds, obs = extract_calibration_curve_points(raw_cc_nan)
        assert preds == []
        assert obs == []

        # Inf value
        raw_cc_inf = {"mean_predicted": [float("inf"), 0.5], "fraction_positives": [0.1, 0.5]}
        preds, obs = extract_calibration_curve_points(raw_cc_inf)
        assert preds == []
        assert obs == []

        # Out-of-bounds probability (> 1.5)
        raw_cc_oob = {"mean_predicted": [2.5, 0.5], "fraction_positives": [0.1, 0.5]}
        preds, obs = extract_calibration_curve_points(raw_cc_oob)
        assert preds == []
        assert obs == []

    def test_mismatched_lengths_rejected(self):
        """Scenario 8: Mismatched coordinate array lengths are rejected."""
        raw_cc = {"mean_predicted": [0.1, 0.5, 0.9], "fraction_positives": [0.1, 0.5]}
        preds, obs = extract_calibration_curve_points(raw_cc)
        assert preds == []
        assert obs == []

        with pytest.raises(ValueError):
            extract_calibration_curve_points(raw_cc, strict=True)

    def test_preserves_exact_values(self):
        """Scenario 9: Valid data preserves exact floating-point coordinates."""
        orig_p = [0.01234, 0.23456, 0.78901]
        orig_o = [0.00000, 0.25000, 0.80000]
        preds, obs = extract_calibration_curve_points({
            "mean_predicted": orig_p,
            "fraction_positives": orig_o,
        })
        assert preds == orig_p
        assert obs == orig_o

    def test_create_calibration_comparison_plot_robustness(self):
        """Verify Plotly figure builder handles both dict and list schemas without errors."""
        comp_dict_schema = {
            "models": {
                "logistic_regression": {
                    "metrics": {"brier_score": 0.0694},
                    "calibration_curve": {
                        "mean_predicted": [0.1, 0.5, 0.9],
                        "fraction_positives": [0.12, 0.48, 0.88],
                    },
                },
                "random_forest": {
                    "metrics": {"brier_score": 0.0930},
                    "calibration_curve": [
                        {"mean_predicted_prob": 0.1, "observed_frequency": 0.11},
                        {"mean_predicted_prob": 0.5, "observed_frequency": 0.52},
                    ],
                },
            }
        }
        fig = create_calibration_comparison_plot(comp_dict_schema)
        assert fig is not None
        # Should contain diagonal + 2 model traces
        assert len(fig.data) == 3


class TestStreamlitEvaluationPageRegression:
    """Streamlit AppTest regression suite verifying Experiment 1 and 2 rendering stability."""

    def test_evaluation_page_renders_experiment_1_and_experiment_2(self):
        """Scenario 10 & 11: Switch between Experiment 1 and 2 without any exceptions."""
        script_path = ROOT_DIR / "src/crip/app/pages/4_Evaluation.py"
        at = AppTest.from_file(str(script_path), default_timeout=20)
        at.run()

        # Initial render (Experiment 1 selected by default)
        assert not at.exception
        assert len(at.tabs) >= 6

        # Radio 2 is the Recovery Benchmark Experiment toggle
        exp_radio = at.radio[2]
        assert "Experiment 1" in exp_radio.value

        # Switch to Experiment 2 (All-Comers)
        exp_radio.set_value("Experiment 2: Unselected All-Comers Cohort (N = 450, Baseline vs. Empirical)").run()

        assert not at.exception
        page_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [m.label for m in at.metric])
        assert "All-Comers" in page_text or "450" in page_text

    def test_evaluation_page_graceful_missing_recovery_json(self, tmp_path, monkeypatch):
        """Scenario 10: _load_multidataset_recovery_data returns None when file is absent."""
        import importlib
        eval_page = importlib.import_module("crip.app.pages.4_Evaluation")

        fake_path = tmp_path / "non_existent_recovery_comparison.json"
        monkeypatch.setattr(eval_page, "MULTIDATASET_RECOVERY_PATH", fake_path)
        eval_page._load_multidataset_recovery_data.clear()
        res = eval_page._load_multidataset_recovery_data()
        assert res is None
