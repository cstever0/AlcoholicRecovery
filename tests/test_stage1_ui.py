"""Unit and integration tests for Assignment Enhancement Stage 1 UI.

Covers:
- Overview Dashboard (0_Overview_Dashboard.py) subsystem health checks and study rendering
- Dedicated Cardiac Measurements page (5_Cardiac_Measurements.py) biventricular metrics,
  reference ranges, paired U-Net vs expert comparison, and session state transfer.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

overview_dash = importlib.import_module("crip.app.pages.0_Overview_Dashboard")
cardiac_meas = importlib.import_module("crip.app.pages.5_Cardiac_Measurements")


class TestOverviewDashboard:
    """Verify overview dashboard subsystem status and data loading."""

    def test_subsystem_artifacts_check(self):
        """Verify health check dictionary reports all 5 subsystems correctly."""
        artifacts = overview_dash._check_system_artifacts()
        required_keys = {"acdc_training", "acdc_testing", "unet_model", "synthetic_cohort", "recovery_model"}
        assert set(artifacts.keys()) == required_keys

        for key, info in artifacts.items():
            assert "name" in info
            assert "available" in info
            assert "detail" in info
            assert isinstance(info["available"], bool)
            assert isinstance(info["name"], str)
            assert len(info["detail"]) > 0

    def test_measurements_table_loading(self):
        """Verify cardiac measurements table can be loaded."""
        df = overview_dash._load_measurements_table()
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert "patient_id" in df.columns
            assert "lvedv_ml" in df.columns
            assert "lvesv_ml" in df.columns
            assert "lvef_percent" in df.columns

    def test_overview_dashboard_renders_with_apptest(self):
        """Verify 0_Overview_Dashboard.py renders cleanly without exceptions."""
        page_path = Path(__file__).resolve().parent.parent / "src/crip/app/pages/0_Overview_Dashboard.py"
        at = AppTest.from_file(str(page_path), default_timeout=10)
        at.run()
        assert not at.exception
        assert len(at.metric) >= 5  # Subsystem metrics + functional cards


class TestCardiacMeasurementsPage:
    """Verify dedicated cardiac measurements page metrics, comparison, and transfer."""

    def test_scmr_reference_ranges_complete(self):
        """Verify clinical reference dictionary contains all essential ventricular parameters."""
        ranges = cardiac_meas.SCMR_REFERENCE_RANGES
        expected_params = {"LVEDV", "LVESV", "LVSV", "LVEF", "RVEDV", "RVESV", "RVSV", "RVEF", "LV_MASS_ED"}
        assert set(ranges.keys()) == expected_params
        for p, d in ranges.items():
            assert "label" in d
            assert "unit" in d
            assert "male" in d
            assert "female" in d
            assert "combined" in d

    def test_lvef_clinical_tagging(self):
        """Verify LVEF clinical categorization logic."""
        tag_norm, col_norm = cardiac_meas._get_lvef_badge(58.5)
        assert "Preserved" in tag_norm

        tag_mild, col_mild = cardiac_meas._get_lvef_badge(45.0)
        assert "Mildly Reduced" in tag_mild

        tag_red, col_red = cardiac_meas._get_lvef_badge(32.0)
        assert "Reduced" in tag_red

    def test_rvef_clinical_tagging(self):
        """Verify RVEF clinical categorization logic."""
        tag_norm, _ = cardiac_meas._get_rvef_badge(48.0)
        assert "Normal" in tag_norm

        tag_red, _ = cardiac_meas._get_rvef_badge(38.0)
        assert "Reduced" in tag_red

    def test_cardiac_measurements_page_renders_with_apptest(self):
        """Verify 5_Cardiac_Measurements.py renders tabs and metrics cleanly."""
        page_path = Path(__file__).resolve().parent.parent / "src/crip/app/pages/5_Cardiac_Measurements.py"
        at = AppTest.from_file(str(page_path), default_timeout=10)
        at.run()
        assert not at.exception
        # Ensure tabs and metric widgets exist
        assert len(at.tabs) >= 3
        assert len(at.metric) >= 4

    def test_transfer_to_recovery_predictor_session_state(self):
        """Verify clicking the transfer button sets session state keys."""
        page_path = Path(__file__).resolve().parent.parent / "src/crip/app/pages/5_Cardiac_Measurements.py"
        at = AppTest.from_file(str(page_path), default_timeout=10)
        at.run()
        assert not at.exception

        # Find and click transfer button if present
        transfer_buttons = [b for b in at.button if "Transfer to Recovery Predictor" in b.label]
        assert len(transfer_buttons) > 0
        transfer_buttons[0].click().run()

        assert not at.exception
        assert "transferred_mri_features" in at.session_state
        transferred = at.session_state["transferred_mri_features"]
        assert "acdc_patient_id" in transferred
        assert "baseline_lvef" in transferred
        assert "edv_ml" in transferred
        assert "esv_ml" in transferred
        assert "stroke_volume_ml" in transferred
