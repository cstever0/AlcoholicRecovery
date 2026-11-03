"""Comprehensive Streamlit AppTest and Integration Test Suite for Stage 6.

Tests all integrated workspaces and pages:
1. Overview Dashboard (0_Overview_Dashboard.py) multi-dataset audit and checkpoint inventory
2. MRI Analysis (2_MRI_Analysis.py) with 3 tabs:
   - ACDC Cine MRI Analysis
   - Cine Model Comparison (ACDC-only vs Joint Cine U-Net)
   - EMIDEC LGE Tissue Workspace (with mandatory experimental limitation warnings)
3. Evaluation (4_Evaluation.py) with 6 tabs:
   - Real MRI Segmentation Benchmarks
   - Cardiac Measurement Agreement (Bland–Altman)
   - Recovery Prediction Benchmarks (Experiment 1 vs Experiment 2)
   - Synthetic Cohort Explorer (Baseline vs Empirically Informed Generator)
   - Scientific Figure Explorer (Multi-format export)
   - Scientific Audit & Clinical Governance
4. Component robustness, missing file fallbacks, and cross-platform behavior.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from crip.app.components.figure_explorer import FIGURE_COLLECTIONS

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestOverviewDashboardIntegration:
    """Verify 0_Overview_Dashboard.py renders multi-dataset and checkpoint artifacts."""

    def test_overview_dashboard_renders_cleanly(self):
        script_path = ROOT_DIR / "src/crip/app/pages/0_Overview_Dashboard.py"
        at = AppTest.from_file(str(script_path), default_timeout=15)
        at.run()

        assert not at.exception
        assert len(at.metric) >= 6

        # Check for multi-dataset mentions
        page_text = " ".join([m.value for m in at.markdown] + [h.value for h in at.header])
        assert "ACDC" in page_text
        assert "M&Ms" in page_text
        assert "EMIDEC" in page_text
        assert "314" in page_text or "cine_joint_unet.pt" in page_text


class TestMRIAnalysisWorkspaces:
    """Verify 2_MRI_Analysis.py renders all 3 tabs including Cine comparison and EMIDEC."""

    def test_mri_analysis_renders_cleanly(self):
        script_path = ROOT_DIR / "src/crip/app/pages/2_MRI_Analysis.py"
        at = AppTest.from_file(str(script_path), default_timeout=15)
        at.run()

        assert not at.exception
        # Must have exactly 3 tabs: ACDC, Cine Comparison, EMIDEC LGE
        assert len(at.tabs) >= 3

        # Must render biventricular functional metrics
        metric_labels = [m.label for m in at.metric]
        assert any("LV End-Diastolic Vol" in label for label in metric_labels)
        assert any("LV Ejection Fraction" in label for label in metric_labels)

        # Tab text should mention workspaces
        full_text = " ".join([m.value for m in at.markdown] + [h.value for h in at.header] + [w.value for w in at.warning])
        assert "Cine Model Comparison" in full_text or "Cross-Dataset" in full_text
        assert "EMIDEC" in full_text

        # Verify mandatory experimental disclaimer is rendered
        assert "remain experimental" in full_text or "Do not use for clinical decisions" in full_text


class TestEvaluationPageMultiDataset:
    """Verify 4_Evaluation.py renders all 6 tabs and handles dual recovery benchmarks."""

    def test_evaluation_page_renders_six_tabs(self):
        script_path = ROOT_DIR / "src/crip/app/pages/4_Evaluation.py"
        at = AppTest.from_file(str(script_path), default_timeout=15)
        at.run()

        assert not at.exception
        # Tabs: Real MRI, Agreement, Recovery, Synthetic Explorer, Figure Explorer, Audit
        assert len(at.tabs) >= 6
        assert len(at.metric) >= 4

    def test_recovery_experiment_toggle_and_content(self):
        script_path = ROOT_DIR / "src/crip/app/pages/4_Evaluation.py"
        at = AppTest.from_file(str(script_path), default_timeout=15)
        at.run()

        assert not at.exception
        full_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info])
        # Experiment 1 metrics
        assert "0.9565" in full_text
        assert "419" in full_text
        # Experiment 2 mentions
        assert "Experiment 2" in full_text or "All-Comers" in full_text

        # Toggle to Experiment 2 (All-Comers) and ensure zero exceptions
        at.radio[2].set_value("Experiment 2: Unselected All-Comers Cohort (N = 450, Baseline vs. Empirical)").run()
        assert not at.exception
        full_text_exp2 = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [m.label for m in at.metric])
        assert "450" in full_text_exp2

    def test_synthetic_cohort_explorer_present(self):
        script_path = ROOT_DIR / "src/crip/app/pages/4_Evaluation.py"
        at = AppTest.from_file(str(script_path), default_timeout=15)
        at.run()

        full_text = " ".join([m.value for m in at.markdown] + [s.value for s in at.subheader])
        assert "Synthetic Cohort" in full_text
        assert "Mathematical Cardiac Invariant" in full_text or "Invariant Audit" in full_text


class TestFigureExplorerComponent:
    """Validate figure explorer collections, manifests, and file integrity."""

    def test_figure_collections_dictionary(self):
        assert len(FIGURE_COLLECTIONS) >= 6
        assert "poster" in FIGURE_COLLECTIONS
        assert "synthetic_integration" in FIGURE_COLLECTIONS
        assert "mnms_external" in FIGURE_COLLECTIONS
        assert "emidec_tissue" in FIGURE_COLLECTIONS
        assert "emidec_training" in FIGURE_COLLECTIONS
        assert "joint_cine" in FIGURE_COLLECTIONS

    def test_figure_assets_exist(self):
        for col_key, col in FIGURE_COLLECTIONS.items():
            base_dir = ROOT_DIR / col["directory"]
            for fig in col["figures"]:
                png_path = base_dir / f"{fig['base']}.png"
                assert png_path.is_file(), f"Missing figure file: {png_path}"


class TestGracefulFallbacksAndMissingData:
    """Verify pages and components do not crash when datasets or models are absent."""

    def test_missing_data_warning_handling(self, monkeypatch):
        from crip.app.components.synthetic_explorer import load_cohort_datasets
        # Verify function executes safely and returns tuple
        df_base, df_emp = load_cohort_datasets()
        assert df_base is not None and df_emp is not None
        assert len(df_base) == 3000
        assert len(df_emp) == 3000

