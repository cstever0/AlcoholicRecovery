"""Tests for EMIDEC delayed-enhancement tissue characterization and metrics.

Validates physical voxel conversions, tissue compartment volume calculations,
myocardial denominator isolation, relative burdens, error handling, and figures.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytest

from crip.datasets.emidec_loader import EMIDECDataLoader
from crip.datasets.labels import EMIDECLabel
from crip.emidec.tissue_metrics import (
    DEFAULT_MEASUREMENTS_CSV,
    DEFAULT_MEASUREMENTS_SUMMARY,
    EMIDECTissueAnalyzer,
    EMIDECTissueMeasurements,
    compute_emidec_tissue_metrics,
    validate_physical_spacing,
)


# ==============================================================================
# Unit Tests (Deterministic fixtures, runs on any CI / environment)
# ==============================================================================


class TestEMIDECTissueMetricsUnit:
    """Deterministic mathematical tests validating tissue volume and burden logic."""

    def test_validate_physical_spacing_valid(self):
        sx, sy, sz = validate_physical_spacing((1.5, 1.5, 8.0))
        assert sx == 1.5
        assert sy == 1.5
        assert sz == 8.0

    def test_validate_physical_spacing_invalid(self):
        with pytest.raises(ValueError, match="at least 3"):
            validate_physical_spacing((1.5, 1.5))

        with pytest.raises(ValueError, match="Non-positive"):
            validate_physical_spacing((1.5, -1.0, 8.0))

        with pytest.raises(ValueError, match="Non-positive"):
            validate_physical_spacing((0.0, 1.5, 8.0))

        with pytest.raises(ValueError, match="Non-finite"):
            validate_physical_spacing((1.5, float("nan"), 8.0))

    def test_compute_metrics_known_volumes(self):
        """Construct synthetic array with exactly known voxel counts:

        Shape: (10, 10, 10) = 1,000 voxels total
        Voxel spacing: (2.0, 2.0, 2.5) mm
        Voxel volume = 2.0 * 2.0 * 2.5 = 10.0 mm^3 = 0.010 mL per voxel

        Labels:
          - Background (0): 500 voxels -> 5.0 mL
          - LV Cavity (1): 200 voxels -> 2.0 mL
          - Normal Myo (2): 200 voxels -> 2.0 mL
          - Infarct (3): 80 voxels -> 0.8 mL
          - No-reflow (4): 20 voxels -> 0.2 mL
        Total Myocardium (2 + 3 + 4) = 300 voxels = 3.0 mL
        Infarct burden = 0.8 / 3.0 = 26.6667%
        No-reflow burden = 0.2 / 3.0 = 6.6667%
        Combined burden = 1.0 / 3.0 = 33.3333%
        """
        mask = np.zeros((10, 10, 10), dtype=np.int16)
        flat = mask.ravel()
        # [0:500] is 0 (Background)
        flat[500:700] = EMIDECLabel.LV_CAVITY.value          # 200
        flat[700:900] = EMIDECLabel.NORMAL_MYOCARDIUM.value   # 200
        flat[900:980] = EMIDECLabel.MYOCARDIAL_INFARCTION.value # 80
        flat[980:1000] = EMIDECLabel.NO_REFLOW.value         # 20

        metrics = compute_emidec_tissue_metrics(
            mask_3d=mask,
            voxel_spacing_mm=(2.0, 2.0, 2.5),
            case_id="MOCK_P001",
            pathology_group="Myocardial Infarction",
        )

        assert metrics.case_id == "MOCK_P001"
        assert metrics.pathology_group == "Myocardial Infarction"
        assert metrics.voxel_volume_mm3 == 10.0
        assert round(metrics.lv_cavity_volume_ml, 4) == 2.0
        assert round(metrics.normal_myocardium_volume_ml, 4) == 2.0
        assert round(metrics.infarction_volume_ml, 4) == 0.8
        assert round(metrics.no_reflow_volume_ml, 4) == 0.2
        assert round(metrics.total_myocardium_volume_ml, 4) == 3.0
        assert round(metrics.estimated_myocardial_mass_g, 4) == round(3.0 * 1.05, 4)
        assert round(metrics.infarction_burden_percent, 2) == 26.67
        assert round(metrics.no_reflow_burden_percent, 2) == 6.67
        assert round(metrics.combined_abnormal_burden_percent, 2) == 33.33
        assert metrics.has_microvascular_obstruction is True
        assert metrics.is_valid is True

    def test_lv_cavity_not_counted_in_myocardium(self):
        """Crucial invariant: LV blood pool (label 1) must NEVER inflate myocardial volume."""
        mask = np.zeros((10, 10, 5), dtype=np.int16)
        mask[:, :, :] = EMIDECLabel.LV_CAVITY.value  # Entirely LV cavity

        metrics = compute_emidec_tissue_metrics(
            mask_3d=mask,
            voxel_spacing_mm=(1.0, 1.0, 5.0),
            case_id="MOCK_LV_ONLY",
            pathology_group="Normal",
        )
        assert metrics.lv_cavity_volume_ml > 0.0
        assert metrics.total_myocardium_volume_ml == 0.0
        assert metrics.infarction_burden_percent == 0.0
        assert metrics.no_reflow_burden_percent == 0.0

    def test_zero_myocardial_volume_handles_cleanly(self):
        mask = np.zeros((10, 10, 5), dtype=np.int16)  # All background

        metrics = compute_emidec_tissue_metrics(
            mask_3d=mask,
            voxel_spacing_mm=(1.0, 1.0, 1.0),
            case_id="MOCK_ZERO",
        )
        assert metrics.total_myocardium_volume_ml == 0.0
        assert metrics.infarction_burden_percent == 0.0
        assert metrics.no_reflow_burden_percent == 0.0
        assert len(metrics.validation_warnings) > 0
        assert "zero" in metrics.validation_warnings[0].lower()

    def test_normal_case_zero_infarct_and_no_reflow(self):
        """In a normal control case, only labels 0, 1, 2 should exist."""
        mask = np.zeros((10, 10, 5), dtype=np.int16)
        mask[2:8, 2:8, :] = EMIDECLabel.NORMAL_MYOCARDIUM.value
        mask[4:6, 4:6, :] = EMIDECLabel.LV_CAVITY.value

        metrics = compute_emidec_tissue_metrics(
            mask_3d=mask,
            voxel_spacing_mm=(1.5, 1.5, 6.0),
            case_id="MOCK_NORM",
            pathology_group="Normal",
        )
        assert metrics.infarction_volume_ml == 0.0
        assert metrics.no_reflow_volume_ml == 0.0
        assert metrics.infarction_burden_percent == 0.0
        assert metrics.no_reflow_burden_percent == 0.0
        assert metrics.has_microvascular_obstruction is False

    def test_non_3d_mask_raises(self):
        mask_2d = np.zeros((10, 10), dtype=np.int16)
        with pytest.raises(ValueError, match="3D NumPy array"):
            compute_emidec_tissue_metrics(mask_2d, (1.0, 1.0, 1.0))

    def test_unrecognized_labels_warning(self):
        mask = np.zeros((10, 10, 5), dtype=np.int16)
        mask[2:4, 2:4, :] = 99  # Invalid label
        metrics = compute_emidec_tissue_metrics(mask, (1.0, 1.0, 1.0))
        assert any("Unrecognized" in w for w in metrics.validation_warnings)


# ==============================================================================
# Integration Tests (Run when real local EMIDEC data is available)
# ==============================================================================

REAL_EMIDEC_PATH = Path("data/raw/emidec/emidec-dataset-1.0.1")


@pytest.mark.skipif(
    not REAL_EMIDEC_PATH.is_dir(),
    reason="EMIDEC raw data directory not found",
)
class TestRealEMIDECTissueIntegration:
    """Empirical integration tests validating actual EMIDEC dataset outputs."""

    def test_real_normal_case_measurements(self):
        loader = EMIDECDataLoader()
        case_data = loader.load_case("Case_N006")
        metrics = compute_emidec_tissue_metrics(
            mask_3d=case_data["contour"],
            voxel_spacing_mm=case_data["voxel_spacing"],
            case_id="Case_N006",
            pathology_group="Normal",
        )
        assert metrics.total_myocardium_volume_ml > 50.0
        assert metrics.normal_myocardium_volume_ml == metrics.total_myocardium_volume_ml
        assert metrics.infarction_volume_ml == 0.0
        assert metrics.no_reflow_volume_ml == 0.0
        assert metrics.infarction_burden_percent == 0.0

    def test_real_pathological_case_measurements(self):
        loader = EMIDECDataLoader()
        case_data = loader.load_case("Case_P001")
        metrics = compute_emidec_tissue_metrics(
            mask_3d=case_data["contour"],
            voxel_spacing_mm=case_data["voxel_spacing"],
            case_id="Case_P001",
            pathology_group="Myocardial Infarction",
        )
        assert metrics.total_myocardium_volume_ml > 50.0
        assert metrics.infarction_volume_ml > 0.0
        assert metrics.has_microvascular_obstruction is True
        assert metrics.no_reflow_volume_ml > 0.0
        assert metrics.infarction_burden_percent > 0.0

    def test_measurements_csv_and_summary_consistency(self):
        assert DEFAULT_MEASUREMENTS_CSV.is_file()
        assert DEFAULT_MEASUREMENTS_SUMMARY.is_file()

        df = pd.read_csv(DEFAULT_MEASUREMENTS_CSV)
        assert len(df) == 100
        assert (df["pathology_group"] == "Normal").sum() == 33
        assert (df["pathology_group"] == "Myocardial Infarction").sum() == 67

        # In all normal controls, infarct volume must be 0
        norm_infarctions = df[df["pathology_group"] == "Normal"]["infarction_volume_ml"]
        assert (norm_infarctions == 0.0).all()

        # In pathological cases, infarct volume must be > 0
        path_infarctions = df[df["pathology_group"] == "Myocardial Infarction"]["infarction_volume_ml"]
        assert (path_infarctions > 0.0).all()

        # MVO cases count
        path_df = df[df["pathology_group"] == "Myocardial Infarction"]
        assert path_df["has_microvascular_obstruction"].sum() == 40

    def test_all_emidec_poster_figures_exist_and_multiformat(self):
        fig_dir = Path("docs/evaluation/external_datasets/emidec")
        for num in range(1, 7):
            for ext in (".png", ".svg", ".pdf"):
                pattern = f"figure{num}_*{ext}"
                matches = list(fig_dir.glob(pattern))
                assert len(matches) > 0, f"Missing figure {pattern} in {fig_dir}"
                assert matches[0].stat().st_size > 0
