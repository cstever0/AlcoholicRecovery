"""Unit and integration tests for cardiac volumetric measurements and batch pipeline.

Covers all Phase 7 testing requirements:
1. Correct LV cavity volume
2. Correct RV cavity volume
3. Correct myocardial volume & mass
4. Correct stroke volume
5. Correct LVEF
6. Correct RVEF
7. Non-isotropic voxel spacing
8. Empty segmentation masks
9. Zero EDV handling
10. ESV greater than EDV handling
11. Invalid segmentation labels
12. Missing ED or ES frames
13. Patient-level processing
14. Batch processing
15. CSV and JSON exports
16. Streamlit measurement display
17. Reproducibility of measurement outputs
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crip.mri.acdc_loader import ACDCDataError, ACDCDataLoader
from crip.mri.cardiac_metrics import (
    MYOCARDIAL_DENSITY_G_PER_ML,
    LABEL_BACKGROUND,
    LABEL_RV_CAVITY,
    LABEL_MYOCARDIUM,
    LABEL_LV_CAVITY,
    calculate_ejection_fraction,
    calculate_myocardial_mass_g,
    calculate_structure_volume_ml,
    compute_biventricular_measurements,
)
from crip.mri.pipeline import (
    compute_cohort_descriptive_stats,
    process_single_patient,
    run_pipeline,
)


class TestCardiacMeasurementsAnalytical:
    """Analytical unit tests using known geometric volumes and label arrays."""

    def test_correct_lv_cavity_volume(self):
        # 10x10x10 voxels = 1000 voxels of label 3 (LV)
        # Spacing: 1.0 x 1.0 x 1.0 mm -> 1 mm³ per voxel -> 0.001 mL per voxel
        # 1000 voxels * 0.001 mL = 1.0 mL
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[:, :, :] = LABEL_LV_CAVITY
        vol_ml, count, vox_vol = calculate_structure_volume_ml(mask, (1.0, 1.0, 1.0), target_label=LABEL_LV_CAVITY)
        assert count == 1000
        assert vox_vol == pytest.approx(1.0)
        assert vol_ml == pytest.approx(1.0)

    def test_correct_rv_cavity_volume(self):
        # 500 voxels of label 1 (RV) with spacing 2.0 x 2.0 x 5.0 mm = 20 mm³ = 0.02 mL per voxel
        # 500 * 0.02 = 10.0 mL
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[:5, :, :] = LABEL_RV_CAVITY
        vol_ml, count, vox_vol = calculate_structure_volume_ml(mask, (2.0, 2.0, 5.0), target_label=LABEL_RV_CAVITY)
        assert count == 500
        assert vox_vol == pytest.approx(20.0)
        assert vol_ml == pytest.approx(10.0)

    def test_correct_myocardial_volume_and_mass(self):
        # 250 voxels of label 2 (Myo) with spacing 2.0 x 2.0 x 10.0 mm = 40 mm³ = 0.04 mL per voxel
        # 250 * 0.04 = 10.0 mL
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[:5, :5, :] = LABEL_MYOCARDIUM
        vol_ml, count, _ = calculate_structure_volume_ml(mask, (2.0, 2.0, 10.0), target_label=LABEL_MYOCARDIUM)
        assert count == 250
        assert vol_ml == pytest.approx(10.0)

        # Mass = 10.0 mL * 1.05 g/mL = 10.5 g
        mass_g = calculate_myocardial_mass_g(vol_ml)
        assert mass_g == pytest.approx(10.5)

    def test_correct_stroke_volume(self):
        # EDV = 150.0 mL, ESV = 60.0 mL -> SV = 90.0 mL
        sv, _ = calculate_ejection_fraction(150.0, 60.0)
        assert sv == pytest.approx(90.0)

    def test_correct_lvef(self):
        # EDV = 200.0 mL, ESV = 80.0 mL -> SV = 120.0 mL -> LVEF = 60.0%
        sv, ef = calculate_ejection_fraction(200.0, 80.0)
        assert sv == pytest.approx(120.0)
        assert ef == pytest.approx(60.0)

    def test_correct_rvef(self):
        # RVEDV = 180.0 mL, RVESV = 90.0 mL -> RVSV = 90.0 mL -> RVEF = 50.0%
        sv, ef = calculate_ejection_fraction(180.0, 90.0)
        assert sv == pytest.approx(90.0)
        assert ef == pytest.approx(50.0)

    def test_non_isotropic_voxel_spacing(self):
        # Spacing: dx=1.25 mm, dy=1.75 mm, dz=8.0 mm
        # Voxel volume = 1.25 * 1.75 * 8.0 = 17.5 mm³ = 0.0175 mL
        # 800 voxels * 0.0175 mL = 14.0 mL
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[:8, :, :] = LABEL_LV_CAVITY
        spacing = (1.25, 1.75, 8.0)
        vol_ml, count, vox_vol = calculate_structure_volume_ml(mask, spacing, target_label=LABEL_LV_CAVITY)
        assert count == 800
        assert vox_vol == pytest.approx(17.5)
        assert vol_ml == pytest.approx(14.0)

    def test_empty_segmentation_masks(self):
        # Mask of all zeros (background)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        vol_ml, count, _ = calculate_structure_volume_ml(mask, (1.0, 1.0, 10.0), target_label=LABEL_LV_CAVITY)
        assert count == 0
        assert vol_ml == 0.0

    def test_zero_edv_handling(self):
        # When ED mask is empty for LV, EDV is 0
        ed_mask = np.zeros((4, 4, 4), dtype=np.uint8)
        es_mask = np.zeros((4, 4, 4), dtype=np.uint8)
        es_mask[1, 1, 1] = LABEL_LV_CAVITY

        measurements = compute_biventricular_measurements(ed_mask, es_mask, (1.0, 1.0, 1.0))
        assert measurements.is_physiologically_valid is False
        assert any("LVEDV is zero" in w for w in measurements.validation_warnings)
        assert measurements.lvef_percent == 0.0

    def test_esv_greater_than_edv_handling(self):
        # Non-physiologic: ES mask has more LV voxels than ED mask
        ed_mask = np.zeros((4, 4, 4), dtype=np.uint8)
        es_mask = np.zeros((4, 4, 4), dtype=np.uint8)

        ed_mask[:1, :, :] = LABEL_LV_CAVITY  # 16 voxels
        es_mask[:3, :, :] = LABEL_LV_CAVITY  # 48 voxels

        measurements = compute_biventricular_measurements(ed_mask, es_mask, (1.0, 1.0, 1.0))
        assert measurements.is_physiologically_valid is False
        assert any("negative stroke volume" in w for w in measurements.validation_warnings)
        assert measurements.lvsv_ml < 0
        assert measurements.lvef_percent < 0

    def test_invalid_segmentation_labels_raise_error(self):
        # Mask with unexpected label 7
        ed_mask = np.zeros((4, 4, 4), dtype=np.uint8)
        es_mask = np.zeros((4, 4, 4), dtype=np.uint8)
        ed_mask[0, 0, 0] = 7

        with pytest.raises(ValueError) as excinfo:
            compute_biventricular_measurements(ed_mask, es_mask, (1.0, 1.0, 1.0))
        assert "invalid labels" in str(excinfo.value)

    def test_missing_masks_raise_error(self):
        with pytest.raises(ValueError):
            compute_biventricular_measurements(None, None, (1.0, 1.0, 1.0))


class TestPipelinePatientAndBatchProcessing:
    """Tests for patient-level and batch measurement extraction and exports."""

    def test_patient_level_processing_synthetic(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        rec = process_single_patient(loader, "patient001")

        assert rec["patient_id"] == "patient001"
        assert rec["diagnostic_group"] == "DCM"
        assert rec["lvedv_ml"] > 0
        assert rec["lvesv_ml"] > 0
        assert rec["lvsv_ml"] > 0
        assert rec["lvef_percent"] > 0
        assert rec["rvedv_ml"] > 0
        assert rec["myocardial_mass_ed_g"] > 0
        assert Path(rec["ed_image_source_file"]).exists()
        assert Path(rec["ed_mask_source_file"]).exists()

    def test_batch_processing_and_exports(self, synthetic_acdc_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "processed_output"
        df, report = run_pipeline(synthetic_acdc_dir, output_dir=out_dir, generate_figures=True)

        assert report["pipeline_status"] == "SUCCESS"
        assert report["total_discovered"] == 1
        assert report["total_processed_successfully"] == 1
        assert report["total_failed"] == 0

        # Verify CSV export
        csv_file = out_dir / "cardiac_measurements.csv"
        assert csv_file.is_file()
        loaded_df = pd.read_csv(csv_file)
        assert len(loaded_df) == 1
        assert "lvef_percent" in loaded_df.columns
        assert "rvef_percent" in loaded_df.columns

        # Verify JSON export
        json_file = out_dir / "cardiac_measurements.json"
        assert json_file.is_file()
        with open(json_file, "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["patient_id"] == "patient001"

        # Verify Validation Report JSON
        rep_file = out_dir / "measurement_validation_report.json"
        assert rep_file.is_file()
        with open(rep_file, "r") as f:
            rep_data = json.load(f)
        assert rep_data["cohort_statistics"]["cohort_size"] == 1

        # Verify figures generated
        fig_dir = out_dir / "figures"
        assert (fig_dir / "distribution_lvef.png").is_file()
        assert (fig_dir / "distribution_lvedv.png").is_file()
        assert (fig_dir / "lvef_by_diagnostic_group.png").is_file()
        assert (fig_dir / "lvedv_vs_lvef.png").is_file()

    def test_reproducibility_of_measurement_outputs(self, synthetic_acdc_dir: Path, tmp_path: Path):
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        df1, _ = run_pipeline(synthetic_acdc_dir, output_dir=out1, generate_figures=False)
        df2, _ = run_pipeline(synthetic_acdc_dir, output_dir=out2, generate_figures=False)

        # Verify deterministic equality
        pd.testing.assert_frame_equal(df1, df2)

    def test_streamlit_measurement_display_apptest(self):
        from streamlit.testing.v1 import AppTest

        script_path = Path(__file__).resolve().parent.parent / "src/crip/app/pages/2_MRI_Analysis.py"
        at = AppTest.from_file(str(script_path), default_timeout=15)
        at.run()
        assert not at.exception

        # Assert metrics are present in the rendered page
        metric_labels = [m.label for m in at.metric]
        assert any("LV End-Diastolic Vol" in l for l in metric_labels)
        assert any("LV Ejection Fraction" in l for l in metric_labels)
        assert any("RV End-Diastolic Vol" in l for l in metric_labels)
        assert any("RV Ejection Fraction" in l for l in metric_labels)
        assert any("Estimated Myocardial Mass" in l for l in metric_labels)
