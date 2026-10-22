"""Unit tests for cardiac volumetric and functional metric calculations."""

import numpy as np
import pytest

from crip.mri.cardiac_metrics import (
    calculate_ejection_fraction,
    calculate_structure_volume_ml,
    compute_cardiac_volumetrics_from_masks,
)


class TestCardiacMetrics:
    """Tests for physical volume calculations and mathematical formulas."""

    def test_calculate_structure_volume_ml(self):
        # Spacing: 2 mm x 2 mm x 10 mm = 40 mm^3 per voxel = 0.04 mL per voxel
        # 100 voxels of label 3 -> 100 * 0.04 = 4.0 mL
        mask = np.zeros((10, 20, 20), dtype=np.uint8)
        mask[:5, :5, :4] = 3  # 5 * 5 * 4 = 100 voxels

        spacing = (2.0, 2.0, 10.0)
        volume_ml, count, vox_vol = calculate_structure_volume_ml(mask, spacing, target_label=3)

        assert count == 100
        assert vox_vol == pytest.approx(40.0)
        assert volume_ml == pytest.approx(4.0)

    def test_calculate_ejection_fraction(self):
        # EDV = 200 mL, ESV = 120 mL -> SV = 80 mL, LVEF = 40.0%
        sv, lvef = calculate_ejection_fraction(200.0, 120.0)
        assert sv == pytest.approx(80.0)
        assert lvef == pytest.approx(40.0)

    def test_invalid_negative_or_zero_edv_raises(self):
        with pytest.raises(ValueError):
            calculate_ejection_fraction(0.0, 50.0)
        with pytest.raises(ValueError):
            calculate_ejection_fraction(-10.0, 50.0)

    def test_compute_cardiac_volumetrics_from_masks(self):
        # Create ED and ES masks with spacing (1.5, 1.5, 10.0) -> voxel volume = 22.5 mm^3 = 0.0225 mL
        spacing = (1.5, 1.5, 10.0)
        ed_mask = np.zeros((5, 10, 10), dtype=np.uint8)
        es_mask = np.zeros((5, 10, 10), dtype=np.uint8)

        # ED: 400 voxels -> 400 * 0.0225 = 9.0 mL
        ed_mask[:4, :10, :10] = 3
        # ES: 200 voxels -> 200 * 0.0225 = 4.5 mL
        es_mask[:2, :10, :10] = 3

        volumetrics = compute_cardiac_volumetrics_from_masks(ed_mask, es_mask, spacing, lv_label=3)

        assert volumetrics.edv_ml == pytest.approx(9.0)
        assert volumetrics.esv_ml == pytest.approx(4.5)
        assert volumetrics.stroke_volume_ml == pytest.approx(4.5)
        assert volumetrics.lvef_percent == pytest.approx(50.0)
        assert volumetrics.is_validated is False
