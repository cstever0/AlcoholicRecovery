"""Integration tests against the real installed ACDC cardiac MRI dataset.

DISCLAIMER:
These integration tests verify compatibility with the actual local ACDC dataset.
They are executed only if the ACDC training dataset is present on disk.
"""

from pathlib import Path
import pytest

from crip.mri.acdc_loader import (
    ACDCDataLoader,
    DEFAULT_ACDC_TRAINING_DIR,
    VALID_SEGMENTATION_LABELS,
)
from crip.mri.cardiac_metrics import compute_cardiac_volumetrics_from_masks

ACDC_EXISTS = DEFAULT_ACDC_TRAINING_DIR.is_dir() and len(list(DEFAULT_ACDC_TRAINING_DIR.glob("patient*"))) > 0


@pytest.mark.skipif(not ACDC_EXISTS, reason="Real ACDC dataset not installed locally at data/raw/acdc/training")
class TestACDCRealDatasetIntegration:
    """Integration verification using actual downloaded ACDC NIfTI scans."""

    def test_training_cohort_patient_count(self):
        loader = ACDCDataLoader(DEFAULT_ACDC_TRAINING_DIR)
        patients = loader.list_patients()
        assert len(patients) == 100, f"Expected 100 training patients, found {len(patients)}"
        assert patients[0] == "patient001"
        assert patients[-1] == "patient100"

    def test_load_patient001_complete_record(self):
        loader = ACDCDataLoader(DEFAULT_ACDC_TRAINING_DIR)
        meta = loader.get_patient_metadata("patient001")
        assert meta.patient_id == "patient001"
        assert meta.ed_frame == 1
        assert meta.es_frame == 12
        assert meta.group == "DCM"

        # Load ED frame and mask
        ed_frame = loader.load_cardiac_frame("patient001", phase="ED", load_mask=True)
        assert ed_frame.num_slices == 10
        assert ed_frame.mask_data is not None
        assert ed_frame.image_data.shape == ed_frame.mask_data.shape

        ed_labels = set(ed_frame.mask_data.flatten().tolist())
        assert ed_labels.issubset(VALID_SEGMENTATION_LABELS)
        assert 3 in ed_labels  # LV cavity present

        # Load ES frame and mask
        es_frame = loader.load_cardiac_frame("patient001", phase="ES", load_mask=True)
        assert es_frame.num_slices == 10
        assert es_frame.mask_data is not None
        assert es_frame.image_data.shape == es_frame.mask_data.shape

        es_labels = set(es_frame.mask_data.flatten().tolist())
        assert es_labels.issubset(VALID_SEGMENTATION_LABELS)

    def test_patient001_volumetrics_computation(self):
        loader = ACDCDataLoader(DEFAULT_ACDC_TRAINING_DIR)
        ed_frame = loader.load_cardiac_frame("patient001", phase="ED", load_mask=True)
        es_frame = loader.load_cardiac_frame("patient001", phase="ES", load_mask=True)

        assert ed_frame.mask_data is not None and es_frame.mask_data is not None
        metrics = compute_cardiac_volumetrics_from_masks(
            ed_frame.mask_data,
            es_frame.mask_data,
            ed_frame.voxel_spacing,
            lv_label=3,
        )

        assert metrics.edv_ml > metrics.esv_ml
        assert metrics.stroke_volume_ml > 0
        assert 10.0 <= metrics.lvef_percent <= 80.0

    def test_load_patient001_4d_cine(self):
        loader = ACDCDataLoader(DEFAULT_ACDC_TRAINING_DIR)
        cine = loader.load_cine_4d("patient001")
        assert cine.num_frames == 30
        assert cine.num_slices == 10
        assert cine.cine_data.ndim == 4
