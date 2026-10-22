"""Unit tests for ACDC Data Loader using synthetic deterministic NIfTI fixtures.

DISCLAIMER:
All fixtures created herein are small, deterministic synthetic NIfTI volumes used
strictly for unit testing data loading logic. They do NOT represent real clinical MRI scans.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import pytest

from crip.mri.acdc_loader import (
    ACDCDataError,
    ACDCDataLoader,
    ACDCCardiacFrame,
    normalize_for_display,
)


class TestACDCDataLoaderUnit:
    """Unit tests verifying data loader operations against synthetic fixtures."""

    def test_valid_patient_discovery(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        assert loader.is_available() is True
        patients = loader.list_patients()
        assert patients == ["patient001"]

    def test_valid_info_cfg_parsing(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        meta = loader.get_patient_metadata("patient001")
        assert meta.patient_id == "patient001"
        assert meta.ed_frame == 1
        assert meta.es_frame == 6
        assert meta.group == "DCM"
        assert meta.height_cm == 180.0
        assert meta.weight_kg == 80.0
        assert meta.num_frames == 10

    def test_missing_patient_directories(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        with pytest.raises(ACDCDataError) as excinfo:
            loader.get_patient_metadata("patient999")
        assert "does not exist" in str(excinfo.value)

    def test_missing_mri_images(self, synthetic_acdc_dir: Path):
        # Remove ED image file
        ed_file = synthetic_acdc_dir / "patient001" / "patient001_frame01.nii.gz"
        ed_file.unlink()

        loader = ACDCDataLoader(synthetic_acdc_dir)
        with pytest.raises(ACDCDataError) as excinfo:
            loader.load_cardiac_frame("patient001", phase="ED")
        assert "image file not found" in str(excinfo.value)

    def test_missing_segmentation_masks(self, synthetic_acdc_dir: Path):
        # Remove ED ground truth mask
        gt_file = synthetic_acdc_dir / "patient001" / "patient001_frame01_gt.nii.gz"
        gt_file.unlink()

        loader = ACDCDataLoader(synthetic_acdc_dir)
        with pytest.raises(ACDCDataError) as excinfo:
            loader.load_cardiac_frame("patient001", phase="ED", load_mask=True)
        assert "mask file not found" in str(excinfo.value)

    def test_corrupted_nifti_files(self, synthetic_acdc_dir: Path):
        # Overwrite NIfTI with random invalid bytes
        ed_file = synthetic_acdc_dir / "patient001" / "patient001_frame01.nii.gz"
        ed_file.write_bytes(b"CORRUPTED_BINARY_DATA_NOT_A_NIFTI")

        loader = ACDCDataLoader(synthetic_acdc_dir)
        with pytest.raises(ACDCDataError) as excinfo:
            loader.load_cardiac_frame("patient001", phase="ED")
        assert "Failed to load" in str(excinfo.value)

    def test_ed_es_frame_identification(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        ed_frame = loader.load_cardiac_frame("patient001", phase="ED")
        assert ed_frame.phase_name == "ED"
        assert ed_frame.frame_index == 1

        es_frame = loader.load_cardiac_frame("patient001", phase="ES")
        assert es_frame.phase_name == "ES"
        assert es_frame.frame_index == 6

    def test_mri_and_mask_shape_compatibility(self, synthetic_acdc_dir: Path):
        # Create incompatible mask shape (e.g. 8 slices instead of 4)
        gt_file = synthetic_acdc_dir / "patient001" / "patient001_frame01_gt.nii.gz"
        incomp_mask = np.zeros((16, 16, 8), dtype=np.uint8)
        nib.save(nib.Nifti1Image(incomp_mask, np.eye(4)), str(gt_file))

        loader = ACDCDataLoader(synthetic_acdc_dir)
        with pytest.raises(ACDCDataError) as excinfo:
            loader.load_cardiac_frame("patient001", phase="ED", load_mask=True)
        assert "Shape mismatch" in str(excinfo.value)

    def test_segmentation_label_correctness(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        ed_frame = loader.load_cardiac_frame("patient001", phase="ED")
        assert ed_frame.mask_data is not None
        labels = set(np.unique(ed_frame.mask_data).tolist())
        assert labels.issubset({0, 1, 2, 3})

        # Inject invalid label (e.g. label 9)
        gt_file = synthetic_acdc_dir / "patient001" / "patient001_frame01_gt.nii.gz"
        bad_mask = np.zeros((16, 16, 4), dtype=np.uint8)
        bad_mask[0, 0, 0] = 9
        nib.save(nib.Nifti1Image(bad_mask, np.eye(4)), str(gt_file))

        with pytest.raises(ACDCDataError) as excinfo:
            loader.load_cardiac_frame("patient001", phase="ED", load_mask=True)
        assert "Unexpected segmentation label" in str(excinfo.value)

    def test_voxel_spacing_and_affine_extraction(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        ed_frame = loader.load_cardiac_frame("patient001", phase="ED")
        assert len(ed_frame.voxel_spacing) == 3
        assert ed_frame.voxel_spacing[0] == pytest.approx(1.5, rel=1e-2)
        assert ed_frame.voxel_spacing[1] == pytest.approx(1.5, rel=1e-2)
        assert ed_frame.voxel_spacing[2] == pytest.approx(10.0, rel=1e-2)
        assert ed_frame.affine.shape == (4, 4)

    def test_valid_slice_selection(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        ed_frame = loader.load_cardiac_frame("patient001", phase="ED")
        # num_slices is 4
        slice_0 = ed_frame.get_slice_image(0)
        assert slice_0.shape == (ed_frame.height, ed_frame.width)
        mask_0 = ed_frame.get_slice_mask(0)
        assert mask_0.shape == (ed_frame.height, ed_frame.width)

    def test_invalid_slice_selection_raises_error(self, synthetic_acdc_dir: Path):
        loader = ACDCDataLoader(synthetic_acdc_dir)
        ed_frame = loader.load_cardiac_frame("patient001", phase="ED")
        with pytest.raises(IndexError):
            ed_frame.get_slice_image(99)
        with pytest.raises(IndexError):
            ed_frame.get_slice_mask(-1)

    def test_normalize_for_display(self):
        arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        norm = normalize_for_display(arr)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

        # Edge case: zero array
        zeros = np.zeros((5, 5), dtype=np.float32)
        norm_zeros = normalize_for_display(zeros)
        assert np.all(norm_zeros == 0.0)

    def test_graceful_handling_of_unavailable_dataset(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist"
        loader = ACDCDataLoader(nonexistent)
        assert loader.is_available() is False
        assert loader.list_patients() == []
