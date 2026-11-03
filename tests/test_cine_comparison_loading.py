"""Automated regression and unit test suite for Cine Model Comparison workspace loading.

Tests:
1. Loading ACDC patient001 (training smoke case) in ED and ES phases.
2. Loading held-out ACDC patient007 (test split case) in ED and ES phases.
3. Proper CardiacPhase enum handling and string normalization.
4. Robust rejection of invalid phase types (e.g. integer frame index passed instead of phase string).
5. M&Ms dataset compatibility and canonical label mapping.
6. Inference engine slice prediction and visual overlay dimensions.
7. Graceful degradation when checkpoints or raw data directories are missing.
8. Streamlit AppTest execution ensuring Tab 2 renders without the "'int' object has no attribute 'upper'" error.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from crip.app.components.cine_comparison import (
    BASELINE_CHECKPOINT,
    JOINT_CHECKPOINT,
    create_cine_overlay,
    create_difference_map,
    get_cine_engines,
)
from crip.datasets.mnms_loader import MNMSDataLoader
from crip.datasets.schemas import CardiacPhase
from crip.mri.acdc_loader import ACDCCardiacFrame, ACDCDataLoader
from crip.mri.segmentation_inference import SegmentationInferenceEngine

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestACDCLoaderPhaseHandling:
    """Verify ACDCDataLoader.load_cardiac_frame accepts valid phase types and rejects invalid ones."""

    def test_load_cardiac_frame_accepts_valid_strings(self):
        loader = ACDCDataLoader()
        if not loader.is_available():
            pytest.skip("ACDC raw dataset not available locally.")

        for pid in ("patient001", "patient007"):
            for phase_str in ("ED", "ed", "ES", "es", "End-Diastole", "End-Systole"):
                frame = loader.load_cardiac_frame(pid, phase=phase_str, load_mask=True)
                assert isinstance(frame, ACDCCardiacFrame)
                assert frame.patient_id == pid
                assert frame.phase_name in ("ED", "ES")
                assert frame.image_data.ndim == 3
                assert frame.ground_truth_mask is not None
                assert frame.ground_truth_mask.shape == frame.image_data.shape

    def test_load_cardiac_frame_accepts_cardiac_phase_enum(self):
        loader = ACDCDataLoader()
        if not loader.is_available():
            pytest.skip("ACDC raw dataset not available locally.")

        ed_frame = loader.load_cardiac_frame("patient001", phase=CardiacPhase.ED, load_mask=True)
        assert ed_frame.phase_name == "ED"

        es_frame = loader.load_cardiac_frame("patient007", phase=CardiacPhase.ES, load_mask=True)
        assert es_frame.phase_name == "ES"

    def test_load_cardiac_frame_rejects_integer_with_type_error(self):
        loader = ACDCDataLoader()
        if not loader.is_available():
            pytest.skip("ACDC raw dataset not available locally.")

        # Passing an int (e.g. frame number) must raise TypeError, not AttributeError
        with pytest.raises(TypeError) as exc_info:
            loader.load_cardiac_frame("patient001", phase=1)  # type: ignore[arg-type]
        assert "Cardiac phase must be a string" in str(exc_info.value)
        assert "got int" in str(exc_info.value)

    def test_load_cardiac_frame_rejects_invalid_string_with_value_error(self):
        loader = ACDCDataLoader()
        if not loader.is_available():
            pytest.skip("ACDC raw dataset not available locally.")

        with pytest.raises(ValueError) as exc_info:
            loader.load_cardiac_frame("patient001", phase="MID_SYSTOLE")
        assert "Invalid cardiac phase" in str(exc_info.value)

    def test_load_cardiac_frame_rejects_incompatible_enum(self):
        loader = ACDCDataLoader()
        if not loader.is_available():
            pytest.skip("ACDC raw dataset not available locally.")

        with pytest.raises(ValueError) as exc_info:
            loader.load_cardiac_frame("patient001", phase=CardiacPhase.STATIC)
        assert "cannot be resolved to an ACDC 3D frame" in str(exc_info.value)


class TestCineComparisonDataExtraction:
    """Verify 3D volume transposition and slice extraction across ACDC and M&Ms."""

    def test_acdc_patient001_transposition_and_shapes(self):
        loader = ACDCDataLoader()
        if not loader.is_available():
            pytest.skip("ACDC raw dataset not available locally.")

        for pid in ("patient001", "patient007"):
            for phase_key in ("ED", "ES"):
                frame = loader.load_cardiac_frame(pid, phase=phase_key, load_mask=True)
                # Standardize to (H, W, num_slices)
                img_3d = np.transpose(frame.image_data, (1, 2, 0))
                gt_3d = np.transpose(frame.mask_data, (1, 2, 0)) if frame.mask_data is not None else None

                assert img_3d.ndim == 3
                num_slices = img_3d.shape[2]
                assert num_slices >= 5
                assert gt_3d is not None
                assert gt_3d.shape == img_3d.shape

                # Check middle slice
                mid_idx = num_slices // 2
                slice_img = img_3d[:, :, mid_idx]
                slice_gt = gt_3d[:, :, mid_idx]
                assert slice_img.shape == (img_3d.shape[0], img_3d.shape[1])
                assert slice_gt.shape == slice_img.shape
                assert set(np.unique(slice_gt)).issubset({0, 1, 2, 3})

    def test_mnms_case_loading_and_canonical_mapping(self):
        mnms = MNMSDataLoader()
        if not mnms.is_available:
            pytest.skip("M&Ms raw dataset not available locally.")

        paired_cases = mnms.get_case_ids(only_paired=True)
        if not paired_cases:
            pytest.skip("No paired M&Ms cases found.")

        case_id = paired_cases[0]
        p_frames = mnms.load_phase_frames(case_id, mapped_to_canonical=True)
        assert "ED" in p_frames
        assert "ES" in p_frames

        for phase_key in ("ED", "ES"):
            img = p_frames[phase_key]["image"]
            mask = p_frames[phase_key]["mask"]
            assert img.ndim == 3
            assert mask.shape == img.shape
            assert set(np.unique(mask)).issubset({0, 1, 2, 3})


class TestInferenceAndOverlays:
    """Verify model inference, visual overlays, and difference maps."""

    def test_overlay_and_difference_generation(self):
        # Synthetic slice data
        H, W = 64, 64
        slice_img = np.random.uniform(0.0, 1000.0, size=(H, W)).astype(np.float32)
        mask_base = np.zeros((H, W), dtype=np.uint8)
        mask_joint = np.zeros((H, W), dtype=np.uint8)

        # Region with label 1 (RV), label 2 (MYO), label 3 (LV)
        mask_base[10:30, 10:30] = 3
        mask_joint[10:32, 10:32] = 3  # Slight boundary disagreement

        overlay = create_cine_overlay(slice_img, mask_base)
        assert overlay.shape == (H, W, 3)
        assert overlay.min() >= 0.0 and overlay.max() <= 1.0

        diff_map = create_difference_map(slice_img, mask_base, mask_joint)
        assert diff_map.shape == (H, W, 3)
        assert diff_map.min() >= 0.0 and diff_map.max() <= 1.0

    def test_real_model_inference_on_acdc_test_case(self):
        loader = ACDCDataLoader()
        if not loader.is_available():
            pytest.skip("ACDC raw dataset not available locally.")
        if not BASELINE_CHECKPOINT.is_file() or not JOINT_CHECKPOINT.is_file():
            pytest.skip("Checkpoints not available locally.")

        base_eng, joint_eng = get_cine_engines()
        assert base_eng is not None and base_eng.is_model_available
        assert joint_eng is not None and joint_eng.is_model_available

        frame = loader.load_cardiac_frame("patient007", phase="ED", load_mask=True)
        img_3d = np.transpose(frame.image_data, (1, 2, 0))
        slice_img = img_3d[:, :, img_3d.shape[2] // 2]

        pred_base = base_eng.predict_slice_mask(slice_img)
        pred_joint = joint_eng.predict_slice_mask(slice_img)

        assert pred_base.shape == slice_img.shape
        assert pred_joint.shape == slice_img.shape
        assert set(np.unique(pred_base)).issubset({0, 1, 2, 3})
        assert set(np.unique(pred_joint)).issubset({0, 1, 2, 3})


class TestMissingCheckpointsAndDatasetHandling:
    """Verify graceful handling when files or directories are missing."""

    def test_missing_checkpoints_do_not_crash_engine_getter(self, tmp_path):
        get_cine_engines.clear()
        with patch("crip.app.components.cine_comparison.BASELINE_CHECKPOINT", tmp_path / "nonexistent1.pt"), \
             patch("crip.app.components.cine_comparison.JOINT_CHECKPOINT", tmp_path / "nonexistent2.pt"):
            base_eng, joint_eng = get_cine_engines()
            assert base_eng is None
            assert joint_eng is None
        get_cine_engines.clear()


class TestStreamlitCineComparisonAppTest:
    """Verify Streamlit 2_MRI_Analysis.py renders Cine Model Comparison Tab without exceptions."""

    def test_cine_comparison_renders_without_loading_error(self):
        script_path = ROOT_DIR / "src/crip/app/pages/2_MRI_Analysis.py"
        at = AppTest.from_file(str(script_path), default_timeout=25)
        at.run()

        # No uncaught exceptions
        assert not at.exception

        # Check errors displayed in st.error
        error_texts = [e.value for e in at.error]
        for err in error_texts:
            assert "has no attribute 'upper'" not in err
            assert "Error loading patient001" not in err

        # Verify page content
        full_text = " ".join([m.value for m in at.markdown] + [h.value for h in at.header])
        assert "Cine Model Comparison" in full_text

    def test_cine_comparison_patient_and_phase_switching(self):
        script_path = ROOT_DIR / "src/crip/app/pages/2_MRI_Analysis.py"
        at = AppTest.from_file(str(script_path), default_timeout=25)
        at.run()
        assert not at.exception

        # Switch to held-out test case patient007 if present
        patient_select = None
        for sb in at.selectbox:
            if "Select ACDC Patient" in sb.label:
                patient_select = sb
                break

        if patient_select is not None and "patient007" in patient_select.options:
            patient_select.select("patient007")
            at.run()
            assert not at.exception
            error_texts = [e.value for e in at.error]
            for err in error_texts:
                assert "Error loading" not in err

        # Switch to ES phase
        phase_select = None
        for sb in at.selectbox:
            if "Cardiac Phase" in sb.label:
                phase_select = sb
                break

        if phase_select is not None:
            es_opt = [o for o in phase_select.options if "ES" in o]
            if es_opt:
                phase_select.select(es_opt[0])
                at.run()
                assert not at.exception

        # Switch dataset to M&Ms
        dataset_select = None
        for sb in at.selectbox:
            if "Select Cine Dataset" in sb.label:
                dataset_select = sb
                break

        if dataset_select is not None:
            mnms_opt = [o for o in dataset_select.options if "M&Ms" in o]
            if mnms_opt:
                dataset_select.select(mnms_opt[0])
                at.run()
                assert not at.exception
                error_texts = [e.value for e in at.error]
                for err in error_texts:
                    assert "has no attribute 'upper'" not in err
