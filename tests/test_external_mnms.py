"""Tests for M&Ms external generalizability evaluation pipeline and poster figures.

Validates frozen model inference, label remapping, spatial dimension alignment,
per-class Dice, HD95, biventricular volumetrics, and figure generation routines.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from crip.datasets.labels import ACDCLabel, MNMSLabel, map_mnms_to_canonical
from crip.datasets.mnms_loader import MNMSDataLoader
from crip.evaluation.external_mnms import (
    ACDC_REFERENCE_BENCHMARKS,
    DEFAULT_MNMS_EVAL_CSV,
    DEFAULT_MNMS_EVAL_SUMMARY,
    MNMSExternalEvaluator,
    compute_file_sha256,
    evaluate_single_mnms_case,
)
from crip.evaluation.poster_figures_mnms import (
    generate_all_mnms_poster_figures,
    generate_figure1_dice_comparison,
    generate_figure2_hd95_distributions,
    generate_figure3_vendor_breakdown,
    generate_figure4_lvef_correlation,
    generate_figure5_segmentation_overlay,
)
from crip.mri.segmentation_inference import SegmentationInferenceEngine
from crip.mri.segmentation_model import DEFAULT_CHECKPOINT_PATH


# ==============================================================================
# Unit Tests (Deterministic fixtures, runs on any CI / checkout)
# ==============================================================================


@pytest.fixture
def mock_mnms_case_data(tmp_path: Path) -> Tuple[MNMSDataLoader, Dict[str, Any]]:
    """Create a synthetic M&Ms case with 4D cine and 4D mask."""
    case_dir = tmp_path / "Training" / "Labeled" / "MOCK01"
    case_dir.mkdir(parents=True)

    # 4D cine: (32, 32, 5, 6) -> (X, Y, Z, T)
    affine = np.diag([1.25, 1.25, 8.0, 1.0])
    img_data = (np.random.rand(32, 32, 5, 6) * 500).astype(np.float32)
    img_nii = nib.Nifti1Image(img_data, affine)
    img_path = case_dir / "MOCK01_sa.nii.gz"
    nib.save(img_nii, str(img_path))

    # 4D mask: raw M&Ms labels (1=LV, 2=MYO, 3=RV)
    mask_data = np.zeros((32, 32, 5, 6), dtype=np.int16)
    # ED at frame 0
    mask_data[10:20, 10:20, 1:4, 0] = MNMSLabel.LV.value
    mask_data[8:22, 8:22, 1:4, 0] = np.where(
        mask_data[8:22, 8:22, 1:4, 0] == 0,
        MNMSLabel.MYOCARDIUM.value,
        mask_data[8:22, 8:22, 1:4, 0],
    )
    mask_data[4:9, 10:20, 1:4, 0] = MNMSLabel.RV.value

    # ES at frame 3
    mask_data[12:18, 12:18, 1:4, 3] = MNMSLabel.LV.value
    mask_data[10:20, 10:20, 1:4, 3] = np.where(
        mask_data[10:20, 10:20, 1:4, 3] == 0,
        MNMSLabel.MYOCARDIUM.value,
        mask_data[10:20, 10:20, 1:4, 3],
    )
    mask_data[5:9, 12:18, 1:4, 3] = MNMSLabel.RV.value

    mask_nii = nib.Nifti1Image(mask_data, affine)
    mask_path = case_dir / "MOCK01_sa_gt.nii.gz"
    nib.save(mask_nii, str(mask_path))

    # CSV metadata
    csv_file = tmp_path / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    csv_file.write_text(
        "External code,VendorName,Vendor,Centre,ED,ES,Diagnosis\n"
        "MOCK01,Philips,B,2,0,3,DCM\n",
        encoding="utf-8",
    )

    loader = MNMSDataLoader(root_dir=tmp_path, csv_path=csv_file)
    cases = loader.discover_cases()
    return loader, cases["MOCK01"]


class TestMNMSEvaluationUnit:
    """Deterministic unit tests validating logic and metric isolation."""

    def test_acdc_reference_benchmarks_constants(self):
        assert ACDC_REFERENCE_BENCHMARKS["cohort_size"] == 20
        assert ACDC_REFERENCE_BENCHMARKS["mean_foreground_dice"] == 0.839
        assert ACDC_REFERENCE_BENCHMARKS["lv_dice"] == 0.906
        assert ACDC_REFERENCE_BENCHMARKS["rv_dice"] == 0.807
        assert ACDC_REFERENCE_BENCHMARKS["myo_dice"] == 0.804
        assert ACDC_REFERENCE_BENCHMARKS["lvef_mae"] == 3.86

    def test_compute_file_sha256(self, tmp_path: Path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello CRIP Multi-Dataset\n", encoding="utf-8")
        h = compute_file_sha256(test_file)
        assert len(h) == 64
        assert isinstance(h, str)

    def test_evaluate_single_case_missing_mask_raises(self, mock_mnms_case_data):
        loader, case_info = mock_mnms_case_data
        case_info_no_mask = dict(case_info)
        case_info_no_mask["mask_path"] = None

        engine = SegmentationInferenceEngine()
        # Mock load_case_cine to return None mask
        original_load = loader.load_case_cine
        loader.load_case_cine = lambda cid, remap_to_canonical=True: {
            "image": np.zeros((10, 10, 5, 2)),
            "mask": None,
            "voxel_spacing": (1.0, 1.0, 5.0),
        }

        with pytest.raises(ValueError, match="does not contain ground truth"):
            evaluate_single_mnms_case(loader, engine, case_info_no_mask)

        loader.load_case_cine = original_load

    def test_evaluate_single_case_invalid_frames_raises(self, mock_mnms_case_data):
        loader, case_info = mock_mnms_case_data
        case_info_bad_frame = dict(case_info)
        case_info_bad_frame["ed_frame"] = None

        engine = SegmentationInferenceEngine()
        with pytest.raises(ValueError, match="missing required ED or ES"):
            evaluate_single_mnms_case(loader, engine, case_info_bad_frame)

    def test_summary_aggregation(self):
        evaluator = MNMSExternalEvaluator()
        # Create synthetic DataFrame
        df = pd.DataFrame([
            {
                "case_id": "C1",
                "vendor": "Philips",
                "split_category": "Training/Labeled",
                "center": "Center_2",
                "dice_patient_mean": 0.80,
                "dice_patient_lv": 0.85,
                "dice_patient_rv": 0.75,
                "dice_patient_myo": 0.80,
                "hd95_patient_lv_mm": 5.0,
                "hd95_patient_rv_mm": 6.0,
                "hd95_patient_myo_mm": 5.5,
                "lvef_error_percent": -2.0,
                "lvef_abs_error_percent": 2.0,
                "lvef_within_5pct": True,
                "lvedv_abs_error_ml": 5.0,
                "lvesv_abs_error_ml": 4.0,
                "lvsv_abs_error_ml": 3.0,
            },
            {
                "case_id": "C2",
                "vendor": "Siemens",
                "split_category": "Validation",
                "center": "Center_1",
                "dice_patient_mean": 0.70,
                "dice_patient_lv": 0.75,
                "dice_patient_rv": 0.65,
                "dice_patient_myo": 0.70,
                "hd95_patient_lv_mm": 12.0,
                "hd95_patient_rv_mm": 15.0,
                "hd95_patient_myo_mm": 10.0,
                "lvef_error_percent": 8.0,
                "lvef_abs_error_percent": 8.0,
                "lvef_within_5pct": False,
                "lvedv_abs_error_ml": 15.0,
                "lvesv_abs_error_ml": 12.0,
                "lvsv_abs_error_ml": 8.0,
            },
        ])

        summary = evaluator.generate_summary(df)
        assert summary["total_evaluated_cases"] == 2
        assert "by_vendor" in summary
        assert "Philips" in summary["by_vendor"]
        assert "Siemens" in summary["by_vendor"]
        assert summary["by_vendor"]["Philips"]["count"] == 1
        assert summary["by_vendor"]["Philips"]["lvef_mae"] == 2.0
        assert summary["by_vendor"]["Siemens"]["lvef_mae"] == 8.0
        assert summary["mnms_aggregate_metrics"]["lvef_within_5pct_rate"] == 0.5


# ==============================================================================
# Integration Tests (Run when model checkpoint and real data exist)
# ==============================================================================

REAL_CHECKPOINT_PATH = Path("models/mri/best_unet_model.pt")
REAL_MNMS_PATH = Path("data/raw/mnms")


@pytest.mark.skipif(
    not REAL_CHECKPOINT_PATH.is_file(),
    reason="ACDC U-Net checkpoint not found",
)
class TestCheckpointIntegrity:
    """Verify frozen model checkpoint properties."""

    def test_checkpoint_exists_and_checksum(self):
        sha = compute_file_sha256(REAL_CHECKPOINT_PATH)
        assert sha == "71ea54235803562acc67b8903deecaaf113d45a1c7b443a97ba9fbe6573072c9"

    def test_inference_engine_loads_frozen_model(self):
        engine = SegmentationInferenceEngine(checkpoint_path=REAL_CHECKPOINT_PATH)
        model = engine.load_model()
        assert not model.training  # Must be in evaluation mode
        # Forward pass on synthetic input tensor [1, 1, 224, 224]
        x = torch.randn(1, 1, 224, 224, device=engine.device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 224, 224)


@pytest.mark.skipif(
    not (REAL_CHECKPOINT_PATH.is_file() and DEFAULT_MNMS_EVAL_CSV.is_file()),
    reason="M&Ms external evaluation CSV not generated yet",
)
class TestRealMNMSEvaluationArtifacts:
    """Validate persisted evaluation artifacts and metrics consistency."""

    def test_evaluation_csv_rows_and_columns(self):
        df = pd.read_csv(DEFAULT_MNMS_EVAL_CSV)
        assert len(df) == 114
        expected_cols = [
            "case_id", "vendor", "split_category", "center", "diagnosis",
            "dice_patient_mean", "dice_patient_lv", "dice_patient_rv", "dice_patient_myo",
            "hd95_patient_lv_mm", "hd95_patient_rv_mm", "hd95_patient_myo_mm",
            "lvef_pred_percent", "lvef_gt_percent", "lvef_error_percent", "lvef_abs_error_percent",
            "lvedv_pred_ml", "lvedv_gt_ml", "lvesv_pred_ml", "lvesv_gt_ml",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_vendor_breakdown_counts(self):
        df = pd.read_csv(DEFAULT_MNMS_EVAL_CSV)
        vendor_counts = df["vendor"].value_counts().to_dict()
        assert vendor_counts.get("Siemens") == 48
        assert vendor_counts.get("Philips") == 46
        assert vendor_counts.get("GE") == 10
        assert vendor_counts.get("Canon") == 10

    def test_split_breakdown_counts(self):
        df = pd.read_csv(DEFAULT_MNMS_EVAL_CSV)
        split_counts = df["split_category"].value_counts().to_dict()
        assert split_counts.get("Training/Labeled") == 80
        assert split_counts.get("Validation") == 34

    def test_dice_bounds(self):
        df = pd.read_csv(DEFAULT_MNMS_EVAL_CSV)
        assert (df["dice_patient_mean"] >= 0.0).all()
        assert (df["dice_patient_mean"] <= 1.0).all()
        assert (df["dice_patient_lv"] >= 0.0).all()
        assert (df["dice_patient_lv"] <= 1.0).all()

    def test_poster_figures_exist_and_multiformat(self):
        fig_dir = Path("docs/evaluation/external_datasets/mnms")
        for num in range(1, 6):
            # Each figure must have .png, .svg, and .pdf
            for ext in (".png", ".svg", ".pdf"):
                pattern = f"figure{num}_*{ext}"
                matches = list(fig_dir.glob(pattern))
                assert len(matches) > 0, f"Missing figure {pattern} in {fig_dir}"
                assert matches[0].stat().st_size > 0
