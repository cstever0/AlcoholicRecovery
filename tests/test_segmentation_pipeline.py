"""Automated Unit and Integration Tests for Milestone 2C Segmentation Pipeline.

Covers:
  - Patient-level stratified dataset splitting
  - Short-axis slice preprocessing and nearest-neighbor label restoration
  - ACDCSegmentationDataset slice extraction and augmentation
  - MONAI 2D U-Net model forward pass, CompoundDiceCELoss, and device selection
  - Checkpoint serialization and roundtrip loading
  - Segmentation inference engine and 3D volume reconstruction
  - Multi-class Dice, HD95 physical distance, and volumetric evaluation metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from crip.evaluation.segmentation_metrics import (
    calculate_dice_coefficient,
    calculate_hd95_mm,
    calculate_multiclass_dice,
    calculate_multiclass_hd95,
    evaluate_patient_segmentation,
)
from crip.mri.cardiac_metrics import BiventricularCardiacMeasurements
from crip.mri.dataset_splits import (
    create_stratified_acdc_splits,
    get_or_create_splits,
    load_dataset_splits,
    save_dataset_splits,
)
from crip.mri.preprocessing import (
    DEFAULT_TARGET_SPATIAL_SIZE,
    normalize_intensity,
    resize_slice,
    restore_mask_to_original_shape,
    restore_volume_mask,
)
from crip.mri.segmentation_dataset import ACDCSegmentationDataset
from crip.mri.segmentation_inference import (
    PatientInferenceResult,
    SegmentationInferenceEngine,
)
from crip.mri.segmentation_model import (
    CompoundDiceCELoss,
    build_unet_model,
    get_optimal_device,
    load_checkpoint,
    save_checkpoint,
)


# ==============================================================================
# 1. Dataset Splitting Tests
# ==============================================================================

class TestDatasetSplitting:
    """Verifies deterministic, patient-level, stratified dataset splitting."""

    def test_split_proportions_and_stratification(self):
        manifest = create_stratified_acdc_splits(seed=42)
        assert manifest["total_patients"] == 100
        assert manifest["counts"]["train"] == 60
        assert manifest["counts"]["val"] == 20
        assert manifest["counts"]["test"] == 20

        # Verify equal representation across all 5 diagnostic groups
        for group in ["DCM", "HCM", "MINF", "NOR", "RV"]:
            assert manifest["group_distribution"]["train"][group] == 12
            assert manifest["group_distribution"]["val"][group] == 4
            assert manifest["group_distribution"]["test"][group] == 4

    def test_mutual_exclusivity_no_patient_overlap(self):
        manifest = create_stratified_acdc_splits(seed=42)
        train_set = set(manifest["splits"]["train"])
        val_set = set(manifest["splits"]["val"])
        test_set = set(manifest["splits"]["test"])

        assert len(train_set) == 60
        assert len(val_set) == 20
        assert len(test_set) == 20

        # No patient leaks between splits
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
        assert len(train_set | val_set | test_set) == 100

    def test_determinism_across_identical_seeds(self):
        split_a = create_stratified_acdc_splits(seed=42)
        split_b = create_stratified_acdc_splits(seed=42)
        assert split_a["splits"] == split_b["splits"]

    def test_save_and_load_manifest(self, tmp_path: Path):
        manifest = create_stratified_acdc_splits(seed=42)
        target_path = tmp_path / "splits.json"
        save_dataset_splits(manifest, target_path)
        loaded = load_dataset_splits(target_path)
        assert loaded["counts"] == manifest["counts"]
        assert loaded["splits"] == manifest["splits"]


# ==============================================================================
# 2. Preprocessing & Spatial Restoration Tests
# ==============================================================================

class TestPreprocessingAndRestoration:
    """Verifies intensity normalization, slice resizing, and exact restoration."""

    def test_normalize_intensity_range(self):
        raw_slice = np.random.uniform(10.0, 500.0, size=(128, 128)).astype(np.float32)
        norm_slice = normalize_intensity(raw_slice)
        assert norm_slice.dtype == np.float32
        assert norm_slice.min() >= 0.0
        assert norm_slice.max() <= 1.0

    def test_normalize_intensity_constant_or_zero(self):
        zero_slice = np.zeros((64, 64), dtype=np.float32)
        norm_zero = normalize_intensity(zero_slice)
        assert np.all(norm_zero == 0.0)

        const_slice = np.ones((64, 64), dtype=np.float32) * 42.0
        norm_const = normalize_intensity(const_slice)
        assert np.all(norm_const == 0.0)

    def test_slice_resize_image_and_mask(self):
        img = np.random.rand(180, 210).astype(np.float32)
        resized_img = resize_slice(img, target_size=(224, 224), is_mask=False)
        assert resized_img.shape == (224, 224)

        # Mask containing all labels {0, 1, 2, 3}
        mask = np.zeros((180, 210), dtype=np.int64)
        mask[20:50, 20:50] = 1
        mask[60:90, 60:90] = 2
        mask[100:130, 100:130] = 3

        resized_mask = resize_slice(mask, target_size=(224, 224), is_mask=True)
        assert resized_mask.shape == (224, 224)
        # Nearest neighbor must strictly preserve integer discrete labels
        unique_labels = set(np.unique(resized_mask).tolist())
        assert unique_labels.issubset({0, 1, 2, 3})
        assert unique_labels == {0, 1, 2, 3}

    def test_restore_mask_to_original_shape(self):
        orig_shape = (216, 256)
        pred_mask_224 = np.zeros((224, 224), dtype=np.int64)
        pred_mask_224[50:100, 50:100] = 3  # LV cavity

        restored = restore_mask_to_original_shape(pred_mask_224, original_shape=orig_shape)
        assert restored.shape == orig_shape
        assert set(np.unique(restored).tolist()) == {0, 3}

    def test_restore_volume_mask(self):
        orig_volume_shape = (10, 200, 240)
        pred_stack = np.zeros((10, 224, 224), dtype=np.int64)
        pred_stack[:, 40:80, 40:80] = 2  # Myocardium

        restored_vol = restore_volume_mask(pred_stack, orig_volume_shape)
        assert restored_vol.shape == orig_volume_shape
        assert set(np.unique(restored_vol).tolist()) == {0, 2}


# ==============================================================================
# 3. ACDCSegmentationDataset Tests
# ==============================================================================

class TestACDCSegmentationDataset:
    """Verifies dataset extraction, caching, and batch tensor shapes."""

    def test_dataset_item_structure(self):
        ds = ACDCSegmentationDataset(
            patient_ids=["patient001"],
            augment=False,
            cache_data=True,
        )
        assert len(ds) > 0
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert "patient_id" in sample
        assert "phase" in sample
        assert "voxel_spacing" in sample

        # Check tensor shapes
        assert sample["image"].shape == (1, 224, 224)
        assert sample["mask"].shape == (224, 224)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.int64

    def test_dataset_augmentation_preserves_valid_labels(self):
        ds = ACDCSegmentationDataset(
            patient_ids=["patient001"],
            augment=True,
            seed=42,
            cache_data=True,
        )
        for i in range(min(5, len(ds))):
            sample = ds[i]
            unique_labels = torch.unique(sample["mask"]).tolist()
            assert all(lbl in {0, 1, 2, 3} for lbl in unique_labels)


# ==============================================================================
# 4. Model Architecture & Loss Tests
# ==============================================================================

class TestSegmentationModel:
    """Verifies U-Net creation, forward pass, hybrid loss, and checkpointing."""

    def test_model_forward_pass_shape(self):
        model = build_unet_model()
        dummy_input = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 4, 224, 224)

    def test_compound_loss_backward(self):
        criterion = CompoundDiceCELoss(dice_weight=0.5, ce_weight=0.5)
        logits = torch.randn(2, 4, 64, 64, requires_grad=True)
        target = torch.randint(0, 4, (2, 64, 64))

        loss = criterion(logits, target)
        assert loss.item() > 0.0
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_get_optimal_device(self):
        device = get_optimal_device()
        assert isinstance(device, torch.device)
        assert device.type in {"cpu", "mps", "cuda"}

    def test_checkpoint_save_and_load(self, tmp_path: Path):
        model = build_unet_model()
        ckpt_path = tmp_path / "test_unet.pt"
        metrics = {"val_loss": 0.35, "mean_dice": 0.82}

        save_checkpoint(model, epoch=5, metrics=metrics, path=ckpt_path)
        assert ckpt_path.is_file()

        model_loaded = build_unet_model()
        ckpt_data = load_checkpoint(model_loaded, path=ckpt_path, device=torch.device("cpu"))
        assert ckpt_data["epoch"] == 5
        assert ckpt_data["metrics"]["mean_dice"] == 0.82

        # Verify weights match
        for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
            assert torch.allclose(p1, p2)


# ==============================================================================
# 5. Evaluation Metrics Tests
# ==============================================================================

class TestEvaluationMetrics:
    """Verifies multi-class Dice, HD95, and patient volumetric error computation."""

    def test_dice_identical_and_disjoint(self):
        m1 = np.ones((10, 10), dtype=np.int64)
        m2 = np.ones((10, 10), dtype=np.int64)
        assert calculate_dice_coefficient(m1, m2) == 1.0

        m3 = np.zeros((10, 10), dtype=np.int64)
        assert calculate_dice_coefficient(m1, m3) == 0.0

        # Both empty
        assert calculate_dice_coefficient(m3, m3) == 1.0

    def test_multiclass_dice(self):
        gt = np.zeros((20, 20), dtype=np.int64)
        gt[2:6, 2:6] = 1  # RV
        gt[7:12, 7:12] = 2  # MYO
        gt[13:18, 13:18] = 3  # LV

        pred = gt.copy()
        dices = calculate_multiclass_dice(pred, gt)
        assert dices["rv"] == 1.0
        assert dices["myo"] == 1.0
        assert dices["lv"] == 1.0
        assert dices["mean"] == 1.0

    def test_hd95_identical_masks(self):
        m = np.zeros((20, 20), dtype=np.int64)
        m[5:15, 5:15] = 3
        hd = calculate_hd95_mm(m, m, voxel_spacing=(1.5, 1.5), label=3)
        assert hd == 0.0

    def test_evaluate_patient_segmentation_5pct_target_flag(self):
        # Construct synthetic PatientInferenceResult
        pred_metrics = BiventricularCardiacMeasurements(
            patient_id="synth001",
            lvedv_ml=120.0,
            lvesv_ml=60.0,
            lvsv_ml=60.0,
            lvef_percent=50.0,
            rvedv_ml=110.0,
            rvesv_ml=55.0,
            rvsv_ml=55.0,
            rvef_percent=50.0,
            myocardial_volume_ed_ml=80.0,
            myocardial_volume_es_ml=85.0,
            myocardial_mass_ed_g=84.0,
            myocardial_mass_es_g=89.25,
            voxel_spacing_mm=(1.4, 1.4, 10.0),
            voxel_volume_mm3=19.6,
        )
        gt_metrics = BiventricularCardiacMeasurements(
            patient_id="synth001",
            lvedv_ml=125.0,
            lvesv_ml=60.0,
            lvsv_ml=65.0,
            lvef_percent=52.0,  # Difference: 2.0% (<= 5.0%)
            rvedv_ml=112.0,
            rvesv_ml=56.0,
            rvsv_ml=56.0,
            rvef_percent=50.0,
            myocardial_volume_ed_ml=82.0,
            myocardial_volume_es_ml=86.0,
            myocardial_mass_ed_g=86.1,
            myocardial_mass_es_g=90.3,
            voxel_spacing_mm=(1.4, 1.4, 10.0),
            voxel_volume_mm3=19.6,
        )

        dummy_mask = np.zeros((5, 32, 32), dtype=np.int64)
        result = PatientInferenceResult(
            patient_id="synth001",
            pred_ed_mask=dummy_mask,
            pred_es_mask=dummy_mask,
            pred_metrics=pred_metrics,
            gt_ed_mask=dummy_mask,
            gt_es_mask=dummy_mask,
            gt_metrics=gt_metrics,
        )

        eval_rec = evaluate_patient_segmentation(result, group="NOR")
        assert eval_rec["within_5pct_target"] is True
        assert eval_rec["lvef_abs_error_pct"] == 2.0


# ==============================================================================
# 6. Inference Engine Tests
# ==============================================================================

class TestSegmentationInferenceEngine:
    """Verifies inference execution, graceful missing checkpoint handling, and slice prediction."""

    def test_missing_checkpoint_raises_filenotfound(self, tmp_path: Path):
        engine = SegmentationInferenceEngine(checkpoint_path=tmp_path / "nonexistent.pt")
        assert engine.is_model_available is False
        with pytest.raises(FileNotFoundError):
            engine.load_model()

    def test_predict_slice_mask_shape_and_labels(self):
        engine = SegmentationInferenceEngine()
        model = build_unet_model().to(engine.device)
        model.eval()

        orig_shape = (192, 230)
        slice_2d = np.random.uniform(20.0, 400.0, size=orig_shape).astype(np.float32)

        pred_mask = engine.predict_slice_mask(slice_2d, model=model)
        assert pred_mask.shape == orig_shape
        unique_labels = set(np.unique(pred_mask).tolist())
        assert unique_labels.issubset({0, 1, 2, 3})

    def test_cohort_evaluation_summary_exists(self):
        from crip.evaluation.segmentation_metrics import DEFAULT_EVAL_SUMMARY_PATH, DEFAULT_EVAL_CSV_PATH
        assert DEFAULT_EVAL_CSV_PATH.is_file()
        assert DEFAULT_EVAL_SUMMARY_PATH.is_file()

        with open(DEFAULT_EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["split_evaluated"] == "test"
        assert summary["n_patients"] == 20
        assert summary["lvef_metrics"]["mean_absolute_error_pct"] > 0.0
        assert summary["geometric_metrics"]["dice_mean_overall"] > 0.70
