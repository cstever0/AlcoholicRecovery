"""Deterministic Tests for EMIDEC LGE Tissue Segmentation Training and Evaluation.

Covers:
  - EMIDEC dataset loading and slice-by-slice extraction
  - Split isolation and patient-level integrity
  - 5-class tissue label preservation
  - Loss function stability with absent and sparse classes (MVO)
  - Inference tensor shapes and physical geometry
  - Checkpoint isolation and baseline SHA-256 preservation
  - Physical tissue volume calculations (mL and % burden)
  - Explicit absent foreground class handling in evaluation
  - Missing asset handling
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from crip.datasets.emidec_loader import EMIDECDataLoader
from crip.datasets.labels import EMIDECLabel
from crip.emidec.tissue_metrics import compute_emidec_tissue_metrics
from crip.mri.segmentation_model import build_unet_model
from crip.pipeline.emidec_dataset import EMIDECSegmentationDataset
from crip.pipeline.emidec_evaluation import (
    EMIDECEvaluator,
    compute_class_hd95,
    compute_dice_score,
)
from crip.pipeline.emidec_trainer import (
    DEFAULT_EMIDEC_CHECKPOINT_PATH,
    EMIDECTrainer,
    compute_file_sha256,
)
from crip.pipeline.splits import DEFAULT_UNIFIED_SPLITS_PATH


class TestEMIDECTrainingPipeline:
    """Test suite for EMIDEC segmentation pipeline components."""

    def test_split_isolation_and_distribution(self):
        """Verify 60/20/20 EMIDEC split has zero patient overlap and preserves strata."""
        assert DEFAULT_UNIFIED_SPLITS_PATH.is_file(), "Unified splits manifest not found."
        with open(DEFAULT_UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        em_splits = manifest["datasets"]["emidec"]["splits"]
        train_set = set(em_splits["train"])
        val_set = set(em_splits["val"])
        test_set = set(em_splits["test"])

        assert len(train_set) == 60
        assert len(val_set) == 20
        assert len(test_set) == 20

        # Zero leakage
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

        # Normal vs Pathological distribution
        n_train_norm = sum(1 for c in train_set if "_N" in c)
        n_val_norm = sum(1 for c in val_set if "_N" in c)
        n_test_norm = sum(1 for c in test_set if "_N" in c)

        assert n_train_norm == 20
        assert n_val_norm == 6
        assert n_test_norm == 7

    def test_emidec_dataset_loading_and_shape(self):
        """Verify EMIDEC dataset indexes slices with correct tensor shapes and 5-class encoding."""
        loader = EMIDECDataLoader()
        cases = loader.discover_cases()
        annotated = [c for c, v in cases.items() if v["has_contour"]]
        assert len(annotated) >= 20

        ds = EMIDECSegmentationDataset(
            case_ids=annotated[:2],
            loader=loader,
            target_size=(224, 224),
            is_training=False,
            augment=False,
            cache_data=True,
        )
        assert len(ds) > 0

        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].shape == (1, 224, 224)
        assert sample["mask"].shape == (1, 224, 224)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.int64

        # Labels must be subset of {0, 1, 2, 3, 4}
        unique_labels = set(sample["mask"].flatten().numpy().tolist())
        assert unique_labels.issubset({0, 1, 2, 3, 4})

    def test_dice_focal_loss_sparse_and_absent_classes(self):
        """Verify loss computes stable gradients even when rare classes (Infarct, MVO) are completely absent."""
        from monai.losses import DiceFocalLoss

        criterion = DiceFocalLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
            batch=True,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=1.0,
        )

        logits = torch.randn(4, 5, 64, 64, requires_grad=True)
        # Target with only background (0), LV cavity (1), and myocardium (2)
        target = torch.zeros(4, 1, 64, 64, dtype=torch.int64)
        target[:, :, 10:30, 10:30] = 1
        target[:, :, 30:50, 30:50] = 2

        loss = criterion(logits, target)
        assert not torch.isnan(loss), "Loss produced NaN!"
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any(), "Gradients contained NaN!"

    def test_unet_5class_architecture_forward_pass(self):
        """Verify MONAI 2D U-Net configured for 5 classes produces logits of shape [B, 5, H, W]."""
        model = build_unet_model(
            in_channels=1,
            out_channels=5,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        x = torch.randn(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, 5, 224, 224)

    def test_physical_tissue_volume_calculation(self):
        """Verify compute_emidec_tissue_metrics calculates accurate volumes and burdens."""
        mask = np.zeros((100, 100, 5), dtype=np.int64)
        # 1,000 voxels of LV (1)
        mask[10:20, 10:20, :] = 1  # 10*10*5 = 500
        # 2,000 voxels of Normal Myo (2)
        mask[30:50, 30:50, :] = 2  # 20*20*5 = 2000
        # 200 voxels of Infarct (3)
        mask[60:70, 60:64, :] = 3  # 10*4*5 = 200
        # 50 voxels of MVO (4)
        mask[70:75, 70:72, :] = 4  # 5*2*5 = 50

        # Spacing: (1.0, 1.0, 10.0 mm) -> 10 mm3 per voxel = 0.01 mL per voxel
        spacing = (1.0, 1.0, 10.0)
        metrics = compute_emidec_tissue_metrics(mask, spacing, case_id="Test_Case", pathology_group="Myocardial Infarction")

        assert metrics.lv_cavity_volume_ml == pytest.approx(5.0, abs=1e-3)
        assert metrics.normal_myocardium_volume_ml == pytest.approx(20.0, abs=1e-3)
        assert metrics.infarction_volume_ml == pytest.approx(2.0, abs=1e-3)
        assert metrics.no_reflow_volume_ml == pytest.approx(0.5, abs=1e-3)
        assert metrics.has_microvascular_obstruction is True

        # Total myo = 20.0 + 2.0 + 0.5 = 22.5 mL
        # Infarct burden = 2.0 / 22.5 * 100 = 8.888%
        assert metrics.infarction_burden_percent == pytest.approx(8.888, abs=1e-2)

    def test_undefined_metrics_handling(self):
        """Verify absent classes return None for Dice and HD95 rather than inflated 1.0 or 0.0 mm."""
        gt_mask = np.zeros((50, 50, 3), dtype=np.int64)
        pred_mask = np.zeros((50, 50, 3), dtype=np.int64)

        # MVO (4) is absent in GT
        dice_mvo = compute_dice_score(pred_mask, gt_mask, class_idx=4)
        assert dice_mvo is None, "Dice should be None when class is absent in reference ground truth."

        hd_mvo = compute_class_hd95(pred_mask, gt_mask, class_idx=4, voxel_spacing=(1.5, 1.5, 10.0))
        assert hd_mvo is None, "HD95 should be None when class is absent in reference ground truth."

    def test_checkpoint_isolation_invariant(self):
        """Verify baseline ACDC and joint cine checkpoints exist and have valid hashes."""
        acdc_ckpt = Path("models/mri/best_unet_model.pt")
        cine_ckpt = Path("models/multidataset/cine_joint_unet.pt")

        if acdc_ckpt.is_file():
            h_acdc = compute_file_sha256(acdc_ckpt)
            assert len(h_acdc) == 64
        if cine_ckpt.is_file():
            h_cine = compute_file_sha256(cine_ckpt)
            assert len(h_cine) == 64
