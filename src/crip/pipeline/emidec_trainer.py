"""EMIDEC Delayed-Enhancement CMR Segmentation Training Engine.

Trains a dedicated 5-class 2D U-Net on EMIDEC LGE imaging:
  - Classes: 0=BG, 1=LV Cavity, 2=Normal Myo, 3=Infarction, 4=MVO/No-Reflow
  - Extreme class imbalance handling via MONAI DiceFocalLoss
  - Patient-level split isolation (60 train / 20 val / 20 held-out test)
  - Strict checkpoint preservation: saves to models/multidataset/emidec_lge_unet.pt
    leaving models/mri/best_unet_model.pt and models/multidataset/cine_joint_unet.pt
    completely inviolate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from crip.mri.segmentation_model import (
    DEFAULT_CHECKPOINT_PATH as ORIGINAL_ACDC_CHECKPOINT,
    build_unet_model,
    get_optimal_device,
)
from crip.pipeline.emidec_dataset import EMIDECSegmentationDataset
from crip.pipeline.models_config import LGESegmentationConfig
from crip.pipeline.splits import DEFAULT_UNIFIED_SPLITS_PATH

logger = logging.getLogger("crip.pipeline.emidec_trainer")

DEFAULT_EMIDEC_CHECKPOINT_PATH = Path("models/multidataset/emidec_lge_unet.pt")
DEFAULT_EMIDEC_HISTORY_PATH = Path("models/multidataset/emidec_lge_unet_history.json")
DEFAULT_EMIDEC_METADATA_PATH = Path("models/multidataset/emidec_lge_unet_metadata.json")
JOINT_CINE_CHECKPOINT = Path("models/multidataset/cine_joint_unet.pt")


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class EMIDECTrainer:
    """Orchestrates LGE CMR 5-class tissue characterization U-Net training."""

    def __init__(
        self,
        config: Optional[LGESegmentationConfig] = None,
        device: Optional[torch.device] = None,
        splits_path: Path | str = DEFAULT_UNIFIED_SPLITS_PATH,
        checkpoint_path: Path | str = DEFAULT_EMIDEC_CHECKPOINT_PATH,
    ) -> None:
        self.config = config or LGESegmentationConfig()
        self.device = device or get_optimal_device()
        self.splits_path = Path(splits_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Baseline hashes for strict preservation audit
        self.original_acdc_sha: Optional[str] = None
        self.joint_cine_sha: Optional[str] = None

        if ORIGINAL_ACDC_CHECKPOINT.is_file():
            self.original_acdc_sha = compute_file_sha256(ORIGINAL_ACDC_CHECKPOINT)
        if JOINT_CINE_CHECKPOINT.is_file():
            self.joint_cine_sha = compute_file_sha256(JOINT_CINE_CHECKPOINT)

    def load_cohort_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """Load EMIDEC train, val, and test splits from unified manifest."""
        if not self.splits_path.is_file():
            from crip.pipeline.splits import build_unified_cohort_manifest
            build_unified_cohort_manifest(output_path=self.splits_path)

        with open(self.splits_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        em_splits = manifest["datasets"]["emidec"]["splits"]
        train_ids = em_splits["train"]
        val_ids = em_splits["val"]
        test_ids = em_splits["test"]

        # Strict isolation check
        s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
        assert len(s_tr & s_va) == 0, "EMIDEC train and val splits overlap!"
        assert len(s_tr & s_te) == 0, "EMIDEC train and test splits overlap!"
        assert len(s_va & s_te) == 0, "EMIDEC val and test splits overlap!"

        logger.info(
            f"Loaded EMIDEC splits: train={len(train_ids)}, val={len(val_ids)}, held_out_test={len(test_ids)}"
        )
        return train_ids, val_ids, test_ids

    def train_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        """Execute single training epoch across all batches."""
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            batch_size = images.size(0)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            total_samples += batch_size

        return running_loss / max(total_samples, 1)

    @torch.no_grad()
    def evaluate_validation(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
    ) -> Dict[str, float]:
        """Evaluate validation metrics across the 4 foreground tissue classes."""
        model.eval()
        running_loss = 0.0
        total_samples = 0

        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        dice_metric.reset()

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # One-hot encode for DiceMetric
            preds = torch.argmax(logits, dim=1)
            pred_oh = F.one_hot(preds, num_classes=5).permute(0, 3, 1, 2)
            target_oh = F.one_hot(masks.squeeze(1), num_classes=5).permute(0, 3, 1, 2)

            dice_metric(pred_oh, target_oh)

        val_loss = running_loss / max(total_samples, 1)
        per_class_dice = dice_metric.aggregate()

        def clean_val(v: torch.Tensor) -> float:
            val = float(v.item())
            return 0.0 if np.isnan(val) else val

        dice_lv = clean_val(per_class_dice[0])
        dice_myo = clean_val(per_class_dice[1])
        dice_infarct = clean_val(per_class_dice[2])
        dice_mvo = clean_val(per_class_dice[3])

        # Combined foreground mean
        dice_mean = float(np.mean([dice_lv, dice_myo, dice_infarct, dice_mvo]))

        return {
            "val_loss": val_loss,
            "dice_mean": dice_mean,
            "dice_lv": dice_lv,
            "dice_myo": dice_myo,
            "dice_infarct": dice_infarct,
            "dice_mvo": dice_mvo,
        }

    def train(
        self,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> Dict[str, Any]:
        """Execute full training workflow for 5-class EMIDEC segmentation."""
        t_start = time.time()
        logger.info(f"Initiating EMIDEC LGE U-Net training on {self.device} for {epochs} epochs...")

        train_ids, val_ids, test_ids = self.load_cohort_splits()

        # Build datasets
        train_ds = EMIDECSegmentationDataset(
            case_ids=train_ids,
            target_size=(224, 224),
            is_training=True,
            augment=True,
            cache_data=True,
        )
        val_ds = EMIDECSegmentationDataset(
            case_ids=val_ids,
            target_size=(224, 224),
            is_training=False,
            augment=False,
            cache_data=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        # Build 5-class U-Net
        model = build_unet_model(
            in_channels=1,
            out_channels=5,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.15,
        ).to(self.device)

        # Foreground class weights: [LV, Myo, Infarct, MVO]
        # Infarct (0.2% voxels) weighted 3.0x, MVO (0.028% voxels) weighted 6.0x
        fg_weights = torch.tensor([1.0, 1.0, 3.0, 6.0], device=self.device)
        criterion = DiceFocalLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
            batch=True,
            gamma=2.0,
            weight=fg_weights,
            lambda_dice=1.0,
            lambda_focal=1.0,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        history: List[Dict[str, Any]] = []
        best_metric = -1.0
        best_epoch = -1
        best_val_metrics: Dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.evaluate_validation(model, val_loader, criterion)
            scheduler.step()
            ep_dur = time.time() - ep_start

            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["val_loss"]),
                "dice_mean": float(val_metrics["dice_mean"]),
                "dice_lv": float(val_metrics["dice_lv"]),
                "dice_myo": float(val_metrics["dice_myo"]),
                "dice_infarct": float(val_metrics["dice_infarct"]),
                "dice_mvo": float(val_metrics["dice_mvo"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "duration_seconds": round(ep_dur, 2),
            }
            history.append(record)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | LV: {val_metrics['dice_lv']:.3f} | "
                f"MYO: {val_metrics['dice_myo']:.3f} | INF: {val_metrics['dice_infarct']:.3f} | "
                f"MVO: {val_metrics['dice_mvo']:.3f} | Comb: {val_metrics['dice_mean']:.3f} ({ep_dur:.1f}s)"
            )

            # Checkpoint selection: track highest combined foreground Dice
            comb_dice = val_metrics["dice_mean"]
            if comb_dice > best_metric or epoch == 1:
                best_metric = comb_dice
                best_epoch = epoch
                best_val_metrics = val_metrics.copy()

                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "config": self.config.to_dict(),
                    "classes": {
                        0: "Background",
                        1: "LV_Cavity",
                        2: "Normal_Myocardium",
                        3: "Myocardial_Infarction",
                        4: "Microvascular_Obstruction",
                    },
                }
                torch.save(checkpoint_data, self.checkpoint_path)

        total_dur = time.time() - t_start
        logger.info(
            f"EMIDEC LGE training concluded in {total_dur:.1f}s. "
            f"Best Combined Dice: {best_metric:.4f} at epoch {best_epoch}."
        )

        # Save training history JSON
        with open(DEFAULT_EMIDEC_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Verify baseline preservation
        acdc_intact = True
        cine_joint_intact = True
        if self.original_acdc_sha and ORIGINAL_ACDC_CHECKPOINT.is_file():
            curr_acdc_sha = compute_file_sha256(ORIGINAL_ACDC_CHECKPOINT)
            acdc_intact = (curr_acdc_sha == self.original_acdc_sha)
        if self.joint_cine_sha and JOINT_CINE_CHECKPOINT.is_file():
            curr_cine_sha = compute_file_sha256(JOINT_CINE_CHECKPOINT)
            cine_joint_intact = (curr_cine_sha == self.joint_cine_sha)

        checkpoint_sha = compute_file_sha256(self.checkpoint_path)
        splits_sha = compute_file_sha256(self.splits_path) if self.splits_path.is_file() else "N/A"

        metadata = {
            "model_identity": "CRIP Dedicated EMIDEC LGE 2D U-Net (5-Class)",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "splits_sha256": splits_sha,
            "best_epoch": best_epoch,
            "total_epochs_trained": epochs,
            "training_duration_seconds": round(total_dur, 2),
            "hardware_device": str(self.device),
            "best_validation_metrics": best_val_metrics,
            "cohort_sizes": {
                "emidec_train_patients": len(train_ids),
                "emidec_val_patients": len(val_ids),
                "emidec_held_out_test_patients": len(test_ids),
                "total_annotated_cases": len(train_ids) + len(val_ids) + len(test_ids),
            },
            "preserved_checkpoints": {
                "original_acdc_checkpoint": str(ORIGINAL_ACDC_CHECKPOINT),
                "original_acdc_intact": acdc_intact,
                "original_acdc_sha256": self.original_acdc_sha,
                "joint_cine_checkpoint": str(JOINT_CINE_CHECKPOINT),
                "joint_cine_intact": cine_joint_intact,
                "joint_cine_sha256": self.joint_cine_sha,
            },
        }

        with open(DEFAULT_EMIDEC_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata
