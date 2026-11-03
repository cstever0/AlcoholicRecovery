"""Joint Multi-Dataset Cine Segmentation Training Engine.

Trains a unified 2D U-Net on ACDC and M&Ms cine CMR datasets:
- Canonical 4-class segmentation: 0=BG, 1=RV, 2=MYO, 3=LV
- Balanced dataset-aware batch sampling
- Multi-device execution (Apple Silicon MPS, NVIDIA CUDA, CPU fallback)
- Independent validation metrics tracking (ACDC Val, M&Ms Val, Combined)
- Strict checkpoint isolation: saves to models/multidataset/cine_joint_unet.pt
  preserving models/mri/best_unet_model.pt inviolate.
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
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from crip.mri.segmentation_model import (
    CompoundDiceCELoss,
    DEFAULT_CHECKPOINT_PATH as ORIGINAL_ACDC_CHECKPOINT,
    build_unet_model,
    get_optimal_device,
)
from crip.pipeline.joint_dataset import (
    BalancedDatasetBatchSampler,
    JointCineSegmentationDataset,
)
from crip.pipeline.models_config import CineSegmentationConfig
from crip.pipeline.splits import DEFAULT_UNIFIED_SPLITS_PATH

logger = logging.getLogger("crip.pipeline.joint_trainer")

DEFAULT_JOINT_CHECKPOINT_PATH = Path("models/multidataset/cine_joint_unet.pt")
DEFAULT_JOINT_HISTORY_PATH = Path("models/multidataset/cine_joint_unet_history.json")
DEFAULT_JOINT_METADATA_PATH = Path("models/multidataset/cine_joint_unet_metadata.json")


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class JointCineTrainer:
    """Orchestrates joint cine CMR segmentation training across ACDC and M&Ms."""

    def __init__(
        self,
        config: Optional[CineSegmentationConfig] = None,
        splits_path: Path = DEFAULT_UNIFIED_SPLITS_PATH,
        checkpoint_path: Path = DEFAULT_JOINT_CHECKPOINT_PATH,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or CineSegmentationConfig()
        self.splits_path = Path(splits_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or get_optimal_device()

        # Seed everything for determinism
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Baseline checkpoint SHA preservation check
        self.original_acdc_sha: Optional[str] = None
        if ORIGINAL_ACDC_CHECKPOINT.is_file():
            self.original_acdc_sha = compute_file_sha256(ORIGINAL_ACDC_CHECKPOINT)

    def load_cohort_splits(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Load ACDC and M&Ms train/val splits from unified manifest."""
        if not self.splits_path.is_file():
            from crip.pipeline.splits import build_unified_cohort_manifest
            build_unified_cohort_manifest(output_path=self.splits_path)

        with open(self.splits_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        acdc_train = manifest["datasets"]["acdc"]["splits"]["train"]
        acdc_val = manifest["datasets"]["acdc"]["splits"]["val"]
        mnms_train = manifest["datasets"]["mnms"]["splits"]["train"]
        mnms_val = manifest["datasets"]["mnms"]["splits"]["val"]

        logger.info(
            f"Loaded splits: ACDC (train={len(acdc_train)}, val={len(acdc_val)}), "
            f"M&Ms (train={len(mnms_train)}, val={len(mnms_val)})"
        )
        return acdc_train, acdc_val, mnms_train, mnms_val

    def train_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        """Run single training epoch."""
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            batch_size = images.size(0)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks.squeeze(1))
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
        """Evaluate validation metrics separated by dataset (ACDC and M&Ms)."""
        model.eval()
        running_loss = 0.0
        total_samples = 0

        dice_metric_acdc = DiceMetric(include_background=False, reduction="mean_batch")
        dice_metric_mnms = DiceMetric(include_background=False, reduction="mean_batch")

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            datasets = batch["dataset"]
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, masks.squeeze(1))
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            pred_labels = torch.argmax(logits, dim=1, keepdim=True)
            pred_onehot = torch.zeros(batch_size, 4, images.size(2), images.size(3), device=self.device)
            pred_onehot.scatter_(1, pred_labels, 1.0)
            true_onehot = torch.zeros(batch_size, 4, images.size(2), images.size(3), device=self.device)
            true_onehot.scatter_(1, masks, 1.0)

            # Separate by dataset
            acdc_mask = [d == "acdc" for d in datasets]
            mnms_mask = [d == "mnms" for d in datasets]

            if any(acdc_mask):
                idx = torch.tensor([i for i, m in enumerate(acdc_mask) if m], device=self.device)
                dice_metric_acdc(y_pred=pred_onehot[idx], y=true_onehot[idx])
            if any(mnms_mask):
                idx = torch.tensor([i for i, m in enumerate(mnms_mask) if m], device=self.device)
                dice_metric_mnms(y_pred=pred_onehot[idx], y=true_onehot[idx])

        # Compute batch-aggregated dice
        try:
            acdc_dices = dice_metric_acdc.aggregate().cpu().numpy()
        except Exception:
            acdc_dices = np.array([0.0, 0.0, 0.0])
        try:
            mnms_dices = dice_metric_mnms.aggregate().cpu().numpy()
        except Exception:
            mnms_dices = np.array([0.0, 0.0, 0.0])

        dice_metric_acdc.reset()
        dice_metric_mnms.reset()

        acdc_rv, acdc_myo, acdc_lv = float(acdc_dices[0]), float(acdc_dices[1]), float(acdc_dices[2])
        mnms_rv, mnms_myo, mnms_lv = float(mnms_dices[0]), float(mnms_dices[1]), float(mnms_dices[2])

        acdc_mean = (acdc_rv + acdc_myo + acdc_lv) / 3.0
        mnms_mean = (mnms_rv + mnms_myo + mnms_lv) / 3.0
        combined_mean = (acdc_mean + mnms_mean) / 2.0

        return {
            "val_loss": running_loss / max(total_samples, 1),
            "acdc_dice_mean": acdc_mean,
            "acdc_dice_lv": acdc_lv,
            "acdc_dice_rv": acdc_rv,
            "acdc_dice_myo": acdc_myo,
            "mnms_dice_mean": mnms_mean,
            "mnms_dice_lv": mnms_lv,
            "mnms_dice_rv": mnms_rv,
            "mnms_dice_myo": mnms_myo,
            "combined_dice_mean": combined_mean,
        }

    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        early_stopping_patience: int = 8,
    ) -> Dict[str, Any]:
        """Execute complete joint training process."""
        num_epochs = epochs or self.config.epochs
        b_size = batch_size or self.config.batch_size

        logger.info(f"Initiating Joint Cine U-Net training on {self.device} for {num_epochs} epochs...")
        start_time = time.time()

        acdc_tr, acdc_va, mnms_tr, mnms_va = self.load_cohort_splits()

        # Build datasets
        train_ds = JointCineSegmentationDataset(
            acdc_patient_ids=acdc_tr,
            mnms_case_ids=mnms_tr,
            target_size=self.config.target_spatial_size,
            augment=True,
            seed=self.config.random_seed,
            cache_data=True,
        )
        val_ds = JointCineSegmentationDataset(
            acdc_patient_ids=acdc_va,
            mnms_case_ids=mnms_va,
            target_size=self.config.target_spatial_size,
            augment=False,
            seed=self.config.random_seed,
            cache_data=True,
        )

        train_sampler = BalancedDatasetBatchSampler(train_ds, batch_size=b_size, seed=self.config.random_seed)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=False, num_workers=0)

        model = build_unet_model().to(self.device)
        criterion = CompoundDiceCELoss(
            dice_weight=self.config.dice_weight,
            ce_weight=self.config.ce_weight,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

        history: List[Dict[str, Any]] = []
        best_dice = -1.0
        best_epoch = -1
        best_metrics: Dict[str, float] = {}
        best_state_dict: Optional[Dict[str, Any]] = None
        no_improve = 0

        for epoch in range(1, num_epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.evaluate_validation(model, val_loader, criterion)
            scheduler.step()
            ep_dur = time.time() - ep_start

            comb_dice = val_metrics["combined_dice_mean"]
            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "duration_seconds": round(ep_dur, 2),
                **val_metrics,
            }
            history.append(rec)

            logger.info(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"ACDC Dice: {val_metrics['acdc_dice_mean']:.3f} | "
                f"M&Ms Dice: {val_metrics['mnms_dice_mean']:.3f} | "
                f"Comb Dice: {comb_dice:.3f} ({ep_dur:.1f}s)"
            )

            if comb_dice > best_dice:
                best_dice = comb_dice
                best_epoch = epoch
                best_metrics = val_metrics
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stopping_patience and epoch >= 15:
                    logger.info(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                    break

        total_training_duration = time.time() - start_time
        logger.info(
            f"Training concluded in {total_training_duration:.1f}s. "
            f"Best Combined Dice: {best_dice:.4f} at epoch {best_epoch}."
        )

        # Save joint model checkpoint
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            "model_state_dict": best_state_dict,
            "epoch": best_epoch,
            "metrics": best_metrics,
            "config": self.config.to_dict(),
            "splits": {
                "acdc_train_count": len(acdc_tr),
                "acdc_val_count": len(acdc_va),
                "mnms_train_count": len(mnms_tr),
                "mnms_val_count": len(mnms_va),
            },
            "training_duration_seconds": round(total_training_duration, 2),
            "software_versions": {
                "torch": torch.__version__,
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
        }
        torch.save(checkpoint_data, self.checkpoint_path)
        logger.info(f"Saved new joint checkpoint to {self.checkpoint_path}")

        # Verify baseline ACDC checkpoint remained untouched
        if self.original_acdc_sha:
            current_sha = compute_file_sha256(ORIGINAL_ACDC_CHECKPOINT)
            assert current_sha == self.original_acdc_sha, "FATAL: Baseline ACDC checkpoint was altered!"
            logger.info("Verified: Original ACDC baseline checkpoint remains 100% inviolate.")

        # Save training history and metadata JSON
        DEFAULT_JOINT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_JOINT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        checkpoint_sha = compute_file_sha256(self.checkpoint_path)
        metadata = {
            "model_identity": "CRIP Joint Cine 2D U-Net (ACDC + M&Ms)",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "best_epoch": best_epoch,
            "total_epochs_trained": len(history),
            "training_duration_seconds": round(total_training_duration, 2),
            "hardware_device": str(self.device),
            "best_validation_metrics": best_metrics,
            "cohort_sizes": {
                "acdc_train_patients": len(acdc_tr),
                "acdc_val_patients": len(acdc_va),
                "mnms_train_studies": len(mnms_tr),
                "mnms_val_studies": len(mnms_va),
                "total_training_patients": len(acdc_tr) + len(mnms_tr),
                "total_validation_patients": len(acdc_va) + len(mnms_va),
            },
            "original_acdc_checkpoint_preserved": True,
            "original_acdc_checkpoint_sha256": self.original_acdc_sha,
        }
        with open(DEFAULT_JOINT_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata
