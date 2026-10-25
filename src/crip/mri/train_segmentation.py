"""Automated Cardiac MRI Segmentation Training Pipeline.

Trains a 2D U-Net using MONAI and PyTorch on stratified patient-level ACDC data.
Monitors validation Dice (RV, Myocardium, LV) and saves the best model checkpoint
along with complete training history.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from crip.mri.dataset_splits import get_or_create_splits
from crip.mri.segmentation_dataset import ACDCSegmentationDataset
from crip.mri.segmentation_model import (
    CompoundDiceCELoss,
    DEFAULT_CHECKPOINT_PATH,
    build_unet_model,
    get_optimal_device,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crip.mri.train_segmentation")

DEFAULT_HISTORY_PATH = Path("models/mri/training_history.json")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch across 2D slices."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
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
def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation slices and compute multi-class Dice."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        batch_size = images.size(0)

        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Convert logits to discrete predictions
        pred_labels = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, H, W]

        # Convert pred and true masks to one-hot for MONAI DiceMetric
        # Classes: 4 (0=BG, 1=RV, 2=MYO, 3=LV)
        pred_onehot = torch.zeros(batch_size, 4, images.size(2), images.size(3), device=device)
        pred_onehot.scatter_(1, pred_labels, 1.0)

        target_expanded = masks.unsqueeze(1)  # [B, 1, H, W]
        target_onehot = torch.zeros(batch_size, 4, images.size(2), images.size(3), device=device)
        target_onehot.scatter_(1, target_expanded, 1.0)

        dice_metric(pred_onehot, target_onehot)

    epoch_loss = running_loss / max(total_samples, 1)

    try:
        class_dices = dice_metric.aggregate().cpu().numpy()
        # class_dices has 3 elements: RV (class 1), MYO (class 2), LV (class 3)
        rv_dice = float(class_dices[0]) if len(class_dices) > 0 else 0.0
        myo_dice = float(class_dices[1]) if len(class_dices) > 1 else 0.0
        lv_dice = float(class_dices[2]) if len(class_dices) > 2 else 0.0
    except Exception as e:
        logger.warning(f"Error aggregating Dice metrics: {e}")
        rv_dice, myo_dice, lv_dice = 0.0, 0.0, 0.0

    mean_fg_dice = float((rv_dice + myo_dice + lv_dice) / 3.0)

    metrics = {
        "val_loss": float(epoch_loss),
        "rv_dice": rv_dice,
        "myo_dice": myo_dice,
        "lv_dice": lv_dice,
        "mean_dice": mean_fg_dice,
    }
    return epoch_loss, metrics


def run_training(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device_name: Optional[str] = None,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
    smoke_test: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete training and validation cycle."""
    # Deterministic seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection
    if device_name and device_name.lower() != "auto":
        device = torch.device(device_name)
    else:
        device = get_optimal_device()

    logger.info(f"Initiating segmentation training pipeline on device: {device}")

    # Load reproducible patient splits
    splits_manifest = get_or_create_splits(seed=seed)
    train_pids = splits_manifest["splits"]["train"]
    val_pids = splits_manifest["splits"]["val"]

    if smoke_test:
        logger.info("Executing SMOKE TEST run on minimal patient cohort (2 train, 1 val, 1 epoch)...")
        train_pids = train_pids[:2]
        val_pids = val_pids[:1]
        epochs = 1

    logger.info(f"Training cohort: {len(train_pids)} patients | Validation cohort: {len(val_pids)} patients")

    train_ds = ACDCSegmentationDataset(train_pids, augment=True, seed=seed, cache_data=True)
    val_ds = ACDCSegmentationDataset(val_pids, augment=False, seed=seed, cache_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_unet_model().to(device)
    criterion = CompoundDiceCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_mean_dice = -1.0
    history: List[Dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        mean_dice = val_metrics["mean_dice"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "rv_dice": float(val_metrics["rv_dice"]),
            "myo_dice": float(val_metrics["myo_dice"]),
            "lv_dice": float(val_metrics["lv_dice"]),
            "mean_dice": float(mean_dice),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "duration_sec": round(elapsed, 2),
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Dice -> LV: {val_metrics['lv_dice']:.3f}, RV: {val_metrics['rv_dice']:.3f}, "
            f"MYO: {val_metrics['myo_dice']:.3f} (Mean: {mean_dice:.3f})"
        )

        # Checkpoint selection: prioritize highest Mean Foreground Dice
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            save_checkpoint(
                model=model,
                epoch=epoch,
                metrics=val_metrics,
                path=checkpoint_path,
                optimizer=optimizer,
            )

    total_training_time = time.time() - start_time
    logger.info(
        f"Training complete in {total_training_time:.1f}s. "
        f"Best Validation Mean Dice: {best_mean_dice:.4f}"
    )

    # Save training history to JSON
    hist_dest = Path(history_path)
    hist_dest.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_dest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": str(device),
                "best_mean_dice": best_mean_dice,
                "total_duration_sec": total_training_time,
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "best_mean_dice": best_mean_dice,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "history": history,
    }


def main() -> None:
    """CLI Entry point for segmentation model training."""
    parser = argparse.ArgumentParser(description="Train 2D U-Net for ACDC Cardiac MRI Segmentation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, mps, cuda, auto)")
    parser.add_argument("--smoke-test", action="store_true", help="Quick verification smoke test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--history-path", type=str, default=str(DEFAULT_HISTORY_PATH))

    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
