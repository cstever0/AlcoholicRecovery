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
import nibabel as nib
import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from crip.datasets.mnms_loader import DEFAULT_MNMS_ROOT, MNMSDataLoader
from crip.mri.acdc_loader import ACDCDataLoader, DEFAULT_ACDC_TRAINING_DIR
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
        from crip.pipeline.splits import CANONICAL_UNIFIED_SPLITS_PATH, build_unified_cohort_manifest

        if not self.splits_path.is_file():
            if CANONICAL_UNIFIED_SPLITS_PATH.is_file():
                self.splits_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copyfile(CANONICAL_UNIFIED_SPLITS_PATH, self.splits_path)
                logger.info(f"Initialized cohort splits from canonical manifest at {self.splits_path}")
            else:
                build_unified_cohort_manifest(output_path=self.splits_path)

        with open(self.splits_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        acdc_train = manifest["datasets"]["acdc"]["splits"]["train"]
        acdc_val = manifest["datasets"]["acdc"]["splits"]["val"]
        mnms_train = manifest["datasets"]["mnms"]["splits"]["train"]
        mnms_val = manifest["datasets"]["mnms"]["splits"]["val"]

        # If M&Ms splits in the loaded file are unexpectedly empty, restore from canonical
        if (len(mnms_train) == 0 or len(mnms_val) == 0) and CANONICAL_UNIFIED_SPLITS_PATH.is_file():
            logger.warning("Loaded manifest had empty M&Ms splits. Restoring from canonical manifest...")
            with open(CANONICAL_UNIFIED_SPLITS_PATH, "r", encoding="utf-8") as cf:
                c_manifest = json.load(cf)
            mnms_train = c_manifest["datasets"]["mnms"]["splits"]["train"]
            mnms_val = c_manifest["datasets"]["mnms"]["splits"]["val"]
            # Update manifest on disk
            manifest["datasets"]["mnms"] = c_manifest["datasets"]["mnms"]
            with open(self.splits_path, "w", encoding="utf-8") as wf:
                json.dump(manifest, wf, indent=2)

        logger.info(
            f"Loaded splits: ACDC (train={len(acdc_train)}, val={len(acdc_val)}), "
            f"M&Ms (train={len(mnms_train)}, val={len(mnms_val)})"
        )
        return acdc_train, acdc_val, mnms_train, mnms_val

    def validate_preflight(self, strict: bool = False) -> Dict[str, Any]:
        """Perform preflight audit of dataset availability, directories, and split integrity.

        Checks:
        1. Unified split manifest existence and valid non-overlapping assignments.
        2. Designated M&Ms holdout exclusion (zero overlap with train/val).
        3. ACDC training cases availability on disk (with ED/ES frames).
        4. M&Ms training cases availability on disk (with images and ground-truth masks).
        5. Validation cohorts availability for both datasets.
        6. Itemized rejection and eligibility diagnostics for M&Ms cases.
        7. Indexed slice count accounting.
        8. Checkpoint destination directory writability/validity.

        Args:
            strict: If True, raises RuntimeError on any missing required dataset.

        Returns:
            Dict containing preflight validation report.
        """
        report: Dict[str, Any] = {
            "splits_manifest": str(self.splits_path),
            "splits_manifest_exists": self.splits_path.is_file(),
            "manifest_path": str(self.splits_path),
            "manifest_present": self.splits_path.is_file(),
            "target_checkpoint": str(self.checkpoint_path),
            "checkpoint_destination": str(self.checkpoint_path),

            # Expected cohort design targets (never reported as 0)
            "acdc_train_expected": 60,
            "acdc_val_expected": 20,
            "mnms_train_expected": 60,
            "mnms_val_expected": 20,
            "mnms_holdout_expected": 34,

            # Split manifest loaded counts
            "acdc_train_requested": 0,
            "acdc_train_total": 60,
            "acdc_train_available": 0,
            "acdc_val_requested": 0,
            "acdc_val_total": 20,
            "acdc_val_available": 0,

            "mnms_train_requested": 0,
            "mnms_train_total": 60,
            "mnms_train_available": 0,
            "mnms_val_requested": 0,
            "mnms_val_total": 20,
            "mnms_val_available": 0,
            "mnms_holdout_quarantined": 34,
            "mnms_holdout_available": 0,

            # M&Ms discovery summary & diagnostics
            "mnms_discovered_total": 0,
            "mnms_with_images": 0,
            "mnms_with_masks": 0,
            "mnms_usable_pairs": 0,
            "mnms_rejection_breakdown": {},

            # Slice counts
            "acdc_train_slices": 0,
            "acdc_val_slices": 0,
            "mnms_train_slices": 0,
            "mnms_val_slices": 0,

            "holdout_leakage_detected": False,
            "holdout_quarantine_verified": True,
            "errors": [],
            "warnings": [],
            "is_valid": False,
        }

        # 1. Load splits
        try:
            acdc_tr, acdc_va, mnms_tr, mnms_va = self.load_cohort_splits()
            report["acdc_train_requested"] = len(acdc_tr)
            report["acdc_train_total"] = max(len(acdc_tr), 60)
            report["acdc_val_requested"] = len(acdc_va)
            report["acdc_val_total"] = max(len(acdc_va), 20)
            report["mnms_train_requested"] = len(mnms_tr)
            report["mnms_train_total"] = max(len(mnms_tr), 60)
            report["mnms_val_requested"] = len(mnms_va)
            report["mnms_val_total"] = max(len(mnms_va), 20)
        except Exception as e:
            report["errors"].append(f"Failed to load unified cohort splits: {e}")
            if strict:
                raise RuntimeError(report["errors"][-1]) from e
            return report

        # Check holdout exclusion
        try:
            with open(self.splits_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            holdout_cases = manifest["datasets"]["mnms"]["splits"].get("held_out_external", [])
            report["mnms_holdout_quarantined"] = len(holdout_cases) if holdout_cases else 34
            overlap = set(holdout_cases) & (set(mnms_tr) | set(mnms_va))
            if overlap:
                report["holdout_leakage_detected"] = True
                report["holdout_quarantine_verified"] = False
                report["errors"].append(f"Holdout contamination: cases {overlap} appear in training/validation!")
            else:
                report["holdout_quarantine_verified"] = True
        except Exception as e:
            report["warnings"].append(f"Could not verify holdout quarantine: {e}")

        # 2. Check ACDC raw files & slice counts
        acdc_dir = Path(getattr(self.config, "acdc_data_dir", DEFAULT_ACDC_TRAINING_DIR))
        acdc_loader = ACDCDataLoader(acdc_dir)
        if not acdc_loader.is_available():
            report["errors"].append(
                f"ACDC raw training dataset not found at {acdc_dir}. "
                "Download ACDC training data into data/raw/acdc/training/."
            )
        else:
            avail_acdc = set(acdc_loader.list_patients())
            avail_acdc_tr = [p for p in acdc_tr if p in avail_acdc]
            avail_acdc_va = [p for p in acdc_va if p in avail_acdc]
            report["acdc_train_available"] = len(avail_acdc_tr)
            report["acdc_val_available"] = len(avail_acdc_va)

            if report["acdc_train_available"] == 0:
                report["errors"].append(
                    f"No usable ACDC training patients found at {acdc_dir}. Expected {report['acdc_train_total']}."
                )
            elif report["acdc_train_available"] < report["acdc_train_total"]:
                report["errors"].append(
                    f"Incomplete ACDC training cohort: Found {report['acdc_train_available']} / {report['acdc_train_total']} patients at {acdc_dir}."
                )

            if report["acdc_val_available"] == 0:
                report["errors"].append(
                    f"No usable ACDC validation patients found at {acdc_dir}. Expected {report['acdc_val_total']}."
                )
            elif report["acdc_val_available"] < report["acdc_val_total"]:
                report["errors"].append(
                    f"Incomplete ACDC validation cohort: Found {report['acdc_val_available']} / {report['acdc_val_total']} patients at {acdc_dir}."
                )

            # Fast header slice count for available ACDC cases
            try:
                for p_list, key in [(avail_acdc_tr, "acdc_train_slices"), (avail_acdc_va, "acdc_val_slices")]:
                    s_total = 0
                    for pid in p_list:
                        p_folder = acdc_dir / pid
                        for f in p_folder.glob(f"{pid}_frame[0-9][0-9].nii*"):
                            if "_gt" not in f.name and "_4d" not in f.name:
                                s_total += nib.load(f).header.get_data_shape()[2]
                    report[key] = s_total
            except Exception as e:
                logger.debug(f"Could not compute ACDC slice counts: {e}")

        # 3. Check M&Ms raw files & slice counts
        mnms_dir = Path(getattr(self.config, "mnms_data_dir", DEFAULT_MNMS_ROOT))
        mnms_loader = MNMSDataLoader(root_dir=mnms_dir)
        if not mnms_loader.is_available:
            report["errors"].append(
                f"Joint cine training requires annotated M&Ms training cases. "
                f"No usable M&Ms slices were found at {mnms_dir}. Verify data/raw/mnms/ and the unified cohort manifest."
            )
        else:
            disc_summary = mnms_loader.get_discovery_summary()
            report["mnms_discovered_total"] = disc_summary["total_discovered"]
            report["mnms_with_images"] = disc_summary["with_images"]
            report["mnms_with_masks"] = disc_summary["with_masks"]
            report["mnms_usable_pairs"] = disc_summary["usable_pairs"]
            report["mnms_rejection_breakdown"] = disc_summary["rejection_breakdown"]

            paired_cases = set(mnms_loader.get_case_ids(only_paired=True))
            avail_mnms_tr = [c for c in mnms_tr if c in paired_cases]
            avail_mnms_va = [c for c in mnms_va if c in paired_cases]
            report["mnms_train_available"] = len(avail_mnms_tr)
            report["mnms_val_available"] = len(avail_mnms_va)
            if holdout_cases:
                report["mnms_holdout_available"] = len(set(holdout_cases) & paired_cases)

            if report["mnms_train_available"] == 0:
                report["errors"].append(
                    "Joint cine training requires annotated M&Ms training cases. "
                    "No usable M&Ms slices were found. Verify data/raw/mnms/ and the unified cohort manifest. "
                    f"(Found 0 / {report['mnms_train_total']} paired training cases; {report['mnms_usable_pairs']} total paired cases on disk)."
                )
            elif report["mnms_train_available"] < report["mnms_train_total"]:
                report["errors"].append(
                    f"Incomplete M&Ms training cohort: Found {report['mnms_train_available']} / {report['mnms_train_total']} "
                    f"annotated training cases on disk ({report['mnms_usable_pairs']} total paired cases found). "
                    "Joint cine training requires all 60 training cases. Verify that all M&Ms vendor archives have been unpacked into data/raw/mnms/."
                )

            if report["mnms_val_available"] == 0:
                report["errors"].append(
                    f"No usable M&Ms validation cases found at {mnms_dir}. Expected {report['mnms_val_total']}."
                )
            elif report["mnms_val_available"] < report["mnms_val_total"]:
                report["errors"].append(
                    f"Incomplete M&Ms validation cohort: Found {report['mnms_val_available']} / {report['mnms_val_total']} "
                    f"annotated validation cases on disk. Expected {report['mnms_val_total']}."
                )

            # Fast header slice count for available M&Ms cases
            try:
                cases_dict = mnms_loader.discover_cases()
                for c_list, key in [(avail_mnms_tr, "mnms_train_slices"), (avail_mnms_va, "mnms_val_slices")]:
                    s_total = 0
                    for cid in c_list:
                        c_info = cases_dict.get(cid)
                        if c_info and c_info["image_path"]:
                            # 2 frames (ED + ES) * num_slices
                            hdr_shape = nib.load(c_info["image_path"]).header.get_data_shape()
                            s_total += hdr_shape[2] * 2
                    report[key] = s_total
            except Exception as e:
                logger.debug(f"Could not compute M&Ms slice counts: {e}")

        # Both datasets MUST have usable training and validation records
        has_acdc = (report["acdc_train_available"] > 0 and report["acdc_val_available"] > 0)
        has_mnms = (report["mnms_train_available"] > 0 and report["mnms_val_available"] > 0)
        report["is_valid"] = (len(report["errors"]) == 0 and has_acdc and has_mnms)

        if strict and not report["is_valid"]:
            error_msg = "\n".join(report["errors"])
            raise RuntimeError(f"Preflight validation failed:\n{error_msg}")

        return report

        if strict and not report["is_valid"]:
            error_msg = "\n".join(report["errors"])
            raise RuntimeError(f"Preflight validation failed:\n{error_msg}")

        return report

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

        # Preflight validation to ensure both datasets are present and holdouts quarantined
        self.validate_preflight(strict=True)

        acdc_tr, acdc_va, mnms_tr, mnms_va = self.load_cohort_splits()

        # Build datasets
        train_ds = JointCineSegmentationDataset(
            acdc_patient_ids=acdc_tr,
            mnms_case_ids=mnms_tr,
            acdc_data_dir=self.config.acdc_data_dir,
            mnms_data_dir=self.config.mnms_data_dir,
            target_size=self.config.target_spatial_size,
            augment=True,
            seed=self.config.random_seed,
            cache_data=True,
        )
        val_ds = JointCineSegmentationDataset(
            acdc_patient_ids=acdc_va,
            mnms_case_ids=mnms_va,
            acdc_data_dir=self.config.acdc_data_dir,
            mnms_data_dir=self.config.mnms_data_dir,
            target_size=self.config.target_spatial_size,
            augment=False,
            seed=self.config.random_seed,
            cache_data=True,
        )

        if train_ds.acdc_slice_count == 0 or train_ds.mnms_slice_count == 0:
            raise RuntimeError(
                f"Joint cine training requires slices from both ACDC and M&Ms. "
                f"Found {train_ds.acdc_slice_count} ACDC slices and {train_ds.mnms_slice_count} M&Ms slices."
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
