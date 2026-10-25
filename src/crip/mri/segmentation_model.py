"""Cardiac MRI 2D U-Net Model Definition and Device Selection.

Defines the multi-class convolutional architecture using MONAI's UNet,
compound Dice + CrossEntropy loss function, and robust checkpoint serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.networks.nets import UNet

logger = logging.getLogger("crip.mri.segmentation_model")

DEFAULT_CHECKPOINT_PATH = Path("models/mri/best_unet_model.pt")


def get_optimal_device() -> torch.device:
    """Detect and return the highest-performance available hardware acceleration device.

    Selection priority:
      1. CUDA (Ubuntu/WSL GPU acceleration)
      2. MPS (macOS Apple Silicon GPU acceleration)
      3. CPU (standard portable fallback)
    """
    if torch.cuda.is_available():
        logger.info("Using NVIDIA CUDA acceleration.")
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            # Smoke test MPS execution to guarantee kernel support
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
            logger.info("Using Apple Silicon MPS acceleration.")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"Apple Silicon MPS detected but initialization failed: {e}. Falling back to CPU.")

    logger.info("Using standard CPU execution.")
    return torch.device("cpu")


def build_unet_model(
    in_channels: int = 1,
    out_channels: int = 4,
    channels: Tuple[int, ...] = (16, 32, 64, 128),
    strides: Tuple[int, ...] = (2, 2, 2),
    num_res_units: int = 2,
    norm: str = "batch",
    dropout: float = 0.1,
) -> nn.Module:
    """Build a 2D MONAI U-Net for multi-class cardiac segmentation.

    Args:
        in_channels: Number of input MRI intensity channels (default: 1).
        out_channels: Number of segmentation classes (default: 4).
        channels: Feature map dimensions across hierarchical levels.
        strides: Striding downsampling factors between levels.
        num_res_units: Number of residual convolution units per level.
        norm: Normalization layer type ('batch' or 'instance').
        dropout: Spatial dropout probability.

    Returns:
        PyTorch nn.Module representing the U-Net.
    """
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )
    return model


class CompoundDiceCELoss(nn.Module):
    """Hybrid loss combining soft Dice loss with multi-class Cross-Entropy.

    Mitigates severe class imbalance between the background and smaller cardiac structures
    (such as thin myocardium and RV apex).
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_fn = DiceLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=include_background,
        )
        self.ce_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate compound loss.

        Args:
            logits: Predicted class logits of shape (B, C, H, W).
            target: Ground truth discrete integer labels of shape (B, H, W).
        """
        # Ensure target is shaped [B, 1, H, W] for MONAI DiceLoss
        target_dice = target.unsqueeze(1) if target.ndim == 3 else target

        loss_dice = self.dice_fn(logits, target_dice)
        loss_ce = self.ce_fn(logits, target)

        total_loss = (self.dice_weight * loss_dice) + (self.ce_weight * loss_ce)
        return total_loss


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: Dict[str, float],
    path: Path | str = DEFAULT_CHECKPOINT_PATH,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Path:
    """Save model checkpoint and training metadata to disk."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, dest)
    logger.info(f"Saved model checkpoint to {dest} (epoch {epoch})")
    return dest


def load_checkpoint(
    model: nn.Module,
    path: Path | str = DEFAULT_CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model weights and metadata from disk."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {src}")

    target_device = device or get_optimal_device()
    checkpoint = torch.load(src, map_location=target_device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from {src} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint
