"""Multi-Task Model Configurations and Training Abstractions.

Defines reproducible configuration dataclasses, loss specifications, normalization
protocols, and checkpoint metadata contracts for:
1. Cine Ventricular Segmentation (ACDC + M&Ms joint cine)
2. LGE Viability Tissue Segmentation (EMIDEC necrosis & microvascular obstruction)

Architecture Strategy:
- Recommends and specifies Architecture A (modular task-specialized pipelines)
  over Architecture B (forced shared encoder), avoiding negative cross-modal transfer
  between dynamic bSSFP cine and static inversion-recovery LGE.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LossFunctionType(str, Enum):
    """Supported loss functions for multi-task cardiac segmentation."""
    COMPOUND_DICE_CE = "CompoundDiceCELoss"
    FOCAL_TVERSKY = "FocalTverskyLoss"
    WEIGHTED_CROSS_ENTROPY = "WeightedCELoss"
    DICE_FOCAL = "DiceFocalLoss"


class NormalizationMethod(str, Enum):
    """Voxel intensity normalization strategies."""
    PERCENTILE_ZERO_MEAN_UNIT_VAR = "percentile_zero_mean_unit_variance"  # 1st-99th clipping + standard scaling
    MIN_MAX_SCALING = "min_max_scaling"                                  # [0, 1] scaling after clipping


@dataclass
class CineSegmentationConfig:
    """Hyperparameter and architectural specification for Cine Ventricular Segmentation."""
    task_name: str = "CINE_VENTRICULAR_SEGMENTATION"
    architecture: str = "UNet_2D"
    spatial_dimensions: int = 2
    in_channels: int = 1
    out_channels: int = 4  # 0: BG, 1: RV, 2: MYO, 3: LV
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    target_spatial_size: Tuple[int, int] = (224, 224)
    normalization: NormalizationMethod = NormalizationMethod.PERCENTILE_ZERO_MEAN_UNIT_VAR
    loss_type: LossFunctionType = LossFunctionType.COMPOUND_DICE_CE
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 25
    random_seed: int = 42
    target_checkpoint_dir: str = "models/multidataset/"
    supported_datasets: List[str] = field(default_factory=lambda: ["acdc", "mnms"])
    acdc_data_dir: str = "data/raw/acdc/training"
    mnms_data_dir: str = "data/raw/mnms"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        d["normalization"] = self.normalization.value
        d["loss_type"] = self.loss_type.value
        return d


@dataclass
class LGESegmentationConfig:
    """Hyperparameter and architectural specification for LGE Viability Tissue Segmentation."""
    task_name: str = "LGE_TISSUE_SEGMENTATION"
    architecture: str = "DynUNet_2D_or_3D"
    spatial_dimensions: int = 2  # Slice-based short axis or 3D patch
    in_channels: int = 1
    out_channels: int = 5  # 0: BG, 1: LV Cavity, 2: Normal Myo, 3: Infarct, 4: No-Reflow (MVO)
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    target_spatial_size: Tuple[int, int] = (256, 256)
    normalization: NormalizationMethod = NormalizationMethod.MIN_MAX_SCALING
    loss_type: LossFunctionType = LossFunctionType.FOCAL_TVERSKY
    # Loss penalties for extreme class imbalance (MVO < 0.1% of voxels)
    class_weights: Tuple[float, ...] = (0.1, 1.0, 1.0, 10.0, 25.0)
    tversky_alpha: float = 0.3  # False-positive penalty
    tversky_beta: float = 0.7   # False-negative penalty (recall prioritized for small MVO core)
    tversky_gamma: float = 1.33 # Focal focusing parameter
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 40
    random_seed: int = 42
    target_checkpoint_dir: str = "models/pipeline/lge/"
    supported_datasets: List[str] = field(default_factory=lambda: ["emidec"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        d["normalization"] = self.normalization.value
        d["loss_type"] = self.loss_type.value
        return d


@dataclass
class CheckpointMetadata:
    """Standardized metadata tracking provenance of any serialized model weights."""
    model_name: str
    task_name: str
    architecture: str
    checkpoint_path: str
    sha256_hash: str
    creation_timestamp: str
    random_seed: int
    training_datasets: List[str]
    training_case_count: int
    validation_case_count: int
    validation_metrics: Dict[str, float]
    git_commit_hash: Optional[str] = None
    input_spatial_shape: Tuple[int, ...] = field(default_factory=tuple)
    label_mapping: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    def save(self, json_path: Path | str) -> None:
        """Save metadata alongside weights."""
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
