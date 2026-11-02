"""Cardiac MRI Segmentation Label Definitions and Semantic Mapping.

Provides explicit label definitions for heterogeneous cardiac MRI datasets and
safe, validated transformations to canonical conventions:
- ACDC: 0=Background, 1=Right Ventricle, 2=Myocardium, 3=Left Ventricle
- M&Ms: 0=Background, 1=Left Ventricle, 2=Myocardium, 3=Right Ventricle
- EMIDEC: 0=Background, 1=LV Cavity, 2=Normal Myocardium, 3=Infarction, 4=No-Reflow

CRITICAL CLINICAL GOVERNANCE:
1. M&Ms inverts the ventricular indices (1=LV, 3=RV) relative to ACDC (1=RV, 3=LV).
   Direct unmapped reuse of M&Ms masks inside an ACDC-trained pipeline would invert
   left and right ventricular blood pools, severely distorting volumetric calculations.
2. EMIDEC represents delayed-enhancement tissue viability (infarction and no-reflow core).
   It MUST NOT be mapped to ventricular blood pool indices.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Set, Union
import numpy as np


class ACDCLabel(IntEnum):
    """ACDC 2017 Challenge canonical label convention."""
    BACKGROUND = 0
    RV = 1
    RV_CAVITY = 1
    MYOCARDIUM = 2
    LV = 3
    LV_CAVITY = 3


class MNMSLabel(IntEnum):
    """M&Ms 2020 Challenge original label convention."""
    BACKGROUND = 0
    LV = 1
    LV_CAVITY = 1
    MYOCARDIUM = 2
    RV = 3
    RV_CAVITY = 3


class EMIDECLabel(IntEnum):
    """EMIDEC 2020 Challenge tissue viability label convention."""
    BACKGROUND = 0
    LV_CAVITY = 1
    NORMAL_MYOCARDIUM = 2
    MYOCARDIAL_INFARCTION = 3
    NO_REFLOW = 4


# Descriptive label dictionaries
ACDC_LABEL_NAMES: Dict[int, str] = {
    int(ACDCLabel.BACKGROUND): "Background",
    int(ACDCLabel.RV_CAVITY): "Right Ventricular Cavity",
    int(ACDCLabel.MYOCARDIUM): "Left Ventricular Myocardium",
    int(ACDCLabel.LV_CAVITY): "Left Ventricular Cavity",
}
ACDC_LABELS = ACDC_LABEL_NAMES

MNMS_LABEL_NAMES: Dict[int, str] = {
    int(MNMSLabel.BACKGROUND): "Background",
    int(MNMSLabel.LV_CAVITY): "Left Ventricular Cavity",
    int(MNMSLabel.MYOCARDIUM): "Left Ventricular Myocardium",
    int(MNMSLabel.RV_CAVITY): "Right Ventricular Cavity",
}
MNMS_LABELS = MNMS_LABEL_NAMES

EMIDEC_LABEL_NAMES: Dict[int, str] = {
    int(EMIDECLabel.BACKGROUND): "Background",
    int(EMIDECLabel.LV_CAVITY): "Left Ventricular Cavity",
    int(EMIDECLabel.NORMAL_MYOCARDIUM): "Normal Myocardium",
    int(EMIDECLabel.MYOCARDIAL_INFARCTION): "Myocardial Infarction (Scar / Hyperenhancement)",
    int(EMIDECLabel.NO_REFLOW): "Microvascular Obstruction (No-Reflow Core)",
}
EMIDEC_LABELS = EMIDEC_LABEL_NAMES

# Semantic mapping: M&Ms -> Canonical ACDC
# M&Ms 1 (LV) -> ACDC 3 (LV)
# M&Ms 2 (Myo) -> ACDC 2 (Myo)
# M&Ms 3 (RV) -> ACDC 1 (RV)
MNMS_TO_CANONICAL_MAP: Dict[int, int] = {
    int(MNMSLabel.BACKGROUND): int(ACDCLabel.BACKGROUND),
    int(MNMSLabel.LV_CAVITY): int(ACDCLabel.LV_CAVITY),
    int(MNMSLabel.MYOCARDIUM): int(ACDCLabel.MYOCARDIUM),
    int(MNMSLabel.RV_CAVITY): int(ACDCLabel.RV_CAVITY),
}

CANONICAL_TO_MNMS_MAP: Dict[int, int] = {
    int(ACDCLabel.BACKGROUND): int(MNMSLabel.BACKGROUND),
    int(ACDCLabel.LV_CAVITY): int(MNMSLabel.LV_CAVITY),
    int(ACDCLabel.MYOCARDIUM): int(MNMSLabel.MYOCARDIUM),
    int(ACDCLabel.RV_CAVITY): int(MNMSLabel.RV_CAVITY),
}


def map_mnms_to_canonical(mask: np.ndarray) -> np.ndarray:
    """Convert M&Ms label mask to canonical ACDC label convention.

    Preserves exact spatial shapes while remapping integer values:
      M&Ms 0 (Background) -> Canonical 0
      M&Ms 1 (LV Cavity)   -> Canonical 3 (LV Cavity)
      M&Ms 2 (Myocardium)  -> Canonical 2 (Myocardium)
      M&Ms 3 (RV Cavity)   -> Canonical 1 (RV Cavity)

    Args:
        mask: NumPy integer array containing M&Ms raw segmentation labels.

    Returns:
        NumPy integer array with canonical ACDC class convention.
    """
    out = np.copy(mask)
    # Use boolean masks from original array to prevent double substitution
    is_rv = (mask == int(MNMSLabel.RV_CAVITY))
    is_lv = (mask == int(MNMSLabel.LV_CAVITY))
    is_myo = (mask == int(MNMSLabel.MYOCARDIUM))

    out[is_rv] = int(ACDCLabel.RV_CAVITY)   # 3 -> 1
    out[is_lv] = int(ACDCLabel.LV_CAVITY)   # 1 -> 3
    out[is_myo] = int(ACDCLabel.MYOCARDIUM) # 2 -> 2
    return out


def map_canonical_to_mnms(mask: np.ndarray) -> np.ndarray:
    """Convert canonical ACDC label mask to M&Ms challenge label convention.

    Args:
        mask: NumPy integer array with canonical ACDC class convention.

    Returns:
        NumPy integer array with original M&Ms class convention.
    """
    out = np.copy(mask)
    is_lv = (mask == int(ACDCLabel.LV_CAVITY))
    is_rv = (mask == int(ACDCLabel.RV_CAVITY))
    is_myo = (mask == int(ACDCLabel.MYOCARDIUM))

    out[is_lv] = int(MNMSLabel.LV_CAVITY)   # 3 -> 1
    out[is_rv] = int(MNMSLabel.RV_CAVITY)   # 1 -> 3
    out[is_myo] = int(MNMSLabel.MYOCARDIUM) # 2 -> 2
    return out


def validate_label_set(mask: np.ndarray, expected_labels: Set[int]) -> bool:
    """Verify that all non-zero labels present in a mask belong to an expected set."""
    present_labels = set(np.unique(mask).astype(int).tolist())
    return present_labels.issubset(expected_labels)
