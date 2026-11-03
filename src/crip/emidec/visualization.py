"""Delayed-Enhancement Cardiac MRI Visualization Engine.

Generates high-resolution publication-quality overlays and visual inspections of
actual EMIDEC 3D LGE images and expert tissue viability annotations:
- Original Late Gadolinium Enhancement (LGE) T1-weighted CMR.
- Expert reference contours (LV blood pool, normal myocardium, scar, no-reflow).
- Multi-component anatomical colormap with adjustable alpha opacity.
- Embedded clinical tissue quantification metrics (infarction burden %, MVO burden %).

NOTE ON SCIENTIFIC ATTRIBUTION:
All masks rendered by this module are EXPERT GROUND-TRUTH ANNOTATIONS from the EMIDEC
challenge (University Hospital of Dijon), NOT automated model predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

from crip.datasets.emidec_loader import EMIDECDataLoader
from crip.datasets.labels import EMIDECLabel, EMIDEC_LABELS
from crip.emidec.tissue_metrics import (
    EMIDECTissueMeasurements,
    compute_emidec_tissue_metrics,
)

logger = logging.getLogger("crip.emidec.visualization")

# Clinical Viability Colormap:
# 0: Transparent Background
# 1: Left Ventricular Cavity (Bright Cyan / Royal Blue)
# 2: Normal Myocardium (Emerald Green)
# 3: Myocardial Infarction / Scar Hyperenhancement (Vivid Red-Orange)
# 4: Microvascular Obstruction / No-Reflow Core (Dark Violet / Navy)
EMIDEC_RGB_PALETTE = [
    [0.0, 0.0, 0.0, 0.0],   # 0: Background (Transparent)
    [0.12, 0.53, 0.90, 0.65], # 1: LV Cavity (Cyan/Blue)
    [0.18, 0.80, 0.44, 0.65], # 2: Normal Myocardium (Green)
    [0.91, 0.30, 0.24, 0.75], # 3: Infarction / Scar (Vivid Red)
    [0.55, 0.15, 0.65, 0.85], # 4: No-Reflow / MVO (Deep Purple)
]
EMIDEC_CMAP = ListedColormap(EMIDEC_RGB_PALETTE)


def get_emidec_legend_patches() -> List[mpatches.Patch]:
    """Return standard legend patches for EMIDEC tissue classes."""
    return [
        mpatches.Patch(color=EMIDEC_RGB_PALETTE[1][:3], label="LV Cavity (Label 1)"),
        mpatches.Patch(color=EMIDEC_RGB_PALETTE[2][:3], label="Normal Myocardium (Label 2)"),
        mpatches.Patch(color=EMIDEC_RGB_PALETTE[3][:3], label="Myocardial Infarction / Scar (Label 3)"),
        mpatches.Patch(color=EMIDEC_RGB_PALETTE[4][:3], label="Microvascular Obstruction / MVO (Label 4)"),
    ]


def normalize_lge_slice(slice_2d: np.ndarray, p_min: float = 1.0, p_max: float = 99.0) -> np.ndarray:
    """Robust percentile intensity normalization for delayed-enhancement MRI."""
    v_min, v_max = np.percentile(slice_2d, (p_min, p_max))
    if v_max <= v_min:
        v_min, v_max = float(slice_2d.min()), float(slice_2d.max())
    if v_max <= v_min:
        return np.zeros_like(slice_2d, dtype=np.float32)
    clipped = np.clip(slice_2d, v_min, v_max)
    return ((clipped - v_min) / (v_max - v_min)).astype(np.float32)


def render_emidec_slice_comparison(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    case_id: str,
    slice_idx: int,
    total_slices: int,
    metrics: Optional[EMIDECTissueMeasurements] = None,
    alpha: float = 0.65,
    figsize: Tuple[float, float] = (12.0, 5.0),
) -> plt.Figure:
    """Render dual-panel comparison: Native LGE MRI vs. Expert Viability Overlay.

    Args:
        image_slice: 2D array of delayed-enhancement MRI (H, W).
        mask_slice: 2D array of expert segmentation labels (0-4) (H, W).
        case_id: Case identifier (e.g. 'Case_P001').
        slice_idx: Zero-based slice index.
        total_slices: Total short-axis slices in stack.
        metrics: Optional quantitative tissue measurements for metadata annotation.
        alpha: Colormap opacity.
        figsize: Output figure dimension tuple.

    Returns:
        Matplotlib Figure object ready for saving or interactive display.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.0, 1.0]})

    norm_img = normalize_lge_slice(image_slice)

    # Panel 1: Native LGE MRI
    ax1.imshow(norm_img, cmap="gray")
    ax1.set_title(f"A. Native Delayed-Enhancement CMR\n({case_id}, Slice {slice_idx + 1}/{total_slices})", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Expert Tissue Overlay
    ax2.imshow(norm_img, cmap="gray")
    masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax2.imshow(masked_overlay, cmap=EMIDEC_CMAP, vmin=0, vmax=4, alpha=alpha, interpolation="nearest")
    ax2.set_title(f"B. Expert Myocardial Viability Contours\n(Ground-Truth Reference)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Legend
    patches = get_emidec_legend_patches()
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        framealpha=0.92,
        fontsize=9,
    )

    # Text Box with Case Metrics
    if metrics is not None:
        info_text = (
            f"Pathology: {metrics.pathology_group}\n"
            f"Myocardium Vol: {metrics.total_myocardium_volume_ml:.1f} mL\n"
            f"Infarct Vol: {metrics.infarction_volume_ml:.1f} mL ({metrics.infarction_burden_percent:.1f}%)\n"
            f"MVO Vol: {metrics.no_reflow_volume_ml:.1f} mL ({metrics.no_reflow_burden_percent:.1f}%)"
        )
        ax2.text(
            0.03, 0.05, info_text,
            transform=ax2.transAxes,
            fontsize=8.5,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.75, edgecolor="gray"),
        )

    plt.tight_layout()
    return fig


def render_case_viability(
    case_id: str,
    loader: Optional[EMIDECDataLoader] = None,
    slice_idx: Optional[int] = None,
) -> plt.Figure:
    """Convenience helper to load an actual EMIDEC case and render the slice overlay."""
    ld = loader or EMIDECDataLoader()
    case_data = ld.load_case(case_id)
    if not case_data["has_contour"]:
        raise ValueError(f"Case {case_id} has no reference contours.")

    img_3d = case_data["image"]
    cnt_3d = case_data["contour"]
    zooms = case_data["voxel_spacing"]

    metrics = compute_emidec_tissue_metrics(
        mask_3d=cnt_3d,
        voxel_spacing_mm=zooms,
        case_id=case_id,
        pathology_group=case_data["pathology"],
    )

    n_slices = img_3d.shape[2] if len(img_3d.shape) == 3 else img_3d.shape[0]

    # Select slice with maximum abnormal tissue or user-specified slice
    if slice_idx is None:
        # Check if 3D is (H, W, Slices) or (Slices, H, W)
        if img_3d.shape[2] == n_slices:
            abnormal_counts = [np.count_nonzero(cnt_3d[:, :, s] >= 3) for s in range(n_slices)]
            if max(abnormal_counts) > 0:
                s_sel = int(np.argmax(abnormal_counts))
            else:
                myo_counts = [np.count_nonzero(cnt_3d[:, :, s] == 2) for s in range(n_slices)]
                s_sel = int(np.argmax(myo_counts))
            img_slice = img_3d[:, :, s_sel]
            cnt_slice = cnt_3d[:, :, s_sel]
        else:
            abnormal_counts = [np.count_nonzero(cnt_3d[s, :, :] >= 3) for s in range(n_slices)]
            s_sel = int(np.argmax(abnormal_counts)) if max(abnormal_counts) > 0 else n_slices // 2
            img_slice = img_3d[s_sel, :, :]
            cnt_slice = cnt_3d[s_sel, :, :]
    else:
        s_sel = slice_idx
        if img_3d.shape[2] == n_slices:
            img_slice = img_3d[:, :, s_sel]
            cnt_slice = cnt_3d[:, :, s_sel]
        else:
            img_slice = img_3d[s_sel, :, :]
            cnt_slice = cnt_3d[s_sel, :, :]

    return render_emidec_slice_comparison(
        image_slice=img_slice,
        mask_slice=cnt_slice,
        case_id=case_id,
        slice_idx=s_sel,
        total_slices=n_slices,
        metrics=metrics,
    )
