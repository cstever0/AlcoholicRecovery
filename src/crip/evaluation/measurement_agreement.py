"""Statistical measurement agreement, Bland-Altman analysis, and spatial distance evaluation.

Provides rigorous biostatistical functions comparing automated U-Net deep learning
predictions against expert ground-truth manual annotations on held-out CMR test cases:
- Predicted vs Reference Scatter Plots with Identity Line (y = x)
- Bland-Altman Difference Analysis with Mean Bias and 95% Limits of Agreement (LoA)
- Standard Errors and 95% Confidence Intervals for Bias and Limits of Agreement
- HD95 95th-percentile Hausdorff Distance spatial metrics across cardiac structures
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


@dataclass
class BlandAltmanResult:
    """Encapsulates quantitative Bland-Altman agreement and error metrics."""
    variable_name: str
    unit: str
    n_samples: int
    mean_bias: float
    sd_diff: float
    lower_loa: float
    upper_loa: float
    ci_bias_95: Tuple[float, float]
    ci_lower_loa_95: Tuple[float, float]
    ci_upper_loa_95: Tuple[float, float]
    mae: float
    median_ae: float
    rmse: float
    pearson_r: float
    differences: np.ndarray
    means: np.ndarray
    references: np.ndarray
    predictions: np.ndarray
    patient_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistical results to dictionary (omitting large arrays)."""
        return {
            "variable_name": self.variable_name,
            "unit": self.unit,
            "n_samples": self.n_samples,
            "mean_bias": round(float(self.mean_bias), 4),
            "sd_diff": round(float(self.sd_diff), 4),
            "lower_loa": round(float(self.lower_loa), 4),
            "upper_loa": round(float(self.upper_loa), 4),
            "ci_bias_95": (round(float(self.ci_bias_95[0]), 4), round(float(self.ci_bias_95[1]), 4)),
            "ci_lower_loa_95": (round(float(self.ci_lower_loa_95[0]), 4), round(float(self.ci_lower_loa_95[1]), 4)),
            "ci_upper_loa_95": (round(float(self.ci_upper_loa_95[0]), 4), round(float(self.ci_upper_loa_95[1]), 4)),
            "mae": round(float(self.mae), 4),
            "median_ae": round(float(self.median_ae), 4),
            "rmse": round(float(self.rmse), 4),
            "pearson_r": round(float(self.pearson_r), 4),
        }


def compute_measurement_agreement(
    predictions: Union[np.ndarray, List[float], pd.Series],
    references: Union[np.ndarray, List[float], pd.Series],
    variable_name: str,
    unit: str,
    patient_ids: Optional[Union[List[str], pd.Series]] = None,
) -> BlandAltmanResult:
    """Compute rigorous Bland-Altman agreement and error metrics.

    Mathematical Formulation:
      Difference: d_i = pred_i - ref_i
      Mean of methods: m_i = (pred_i + ref_i) / 2.0
      Mean bias: d_bar = (1/n) * sum(d_i)
      Sample variance (ddof=1): s^2 = (1/(n - 1)) * sum((d_i - d_bar)^2)
      Sample standard deviation: s = sqrt(s^2)
      95% Limits of Agreement (LoA): d_bar +/- 1.96 * s

    Standard Errors (Bland & Altman 1999):
      SE(bias) = s / sqrt(n)
      SE(LoA) = sqrt(3 * s^2 / n)
      95% CI(bias) = d_bar +/- t_{n-1, 0.975} * SE(bias)
      95% CI(LoA) = LoA +/- t_{n-1, 0.975} * SE(LoA)

    Args:
        predictions: Array of model predicted measurements
        references: Array of ground-truth reference measurements
        variable_name: Human-readable name of physiological parameter (e.g. 'LVEF')
        unit: Measurement unit (e.g. 'mL' or '%-points')
        patient_ids: Optional list of patient identifiers

    Returns:
        BlandAltmanResult dataclass with comprehensive statistical outputs.
    """
    preds = np.asarray(predictions, dtype=float)
    refs = np.asarray(references, dtype=float)

    if preds.ndim != 1 or refs.ndim != 1:
        raise ValueError(f"Inputs must be 1D arrays, got preds {preds.shape}, refs {refs.shape}")

    if len(preds) != len(refs):
        raise ValueError(
            f"Mismatched array lengths: {len(preds)} predictions vs {len(refs)} references"
        )

    if len(preds) < 2:
        raise ValueError(f"At least 2 samples required for agreement analysis, got {len(preds)}")

    # Filter out NaNs
    valid_mask = np.isfinite(preds) & np.isfinite(refs)
    if not np.all(valid_mask):
        preds = preds[valid_mask]
        refs = refs[valid_mask]

    n = len(preds)
    if n < 2:
        raise ValueError("Insufficient non-NaN sample pairs for agreement analysis")

    if patient_ids is not None:
        p_ids = [str(pid) for i, pid in enumerate(patient_ids) if valid_mask[i]]
    else:
        p_ids = [f"Case_{i+1:02d}" for i in range(n)]

    # Pairwise differences and averages
    diff = preds - refs
    means = (preds + refs) / 2.0

    # Error metrics
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    median_ae = float(np.median(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Correlation
    if np.std(preds) > 0 and np.std(refs) > 0:
        corr_matrix = np.corrcoef(refs, preds)
        pearson_r = float(corr_matrix[0, 1])
    else:
        pearson_r = float("nan")

    # Bland-Altman statistics
    mean_bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    lower_loa = mean_bias - 1.96 * sd_diff
    upper_loa = mean_bias + 1.96 * sd_diff

    # Standard errors and Student-t 95% confidence intervals (df = n - 1)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    se_bias = sd_diff / np.sqrt(n)
    se_loa = np.sqrt(3.0 * (sd_diff ** 2) / n)

    ci_bias = (
        float(mean_bias - t_crit * se_bias),
        float(mean_bias + t_crit * se_bias),
    )
    ci_lower_loa = (
        float(lower_loa - t_crit * se_loa),
        float(lower_loa + t_crit * se_loa),
    )
    ci_upper_loa = (
        float(upper_loa - t_crit * se_loa),
        float(upper_loa + t_crit * se_loa),
    )

    return BlandAltmanResult(
        variable_name=variable_name,
        unit=unit,
        n_samples=n,
        mean_bias=mean_bias,
        sd_diff=sd_diff,
        lower_loa=lower_loa,
        upper_loa=upper_loa,
        ci_bias_95=ci_bias,
        ci_lower_loa_95=ci_lower_loa,
        ci_upper_loa_95=ci_upper_loa,
        mae=mae,
        median_ae=median_ae,
        rmse=rmse,
        pearson_r=pearson_r,
        differences=diff,
        means=means,
        references=refs,
        predictions=preds,
        patient_ids=p_ids,
    )


def create_identity_scatter_plot(
    res: BlandAltmanResult,
    height: int = 420,
    dark_mode: bool = True,
) -> go.Figure:
    """Create interactive Plotly scatter plot comparing predicted vs. reference measurements with identity line (y = x)."""
    fig = go.Figure()

    # Determine common axis limits
    all_vals = np.concatenate([res.references, res.predictions])
    min_val = float(np.min(all_vals))
    max_val = float(np.max(all_vals))
    val_range = max_val - min_val
    padding = max(val_range * 0.08, 1.0)
    axis_min = min_val - padding
    axis_max = max_val + padding

    # 1. Identity line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="Identity Line (y = x)",
            line=dict(color="#A0AAB2", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    # 2. Scatter Points
    hover_texts = [
        f"<b>Patient:</b> {pid}<br>"
        f"<b>Reference:</b> {ref:.2f} {res.unit}<br>"
        f"<b>Predicted:</b> {pred:.2f} {res.unit}<br>"
        f"<b>Signed Error:</b> {diff:+.2f} {res.unit}<br>"
        f"<b>Absolute Error:</b> {abs(diff):.2f} {res.unit}"
        for pid, ref, pred, diff in zip(res.patient_ids, res.references, res.predictions, res.differences)
    ]

    fig.add_trace(
        go.Scatter(
            x=res.references,
            y=res.predictions,
            mode="markers",
            name="Test Cases",
            marker=dict(
                size=9,
                color="#00CC96",
                line=dict(width=1, color="#FFFFFF"),
                opacity=0.9,
            ),
            text=hover_texts,
            hoverinfo="text",
        )
    )

    # Statistical annotation box
    unit_str = res.unit
    stat_annotation = (
        f"<b>Evaluation Cohort (n = {res.n_samples})</b><br>"
        f"MAE: {res.mae:.2f} {unit_str}<br>"
        f"Median AE: {res.median_ae:.2f} {unit_str}<br>"
        f"RMSE: {res.rmse:.2f} {unit_str}<br>"
        f"Pearson r: {res.pearson_r:.4f}"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.04,
        y=0.95,
        text=stat_annotation,
        showarrow=False,
        align="left",
        bgcolor="rgba(30, 37, 48, 0.85)" if dark_mode else "rgba(240, 240, 240, 0.85)",
        bordercolor="#4C566A",
        borderwidth=1,
        borderpad=6,
        font=dict(size=11, color="#ECEFF4" if dark_mode else "#2E3440"),
    )

    fig.update_layout(
        title=f"Predicted vs. Reference {res.variable_name} ({res.unit}) — Held-out Test Cohort",
        xaxis_title=f"Expert Ground-Truth Reference ({res.unit})",
        yaxis_title=f"Automated 2D U-Net Prediction ({res.unit})",
        xaxis=dict(range=[axis_min, axis_max], zeroline=False),
        yaxis=dict(range=[axis_min, axis_max], zeroline=False),
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=50, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_bland_altman_plot(
    res: BlandAltmanResult,
    height: int = 440,
    dark_mode: bool = True,
) -> go.Figure:
    """Create interactive Plotly Bland-Altman agreement plot with bias and 95% LoA."""
    fig = go.Figure()

    min_mean = float(np.min(res.means))
    max_mean = float(np.max(res.means))
    mean_pad = max((max_mean - min_mean) * 0.08, 1.0)
    x_min = min_mean - mean_pad
    x_max = max_mean + mean_pad

    # 1. Zero Reference Line (d = 0)
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[0, 0],
            mode="lines",
            name="Zero Difference",
            line=dict(color="#6E7681", width=1, dash="dot"),
            hoverinfo="skip",
        )
    )

    # 2. Mean Bias Line
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[res.mean_bias, res.mean_bias],
            mode="lines",
            name=f"Mean Bias ({res.mean_bias:+.2f} {res.unit})",
            line=dict(color="#2E86C1", width=2.5),
            hoverinfo="skip",
        )
    )

    # 3. Upper 95% Limit of Agreement (+1.96 SD)
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[res.upper_loa, res.upper_loa],
            mode="lines",
            name=f"Upper 95% LoA ({res.upper_loa:+.2f} {res.unit})",
            line=dict(color="#E74C3C", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    # 4. Lower 95% Limit of Agreement (-1.96 SD)
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[res.lower_loa, res.lower_loa],
            mode="lines",
            name=f"Lower 95% LoA ({res.lower_loa:+.2f} {res.unit})",
            line=dict(color="#E74C3C", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    # 5. Scatter Points
    hover_texts = [
        f"<b>Patient:</b> {pid}<br>"
        f"<b>Mean of Methods:</b> {m:.2f} {res.unit}<br>"
        f"<b>Difference (Pred - Ref):</b> {d:+.2f} {res.unit}<br>"
        f"<b>Reference:</b> {ref:.2f} {res.unit}<br>"
        f"<b>Predicted:</b> {pred:.2f} {res.unit}"
        for pid, m, d, ref, pred in zip(res.patient_ids, res.means, res.differences, res.references, res.predictions)
    ]

    fig.add_trace(
        go.Scatter(
            x=res.means,
            y=res.differences,
            mode="markers",
            name="Patient Differences",
            marker=dict(
                size=9,
                color="#636EFA",
                line=dict(width=1, color="#FFFFFF"),
                opacity=0.9,
            ),
            text=hover_texts,
            hoverinfo="text",
        )
    )

    # Annotations on the right for LoA and Bias lines
    fig.add_annotation(
        x=x_max,
        y=res.upper_loa,
        text=f" +1.96 SD: {res.upper_loa:+.2f}",
        showarrow=False,
        xanchor="left",
        font=dict(color="#E74C3C", size=10),
    )
    fig.add_annotation(
        x=x_max,
        y=res.mean_bias,
        text=f" Bias: {res.mean_bias:+.2f}",
        showarrow=False,
        xanchor="left",
        font=dict(color="#2E86C1", size=10),
    )
    fig.add_annotation(
        x=x_max,
        y=res.lower_loa,
        text=f" -1.96 SD: {res.lower_loa:+.2f}",
        showarrow=False,
        xanchor="left",
        font=dict(color="#E74C3C", size=10),
    )

    # Summary confidence intervals box
    ci_box = (
        f"<b>Bland–Altman Agreement (n = {res.n_samples})</b><br>"
        f"Mean Bias: <b>{res.mean_bias:+.2f} {res.unit}</b> (95% CI: [{res.ci_bias_95[0]:+.2f}, {res.ci_bias_95[1]:+.2f}])<br>"
        f"SD of Differences: <b>{res.sd_diff:.2f} {res.unit}</b><br>"
        f"Upper 95% LoA: <b>{res.upper_loa:+.2f} {res.unit}</b> (95% CI: [{res.ci_upper_loa_95[0]:+.2f}, {res.ci_upper_loa_95[1]:+.2f}])<br>"
        f"Lower 95% LoA: <b>{res.lower_loa:+.2f} {res.unit}</b> (95% CI: [{res.ci_lower_loa_95[0]:+.2f}, {res.ci_lower_loa_95[1]:+.2f}])"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.03,
        y=0.06,
        text=ci_box,
        showarrow=False,
        align="left",
        bgcolor="rgba(30, 37, 48, 0.85)" if dark_mode else "rgba(240, 240, 240, 0.85)",
        bordercolor="#4C566A",
        borderwidth=1,
        borderpad=6,
        font=dict(size=10, color="#ECEFF4" if dark_mode else "#2E3440"),
    )

    fig.update_layout(
        title=f"Bland–Altman Agreement: {res.variable_name} ({res.unit}) — Held-out Test Cohort",
        xaxis_title=f"Mean of Predicted and Reference Measurements ({res.unit})",
        yaxis_title=f"Difference: (Predicted − Reference) ({res.unit})",
        xaxis=dict(range=[x_min, x_max * 1.08], zeroline=False),
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=50, r=70, t=50, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def compute_hd95_statistics(eval_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute comprehensive HD95 summary statistics (ED, ES, and Pooled) for LV, RV, and Myocardium."""
    structures = {
        "LV": {"label": "Left Ventricle Cavity", "key": "lv"},
        "RV": {"label": "Right Ventricle Cavity", "key": "rv"},
        "MYO": {"label": "Myocardium", "key": "myo"},
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, info in structures.items():
        k = info["key"]
        ed_col = f"hd95_ed_{k}_mm"
        es_col = f"hd95_es_{k}_mm"

        ed_vals = eval_df[ed_col].dropna().to_numpy() if ed_col in eval_df.columns else np.array([])
        es_vals = eval_df[es_col].dropna().to_numpy() if es_col in eval_df.columns else np.array([])
        pooled_vals = np.concatenate([ed_vals, es_vals]) if len(ed_vals) and len(es_vals) else ed_vals

        def _get_stats(vals: np.ndarray) -> Dict[str, float]:
            if len(vals) == 0:
                return {"mean": float("nan"), "median": float("nan"), "sd": float("nan"), "iqr_low": float("nan"), "iqr_high": float("nan"), "min": float("nan"), "max": float("nan")}
            return {
                "mean": round(float(np.mean(vals)), 2),
                "median": round(float(np.median(vals)), 2),
                "sd": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
                "iqr_low": round(float(np.percentile(vals, 25)), 2),
                "iqr_high": round(float(np.percentile(vals, 75)), 2),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
            }

        results[name] = {
            "name": info["label"],
            "unit": "mm",
            "ed": _get_stats(ed_vals),
            "es": _get_stats(es_vals),
            "pooled": _get_stats(pooled_vals),
        }

    return results


def create_hd95_box_plot(eval_df: pd.DataFrame, height: int = 380, dark_mode: bool = True) -> go.Figure:
    """Create interactive Plotly box plot illustrating HD95 spatial boundary error across structures."""
    fig = go.Figure()

    palette = {"LV": "#FF4040", "RV": "#26A6FF", "MYO": "#33D959"}

    for struct, col_name, label in [("LV", "hd95_ed_lv_mm", "LV Cavity (ED)"),
                                     ("RV", "hd95_ed_rv_mm", "RV Cavity (ED)"),
                                     ("MYO", "hd95_ed_myo_mm", "Myocardium (ED)")]:
        if col_name in eval_df.columns:
            vals = eval_df[col_name].dropna().to_numpy()
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=label,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.6,
                    marker=dict(color=palette[struct], size=6),
                    line=dict(width=1.5),
                )
            )

    fig.update_layout(
        title="95th-Percentile Hausdorff Distance (HD95 mm) by Anatomical Structure (End-Diastole)",
        yaxis_title="HD95 Boundary Distance (mm)",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        showlegend=False,
    )

    return fig
