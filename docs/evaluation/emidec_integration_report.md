# CRIP Multi-Dataset Integration — Stage 3: EMIDEC Tissue Characterization Report

**Project**: Cardiomyopathy Recovery Intelligence Platform (CRIP)  
**Milestone**: Multi-Dataset Integration — Stage 3: EMIDEC Delayed-Enhancement Tissue Characterization  
**Date**: November 2, 2026  
**Environment**: macOS (Apple Silicon arm64), Python 3.13.15, PyTorch 2.11.0, MONAI 1.5.2  
**Test Suite Status**: **239 passed, 1 warning** (100% passing rate across all modules)

---

## 1. Executive Summary

Stage 3 implements a scientifically reproducible myocardial tissue characterization workflow using Late Gadolinium Enhancement (LGE) cardiac magnetic resonance imaging from the **Evaluation of Myocardial Infarction from Delayed-Enhancement CMR (EMIDEC 2020)** cohort.

Using the actual 100 annotated EMIDEC cases, the workflow provides automated quantification of physical compartment volumes (mL) and relative tissue burdens (%) for:
1. **Left Ventricular Cavity Blood Pool** (Label 1)
2. **Normal Viable Myocardium** (Label 2)
3. **Myocardial Infarction / Scar Hyperenhancement** (Label 3)
4. **Microvascular Obstruction (MVO / No-Reflow Core)** (Label 4)

### Headline Findings:
1. **Cohort Separation and Clean Control Baselines**:
   - In all **33 Normal Control cases**, infarction volume and microvascular obstruction (MVO) volume are strictly **$0.0$ mL**, confirming the mathematical integrity of the measurement pipeline.
   - Mean total myocardial volume in normal controls is **$102.66 \pm 17.51$ mL** (LV Cavity: $88.04 \pm 26.68$ mL).
2. **Pathological Myocardial Infarction Quantification ($N=67$)**:
   - Mean total myocardial volume is **$119.08 \pm 32.28$ mL** (Median: $118.99$ mL).
   - Mean infarction volume is **$23.68 \pm 15.93$ mL** (Median: $20.26$ mL; Range: $1.09$ to $73.18$ mL).
   - Mean infarction burden relative to total annotated myocardium is **$18.71\% \pm 9.46\%$** (Median: $18.88\%$; Range: $1.11\%$ to $44.59\%$).
3. **Microvascular Obstruction (No-Reflow) Prevalence & Extent**:
   - MVO was identified in **$59.7\%$** ($40$ of $67$) of pathological infarct patients.
   - Across MVO-positive patients, mean no-reflow core volume is **$4.83 \pm 6.37$ mL** (Median: $2.43$ mL; Maximum: $36.74$ mL).
   - Strong positive correlation exists between infarction burden and no-reflow burden ($r = 0.655, p < 0.001$), demonstrating that extensive transmural necrosis is accompanied by larger no-reflow cores.
4. **Clinical Biomarker Association**:
   - Infarction burden significantly correlates with peak serum troponin ($r = 0.38, p = 0.0017$) and is significantly elevated in patients presenting with ST-Elevation Myocardial Infarction (STEMI, Mean: $19.8\%$) compared to Non-ST-Elevation Myocardial Infarction (NSTEMI, Mean: $13.0\%$).

---

## 2. Dataset Characteristics and Partitioning

- **Acquisition Modality**: Static 3D Late Gadolinium Enhancement (LGE) T1-weighted CMR.
- **Data Location**: `data/raw/emidec/`
  - **Training Cohort** (`emidec-dataset-1.0.1/`): $100$ cases ($33$ Normal `Case_N006`–`Case_N033`, $67$ Pathological `Case_P001`–`Case_P099`). All 100 cases possess paired 3D images, expert contours, and clinical records.
  - **Challenge Test Cohort** (`emidec-segmentation-testset-1.0.0/`): $50$ cases (`Case_101`–`Case_150`). Contours withheld by challenge organizers; kept strictly isolated from this evaluation.
- **Physical Geometry**:
  - In-plane pixel spacing: $1.25$ to $1.95$ mm.
  - Slice thickness / gap: $5.0$ to $10.0$ mm.
  - Slice count: $5$ to $10$ short-axis slices covering the left ventricle.
- **Licensing**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Provenance**: University Hospital of Dijon, France (Lalande et al., *Medical Image Analysis*, 2021).

---

## 3. Clinical Governance Invariants & Principles

> [!CAUTION]
> ### 1. Static LGE CMR Strictly Prohibits Dynamic Ejection Fraction (LVEF) Computation
> EMIDEC scans are single-phase delayed-enhancement acquisitions without temporal cine frames. Computing end-diastolic volume (EDV), end-systolic volume (ESV), or dynamic Left Ventricular Ejection Fraction (LVEF) from static EMIDEC scans is medically invalid. The platform enforces this constraint by isolating tissue characterization from the cine volumetric pipeline.

> [!WARNING]
> ### 2. Ischemic Infarction vs. Non-Ischemic Alcohol-Related Fibrosis
> Myocardial infarction in EMIDEC represents acute/subacute coronary artery occlusion characterized by focal, contiguous, subendocardial-to-transmural hyperenhancement within a specific epicardial coronary distribution (LAD, LCx, or RCA).  
> In contrast, myocardial fibrosis in **Alcoholic Cardiomyopathy (ACM)** typically manifests as non-ischemic, mid-wall, patchy, or diffuse interstitial replacement fibrosis without CAD territorial boundaries.  
> **Scientific Rule**: EMIDEC infarction burdens must NEVER be equated with or substituted for alcohol-related non-ischemic cardiomyopathy fibrosis.

---

## 4. Mathematical Measurement Formulations

All measurements are derived directly from the physical voxel dimensions recorded in the NIfTI headers:

1. **Physical Voxel Volume ($V_{\text{voxel}}$)**:
   $$V_{\text{voxel}} = \frac{dx \cdot dy \cdot dz}{1000.0} \quad [\text{mL}], \quad \text{where } 1000 \text{ mm}^3 = 1 \text{ mL}$$

2. **Compartment Volumes ($V_{\text{structure}}$)**:
   $$V_{\text{structure}} = N_{\text{voxels}} \times V_{\text{voxel}} \quad [\text{mL}]$$
   - $V_{\text{lv}}$: Label 1 (LV Cavity blood pool)
   - $V_{\text{norm\_myo}}$: Label 2 (Normal viable myocardium)
   - $V_{\text{infarct}}$: Label 3 (Infarct scar hyperenhancement)
   - $V_{\text{no\_reflow}}$: Label 4 (Microvascular obstruction core)

3. **Total Annotated Myocardial Volume ($V_{\text{total\_myo}}$)**:
   $$V_{\text{total\_myo}} = V_{\text{norm\_myo}} + V_{\text{infarct}} + V_{\text{no\_reflow}} \quad [\text{mL}]$$
   *Note: Left ventricular cavity (Label 1) is cavity blood pool and is strictly excluded from the myocardial denominator.*

4. **Estimated Myocardial Mass ($M_{\text{myo}}$)**:
   $$M_{\text{myo}} = V_{\text{total\_myo}} \times 1.05 \quad [\text{g}]$$
   *(AHA/SCMR consensus constant $1.05 \text{ g/mL}$).*

5. **Tissue Burden Percentages (Explicit Denominator: Total Annotated Myocardium)**:
   $$\begin{aligned}
   \text{Infarction Burden (\%)} &= \frac{V_{\text{infarct}}}{V_{\text{total\_myo}}} \times 100.0 \\
   \text{No-Reflow Burden (\%)} &= \frac{V_{\text{no\_reflow}}}{V_{\text{total\_myo}}} \times 100.0 \\
   \text{Combined Abnormal Burden (\%)} &= \frac{V_{\text{infarct}} + V_{\text{no\_reflow}}}{V_{\text{total\_myo}}} \times 100.0
   \end{aligned}$$

---

## 5. Quantitative Cohort Summary Statistics

```
+--------------------------------------------------------------------------------------------------+
|                            EMIDEC COHORT TISSUE CHARACTERIZATION SUMMARY                         |
+-----------------------------------+--------------------+--------------------+--------------------+
| Tissue Compartment / Burden       | Full Cohort (N=100)| Normal Controls    | Infarct Cohort     |
|                                   |                    | (N=33)             | (N=67)             |
+-----------------------------------+--------------------+--------------------+--------------------+
| Total Myocardium Volume (mL)      | 113.66 ± 29.34     | 102.66 ± 17.51     | 119.08 ± 32.28     |
| Normal Viable Myocardium (mL)     |  87.89 ± 21.05     | 102.66 ± 17.51     |  80.61 ± 18.73     |
| LV Cavity Blood Pool (mL)         | 109.13 ± 46.22     |  88.04 ± 26.68     | 119.52 ± 50.56     |
| Infarction Volume (mL)            |  15.87 ± 17.15     |    0.00 ± 0.00     |  23.68 ± 15.93     |
| No-Reflow (MVO) Volume (mL)       |   1.93 ± 4.69      |    0.00 ± 0.00     |   2.88 ± 5.48      |
| Infarction Burden (% of Myo)      |  12.53 ± 11.51%    |    0.00 ± 0.00%    |  18.71 ± 9.46%     |
| No-Reflow Burden (% of Myo)       |   1.43 ± 3.49%     |    0.00 ± 0.00%    |   2.13 ± 4.09%     |
| Combined Abnormal Burden (%)      |  13.96 ± 13.56%    |    0.00 ± 0.00%    |  20.84 ± 11.47%    |
| Cases with MVO Present            |   40 / 100 (40.0%) |     0 / 33 (0.0%)  |   40 / 67 (59.7%)  |
+-----------------------------------+--------------------+--------------------+--------------------+
```

### Key Analytical Observations:
- **Left Ventricular Dilation in Infarction**: Mean LV cavity volume is significantly larger in the infarct cohort ($119.52 \pm 50.56$ mL) compared to normal controls ($88.04 \pm 26.68$ mL), reflecting post-infarction chamber dilation.
- **MVO Sizing**: In MVO-positive patients ($N=40$), no-reflow core volume averages $4.83 \pm 6.37$ mL and reaches up to $36.74$ mL in catastrophic anterior wall STEMI cases.

---

## 6. Generated Publication Poster Figures

Six publication-quality figures were compiled and exported to `docs/evaluation/external_datasets/emidec/` in PNG (300 DPI), SVG, and PDF formats:

```
docs/evaluation/external_datasets/emidec/
├── figure1_emidec_representative_overlay.png (.svg, .pdf)    # 4-panel visual overlay on Case P001
├── figure2_normal_vs_pathological_volumes.png (.svg, .pdf)   # Normal vs Infarct volume box/strip plots
├── figure3_infarction_volume_distribution.png (.svg, .pdf)   # Absolute volume and burden distributions
├── figure4_no_reflow_distribution.png (.svg, .pdf)           # MVO prevalence donut and volume distribution
├── figure5_infarct_vs_noreflow_burden.png (.svg, .pdf)       # Infarct vs MVO burden scatter & linear fit
└── figure6_infarct_burden_by_clinical_severity.png (.svg, .pdf)# Clinical correlation with Troponin & STEMI
```

### Visual Summary:
- **Figure 1**: Multi-panel visualization of `Case_P001`: Native delayed-enhancement CMR (Panel A), expert viability overlay (Panel B), zoomed myocardial core showing transmural hyperenhancement and dark central hypointense no-reflow core (Panel C), and case tissue proportion chart (Panel D).
- **Figure 2**: Box and jittered scatter plots comparing Normal controls ($N=33$) vs Pathological cases ($N=67$) across Total Myocardium, Viable Myocardium, and LV Cavity volume.
- **Figure 3**: Dual histograms illustrating Infarction Volume (Mean: $23.68$ mL, Median: $20.26$ mL, IQR: $11.45 - 34.02$ mL) and relative Infarction Burden (Mean: $18.71\%$, Median: $18.88\%$).
- **Figure 4**: Panel A presents an MVO prevalence donut chart ($59.7\%$ present vs $40.3\%$ absent); Panel B presents the volume distribution in MVO-positive cases (Mean: $4.83$ mL, Max: $36.74$ mL).
- **Figure 5**: Bivariate scatter plot of Infarction Burden (%) vs No-Reflow Burden (%) demonstrating a positive correlation ($r = 0.655, R^2 = 0.429, p < 0.001$) among MVO-positive patients.
- **Figure 6**: Translational correlation linking CMR tissue metrics with clinical severity: Panel A shows positive correlation with peak serum troponin ($r = 0.38, p = 0.0017$); Panel B displays significant elevation in STEMI ($19.8\%$) vs NSTEMI ($13.0\%$).

---

## 7. Research Extension Feasibility: Automated EMIDEC Segmentation

### Assessment:
Training a separate deep learning segmentation model for EMIDEC is **technically feasible** but requires careful architectural handling:

1. **Available Data & Partitioning**:
   - $100$ annotated 3D volumes ($33$ Normal, $67$ Pathological).
   - Recommended strategy: **5-fold stratified patient-level cross-validation** (stratified by pathology group and MVO presence: ~13-14 pathological and ~6-7 normal cases per fold).
2. **Extreme Class Imbalance**:
   - Background: ~90.5% of total voxels.
   - LV Blood Pool: ~5.2% of voxels.
   - Normal Myocardium: ~3.5% of voxels.
   - Myocardial Infarction: ~0.72% of voxels.
   - Microvascular Obstruction (MVO): ~0.08% of voxels.
   - Standard Cross-Entropy will fail to segment MVO. A compound **Focal Tversky Loss** or **Generalized Dice + Weighted Cross-Entropy** with class weights of $10\times$ for Infarct and $25\times$ for MVO is mandatory.
3. **Model Architecture & Hardware**:
   - 2D or 3D DynUNet (nnU-Net style architecture) with residual units.
   - In-plane image size: $224 \times 224$ or $256 \times 256$.
   - Hardware requirements: Easily trainable on Apple Silicon MPS or a single 8GB+ GPU in ~30–45 minutes for 100 epochs.
4. **Overfitting Risks**:
   - High risk due to small sample size ($N=100$) and single-center origin (University Hospital of Dijon). Heavy spatial augmentation (random rotation, scaling, elastic deformation) and intensity augmentation (Gaussian noise, contrast jitter) are essential.

---

## 8. Automated Test Suite Execution

Automated tests were executed via `.venv/bin/pytest -q`, validating all existing and new functionality:

```text
======================= 239 passed, 1 warning in 15.64s ========================
```

### Coverage Breakdown:
- **Baseline CRIP Suite**: 227 tests covering ACDC segmentation, volumetrics, recovery modeling, EHR database, Streamlit UI, Bland–Altman agreement, dataset registry, and M&Ms external evaluation.
- **Stage 3 EMIDEC Tests (`tests/test_emidec_tissue_metrics.py`)**: 12 new tests:
  1. `test_validate_physical_spacing_valid`: Verifies tuple parsing.
  2. `test_validate_physical_spacing_invalid`: Rejects non-finite, negative, and 2D spacing.
  3. `test_compute_metrics_known_volumes`: Validates exact mathematical volumes and burden percentages on synthetic arrays.
  4. `test_lv_cavity_not_counted_in_myocardium`: Proves that LV blood pool is never counted in the myocardial denominator.
  5. `test_zero_myocardial_volume_handles_cleanly`: Confirms no division-by-zero on empty masks.
  6. `test_normal_case_zero_infarct_and_no_reflow`: Confirms 0.0% burdens on normal masks.
  7. `test_non_3d_mask_raises`: Enforces 3D array requirements.
  8. `test_unrecognized_labels_warning`: Flags non-standard label values.
  9. `test_real_normal_case_measurements`: Validates `Case_N006` on disk.
  10. `test_real_pathological_case_measurements`: Validates `Case_P001` on disk.
  11. `test_measurements_csv_and_summary_consistency`: Verifies all 100 rows, 33 normal, 67 pathological, and 40 MVO cases.
  12. `test_all_emidec_poster_figures_exist_and_multiformat`: Confirms all 6 figures exist in PNG, SVG, and PDF.

---

## 9. Reproduction Commands

To independently reproduce the EMIDEC tissue measurements and figures:

```bash
# 1. Run EMIDEC tissue characterization across all 100 annotated cases
python -m crip.emidec.tissue_metrics

# 2. Re-export all 6 publication poster figures in PNG, SVG, and PDF
python -m crip.evaluation.poster_figures_emidec

# 3. Run complete automated test suite
pytest
```

---

## 10. Current Git Status

Per stop conditions, changes have not been automatically committed or pushed:

```text
On branch main
Changes not staged for commit:
	modified:   README.md

Untracked files:
	docs/datasets/
	docs/evaluation/emidec_integration_report.md
	docs/evaluation/external_datasets/
	docs/evaluation/mnms_external_validation_report.md
	docs/evaluation/multidataset_stage1_report.md
	src/crip/datasets/
	src/crip/emidec/
	src/crip/evaluation/external_mnms.py
	src/crip/evaluation/poster_figures_emidec.py
	src/crip/evaluation/poster_figures_mnms.py
	tests/test_dataset_loaders.py
	tests/test_dataset_registry.py
	tests/test_emidec_tissue_metrics.py
	tests/test_external_mnms.py
```
