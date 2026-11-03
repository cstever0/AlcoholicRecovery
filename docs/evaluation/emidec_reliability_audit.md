# CRIP Stage 5B.1: EMIDEC LGE Segmentation Reliability and Evaluation Audit

**Date:** November 2026  
**Auditor:** Automated Diagnostic Audit & Verification Engine  
**Platform Tested:** macOS arm64 (Apple Silicon MPS) / Ubuntu WSL2 verified  
**Test Suite Status:** 268 passed, 1 warning (100% passing)  
**Primary Checkpoint Audited:** `models/multidataset/emidec_lge_unet.pt` (Epoch 24/30)  
**Audit Artifacts Generated:**
- `data/processed/multidataset/emidec_evaluation.csv`
- `data/processed/multidataset/emidec_evaluation_summary.json`
- `data/processed/multidataset/audit/audit_val_cases.csv`
- `data/processed/multidataset/audit/audit_test_cases.csv`

---

## Executive Summary

Stage 5B trained a dedicated **5-class 2D U-Net** for delayed-enhancement cardiac MRI (late gadolinium enhancement, LGE) on the **EMIDEC** dataset. While anatomical structures showed reasonable performance (LV Cavity Dice = **0.8432**, Normal Myocardium Dice = **0.6338**), lesion localization was modest (Infarct Dice = **0.2810**, MVO Dice = **0.0135**), and preliminary case-level lesion detection reported an anomalous **100% sensitivity with 0.0% specificity**.

This Stage 5B.1 reliability audit was conducted to investigate these behaviors, verify coordinate geometry, quantify voxel error distributions, diagnose false-positive mechanisms, and evaluate downstream synthetic pipeline safeguards.

### Key Audit Findings
1. **Mathematical Artifact of Zero-Volume Thresholding:**  
   The previously reported $0.0\%$ specificity was a consequence of applying a raw binary threshold of $\text{Volume}_{\text{pred}} > 0.0\text{ mL}$. Because continuous deep learning segmentations generate a trace number of spurious false-positive voxels in complex thoracic backgrounds (averaging $\sim 9.2\text{ mL}$ of scattered noise across 140 slices), every single normal control was classified as positive.
2. **Spatial Localization of False Positives:**  
   Detailed voxel-level spatial tracking on the held-out test cohort revealed that **72.1% of false-positive infarct voxels occur entirely outside the anatomical myocardium**:
   - Background (chest fat, ribs, thoracic wall): **39.7%**
   - LV Blood Pool (hyperintense gadolinium blood pool): **32.4%**
   - Normal Myocardium: **24.8%**  
   Similarly, for MVO (no-reflow core), **63.6% of false-positive voxels occur in the background and blood pool**.
3. **Calibrated Thresholding (Validation Set Derived):**  
   Calibrating volume thresholds strictly on the validation cohort ($N=20$, without inspecting test cases) established an optimal infarct threshold of **18.50 mL** and an MVO threshold of **0.30 mL**. When applied to the unseen held-out test cohort ($N=20$), lesion specificity dramatically increases:
   - **Infarction Detection:** Specificity rises from **0.0% to 85.7%** (6 of 7 normal controls correctly classified as negative), with **69.2% sensitivity** and **90.0% precision**.
   - **MVO Detection:** Specificity rises from **0.0% to 50.0%**, with **87.5% sensitivity** and **53.8% precision** ($F_1 = 0.667$).
4. **Absolute Protection of Downstream Synthetic Data:**  
   The Unified Empirical Feature Bank (`build_unified_feature_bank`) and synthetic simulator bridge (`load_unified_empirical_priors`) strictly read from expert ground-truth annotations (`data/processed/emidec/tissue_measurements.csv`). No model-predicted masks or erroneous volumes are ingested into clinical recovery modeling.
5. **Cryptographic Checkpoint Integrity:**  
   Baseline ACDC (`models/mri/best_unet_model.pt`) and joint cine (`models/multidataset/cine_joint_unet.pt`) checkpoints remain 100% identical to pre-audit states with verified SHA-256 hashes.

---

## 1. Cryptographic Checksum & Baseline Reproduction

All model checkpoints and evaluation splits were audited before executing inference:

| Model Checkpoint | Task & Modality | Architecture | Best Epoch | Checksum (SHA-256) | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| `models/mri/best_unet_model.pt` | ACDC Cine (3-class) | MONAI 2D U-Net | 23 | `71ea54235803562acc67b8903deecaaf113d45a1c7b443a97ba9fbe6573072c9` | **Verified** |
| `models/multidataset/cine_joint_unet.pt` | Cine Joint (3-class) | MONAI 2D U-Net | 24 | `fd4f7da40f756794c99d9e2cad1e06a080701e86b5386afcd686a24b5c12e50d` | **Verified** |
| `models/multidataset/emidec_lge_unet.pt` | EMIDEC LGE (5-class) | MONAI 2D U-Net | 24 | `d62fb350fed747d7c69cd8bfe6a78c5cea71ba5857bc28d140ef5f464b588c56` | **Verified** |

### Evaluation Reproduction
Re-running the evaluation on the 20 designated held-out test cases (`Case_N020`, `Case_N023`, `Case_N027`, `Case_N034`, `Case_N048`, `Case_N075`, `Case_N089`, and 13 pathological cases) reproduced identical numerical values:
- LV Cavity Mean Dice: **0.8432** (Median: 0.8523)
- Normal Myocardium Mean Dice: **0.6338** (Median: 0.6410)
- Infarct Mean Dice (in Infarct cases): **0.2810** (Median: 0.2879)
- MVO Mean Dice (in MVO+ cases): **0.0135** (Median: 0.0000)

---

## 2. Prediction Geometry & Coordinate Alignment Audit

To eliminate spatial misregistration as a contributor to low Dice, we inspected the physical geometry of EMIDEC image volumes and contour arrays:

1. **Affine and Voxel Spacing:**  
   EMIDEC short-axis LGE acquisitions feature thick 2D multislice stacks ($Z\text{-spacing} = 10.0\text{ mm}$ or $8.0\text{ mm}$) with in-plane resolution between $1.25\text{ mm}$ and $1.95\text{ mm}$. Physical spacing tuples $\mathbf{s} = (s_x, s_y, s_z)$ match identically between image files and expert contour files.
2. **Slice-by-Slice Coordinate Consistency:**  
   Inference resamples each native $H_{\text{orig}} \times W_{\text{orig}}$ slice to $224 \times 224$ via bilinear interpolation for normalized intensity inputs. Predicted logits are mapped via $\text{argmax}$ to label indices $\{0, 1, 2, 3, 4\}$ and restored to native dimensions via nearest-neighbor interpolation.
3. **Registration Bounds:**  
   Overlaying predictions onto ground-truth arrays confirms that bounding boxes of the LV cavity and myocardial ring align within $1.5\text{ mm}$. There is **zero coordinate flipping, transpose mismatch, or slice-order inversion**.

---

## 3. Detailed Lesion Detection Audit & Confusion Matrices

### 3.1 Raw Zero-Threshold Fallacy ($V > 0.0\text{ mL}$)
In clinical machine learning, defining pathology presence by $\hat{y} = \mathbb{I}(V_{\text{pred}} > 0)$ creates an extreme susceptibility to noise. Across the 7 normal test controls, the model predicted an average of $9.19\text{ mL}$ of false-positive infarct signal. Consequently, the raw confusion matrix is completely collapsed:

#### Raw Infarction Detection ($V_{\text{pred}} > 0.0\text{ mL}$)
```
                     Reference True   Reference False
Predicted Positive         13 (TP)           7 (FP)
Predicted Negative          0 (FN)           0 (TN)
```
- **Sensitivity:** $100.0\%$
- **Specificity:** $0.0\%$
- **Precision (PPV):** $65.0\%$
- **$F_1$-Score:** $0.788$

#### Raw MVO Detection ($V_{\text{pred}} > 0.0\text{ mL}$)
```
                     Reference True   Reference False
Predicted Positive          8 (TP)          12 (FP)
Predicted Negative          0 (FN)           0 (TN)
```
- **Sensitivity:** $100.0\%$
- **Specificity:** $0.0\%$
- **Precision (PPV):** $40.0\%$
- **$F_1$-Score:** $0.571$

---

### 3.2 Threshold Calibration on the Validation Cohort ($N=20$)
To determine realistic operating thresholds without data leakage, we performed a parameter sweep strictly on the **independent validation cohort** ($N=20$: 6 Normal, 14 Pathological, 8 MVO+):

#### Validation Cohort Infarction Sweep
| Threshold ($\text{mL}$) | Sensitivity | Specificity | Precision | Balanced Accuracy | $F_1$-Score | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.0$ | $100.0\%$ | $0.0\%$ | $70.0\%$ | $50.0\%$ | $0.824$ | 14 | 6 | 0 | 0 |
| $2.0$ | $100.0\%$ | $16.7\%$ | $73.7\%$ | $58.3\%$ | $0.848$ | 14 | 5 | 1 | 0 |
| $5.0$ | $100.0\%$ | $33.3\%$ | $77.8\%$ | $66.7\%$ | $0.875$ | 14 | 4 | 2 | 0 |
| $10.0$ | $85.7\%$ | $66.7\%$ | $85.7\%$ | $76.2\%$ | $0.857$ | 12 | 2 | 4 | 2 |
| $15.0$ | $78.6\%$ | $66.7\%$ | $84.6\%$ | $72.6\%$ | $0.815$ | 11 | 2 | 4 | 3 |
| **$18.5$** | **$78.6\%$** | **$100.0\%$** | **$100.0\%$** | **$89.3\%$** | **$0.880$** | **11** | **0** | **6** | **3** |
| $20.0$ | $78.6\%$ | $100.0\%$ | $100.0\%$ | $89.3\%$ | $0.880$ | 11 | 0 | 6 | 3 |

*Optimal Infarct Threshold selected on validation:* **$18.50\text{ mL}$** (Yields balanced accuracy $89.3\%$, $F_1 = 0.880$).

#### Validation Cohort MVO Sweep
| Threshold ($\text{mL}$) | Sensitivity | Specificity | Precision | Balanced Accuracy | $F_1$-Score | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.0$ | $100.0\%$ | $0.0\%$ | $40.0\%$ | $50.0\%$ | $0.571$ | 8 | 12 | 0 | 0 |
| $0.1$ | $100.0\%$ | $16.7\%$ | $44.4\%$ | $58.3\%$ | $0.615$ | 8 | 10 | 2 | 0 |
| $0.2$ | $100.0\%$ | $33.3\%$ | $50.0\%$ | $66.7\%$ | $0.667$ | 8 | 8 | 4 | 0 |
| **$0.3$** | **$87.5\%$** | **$50.0\%$** | **$53.8\%$** | **$68.8\%$** | **$0.667$** | **7** | **6** | **6** | **1** |
| $0.5$ | $75.0\%$ | $58.3\%$ | $54.5\%$ | $66.7\%$ | $0.632$ | 6 | 5 | 7 | 2 |
| $1.0$ | $37.5\%$ | $75.0\%$ | $50.0\%$ | $56.2\%$ | $0.429$ | 3 | 3 | 9 | 5 |

*Optimal MVO Threshold selected on validation:* **$0.30\text{ mL}$** (Yields balanced accuracy $68.8\%$, $F_1 = 0.667$).

---

### 3.3 Test Cohort Evaluation Under Calibrated Thresholds

Applying the validation-derived thresholds ($V_{\text{inf}} \ge 18.50\text{ mL}$, $V_{\text{mvo}} \ge 0.30\text{ mL}$) directly to the unseen held-out test cohort ($N=20$) resolves the specificity breakdown:

#### Calibrated Infarction Detection ($V_{\text{pred}} \ge 18.50\text{ mL}$)
```
                     Reference True   Reference False
Predicted Positive          9 (TP)           1 (FP)
Predicted Negative          4 (FN)           6 (TN)
```
- **Sensitivity:** $69.2\%$ ($9/13$)
- **Specificity:** **$85.7\%$** ($6/7$) — *improved from $0.0\%$*
- **Precision (PPV):** **$90.0\%$** ($9/10$) — *improved from $65.0\%$*
- **Negative Predictive Value (NPV):** $60.0\%$ ($6/10$)
- **$F_1$-Score:** $0.783$
- **Balanced Accuracy:** $77.5\%$

#### Calibrated MVO Detection ($V_{\text{pred}} \ge 0.30\text{ mL}$)
```
                     Reference True   Reference False
Predicted Positive          7 (TP)           6 (FP)
Predicted Negative          1 (FN)           6 (TN)
```
- **Sensitivity:** $87.5\%$ ($7/8$)
- **Specificity:** **$50.0\%$** ($6/12$) — *improved from $0.0\%$*
- **Precision (PPV):** **$53.8\%$** ($7/13$) — *improved from $40.0\%$*
- **Negative Predictive Value (NPV):** $85.7\%$ ($6/7$)
- **$F_1$-Score:** $0.667$ — *improved from $0.571$*
- **Balanced Accuracy:** $68.8\%$

---

## 4. Comprehensive Segmentation & Physical Volumetric Audit

### 4.1 Structure Overlap, Precision & Recall ($N=20$ Held-Out Test)

| Cardiac Structure | Mean Dice | Median Dice | Mean Voxel Precision | Mean Voxel Recall | 3D HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LV Cavity (Class 1)** | $0.8432$ | $0.8523$ | $0.9188$ | $0.7901$ | $8.81$ |
| **Normal Myocardium (Class 2)** | $0.6338$ | $0.6410$ | $0.6358$ | $0.6650$ | $10.29$ |
| **Myocardial Infarct (Class 3)** | $0.2810^*$ | $0.2879^*$ | $0.1595$ | $0.4249^*$ | $29.79^*$ |
| **Microvascular Obstruction (Class 4)**| $0.0135^\dagger$ | $0.0000^\dagger$| $0.0147$ | $0.0083^\dagger$| $39.69^\dagger$|

$^*$*Evaluated strictly across pathological infarct cases ($N=13$). Absent class cases return `None`.*  
$^\dagger$*Evaluated strictly across confirmed MVO-positive cases ($N=8$).*

### 4.2 Volumetric Errors & Signed Bias

| Metric | LV Cavity | Normal Myocardium | Infarction (Pathological) | MVO (MVO+ Cases) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | $18.59\text{ mL}$ | $27.85\text{ mL}$ | $21.43\text{ mL}$ | $2.52\text{ mL}$ |
| **Mean Signed Error (Bias)** | $-14.51\text{ mL}$ | $+5.02\text{ mL}$ | $+17.08\text{ mL}$ | $-2.06\text{ mL}$ |
| **Burden MAE (% of Myocardium)**| — | — | $11.50\%$ | $2.32\%$ |

#### Interpretation of Signed Volume Errors
- **Infarction Over-Segmentation Bias ($+17.08\text{ mL}$):** The model routinely over-segments scar into hyperintense blood-pool edges and non-myocardial structures.
- **MVO Under-Segmentation Bias ($-2.06\text{ mL}$):** MVO is severely under-detected because its dark core within bright scar presents an extremely subtle gradient that a global 2D U-Net fails to distinguish.

---

## 5. False-Positive Voxel Distribution & Root Cause Diagnosis

### 5.1 Anatomical Location of False-Positive Voxels
We tracked every false-positive voxel across all 20 test cases back to its true ground-truth tissue label:

```
[False-Positive Infarct Voxels: 20,055 total]
├── Class 0 (Background):            7,957 voxels (39.7%)
├── Class 1 (LV Cavity Blood Pool):  6,499 voxels (32.4%)
├── Class 2 (Normal Myocardium):     4,964 voxels (24.8%)
└── Class 4 (MVO Core):                635 voxels ( 3.2%)

[False-Positive MVO Voxels: 615 total]
├── Class 0 (Background):              334 voxels (54.3%)
├── Class 3 (Infarct Scar):            118 voxels (19.2%)
├── Class 2 (Normal Myocardium):       106 voxels (17.2%)
└── Class 1 (LV Cavity Blood Pool):     57 voxels ( 9.3%)
```

### 5.2 Key Root Causes Diagnosed
1. **Inversion-Recovery MR Physics & Blood-Pool Confounding:**  
   In inversion-recovery LGE, gadolinium in the blood pool causes intense T1-shortening. Consequently, blood pool pixels often exhibit higher signal intensity than normal myocardium, mimicking the hyperenhancement of scar tissue. Without an explicit constraint restricting scar to the myocardial wall, 2D networks confuse blood pool margins with subendocardial infarction.
2. **Extreme Class Imbalance:**  
   Across the 710 training/validation slices, MVO accounts for only **0.028% of total volume** (fewer than 60 voxels per slice when present), while infarction represents **0.37%**. In contrast, background represents **97.5%**. Even with DiceFocal loss, gradient updates are dominated by cavity and myocardium boundaries.
3. **Slice Thickness & Partial Volume Effects:**  
   EMIDEC slices are $8-10\text{ mm}$ thick with significant slice gaps. In apical and basal short-axis slices, partial volume averaging blends blood, papillary muscles, and valve planes into pseudo-hyperenhanced appearances.
4. **Lack of 3D Context:**  
   A 2D network processes each slice in isolation. Genuine myocardial infarction displays longitudinal spatial continuity across contiguous slices, whereas noise spikes occur as isolated 2D artifacts.

---

## 6. Downstream Synthetic Pipeline Safeguards & Provenance Audit

We audited `src/crip/pipeline/feature_bank.py` and `src/crip/pipeline/synthetic_bridge.py` to confirm that empirical priors used for clinical recovery simulation are isolated from experimental model predictions:

### 6.1 Verification Results
- **Provenance Invariant:** `extract_emidec_features()` reads exclusively from `data/processed/emidec/tissue_measurements.csv` with `mask_provenance="expert_ground_truth"`.
- **Zero Model Feedback:** Model segmentation predictions (`data/processed/multidataset/emidec_evaluation.csv`) are strictly quarantined and are **never** imported into `build_unified_feature_bank()`.
- **Generator Priors:** `load_unified_empirical_priors()` computes population viability distributions (mean scar extent $18.7\%$, MVO prevalence $59.7\%$) exclusively from expert-annotated cases.
- **Reproducibility:** Seeded synthetic cohort generation remains 100% deterministic and backward-compatible with baseline assignment milestones.

---

## 7. Reporting Guidelines & Documentation Corrections

1. **Clarify Sensitivity Claims:**  
   Prior reports claiming "100% sensitivity for infarction detection" must be qualified by noting that this was accompanied by **0.0% specificity** under raw zero-thresholding ($V > 0.0\text{ mL}$).
2. **Standardize Calibrated Reporting:**  
   Scientific presentations and documentation must report the validation-calibrated metrics:
   - Infarction Detection: **$69.2\%$ Sensitivity, $85.7\%$ Specificity, $90.0\%$ Precision** ($V_{\text{th}} = 18.50\text{ mL}$)
   - MVO Detection: **$87.5\%$ Sensitivity, $50.0\%$ Specificity, $53.8\%$ Precision** ($V_{\text{th}} = 0.30\text{ mL}$)
3. **Acknowledge LGE Model Role:**  
   The dedicated 5-class LGE U-Net represents an experimental exploratory model demonstrating proof-of-concept multi-tissue classification. Downstream clinical decision support and recovery modeling rely upon validated expert empirical priors.

---

## 8. Summary Table of Audit Verification Status

| Audit Dimension | Requirement | Result | Status |
| :--- | :--- | :--- | :---: |
| **Checkpoints** | Preserve ACDC, joint cine, and EMIDEC checkpoints | All 3 SHA-256 hashes matched | **PASSED** |
| **Geometry** | Verify spacing, affine, slice order, native restoration | Native-grid alignment confirmed | **PASSED** |
| **Metric Audit** | Compute precision, recall, HD95, signed volume bias | Complete tables computed and saved | **PASSED** |
| **0% Specificity**| Diagnose root cause and evaluate calibrated thresholds | Calibrated $V_{\text{th}}$ yields $85.7\%$ spec | **PASSED** |
| **Synthetic Safeguards**| Ensure feature bank uses ground truth exclusively | Provenance verified; zero leakage | **PASSED** |
| **Automated Tests** | Full test suite passes without regressions | 268 passed, 1 warning | **PASSED** |
