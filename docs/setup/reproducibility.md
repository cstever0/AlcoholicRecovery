# Reproducibility Protocol & Reference Verification Guide

This document records the exact computational parameters and baseline metrics for reproducing experimental findings in the Cardiomyopathy Recovery Intelligence Platform (CRIP).

---

## 1. Random Seeds & Determinism

- **Synthetic Cohort Generation:** Seed = `42`
- **ACDC Stratified Patient Splitting:** Seed = `42`
- **MONAI 2D U-Net Training:** Seed = `42`
- **Scikit-learn Recovery Pipeline Split & Fit:** Seed = `42`

---

## 2. Reconciled Reference Metrics

### A. MRI Volumetric & Segmentation Accuracy (Held-Out Test Set, $N=20$)

| Metric | Reference Value | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Mean Absolute LVEF Error** | **3.86 percentage points** | $\le 5.0$ percentage points | **MET** |
| **Median Absolute LVEF Error** | **3.14 percentage points** | — | Verified |
| **LVEF RMSE** | **4.99 percentage points** | — | Verified |
| **Cases Meeting $\le 5.0\%$ Error Target** | **75.0%** (15 / 20) | $\ge 70.0\%$ | **MET** |
| **Mean Foreground Spatial Dice** | **0.839** | $\ge 0.80$ | **MET** |
| **LV Cavity Dice** | **0.906** | $\ge 0.85$ | **MET** |
| **RV Cavity Dice** | **0.807** | $\ge 0.75$ | **MET** |
| **Myocardium Dice** | **0.804** | $\ge 0.75$ | **MET** |
| **LVEDV Volumetric MAE** | **6.29 mL** | — | Verified |
| **LVESV Volumetric MAE** | **5.86 mL** | — | Verified |
| **LVSV Volumetric MAE** | **5.76 mL** | — | Verified |

---

### B. Recovery Prediction Discrimination & Calibration (Held-Out Test Set, $N=419$)

| Metric | Reference Value | 95% Confidence Interval | Target Benchmark | Status |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | **0.9565** | $[0.9264, 0.9802]$ | $\ge 0.800$ | **MET** |
| **PR-AUC** | **0.9835** | $[0.9678, 0.9940]$ | — | Verified |
| **Brier Score** | **0.0694** | $[0.0524, 0.0879]$ | $< 0.150$ | **MET** |
| **Accuracy** | **89.26%** | $[86.16\%, 92.12\%]$ | — | Verified |
| **Sensitivity** | **89.94%** | $[86.27\%, 93.31\%]$ | — | Verified |
| **Specificity** | **86.81%** | $[79.12\%, 93.41\%]$ | — | Verified |

---

### C. Demonstration Case Deterministic Reference Outputs

| Demonstration Scenario | Input Patient & Profiles | Recovery Probability | Propensity Tier | Sensitivity Bounds |
| :--- | :--- | :---: | :---: | :---: |
| **Case A (`CRIP-DEMO-CASE-A`)** | Real ACDC `patient001` (DCM), PEth $420\text{ ng/mL}$, wildtype *TTN*, negative LGE | **99.62%** ($0.9962$) | `HIGH_PROBABILITY_RECOVERY` | $[99.6\%, 99.6\%]$ |
| **Case B (`CRIP-DEMO-CASE-B`)** | Real ACDC `patient002` (DCM), PEth $140\text{ ng/mL}$, pathogenic *TTNtv*, midwall LGE $12.5\%$ | **0.02%** ($0.0002$) | `LOW_PROBABILITY_RECOVERY` | $[0.02\%, 0.02\%]$ |
| **Case C (`CRIP-DEMO-CASE-C`)** | Real ACDC `patient003` (DCM), unperformed PEth, untested *TTN*, unavailable LGE | **80.90%** ($0.8090$) | `HIGH_PROBABILITY_RECOVERY` | $[0.0\%, 95.4\%]$ |

---

## 3. Cohort Sample Size Accounting

1. **Synthetic Population:**
   - 3,000 total generated records.
   - 70% nominal abstinence probability yielding 2,095 eligible sustained-abstinence cases under seed 42.
   - 60/20/20 train/validation/test partition:
     - Training: 1,257 patients (60.0%)
     - Validation: 419 patients (20.0%)
     - Held-out Test: 419 patients (20.0%)

2. **Real ACDC Cohort:**
   - 100 annotated training scans (DCM: 20, HCM: 20, MINF: 20, NOR: 20, RV: 20).
   - 60 train / 20 validation / 20 held-out test scans (4 per diagnosis).
   - 50 testing scans reserved for challenge benchmark evaluations.
