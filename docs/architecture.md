# CRIP Architecture & System Design

## Overview

The **Cardiomyopathy Recovery Intelligence Platform (CRIP)** is an investigative, clinical AI research proof of concept. Its primary goal is to study interpretable machine learning approaches for 6-month left ventricular ejection fraction (LVEF) recovery trajectories in individuals with cardiomyopathy linked to toxic alcohol exposure, Titin (*TTN*) genetic variants, and cardiac tissue fibrosis.

> [!CAUTION]
> **RESEARCH PROOF OF CONCEPT — NOT A CLINICAL TOOL**  
> This application is strictly an investigative prototype and must NEVER be used for clinical diagnosis, patient risk stratification, or treatment decisions. Synthetic predictions or derived cardiac measurements must never be represented as clinically validated patient probabilities.

---

## Modular Component Structure

The platform is partitioned into discrete, decoupled packages:

```text
src/crip/
├── __init__.py           # Package version and research disclaimer
├── app/                  # Streamlit graphical interface
│   ├── main.py           # Multi-page application entrypoint
│   ├── components/       # Reusable UI widgets (e.g. disclaimer)
│   └── pages/            # Multi-page views (Profile, MRI, Recovery, Evaluation)
├── mri/                  # Medical imaging pipelines (NiBabel, SimpleITK, MONAI)
│   ├── acdc_loader.py    # ACDC NIfTI data loading and coordinate alignment
│   ├── cardiac_metrics.py# Physical volumetric and ejection fraction calculations
│   ├── inspect_acdc.py   # Dataset validation and integrity inspection CLI
│   └── pipeline.py       # Patient and cohort-level measurement pipeline
├── synthetic/            # Patient schemas, clinical constraints, synthetic cohort generation
├── modeling/             # Interpretable predictive modeling & SHAP explainability
├── evaluation/           # Statistical metrics (ROC-AUC, Brier score, calibration curves)
└── ehr/                  # Local SQLite simulation for synthetic records (no external connections)
```

---

## Component Responsibilities & Implementation

### 1. `crip.synthetic.schemas`
- Defines strongly-typed Pydantic schemas enforcing clinical consistency:
  - $ESV < EDV$
  - Consistency between volume-derived LVEF ($\frac{EDV - ESV}{EDV} \times 100\%$) and reported baseline LVEF
  - Biological bounds (age 18–105, LVEF 5–85%, PEth $\ge 0\text{ ng/mL}$)
  - Standardized clinical categorizations for Titin variants (*TTNtv*, VUS, negative) and Late Gadolinium Enhancement (LGE) fibrosis.

### 2. `crip.ehr.db`
- Manages a zero-dependency, local SQLite database for synthetic patient profiles.
- Enforces isolation: strictly no real patient data, no external cloud dependencies, and no remote EHR interfaces.

### 3. `crip.mri` & `crip.evaluation` (Milestones 2A, 2B & 2C)
- **`acdc_loader.py`**:
  - Encapsulates 3D cine frame loading (`ACDCCardiacFrame`), 4D cine series (`ACDCCineSeries`), and patient metadata (`ACDCPatientMetadata`).
  - Standardizes spatial coordinate ordering to `(num_slices, height, width)`:
    - Axis 0 ($z$): Short-axis stack (ventricular base to apex).
    - Axis 1 ($y$): Row height.
    - Axis 2 ($x$): Column width.
  - Normalizes intensity arrays via robust 1st–99th percentile windowing for uniform UI contrast.
- **`cardiac_metrics.py`**:
  - Calculates physical chamber volumes for Left Ventricle (LV, label 3), Right Ventricle (RV, label 1), and Myocardium (label 2).
  - Voxel-to-Volume Formulation:
    $$V_{\text{cavity}} = N_{\text{voxels}} \times \frac{\Delta x \times \Delta y \times \Delta z}{1000.0}\text{ mL}$$
    Accounting for non-isotropic pixel spacing ($\Delta x, \Delta y$) and slice spacing ($\Delta z$).
  - Stroke Volume & Ejection Fraction:
    $$SV = EDV - ESV$$
    $$EF = \frac{EDV - ESV}{EDV} \times 100.0\%$$
  - Estimated Myocardial Mass:
    $$\text{Mass}_{\text{myo}} = V_{\text{myo}}(\text{mL}) \times 1.05\text{ g/mL}$$
    Based on standard AHA/SCMR density consensus (Cerqueira et al., *Circulation* 2002).
- **`dataset_splits.py` (Milestone 2C)**:
  - Partitions the 100 ACDC training cases at the **patient level** into 60 train, 20 validation, and 20 held-out test cases.
  - Stratifies across DCM, HCM, MINF, NOR, and RV (12/4/4 each).
  - Serializes manifests to `data/processed/acdc/dataset_split.json`.
- **`preprocessing.py` (Milestone 2C)**:
  - 1st-99th percentile intensity windowing and scaling to $[0.0, 1.0]$.
  - Bilinear interpolation for image slices to $224 \times 224$.
  - Strict **nearest-neighbor** interpolation for segmentation masks, guaranteeing preservation of discrete integer labels $\{0, 1, 2, 3\}$.
  - Inverse spatial resizing back to native short-axis resolution.
- **`segmentation_dataset.py` (Milestone 2C)**:
  - PyTorch Dataset extracting short-axis slices from ED and ES frames.
  - Online geometric and intensity augmentations (random flips, 90-degree rotations, gamma jitter) preserving label boundaries.
- **`segmentation_model.py` (Milestone 2C)**:
  - MONAI 2D U-Net architecture (`in_channels=1`, `out_channels=4`, `channels=(16, 32, 64, 128)`).
  - Multi-tier device selection: CUDA $\to$ Apple Silicon MPS $\to$ CPU fallback.
  - `CompoundDiceCELoss`: hybrid loss balancing foreground Dice with multi-class Cross-Entropy.
- **`train_segmentation.py` (Milestone 2C)**:
  - Training loop tracking per-epoch train/val loss, RV Dice, Myocardium Dice, and LV Dice.
  - Cosine annealing learning rate scheduling and best-model checkpointing to `models/mri/best_unet_model.pt`.
- **`segmentation_inference.py` (Milestone 2C)**:
  - Slice-by-slice 2D prediction, nearest-neighbor spatial inversion, and 3D volume reconstruction.
  - End-to-end patient inference returning automated biventricular metrics and ground-truth comparison.
- **`crip.evaluation.segmentation_metrics` (Milestone 2C)**:
  - Multi-class Dice, 95th Percentile Hausdorff Distance (HD95 mm) with physical spacing.
  - Volumetric errors (LVEDV, LVESV, LVSV, LVEF MAE/RMSE) and threshold evaluation ($|\Delta \text{LVEF}| \le 5\%$).
  - Exports `data/processed/acdc/segmentation_evaluation.csv` and summary JSON.

### 4. `crip.app`
- Interactive Streamlit dashboard supporting four research workflows:
  - **Patient Profile**: Inspect synthetic demographics, biomarkers, genetics, and MRI volumetrics.
  - **MRI Analysis**: Real ACDC CMR visualization, side-by-side original/expert mask comparison, biventricular metric cards, ED/ES volume comparison, and cohort context scatter chart.
  - **Recovery Estimate**: Multimodal feature attribution design (PEth + genetics + CMR).
  - **Evaluation**: Calibration, discrimination, and subgroup metrics protocol.

### 5. `crip.synthetic.cohort_generator` & Literature Registry (Milestone 3A)
- **Literature & Assumptions Registry (`docs/research/`)**:
  - `literature_registry.csv`: 10 landmark studies detailing PEth kinetics (Ulwelling & Smith 2018), recovery rates after abstinence (Nicolas 2002, Fernández-Solà 2006, Fauchier 2000), Titin truncations (Ware et al. *NEJM* 2016), and replacement fibrosis (Gulati 2013, Halliday 2017).
  - `synthetic_cohort_assumptions.md`: Clinical simulation rules distinguishing empirical priors from engineering modeling choices.
- **Empirical MRI Grounding**:
  - Extracts structural CMR distributions from the 20 ACDC DCM cases ($\mu_{\text{LVEF}} = 18.4\%$, $\mu_{\text{LVEDV}} = 284.6\text{ mL}$, $\mu_{\text{Mass}} = 170.2\text{ g}$) to ground simulated ventricular morphology.
- **Non-Linear Recovery Modeling**:
  - Experimental primary endpoint: Binary recovery = $1$ if $\Delta \text{LVEF} \ge 10.0$ percentage points at 6-month follow-up, else $0$.
  - Multimodal reverse remodeling drivers: cessation of toxic exposure (+), replacement fibrosis (-), pathogenic TTNtv (-), severe baseline dilation (-), patient age (-).
  - Stochastic Gaussian residual noise ($\sigma = 4.25\%$ points) ensuring non-trivial multimodal separability (all univariate ROC-AUCs $< 0.82$).
- **Missingness Mechanisms**:
  - Realistic missingness rates: PEth missing (4.4%), untested TTN genetics (18.0%), contraindicated LGE (10.6%), missing self-report alcohol history (7.1%).
- **Variable-Level Provenance**:
  - Explicit audit tracking for every parameter in every record (`variable_provenance`).
- **SQLite EHR Cohort Persistence (`crip.ehr.db`)**:
  - Automated schema migration, batch insertion, multi-criteria filtering, and cohort summaries.

### 6. `crip.modeling` & `crip.evaluation` (Milestone 3B)
- **`dataset.py`**:
  - Enforces strict baseline feature whitelist (demographics, CMR volumetrics, PEth toxicology, TTN genetics, LGE fibrosis).
  - Explicit data leakage prevention: raises `ValueError` if post-baseline or future outcome variables are present.
  - Stratified 60/20/20 train/validation/test partitioning with isolated preprocessor fitting.
- **`train.py`**:
  - Scikit-learn pipelines with `ColumnTransformer` (median imputation, standard scaling, OHE).
  - Baseline classifiers: `DummyClassifier` (prior prevalence), `RandomForestClassifier` (non-linear tree benchmark), `LogisticRegression` (primary interpretable model).
  - Modality ablation experiments evaluating incremental value of CMR volumetrics, PEth toxicology, TTN genetics, and LGE fibrosis.
- **`recovery_metrics.py`**:
  - Comprehensive statistical evaluation on held-out test split: ROC-AUC, PR-AUC, Brier score, accuracy, sensitivity, specificity, F1 score.
  - Binned reliability calibration diagrams and empirical 95% bootstrap confidence intervals ($B=1,000$).
- **`explain.py`**:
  - Global standardized odds ratios ($\text{OR} = \exp(\beta)$) and directional classification.
  - Patient-level additive log-odds decomposition and plain-language narrative generation.
  - Exact linear SHAP feature attributions with strict non-causal research disclaimers.
- **`sensitivity.py`**:
  - Stress testing across stochastic noise multipliers ($0.5\times$ to $2.0\times$) and population compliance rates.
- **`artifacts.py`**:
  - Serializes model pipelines, feature schemas, patient split manifests, and evaluation summaries to `models/recovery/`.

### 7. `crip.evaluation` (Milestone 4+)
- Empirical model calibration, subgroup fairness analysis, and clinical reporting protocols.

---

## Critical Methodology & Validation Distinctions

### Segmentation Accuracy vs. Volumetric Accuracy
1. **Segmentation Accuracy (Geometric Overlap):**
   - Typically quantified by spatial metrics such as Dice Similarity Coefficient (DSC) and 95% Hausdorff Distance (HD95).
   - Measures how well spatial boundary contours match voxel-by-voxel.
2. **Volumetric & Functional Accuracy (Clinical Aggregates):**
   - Quantified by absolute volume errors ($|\Delta EDV|, |\Delta ESV|$) and ejection fraction differences ($\Delta EF = EF_{\text{pred}} - EF_{\text{ref}}$).
   - A high spatial Dice score (e.g. 0.90) does not guarantee meeting clinical volume targets (e.g. $\pm 5\%$ LVEF), because small contour errors located at basal slice boundaries (where ventricular slice areas are large) introduce disproportionate volume errors.

### Limitations of Expert-Mask Derived Measurements
- **No External Independent Clinical Standard:** The raw ACDC dataset distribution provides expert ground-truth contour masks, but does not provide independent external core-laboratory volumetric tables. Consequently, volume calculations represent voxel integrations of these contours.
- **Voxel Partial Volume Effects:** Standard short-axis acquisitions feature slice thickness of 5–10 mm with small in-plane spacing (~1.4 mm). Voxels intersecting both myocardium and blood pool introduce partial volume averaging.
- **Basal Slice Delineation:** Inclusion or exclusion of the outflow tract and mitral valve plane across the basal-most slice is the largest single source of inter-observer variance in CMR volumetry.

---

## Development Roadmap

| Milestone | Scope | Status | Deliverables |
|---|---|---|---|
| **Milestone 1** | Foundation & Architecture | **Completed** | Modular package, Pydantic schemas, SQLite EHR, Streamlit app, test suite |
| **Milestone 2A** | ACDC Ingestion & Display | **Completed** | ACDC loader, dataset inspection tool, side-by-side Streamlit visualization |
| **Milestone 2B** | Cardiac Volumetrics | **Completed** | Biventricular metrics, cohort extraction pipeline, descriptive stats & plots |
| **Milestone 2C** | Automated U-Net Segmentation | **Completed** | MONAI/PyTorch 2D U-Net pipeline, patient-level splits, Dice/HD95 & LVEF MAE evaluation, Streamlit automated mode |
| **Milestone 3A** | Literature-Grounded Multimodal Cohort | **Completed** | 3,000-patient synthetic cohort, ACDC DCM grounding, PEth/TTN/LGE integration, SQLite EHR filters, 104 tests |
| **Milestone 3B** | Interpretable Predictive Modeling | **Completed** | Logistic Regression/RF pipelines, feature ablations, test ROC-AUC 0.957, SHAP & log-odds attributions, scenario explorer, 129 tests |
| **Milestone 4** | Model Calibration & Validation | Queued | Brier scores, ROC-AUC, subgroup fairness analysis, clinical reporting |
