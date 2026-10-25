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

### 3. `crip.mri` (Milestone 2A & 2B)
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
- **`pipeline.py`**:
  - Batch cohort extraction and statistical profiling.
  - Exports validated metrics to `data/processed/acdc/cardiac_measurements.csv`, `cardiac_measurements.json`, and `measurement_validation_report.json`.

### 4. `crip.app`
- Interactive Streamlit dashboard supporting four research workflows:
  - **Patient Profile**: Inspect synthetic demographics, biomarkers, genetics, and MRI volumetrics.
  - **MRI Analysis**: Real ACDC CMR visualization, side-by-side original/expert mask comparison, biventricular metric cards, ED/ES volume comparison, and cohort context scatter chart.
  - **Recovery Estimate**: Multimodal feature attribution design (PEth + genetics + CMR).
  - **Evaluation**: Calibration, discrimination, and subgroup metrics protocol.

### 5. `crip.modeling` (Milestone 3+)
- Transparent tabular modeling (Scikit-learn ElasticNet / Random Forest) with SHAP attribution.

### 6. `crip.evaluation` (Milestone 4+)
- Empirical model calibration and validation on synthetic cohorts.

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
| **Milestone 2C** | Automated U-Net Segmentation | Queued | MONAI/PyTorch U-Net inference, Dice/HD95 evaluation against expert masks |
| **Milestone 3** | Interpretable Predictive Modeling | Queued | ElasticNet/Ensembles, SHAP feature attributions, recovery trajectories |
| **Milestone 4** | Model Calibration & Validation | Queued | Brier scores, ROC-AUC, subgroup fairness analysis, clinical reporting |
