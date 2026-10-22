# CRIP Architecture & System Design

## Overview

The **Cardiomyopathy Recovery Intelligence Platform (CRIP)** is an investigative, clinical AI research proof of concept. Its primary goal is to study interpretable machine learning approaches for 6-month left ventricular ejection fraction (LVEF) recovery trajectories in individuals with cardiomyopathy linked to toxic alcohol exposure, Titin (*TTN*) genetic variants, and cardiac tissue fibrosis.

> [!CAUTION]
> **RESEARCH PROOF OF CONCEPT — NOT A CLINICAL TOOL**
> This application is strictly an investigative prototype and must NEVER be used for clinical diagnosis, patient risk stratification, or treatment decisions. Synthetic predictions must never be represented as clinically validated patient probabilities.

---

## Modular Component Structure

The platform is partitioned into discrete, decoupled packages:

```
src/crip/
├── __init__.py           # Package version and research disclaimer
├── app/                  # Streamlit graphical interface
│   ├── main.py           # Multi-page application entrypoint
│   ├── components/       # Reusable UI widgets (e.g. disclaimer)
│   └── pages/            # Multi-page views (Profile, MRI, Recovery, Evaluation)
├── mri/                  # Medical imaging pipelines (NiBabel, SimpleITK, MONAI)
├── synthetic/            # Patient schemas, clinical constraints, synthetic cohort generation
├── modeling/             # Interpretable predictive modeling & SHAP explainability
├── evaluation/           # Statistical metrics (ROC-AUC, Brier score, calibration curves)
└── ehr/                  # Local SQLite simulation for synthetic records (no external connections)
```

### Component Responsibilities

1. **`crip.synthetic.schemas`**:
   - Defines strongly-typed Pydantic schemas enforcing clinical consistency:
     - $ESV < EDV$
     - Consistency between volume-derived LVEF ($\frac{EDV - ESV}{EDV} \times 100\%$) and reported baseline LVEF
     - Biological bounds (age 18–105, LVEF 5–85%, PEth $\ge 0\text{ ng/mL}$)
     - Standardized clinical categorizations for Titin variants (*TTNtv*, VUS, negative) and Late Gadolinium Enhancement (LGE) fibrosis.

2. **`crip.ehr.db`**:
   - Manages a zero-dependency, local SQLite database for synthetic patient profiles.
   - Enforces isolation: strictly no real patient data, no external cloud dependencies, and no remote EHR interfaces.

3. **`crip.app`**:
   - Interactive Streamlit dashboard supporting four research workflows:
     - **Patient Profile**: Inspect synthetic demographics, biomarkers, genetics, and MRI volumetrics.
     - **MRI Analysis**: Cardiac MRI contouring and volumetric analysis preview.
     - **Recovery Estimate**: Multimodal feature attribution design (PEth + genetics + CMR).
     - **Evaluation**: Calibration, discrimination, and subgroup metrics protocol.

4. **`crip.mri` (Milestone 2+)**:
   - Preprocessing and deep learning segmentation using PyTorch and MONAI for short-axis cine stacks.

5. **`crip.modeling` (Milestone 3+)**:
   - Transparent tabular modeling (Scikit-learn ElasticNet / Random Forest) with SHAP attribution.

6. **`crip.evaluation` (Milestone 4+)**:
   - Empirical model calibration and validation on synthetic cohorts.
