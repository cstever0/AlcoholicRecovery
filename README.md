# Cardiomyopathy Recovery Intelligence Platform (CRIP)

> [!WARNING]
> ### CLINICAL RESEARCH DISCLAIMER — NOT FOR DIAGNOSIS OR TREATMENT
> **This software is an experimental research proof of concept developed exclusively for academic investigation into left ventricular ejection fraction recovery patterns. It integrates synthetic patient data, simulated genetics, and exploratory cardiac imaging parameters. This platform is NOT a certified medical device and MUST NEVER be used for clinical diagnosis, patient risk stratification, or clinical treatment decisions. Synthetic predictions or derived cardiac measurements must never be represented as clinically validated patient probabilities.**

---

## 1. Project Overview

The **Cardiomyopathy Recovery Intelligence Platform (CRIP)** is a 4-week clinical AI research proof of concept. The platform investigates interpretable machine learning models predicting 6-month Left Ventricular Ejection Fraction (LVEF) recovery in cardiomyopathy associated with:

- **Toxic Alcohol Exposure:** Quantified via Phosphatidylethanol (PEth) blood biomarkers and duration of cessation.
- **Cardiac MRI Volumetrics & Fibrosis:** End-diastolic volume (EDV), end-systolic volume (ESV), stroke volume (SV), ejection fraction (EF), and late gadolinium enhancement (LGE) myocardial scar patterns.
- **Genetic Susceptibility:** Titin (*TTN*) truncating variants (*TTNtv*) and variants of uncertain significance (VUS).

---

## 2. Architecture & Directory Structure

CRIP is structured as a decoupled, modular Python library and interactive Streamlit application:

```text
AlcoholicRecovery/
├── .gitignore                          # Excludes venvs, secrets, caches, raw datasets, weights
├── .streamlit/
│   └── config.toml                     # Streamlit headless and privacy configuration
├── docs/
│   ├── architecture.md                 # Architectural specification and milestone roadmap
│   └── acdc/                           # ACDC dataset documentation & citations
│       ├── LICENSE_TERMS.md            # CC BY-NC-SA 4.0 license details
│       ├── MANDATORY_CITATION.md       # Official publication citation
│       ├── STATE-OF-THE-ART-nnUNet-method.md # nnU-Net benchmark references
│       ├── acdc_validation_report.json # Validation report for training cohort (100 cases)
│       ├── acdc_testing_validation_report.json # Validation report for testing cohort (50 cases)
│       └── figures/                    # Cohort descriptive figures (LVEF & volume distributions)
├── notebooks/
│   └── acdc/
│       └── script_acdc_visu.ipynb      # Reference visualization notebook
├── data/
│   ├── raw/
│   │   └── acdc/                       # Real ACDC dataset (excluded from Git)
│   │       ├── training/               # 100 patients (patient001 to patient100)
│   │       └── testing/                # 50 patients (patient101 to patient150)
│   ├── processed/
│   │   └── acdc/                       # Extracted measurements & reports (excluded from Git)
│   │       ├── cardiac_measurements.csv
│   │       ├── cardiac_measurements.json
│   │       ├── measurement_validation_report.json
│   │       └── figures/
│   └── synthetic/
│       └── ehr_synthetic.db            # Simulated SQLite EHR database
├── src/
│   └── crip/
│       ├── __init__.py                 # CRIP package definition and disclaimer
│       ├── app/                        # Streamlit web application
│       │   ├── __init__.py
│       │   ├── main.py                 # Multi-page application entrypoint
│       │   ├── components/
│       │   │   ├── __init__.py
│       │   │   └── disclaimer.py       # Clinical disclaimer UI banner
│       │   └── pages/
│       │       ├── 1_Patient_Profile.py# Synthetic patient exploration & EHR cohort view
│       │       ├── 2_MRI_Analysis.py   # Real ACDC CMR visualization, metrics & cohort context
│       │       ├── 3_Recovery_Estimate.py # Interpretable modeling & SHAP UI architecture
│       │       └── 4_Evaluation.py     # Statistical metrics, calibration & fairness checklist
│       ├── synthetic/                  # Synthetic clinical data schemas
│       │   ├── __init__.py
│       │   └── schemas.py              # Strongly-typed Pydantic models (PEth, TTN, LGE)
│       ├── ehr/                        # Local synthetic EHR simulation
│       │   ├── __init__.py
│       │   └── db.py                   # Local SQLite synthetic database manager & seeder
│       ├── mri/                        # Cardiac MRI processing & analysis
│       │   ├── __init__.py
│       │   ├── acdc_loader.py          # ACDC dataset ingestion & NIfTI/mask loader
│       │   ├── cardiac_metrics.py      # Quantitative volumetric & LVEF interfaces
│       │   ├── inspect_acdc.py         # Automated dataset validation CLI script
│       │   └── pipeline.py             # Batch patient & cohort measurement pipeline
│       ├── modeling/                   # Predictive modeling & SHAP (Milestone 3)
│       │   └── __init__.py
│       └── evaluation/                 # Metrics & calibration protocols (Milestone 4)
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Shared pytest fixtures & synthetic NIfTI volumes
│   ├── test_schemas.py                 # Schema validation rules and constraints
│   ├── test_ehr_db.py                  # SQLite persistence and retrieval tests
│   ├── test_app_init.py                # App compilation, navigation, and AppTest tests
│   ├── test_mri_loader.py              # Unit tests for ACDC data loader (14 tests)
│   ├── test_cardiac_metrics.py         # Unit tests for physical volume & LVEF math
│   ├── test_inspect_acdc.py            # Unit tests for dataset validation utility
│   ├── test_acdc_integration.py       # Integration tests against real ACDC dataset
│   └── test_pipeline_and_measurements.py # Comprehensive Milestone 2B measurement tests (16 tests)
├── pyproject.toml                      # Modern Python build metadata and pytest config
├── requirements.txt                    # Pinned, tested dependencies with rationale
└── README.md                           # Master setup guide, run instructions, and limitations
```

---

## 3. Cardiac Measurement Methodology (Milestone 2B)

### 3.1 Anatomical Segmentation Labels (ACDC)
- **Label 0:** Background
- **Label 1:** Right Ventricular (RV) cavity
- **Label 2:** Left Ventricular Myocardium
- **Label 3:** Left Ventricular (LV) cavity

### 3.2 Mathematical Formulation & Physical Units
All measurements are calculated using physical voxel dimensions extracted from NIfTI metadata:
1. **Physical Voxel Volume ($\text{mL}$):**
   $$V_{\text{voxel}} = \frac{\Delta x \times \Delta y \times \Delta z}{1000.0}\text{ mL} \quad (\text{since } 1000\text{ mm}^3 = 1\text{ mL})$$
   where $\Delta x, \Delta y$ represent in-plane pixel spacing (typically $1.2$–$1.8\text{ mm}$) and $\Delta z$ represents slice thickness/spacing (typically $5$–$10\text{ mm}$).
2. **Chamber Volumes ($\text{mL}$):**
   $$V_{\text{chamber}} = N_{\text{voxels}} \times V_{\text{voxel}}$$
   Calculated at End-Diastole (LVEDV, RVEDV) and End-Systole (LVESV, RVESV).
3. **Stroke Volume ($SV$, $\text{mL}$):**
   $$LVSV = LVEDV - LVESV, \quad RVSV = RVEDV - RVESV$$
4. **Ejection Fraction ($EF$, $\%$):**
   $$LVEF = \frac{LVEDV - LVESV}{LVEDV} \times 100.0\%, \quad RVEF = \frac{RVEDV - RVESV}{RVEDV} \times 100.0\%$$
5. **Estimated Myocardial Mass ($\text{g}$):**
   $$\text{Mass}_{\text{myo}} = V_{\text{myo}}(\text{mL}) \times 1.05\text{ g/mL}$$
   Using the standard human myocardial tissue density constant ($\rho = 1.05\text{ g/mL}$) established by American Heart Association (AHA) and Society for Cardiovascular Magnetic Resonance (SCMR) consensus guidelines (Cerqueira et al., *Circulation* 2002; Schulz-Menger et al., *JCMR* 2020).

---

## 4. Reproducing the Cohort Measurement Pipeline

Execute the patient-level extraction pipeline across the annotated ACDC training cohort (100 patients):

```bash
python -m crip.mri.pipeline --data-dir data/raw/acdc/training --output-dir data/processed/acdc
```

### Generated Output Files
- **`data/processed/acdc/cardiac_measurements.csv`:** Tabular records of LVEDV, LVESV, LVSV, LVEF, RVEDV, RVESV, RVSV, RVEF, myocardial mass, voxel spacing, and source file paths.
- **`data/processed/acdc/cardiac_measurements.json`:** Structured JSON export of all patient records.
- **`data/processed/acdc/measurement_validation_report.json`:** Comprehensive validation report with cohort parametric (mean, std) and non-parametric (median, IQR) statistics stratified by diagnosis category.
- **Figures (`data/processed/acdc/figures/` and `docs/acdc/figures/`):**
  - `distribution_lvef.png`: Cohort LVEF distribution with 50% normal cutoff.
  - `distribution_lvedv.png`: Cohort LVEDV volume histogram.
  - `lvef_by_diagnostic_group.png`: Boxplot of LVEF stratified across DCM, HCM, MINF, NOR, and RV groups.
  - `lvedv_vs_lvef.png`: Scatter plot demonstrating cavity dilation vs. ejection fraction across groups.

---

## 5. Prerequisites & Environment Compatibility

- **Python:** Version 3.11 or 3.12 is the primary clinical ML benchmark target (tested up to Python 3.13.15).
  > **Note on Python 3.13:** The current development environment runs Python 3.13.15 on macOS arm64 with all core packages functional. When deploying with CUDA acceleration under Ubuntu/WSL, Python 3.11 or 3.12 is strongly recommended for seamless PyTorch/MONAI GPU wheel compatibility.
- **Operating Systems:** macOS (Apple Silicon / Intel) or Ubuntu 22.04 / 24.04 LTS (native or WSL2).
- **Hardware:** Standard CPU (GPU optional; PyTorch / MONAI automatically falls back to CPU).

---

## 6. Environment Setup & Installation

### macOS Setup

```bash
# 1. Clone / Navigate to project
cd AlcoholicRecovery

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Ubuntu / WSL Setup

```bash
# 1. Install system prerequisites
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git build-essential

# 2. Clone / Navigate to project
cd AlcoholicRecovery

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 7. Running the Application

Launch the Streamlit interface:

```bash
streamlit run src/crip/app/main.py
```

Open your browser to `http://localhost:8501`.

### MRI Analysis Viewer Features (Milestone 2B Enhanced)
1. Select **MRI Analysis** (`🫀`) from the sidebar.
2. Select any patient from the 100-case ACDC cohort dropdown.
3. Review computed quantitative parameters:
   - **Left Ventricle:** LVEDV, LVESV, LVSV, LVEF (%)
   - **Right Ventricle:** RVEDV, RVESV, RVSV, RVEF (%)
   - **Tissue:** Estimated Myocardial Mass ($1.05\text{ g/mL}$)
4. Inspect the **Biventricular Phase Volume Comparison** bar chart.
5. Inspect the **ACDC Cohort Context** scatter plot, displaying the selected patient's position against the 100 ACDC cases and calculating their cohort percentile rank.
6. Toggle between **ED** and **ES** and use the **Slice Index** slider to navigate short-axis slices with expert segmentation contours.

---

## 8. Running Automated Tests

Run the full automated test suite:

```bash
# Run all 62 unit, integration, and regression tests
pytest

# Run tests with test coverage breakdown
pytest --cov=src/crip --cov-report=term-missing
```

The test suite covers:
- **Cardiac Measurements & Pipeline (16 tests in `test_pipeline_and_measurements.py`):** Analytical volume verification, RV/LV volumes, myocardial mass, stroke volume, LVEF/RVEF formulas, non-isotropic spacing, empty masks, zero EDV, ESV > EDV handling, invalid label rejection, patient-level processing, batch execution, CSV/JSON integrity, reproducibility, and Streamlit AppTest rendering.
- **ACDC Data Loader (14 tests in `test_mri_loader.py`):** Synthetic NIfTI fixtures validating metadata parsing, missing files, corrupted files, shape mismatches, slice indexing, and intensity normalization.
- **Dataset Inspector (5 tests in `test_inspect_acdc.py`):** `Info.cfg` parsing, missing directory detection, and structured JSON reporting.
- **ACDC Integration (4 tests in `test_acdc_integration.py`):** End-to-end integration testing against the actual 100-patient training dataset.
- **EHR & Schemas (14 tests in `test_schemas.py`, `test_ehr_db.py`):** Pydantic clinical constraints ($ESV < EDV$, LVEF bounds, PEth risk tiers) and SQLite EHR persistence.
- **Application Initialization (5 tests in `test_app_init.py`):** Syntax compilation, page navigation registration, disclaimer verification, and headless AppTest page rendering.

---

## 9. Critical Methodology & Limitations

### 9.1 Segmentation Accuracy vs. Volumetric Accuracy
- **Segmentation Accuracy:** Measures spatial overlap between boundaries (e.g. Dice Similarity Coefficient, Hausdorff distance). A high Dice score ($>0.90$) does not guarantee meeting a clinical volume threshold ($\pm 5\%$ LVEF) because boundary errors on large basal slices carry high volumetric impact.
- **Volumetric Accuracy:** Measures clinical aggregate volume errors ($|\Delta EDV|, |\Delta ESV|, \Delta EF$).

### 9.2 Limitations of Expert-Mask Derived Measurements
- **No External Independent Clinical Consensus Values:** The official ACDC challenge distribution supplies expert ground-truth masks, but does not provide separate core-lab tabulated clinical reference scalars. All volumes reported herein are geometric voxel integrations of these expert masks.
- **Partial Volume Averaging:** Thick short-axis slices ($5$–$10\text{ mm}$) lead to voxel intensity averaging at apex and base.
- **Research Scope:** Measurements from the ACDC cohort (DCM, HCM, MINF, NOR, RV) investigate structural cardiomyopathy and do not prove alcohol-induced cardiomyopathy or recovery following abstinence.

---

## 10. Mandatory Dataset Attribution & License

The ACDC dataset is incorporated under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

In accordance with mandatory challenge terms, users and publications citing this platform must reference:

> **O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.**  
> *"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?"*  
> **IEEE Transactions on Medical Imaging**, vol. 37, no. 11, pp. 2514-2525, Nov. 2018.  
> **DOI:** [10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502)
