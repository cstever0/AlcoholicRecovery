# Cardiomyopathy Recovery Intelligence Platform (CRIP)

> [!WARNING]
> ### CLINICAL RESEARCH DISCLAIMER — NOT FOR DIAGNOSIS OR TREATMENT
> **This software is an experimental research proof of concept developed exclusively for academic investigation into left ventricular ejection fraction recovery patterns. It integrates synthetic patient data, simulated genetics, and exploratory cardiac imaging parameters. This platform is NOT a certified medical device and MUST NEVER be used for clinical diagnosis, patient risk stratification, or clinical treatment decisions. Synthetic predictions must never be represented as clinically validated patient probabilities.**

---

## 1. Project Overview

The **Cardiomyopathy Recovery Intelligence Platform (CRIP)** is a 4-week clinical AI research proof of concept. The platform investigates interpretable machine learning models predicting 6-month Left Ventricular Ejection Fraction (LVEF) recovery in cardiomyopathy associated with:

- **Toxic Alcohol Exposure:** Quantified via Phosphatidylethanol (PEth) blood biomarkers and duration of cessation.
- **Cardiac MRI Volumetrics & Fibrosis:** End-diastolic volume (EDV), end-systolic volume (ESV), stroke volume (SV), and late gadolinium enhancement (LGE) myocardial scar patterns.
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
│       └── acdc_testing_validation_report.json # Validation report for testing cohort (50 cases)
├── notebooks/
│   └── acdc/
│       └── script_acdc_visu.ipynb      # Reference visualization notebook
├── data/
│   ├── raw/
│   │   └── acdc/                       # Real ACDC dataset (excluded from Git)
│   │       ├── training/               # 100 patients (patient001 to patient100)
│   │       └── testing/                # 50 patients (patient101 to patient150)
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
│       │       ├── 2_MRI_Analysis.py   # Real ACDC CMR visualization & volumetric preview
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
│       │   └── inspect_acdc.py         # Automated dataset validation CLI script
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
│   └── test_acdc_integration.py       # Integration tests against real ACDC dataset
├── pyproject.toml                      # Modern Python build metadata and pytest config
├── requirements.txt                    # Pinned, tested dependencies with rationale
└── README.md                           # Master setup guide, run instructions, and limitations
```

---

## 3. ACDC Cardiac MRI Dataset Integration

The platform integrates the **Automated Cardiac Diagnosis Challenge (ACDC)** benchmark dataset for cardiac imaging:

### Dataset Directory Structure
```text
data/raw/acdc/
├── training/
│   ├── MANDATORY_CITATION.md
│   ├── patient001/
│   │   ├── Info.cfg                    # Metadata: ED/ES frame numbers, diagnosis, height, weight
│   │   ├── patient001_4d.nii.gz        # 4D short-axis cine cardiac MRI series
│   │   ├── patient001_frame01.nii.gz   # 3D End-Diastolic (ED) volume
│   │   ├── patient001_frame01_gt.nii.gz# 3D ED expert ground-truth segmentation mask
│   │   ├── patient001_frame12.nii.gz   # 3D End-Systolic (ES) volume
│   │   └── patient001_frame12_gt.nii.gz# 3D ES expert ground-truth segmentation mask
│   └── ... (through patient100)
└── testing/
    ├── MANDATORY_CITATION.md
    └── patient101/ ... through patient150/
```

### Anatomical Segmentation Labels
- **Label 0:** Background
- **Label 1:** Right Ventricular (RV) cavity
- **Label 2:** Left Ventricular Myocardium
- **Label 3:** Left Ventricular (LV) cavity

### Verifying the Installed Dataset
Run the automated validation utility across the dataset:

```bash
# Verify training split (100 patients)
python -m crip.mri.inspect_acdc --data-dir data/raw/acdc/training --output docs/acdc/acdc_validation_report.json

# Verify testing split (50 patients)
python -m crip.mri.inspect_acdc --data-dir data/raw/acdc/testing --output docs/acdc/acdc_testing_validation_report.json
```

The validation tool performs rigorous checks:
- Verifies existence and syntax of `Info.cfg`.
- Validates 3D NIfTI image readability and voxel spacing.
- Confirms shape compatibility between each cine frame and its expert mask.
- Verifies segmentation values strictly belong to `{0, 1, 2, 3}`.
- Categorizes patients by diagnosis (DCM, HCM, MINF, NOR, RV).

---

## 4. Prerequisites & Environment Compatibility

- **Python:** Version 3.11 or 3.12 is the primary clinical ML benchmark target (tested up to Python 3.13.15).
  > **Note on Python 3.13:** The current development environment on macOS runs Python 3.13.15 with all core packages functional. When deploying with CUDA acceleration under Ubuntu/WSL, Python 3.11 or 3.12 is strongly recommended for seamless PyTorch/MONAI GPU wheel compatibility.
- **Operating Systems:** macOS (Apple Silicon / Intel) or Ubuntu 22.04 / 24.04 LTS (native or WSL2).
- **Hardware:** Standard CPU (GPU optional; PyTorch / MONAI automatically falls back to CPU).

---

## 5. Environment Setup & Installation

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

## 6. Running the Application

Launch the Streamlit interface:

```bash
streamlit run src/crip/app/main.py
```

Open your browser to `http://localhost:8501`.

### Navigating to MRI Analysis
1. Open the sidebar navigation and select **MRI Analysis** (`🫀`).
2. Select any patient from the ACDC cohort dropdown (e.g., `patient001`, `patient020`).
3. Toggle between **ED** (End-Diastole) and **ES** (End-Systole).
4. Use the **Slice Index** slider to navigate short-axis slices from ventricular base to apex.
5. Toggle **Show Expert Segmentation Overlay** and adjust **Overlay Opacity (Alpha)**.
6. Observe side-by-side comparison:
   - **Left:** Original grayscale CMR Cine image.
   - **Right:** Expert multi-structure segmentation overlay (RV in Azure, Myocardium in Green, LV in Crimson).
7. Review physical structure volumes ($V = N_{\text{voxels}} \times \frac{\Delta x \cdot \Delta y \cdot \Delta z}{1000}\text{ mL}$) and interactive LVEF simulation.

---

## 7. Running Automated Tests

Run the full automated test suite:

```bash
# Run all 46 unit, integration, and regression tests
pytest

# Run tests with test coverage breakdown
pytest --cov=src/crip --cov-report=term-missing
```

Test suite coverage includes:
- **ACDC Data Loader (14 tests):** Synthetic NIfTI fixtures validating metadata parsing, file missing errors, corrupted files, shape mismatches, label validity, voxel spacing, slice indexing, and intensity normalization.
- **Cardiac Metrics (4 tests):** Mathematical verification of physical voxel volumetrics, stroke volume ($EDV - ESV$), and LVEF calculation ($((EDV - ESV) / EDV) \times 100$).
- **Dataset Inspector (5 tests):** Validation of `Info.cfg` parsing, missing directory detection, and structured JSON reporting.
- **ACDC Integration (4 tests):** End-to-end integration testing against the actual 100-patient training dataset.
- **EHR & Schemas (14 tests):** Pydantic clinical constraints ($ESV < EDV$, LVEF bounds, PEth risk tiers) and SQLite EHR persistence.
- **Application Initialization (5 tests):** Syntax compilation, page navigation registration, disclaimer verification, and headless AppTest page rendering.

---

## 8. Common Error Messages & Troubleshooting

| Error Message | Cause | Resolution |
|---|---|---|
| `ACDC Dataset Not Found at: data/raw/acdc/training` | Dataset folder is missing or located in a custom path. | Ensure the ACDC dataset is extracted into `data/raw/acdc/training` or enter your custom path in the sidebar input field. |
| `Missing Info.cfg for patient: ...` | Incomplete patient folder download. | Re-extract or re-verify patient folder from the official ACDC distribution. |
| `Shape mismatch: Image shape != Mask shape` | Mismatched NIfTI files or corrupted slice counts. | Run `python -m crip.mri.inspect_acdc` to locate corrupted cases. |
| `Unexpected segmentation label(s) in ...` | Mask file contains invalid labels (outside `{0, 1, 2, 3}`). | Ensure official ACDC ground truth masks are used (`_gt.nii.gz`). |
| `PermissionError: [Errno 1] Operation not permitted` | Sandboxed environment blocking external network/port access. | Streamlit is configured in `.streamlit/config.toml` to bind locally without telemetry. |

---

## 9. Mandatory Dataset Attribution & License

The ACDC dataset is incorporated under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

In accordance with mandatory challenge terms, users and publications citing this platform must reference:

> **O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.**  
> *"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?"*  
> **IEEE Transactions on Medical Imaging**, vol. 37, no. 11, pp. 2514-2525, Nov. 2018.  
> **DOI:** [10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502)

---

## 10. Project Limitations & Ethical Constraints

- **Research Demonstration Only:** All predictive modeling in CRIP is for exploratory hypothesis generation and must never be used for clinical decision-making.
- **Expert Contours in Milestone 2A:** Segmentation overlays currently displayed in the MRI Analysis page represent expert human annotations provided by the challenge organizers. **Automatic deep learning segmentation has not yet been executed in Milestone 2A.**
- **No Direct Clinical Claim:** The platform strictly isolates calculated mathematical values from diagnostic inferences.
