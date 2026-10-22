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

## 2. Architecture

CRIP is structured as a decoupled, modular Python library and interactive Streamlit application:

```
AlcoholicRecovery/
├── .streamlit/
│   └── config.toml             # Streamlit headless and privacy configuration
├── docs/
│   └── architecture.md         # Comprehensive architectural specification
├── src/
│   └── crip/
│       ├── __init__.py         # Package metadata and research disclaimer
│       ├── app/                # Streamlit web application
│       │   ├── __init__.py
│       │   ├── main.py         # Main entrypoint with multipage navigation
│       │   ├── components/
│       │   │   ├── __init__.py
│       │   │   └── disclaimer.py # Clinical disclaimer UI component
│       │   └── pages/
│       │       ├── 1_Patient_Profile.py    # Demographics, biomarkers & genetics
│       │       ├── 2_MRI_Analysis.py       # Volumetric analysis & CMR pipeline
│       │       ├── 3_Recovery_Estimate.py  # Interpretable recovery modeling design
│       │       └── 4_Evaluation.py         # Validation protocols & safety metrics
│       ├── synthetic/          # Synthetic clinical data schemas
│       │   ├── __init__.py
│       │   └── schemas.py      # Strongly-typed Pydantic patient models
│       ├── ehr/                # Local SQLite simulation
│       │   ├── __init__.py
│       │   └── db.py           # SQLite manager and cohort seeder
│       ├── mri/                # CMR pipelines (NiBabel / SimpleITK / MONAI)
│       │   └── __init__.py
│       ├── modeling/           # Interpretable ML & SHAP (Milestone 3)
│       │   └── __init__.py
│       └── evaluation/         # Calibration & metrics (Milestone 4)
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures and synthetic cases
│   ├── test_schemas.py         # Schema validation rules and constraints
│   ├── test_ehr_db.py          # SQLite persistence and retrieval tests
│   └── test_app_init.py        # App initialization, disclaimer, and AppTest tests
├── .gitignore                  # Git exclusions for venv, data, checkpoints, caches
├── pyproject.toml              # Modern Python build metadata and pytest config
├── requirements.txt            # Pinned, tested dependencies with rationale
└── README.md                   # Setup guide, run instructions, and limitations
```

---

## 3. Prerequisites

- **Python:** Version 3.11 or 3.12 (also tested on 3.13).
- **Operating Systems:**
  - macOS (Apple Silicon or Intel, macOS 12+).
  - Ubuntu Linux 22.04 / 24.04 LTS (native or under Windows Subsystem for Linux — WSL2).
- **Hardware:** Standard CPU (GPU optional; PyTorch / MONAI will operate in CPU mode if CUDA/MPS is unavailable).

---

## 4. Environment Setup

### macOS Setup

1. **Verify Python 3.11 / 3.12+ installation:**
   ```bash
   python3 --version
   ```
   *(If not installed, install via Homebrew: `brew install python@3.12`)*

2. **Clone / Navigate to project repository:**
   ```bash
   cd AlcoholicRecovery
   ```

3. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

4. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

### Ubuntu / WSL Setup

1. **Update package lists and install Python prerequisites:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-venv python3-pip git build-essential
   ```

2. **Clone / Navigate to project repository:**
   ```bash
   cd AlcoholicRecovery
   ```

3. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

4. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

---

## 5. Dependency Installation

With the virtual environment activated:

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install required dependencies
pip install -r requirements.txt

# Install the CRIP package in editable mode
pip install -e .
```

### Dependency Rationale
- `streamlit`: Declarative web interface with native multipage navigation.
- `numpy` & `pandas`: Vectorized calculations and tabular biomarker manipulation.
- `pydantic`: Schema enforcement for clinical variables (ESV < EDV, LVEF limits, PEth tiers).
- `nibabel` & `SimpleITK`: NIfTI/DICOM medical image handling and spatial resampling.
- `torch` & `monai`: Deep learning framework and medical imaging toolkit for CMR segmentation.
- `scikit-learn`: Transparent, interpretable predictive models (ElasticNet, shallow ensembles).
- `shap`: Explainable AI (XAI) feature attribution for clinician trust.
- `plotly` & `matplotlib`: Dynamic LVEF trajectories and static cardiac MRI slice visualizations.
- `pytest` & `pytest-cov`: Automated unit, integration, and coverage test runner.

---

## 6. Running the Application

Launch the Streamlit interface using:

```bash
# Start Streamlit application
streamlit run src/crip/app/main.py
```

Once running, open your web browser to `http://localhost:8501`.

The application includes four navigation views in the sidebar:
1. **Patient Profile:** Browse synthetic cases, cardiac volumetrics, PEth exposure tiers, and genetic markers.
2. **MRI Analysis:** Interactive ventricular volume calculator and CMR segmentation pipeline design.
3. **Recovery Estimate:** Multimodal feature attribution framework and conceptual trajectory curves.
4. **Evaluation:** Pre-registered statistical metrics, calibration protocols, and fairness checklist.

---

## 7. Running the Automated Tests

Execute the comprehensive test suite with coverage:

```bash
# Run all unit and integration tests
pytest

# Run tests with detailed coverage report
pytest --cov=src/crip --cov-report=term-missing
```

The test suite validates:
- Pydantic schema constraints (e.g. rejection of $ESV \ge EDV$, invalid PEth values, inconsistent LVEF).
- SQLite synthetic database operations (insertion, retrieval, idempotency).
- Static syntax compilation of all application source files.
- Programmatic rendering of every Streamlit page using `streamlit.testing.v1.AppTest`.
- Prominent presence of safety disclaimers across all user interface views.

---

## 8. Project Limitations & Ethical Constraints

- **Synthetic Data Only:** All patient records, clinical markers, and imaging parameters in Milestone 1 are simulated or synthetic.
- **No Diagnostic Authority:** Predictions generated by the system are for exploratory research purposes only and must not be used to guide patient therapy or diagnosis.
- **Milestone 1 Scope:** Models are not trained, external MRI datasets (ACDC) are not downloaded automatically, and live EHR network connections are strictly prohibited.
- **Hardware Independence:** Designed to function without dedicated GPUs; deep learning inference will default to CPU execution if hardware accelerators are unavailable.
