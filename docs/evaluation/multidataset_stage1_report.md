# CRIP Multi-Dataset Integration — Stage 1 Completion Report

**Project**: Cardiomyopathy Recovery Intelligence Platform (CRIP)  
**Milestone**: Multi-Dataset Integration — Stage 1: Dataset Registry and Modular Loaders  
**Date**: November 2, 2026  
**Environment**: macOS (Apple Silicon arm64), Python 3.13.15, PyTorch 2.11.0, MONAI 1.5.2  
**Test Suite Status**: **215 passed, 1 warning** (100% passing across baseline and new integration modules)

---

## 1. Executive Summary

Stage 1 of the Multi-Dataset Integration milestone establishes a unified data ingestion framework for CRIP. The platform expands from its foundational ACDC cine CMR cohort to include **EMIDEC** (delayed-enhancement viability CMR) and **M&Ms** (multi-center, multi-vendor cine CMR), with pre-registered interfaces for **Sunnybrook** and **MyoPS 2020**.

All implementations preserve existing ACDC models, pipelines, and evaluation benchmarks while enforcing clinical and geometric constraints.

### Key Achievements:
1. **Modular Dataset Package (`src/crip/datasets/`)**:
   - `schemas.py`: Standardized dataclasses, enums (`ModalityType`, `CardiacPhase`, `DatasetStatus`), and case metadata models.
   - `labels.py`: Formal label semantics with bidirectional conversion routines (`map_mnms_to_canonical`, `map_canonical_to_mnms`).
   - `registry.py`: Centralized catalog supporting query by modality, clinical task, and availability status.
   - `emidec_loader.py`: Automated discovery, Latin-1 clinical text parsing, 3D volume loading, and affine preservation.
   - `mnms_loader.py`: 4D cine series loading, clinical CSV correlation, phase extraction (ED/ES), and automatic canonical label remapping.
   - `validation.py`: CLI and programmatic dataset auditor producing verifiable JSON manifests with SHA-256 integrity checksums.
2. **Comprehensive Data Discovery & Audit**:
   - **EMIDEC**: 150 cases verified (100 training with contours and clinical data, 50 held-out test cases). Zero integrity issues.
   - **M&Ms**: 345 cases verified across 5 clinical centers and 4 MRI vendors (Siemens, Philips, GE, Canon). 114 paired image+mask series identified. Zero integrity issues.
   - Manifests exported to `data/processed/emidec/dataset_manifest.json` and `data/processed/mnms/dataset_manifest.json`.
3. **Clinical Governance Invariants Enforced**:
   - Explicit isolation of EMIDEC tissue viability labels (infarction and microvascular obstruction) from ventricular blood pools.
   - Strict prohibition of dynamic ejection fraction (LVEF) calculations on static EMIDEC scans.
   - Resolution of the M&Ms ventricular label inversion (1=LV, 3=RV vs ACDC 1=RV, 3=LV) via automated canonical remapping.
4. **Automated Testing**:
   - 37 new unit and integration tests added across `tests/test_dataset_registry.py` and `tests/test_dataset_loaders.py`.
   - Full test suite expanded from 178 to 215 tests with zero regressions.

---

## 2. Baseline Verification

Prior to code modifications, the repository baseline was verified:
- `git status`: Clean working directory on `main`.
- Pytest suite: 178 passed, 1 warning in 23.18s.
- Active Python environment: `.venv` (Python 3.13.15).

---

## 3. Dataset Discovery and Invariant Findings

### 3.1 EMIDEC 2020 Viability CMR Cohort
- **Physical Modality**: Static 3D Late Gadolinium Enhancement (LGE) CMR.
- **Data Location**: `data/raw/emidec/`
  - Training cohort: `emidec-dataset-1.0.1/` containing 100 cases (`Case_N006` to `Case_N033` Normal; `Case_P001` to `Case_P099` Pathological).
  - Test cohort: `emidec-segmentation-testset-1.0.0/` containing 50 cases (`Case_101` to `Case_150`).
- **Anatomical Contours**:
  - Training cases: 100% paired with 3D NIfTI contours (`Contours/Case_*.nii.gz`).
  - Test cases: Reference contours withheld for challenge evaluation.
- **Labels**:
  - `0`: Background
  - `1`: Left Ventricular Cavity
  - `2`: Normal Myocardium
  - `3`: Myocardial Infarction (Scar / Hyperenhancement)
  - `4`: Microvascular Obstruction (No-Reflow Core)
- **Clinical Metadata**:
  - Individual text files (e.g. `Case P001.txt`) containing sex, age, tobacco history, hypertension, diabetes, coronary artery disease family history, ST elevation, troponin, Killip class, reference LVEF (FEVG), and NT-proBNP.
  - Encoding: Latin-1 non-breaking spaces handled cleanly with encoding fallback.
- **Clinical Governance Finding**:
  - EMIDEC acquisitions are static single-phase 3D volumes. They **do not** possess temporal cine phases. The registry explicitly sets `supports_lvef_evaluation = False`.

### 3.2 M&Ms 2020 Multi-Center Cine CMR Cohort
- **Physical Modality**: 4D Short-Axis Cine SSFP CMR (`(H, W, Slices, Frames)`).
- **Data Location**: `data/raw/mnms/`
  - Split subdirectories: `Training/Labeled/` (150 folders), `Training/Unlabeled/` (25 folders), `Validation/` (34 folders), `Testing/` (136 folders).
  - Paired cases with both image (`*_sa.nii.gz`) and ground truth (`*_sa_gt.nii.gz`): **114 cases** (80 in Training/Labeled, 34 in Validation).
  - Ground truth only cases: 65 cases.
  - Empty challenge holder folders: 166 directories.
- **Scanner Vendors & Centers**:
  - Vendors: Philips (125), Siemens (95), GE (75), Canon (50).
  - Diagnostic pathologies: Dilated Cardiomyopathy (97), Normal (89), Hypertrophic Cardiomyopathy (85), Hypertensive Heart Disease (24), Arrhythmogenic RV (15), Ischemic Heart Disease (5), Other (30).
- **Critical Label Inversion Resolved**:
  - Raw M&Ms labels: `1` = LV Cavity, `2` = Myocardium, `3` = RV Cavity.
  - ACDC canonical labels: `1` = RV Cavity, `2` = Myocardium, `3` = LV Cavity.
  - The `MNMSDataLoader` applies `map_mnms_to_canonical` by default, translating `1 -> 3` and `3 -> 1` so downstream segmentation models and volumetric calculations receive proper anatomical channel indices.

### 3.3 Sunnybrook and MyoPS 2020 Status
- `data/raw/sunnybrook/` and `data/raw/myops2020/` are currently empty directory placeholders.
- Registered in `DatasetRegistry` with `DatasetStatus.BLOCKED`, preventing accidental execution while documenting integration schemas.

---

## 4. Software Implementation Summary

### 4.1 Module Inventory

| Module | Location | Purpose |
| :--- | :--- | :--- |
| `schemas.py` | `src/crip/datasets/schemas.py` | Modality types, cardiac phases, dataset statuses, and case metadata dataclasses. |
| `labels.py` | `src/crip/datasets/labels.py` | Enums for ACDC, M&Ms, EMIDEC, descriptive names, and label remapping algorithms. |
| `registry.py` | `src/crip/datasets/registry.py` | Central catalog with modality and task querying; pre-registered defaults. |
| `emidec_loader.py` | `src/crip/datasets/emidec_loader.py` | 3D LGE volume loader, Latin-1 clinical parser, case metadata constructor. |
| `mnms_loader.py` | `src/crip/datasets/mnms_loader.py` | 4D cine loader, CSV indexer, canonical remapper, ED/ES phase volume extractor. |
| `validation.py` | `src/crip/datasets/validation.py` | Dataset audit engine, NIfTI geometric checker, SHA-256 hash generator. |
| `__init__.py` | `src/crip/datasets/__init__.py` | Clean public exports for the package. |

---

## 5. Automated Test Suite Results

The automated test suite was executed via `.venv/bin/pytest -q`.

```
======================= 215 passed, 1 warning in 22.75s ========================
```

### Breakdown of Test Execution:
- **Baseline CRIP Tests**: 178 tests passed (ACDC integration, U-Net inference, volumetric calculation, synthetic recovery models, EHR database, Streamlit UI components, Bland–Altman agreement, HD95 calculations).
- **Dataset Registry Tests (`tests/test_dataset_registry.py`)**: 23 tests passed.
  - Enum validations, singleton behavior, pre-registered catalog queries.
  - Modality filtering (`CINE_MRI` vs `LGE_MRI`), task filtering (`cine_segmentation` vs `viability_assessment`).
  - Label remapping round-trip invariance (`map_canonical_to_mnms(map_mnms_to_canonical(x)) == x`).
  - Label set validation and unknown label preservation.
- **Dataset Loader Tests (`tests/test_dataset_loaders.py`)**: 14 tests passed.
  - Synthetic EMIDEC unit tests (discovery, 3D volume loading, metadata parsing, error handling).
  - Synthetic M&Ms unit tests (discovery, 4D cine loading, canonical label remapping verification, ED/ES phase extraction).
  - Real-data integration tests (actual EMIDEC and M&Ms files loaded, dimensions verified, pathology labels validated).

---

## 6. Audit Manifest Deliverables

Two persistent JSON manifests were compiled and saved:

1. **`data/processed/emidec/dataset_manifest.json`**:
   - Total cases audited: 150
   - Cases with contours: 100
   - Cases with clinical information: 100
   - Total issues found: 0
2. **`data/processed/mnms/dataset_manifest.json`**:
   - Total cases audited: 345
   - Images present: 114
   - Masks present: 179
   - Paired cases (image + mask): 114
   - Total issues found: 0

---

## 7. Downstream Readiness & Next Steps

Stage 1 is complete. The platform possesses verified data loaders and a unified registry.

### Recommendations for Future Stages:
1. **Multi-Center Cine Validation (Stage 2)**:
   - Evaluate the existing ACDC-trained 2D U-Net directly on M&Ms ED/ES volumes across Siemens, Philips, GE, and Canon scanners to measure cross-vendor domain shift.
2. **EMIDEC Viability Exploration**:
   - Utilize EMIDEC contours for dedicated tissue viability experiments (infarct sizing, no-reflow detection).
   - Ensure EMIDEC remains strictly isolated from dynamic ejection fraction pipelines.
