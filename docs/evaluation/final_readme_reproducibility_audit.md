# CRIP — Final README Reproducibility Audit & Verification Report

**Author:** Antigravity AI Engineering & Clinical Research Assistant  
**Date:** November 2, 2026  
**Status:** COMPLETE & VERIFIED  
**Target Environments:**
- Primary Development: macOS Apple Silicon arm64 (`~/Desktop/AlcoholicRecovery`)
- Secondary Reproduction: Ubuntu Linux on WSL2 (`~/college/comp_sci_5542/AlcoholicRecovery`)
- Target Repository: [https://github.com/cstever0/AlcoholicRecovery](https://github.com/cstever0/AlcoholicRecovery)

---

## 1. Executive Summary

A comprehensive documentation and technical reproducibility audit of the Cardiomyopathy Recovery Intelligence Platform (CRIP) was performed. Following the successful cross-platform debugging of the Ubuntu/WSL joint training pipeline, EMIDEC manifest synchronization, feature bank preprocessing order, and Streamlit evaluation calibration schema handling, `README.md` was subjected to an exhaustive line-by-line review.

The objective was to transform `README.md` into an independently executable, beginner-friendly, cross-platform reproduction guide that guarantees a new researcher, student, or peer reviewer can clone the repository and reproduce all verified results without inspecting internal Python source code to discover undocumented preprocessing steps or command-line flags.

All documented commands were verified against the Python source code implementation, all 353 automated regression tests were confirmed passing, model checkpoint SHA-256 digests were verified intact, and two self-contained reproduction companions were authored:
1. `docs/setup/reproduction_checklist.md`: Itemized checklists and operational criteria for both Path A and Path B.
2. `docs/evaluation/final_readme_reproducibility_audit.md`: This comprehensive audit report.

---

## 2. Identified Documentation Gaps in Historical Documentation

Prior to this audit, several critical documentation gaps, ordering discrepancies, and missing instructions were identified:

### Gap 1: Omission of Held-Out Segmentation Evaluation before Streamlit Launch
- **Issue:** In the historical documentation, `python -m crip.evaluation.segmentation_metrics` was briefly mentioned in an evaluation section towards the end of the document, after the Streamlit launch instructions.
- **Consequence:** A user following the README sequentially from start to finish would launch Streamlit without generating `data/processed/acdc/segmentation_evaluation.csv`. Upon opening Page 4 (Model Evaluation) $\to$ Tab 2 (Cardiac Measurement Agreement / Bland–Altman), Streamlit would display a warning banner: `⚠️ segmentation_evaluation.csv not found. Please run the evaluation pipeline first.`
- **Correction:** Established a dedicated **Section 19: Held-Out Model Evaluation & Bland–Altman Inputs** placed strictly before the Streamlit launch instructions. Explicitly documented the command, output files, and verification criteria for the 20 held-out ACDC test cases.

### Gap 2: Incorrect Figure Generation Output Directory Paths
- **Issue:** Historical narrative drafts stated that figure generation scripts exported figures to `docs/figures/multidataset/` or `docs/figures/poster_figures/`.
- **Code Inspection:** Direct examination of `src/crip/evaluation/poster_figures.py`, `poster_figures_mnms.py`, `poster_figures_emidec.py`, and `src/crip/pipeline/recovery_trainer.py` revealed that `docs/figures` does not exist in the codebase. All figures are systematically exported to `docs/evaluation/` subdirectories:
  - Poster figures: `docs/evaluation/poster_figures/`
  - M&Ms figures: `docs/evaluation/external_datasets/mnms/`
  - EMIDEC figures: `docs/evaluation/external_datasets/emidec/`
  - Synthetic recovery multi-dataset figures: `docs/evaluation/multidataset/synthetic_integration/`
- **Correction:** Updated all figure generation documentation to reference the exact default paths defined in code.

### Gap 3: Unexplained Git Tracked Output Modifications on Secondary Platforms
- **Issue:** When reproducing experiments on Ubuntu/WSL, running training and figure generation modified tracked files in the Git working tree (PNGs, SVGs, PDFs, and training history JSONs), creating confusion regarding repository cleanliness.
- **Root Cause:** Minor floating-point variations across hardware accelerators (Apple Silicon Metal Performance Shaders vs. NVIDIA CUDA vs. x86_64 CPU) cause minute variations in loss trajectories, neural network gradients, and Matplotlib anti-aliasing rasterization.
- **Correction:** Authored a dedicated **Section 22: Git Research Provenance & Workflow Isolation** explaining why this happens, distinguishing source code from generated assets, recommending dedicated reproduction branches (`git checkout -b reproduction-run`), and warning against destructive resets (`git reset --hard`).

### Gap 4: Outdated Test Suite Counts
- **Issue:** Historical README text referenced "296 automated pytest unit & integration tests" and "= 296 passed".
- **Current State:** Following the addition of multi-dataset regression tests, EMIDEC manifest repair tests, cine comparison loading tests, and calibration curve schema regression tests, the automated test suite has expanded to **353 passed tests**.
- **Correction:** Synchronized all test count references to 353 passed tests across the repository structure, test instructions, and a new 29-module test inventory table.

### Gap 5: Disorganized Troubleshooting Section
- **Issue:** Historical troubleshooting was presented as an unstructured list of 16 sequential items without domain grouping.
- **Correction:** Restructured into **Section 28: Troubleshooting & Frequently Asked Questions** organized into 6 distinct, logical failure domains:
  1. Python and Virtual Environment Problems
  2. Dataset Acquisition and Organization Problems
  3. Cohort Partition and Split Repair Problems
  4. Preprocessing and Feature Bank Dependency Problems
  5. Model Checkpoint and Evaluation Problems
  6. Streamlit Web Platform Problems (including port conflicts, WSL localhost, and the calibration JSON schema mismatch resolution)

---

## 3. Structural Reorganization of `README.md`

`README.md` was reorganized into 30 clearly numbered, logical sections with a fully synchronized Table of Contents:

| Section # | Title / Domain | Purpose & Enhancements |
| :---: | :--- | :--- |
| **1–9** | Platform Overview & Scientific Background | Core clinical background, architecture, hardware requirements, repository layout (353 tests). |
| **10** | **PATH A — Quick Start: Synthetic Data Only** | Standalone 3–5 min quick start. No MRI downloads, no GPU. Explains what works (Pages 0, 1, 3, 4) vs. graceful notices (Pages 2, 5). |
| **11** | Cross-Platform System Installation Guide | macOS (MPS), Ubuntu native (CUDA), Ubuntu on WSL2 (ext4 storage directive), native Windows (experimental). |
| **12** | **PATH B — Complete Real MRI Dataset Setup** | Overview, download links, expected directory trees, `:Zone.Identifier` diagnostic and cleanup, ingestion audit. |
| **13** | Canonical Cohort Partitioning & Split Verification | Split preservation in `cohort_splits.json`, explain repair command (`prepare --repair-missing-splits`). |
| **14** | Physical Cardiac MRI Preprocessing | Preprocessing dependency map, ACDC (100), M&Ms (114), EMIDEC (100), verification smoke test. |
| **15** | Unified Empirical Feature Bank Assembly | `pipeline features` (314 records), verification command. |
| **16** | Synthetic Cohort Generation (Baseline vs. Empirical) | Baseline generator vs. empirical 5D Gaussian copula simulation ($N=3,000$). |
| **17** | Recovery Prediction Model Training & Benchmarking | Preflight dry-run (`train --task recovery --dry-run`), empirical benchmark, baseline training. |
| **18** | MRI Segmentation Model Training & Inference | Baseline Cine U-Net, Joint Cine U-Net, EMIDEC LGE U-Net, held-out inference smoke test (`patient007`). |
| **19** | **Held-Out Model Evaluation & Bland–Altman Inputs** | **NEW:** ACDC held-out evaluation (`segmentation_metrics`), Joint Cine matched evaluation (`pipeline evaluate`), M&Ms, EMIDEC. |
| **20** | **Scientific Figure Generation & Multi-Format Export** | **NEW:** Multi-format exports (PNG, SVG, PDF) for Poster Figures 1–8, M&Ms Figures 1–5, EMIDEC Figures 1–6, Recovery Figures 1–8. |
| **21** | Model Checkpoint Inventory and Artifact Restoration | Complete checkpoint inventory table, SHA-256 digests, fallback mechanisms. |
| **22** | **Git Research Provenance & Workflow Isolation** | **NEW:** Explains tracked output modifications, experiment branch isolation, non-destructive workflows. |
| **23** | Launching the Streamlit Clinical Research Platform | Launch commands, WSL2 network binding (`--server.address 0.0.0.0`), port conflict handling, stopping with Ctrl+C. |
| **24** | Application Workspaces & Demonstration Tour | Walkthrough of Pages 0–5, curated Case A/B/C execution, complete Streamlit verification checklist. |
| **25** | Mock FHIR Import/Export | FHIR R4 Bundle import/export code, supported LOINC and resource types. |
| **26** | Running Automated Tests | Full suite execution (353 passed), fresh clone skips, 29-module test inventory table. |
| **27** | Reproducing Scientific Evaluation Metrics | Comprehensive benchmark tables, reconciliation of draft targets, clinical governance invariants. |
| **28** | Troubleshooting & Frequently Asked Questions | Categorized into 6 failure domains with verified causes and remedies. |
| **29** | Reproducibility and Limitations | Computational reproducibility, non-clinical research boundaries. |
| **30** | License, Dataset Attribution, and References | ACDC CC BY-NC-SA 4.0 license, academic citation, clinical references. |

---

## 4. Verification of Commands Against Source Code

All command-line interfaces, argument flags, and entry points were audited against repository source code:

| Command Line Invocation | Source Module | Verified CLI Flags | Verified Outputs |
| :--- | :--- | :--- | :--- |
| `python -m crip.pipeline audit` | `src/crip/pipeline/cli.py` | *(none)* | Audit summary across all 3 datasets & models |
| `python -m crip.pipeline prepare --repair-missing-splits` | `src/crip/pipeline/splits.py` | `--repair-missing-splits`, `--force` | `data/processed/unified/cohort_splits.json` |
| `python -m crip.mri.pipeline --data-dir ... --output-dir ...` | `src/crip/mri/pipeline.py` | `--data-dir`, `--output-dir` | `data/processed/acdc/cardiac_measurements.csv` |
| `python -m crip.evaluation.external_mnms` | `src/crip/evaluation/external_mnms.py` | *(none)* | `data/processed/mnms/external_evaluation.csv` |
| `python -m crip.emidec.tissue_metrics` | `src/crip/emidec/tissue_metrics.py` | *(none)* | `data/processed/emidec/tissue_measurements.csv` |
| `python -m crip.pipeline features` | `src/crip/pipeline/feature_bank.py` | *(none)* | `data/processed/unified/empirical_feature_bank.csv` |
| `python -m crip.synthetic.cohort_generator ...` | `src/crip/synthetic/cohort_generator.py` | `--n-patients`, `--seed`, `--abstinence-prob`, `--output-dir` | `data/synthetic/synthetic_cardiomyopathy_cohort.csv` |
| `python -m crip.pipeline synthesize ...` | `src/crip/pipeline/empirical_generator.py`| `--n-patients`, `--empirical`, `--seed`, `--export` | Tabular & SQLite synthetic records |
| `python -m crip.pipeline train --task recovery --dry-run` | `src/crip/pipeline/recovery_trainer.py` | `--task recovery`, `--dry-run` | Validates 5 prerequisite artifacts before training |
| `python -m crip.pipeline train --task recovery` | `src/crip/pipeline/recovery_trainer.py` | `--task recovery` | Trained recovery models + 24 figures |
| `python src/crip/mri/train_segmentation.py --epochs 25` | `src/crip/mri/train_segmentation.py` | `--epochs`, `--batch-size`, `--lr`, `--device` | `models/mri/best_unet_model.pt` |
| `python -m crip.pipeline train --task cine --dry-run` | `src/crip/pipeline/joint_trainer.py` | `--task cine`, `--dry-run` | Preflight slice & holdout validation |
| `python -m crip.pipeline train --task cine --epochs 20` | `src/crip/pipeline/joint_trainer.py` | `--task cine`, `--epochs` | `models/multidataset/cine_joint_unet.pt` |
| `python -m crip.pipeline train --task lge --dry-run` | `src/crip/pipeline/emidec_trainer.py` | `--task lge`, `--dry-run` | Preflight EMIDEC pairing validation |
| `python -m crip.pipeline train --task lge --epochs 30` | `src/crip/pipeline/emidec_trainer.py` | `--task lge`, `--epochs` | `models/multidataset/emidec_lge_unet.pt` |
| `python -m crip.evaluation.segmentation_metrics ...` | `src/crip/evaluation/segmentation_metrics.py` | `--data-dir`, `--checkpoint`, `--split`, `--output-csv`, `--output-summary` | `data/processed/acdc/segmentation_evaluation.csv` |
| `python -m crip.pipeline evaluate` | `src/crip/pipeline/cli.py` | `--summary-only` | Matched 54-case evaluation summary |
| `python -m crip.evaluation.poster_figures` | `src/crip/evaluation/poster_figures.py` | *(none / entry point)* | 8 poster figures in PNG, SVG, PDF |
| `python -m crip.evaluation.poster_figures_mnms` | `src/crip/evaluation/poster_figures_mnms.py` | *(none / entry point)* | 5 M&Ms figures in PNG, SVG, PDF |
| `python -m crip.evaluation.poster_figures_emidec`| `src/crip/evaluation/poster_figures_emidec.py`| *(none / entry point)* | 6 EMIDEC figures in PNG, SVG, PDF |
| `python -m streamlit run src/crip/app/main.py` | `src/crip/app/main.py` | `--server.port`, `--server.address` | Multi-page clinical research web platform |

---

## 5. Artifact Dependency & Pipeline Data Flow

The following directed dependency diagram illustrates the exact data flow enforced by the pipeline:

```
[Phase 12: Raw Medical Imaging Datasets]
  ├── ACDC 2017   (data/raw/acdc/training/)   [100 patients]
  ├── M&Ms 2020   (data/raw/mnms/)            [114 paired studies]
  └── EMIDEC 2020 (data/raw/emidec/)          [100 annotated cases]
           │
           ▼ [Phase 13: Partition Synchronization]
  └── data/processed/unified/cohort_splits.json (314 partition assignments)
           │
           ▼ [Phase 14: Source Measurement Extraction]
  ├── data/processed/acdc/cardiac_measurements.csv       (100 rows)
  ├── data/processed/mnms/external_evaluation.csv        (114 rows)
  └── data/processed/emidec/tissue_measurements.csv      (100 rows)
           │
           ▼ [Phase 15: Unified Feature Assembly]
  └── data/processed/unified/empirical_feature_bank.csv  (314 rows)
           │
           ├─────────────────────────────────────────────────┐
           ▼ [Phase 16: Synthetic Cohort Simulation]         ▼ [Phase 17: Recovery Benchmarking]
  └── data/synthetic/synthetic_cardiomyopathy_cohort.csv  └── models/recovery/recovery_pipeline.joblib
           │                                                 └── docs/evaluation/multidataset/synthetic_integration/
           ▼ [Phase 18: MRI Segmentation Training]               (24 recovery figures)
  ├── models/mri/best_unet_model.pt
  ├── models/multidataset/cine_joint_unet.pt
  └── models/multidataset/emidec_lge_unet.pt
           │
           ▼ [Phase 19: Held-Out Model Evaluation]
  ├── data/processed/acdc/segmentation_evaluation.csv    (20 held-out ACDC cases)
  └── data/processed/acdc/segmentation_evaluation_summary.json
           │
           ▼ [Phase 20: Scientific Figure Generation]
  ├── docs/evaluation/poster_figures/                    (8 poster figures)
  ├── docs/evaluation/external_datasets/mnms/            (5 M&Ms figures)
  └── docs/evaluation/external_datasets/emidec/          (6 EMIDEC figures)
           │
           ▼ [Phase 23 & 24: Interactive Clinical Platform]
  └── Streamlit Multi-Page Web Platform (http://localhost:8501)
      - Page 0: Overview Dashboard
      - Page 1: Patient Profile & Cohort Explorer
      - Page 2: Cardiac MRI Analysis (ACDC, Cine Comparison, EMIDEC LGE)
      - Page 3: Recovery Estimation & Multimodal Integration
      - Page 4: Model Evaluation (Bland-Altman, Calibration, Figures)
      - Page 5: Cardiac Measurements (AI vs Expert Comparison)
```

---

## 6. Automated Test Suite Results & Checkpoint Integrity

### Automated Test Suite Execution:
- **Test Command:** `MPLCONFIGDIR=/tmp/matplotlib .venv/bin/pytest -q`
- **Result:** **353 passed, 1 warning in 23.57 seconds**
- **Test Modules:** 29 test files covering data loaders, registries, Pydantic schemas, U-Net architectures, joint training, EMIDEC tissue metrics, synthetic generator, recovery modeling, FHIR R4 interoperability, Streamlit page initialization, and calibration schema regression.

### Model Checkpoint SHA-256 Digest Verification:
- **Command:** `shasum -a 256 models/mri/best_unet_model.pt models/multidataset/cine_joint_unet.pt models/multidataset/emidec_lge_unet.pt`
- **Verified Digests:**
  - `models/mri/best_unet_model.pt`: `71ea54235803562acc67b8903deecaaf113d45a1c7b443a97ba9fbe6573072c9`
  - `models/multidataset/cine_joint_unet.pt`: `fd4f7da40f756794c99d9e2cad1e06a080701e86b5386afcd686a24b5c12e50d`
  - `models/multidataset/emidec_lge_unet.pt`: `630d637ee17beea7580d40b06f0e581f059018f22f6f82fde94d923c454f9ad7`
- **Integrity Status:** All model checkpoints remain byte-identical to their verified baselines.

---

## 7. Remaining Limitations & Non-Clinical Governance Disclosures

In accordance with strict clinical AI governance principles:
1. **Academic Proof of Concept:** CRIP is strictly an academic research exploration into LVEF recovery modeling. It is neither CE-marked nor FDA-cleared and must never be used for clinical patient care or medical decision-making.
2. **Computational Synthetic Endpoints:** Longitudinal recovery endpoints ($\Delta\text{LVEF} \ge 10.0\%$) and PEth biomarker levels are generated from published clinical distributions and empirical priors; they do not represent prospectively followed clinical trial subjects.
3. **Challenge Cohort Composition:** ACDC comprises DCM, HCM, MINF, and RV abnormalities from public challenge imaging; it does not contain confirmed clinical alcoholic cardiomyopathy cases.
4. **Kinematic vs. Viability Invariant:** EMIDEC static delayed-enhancement CMR cannot be used to calculate dynamic ejection fraction (LVEF).
5. **Pathology Separation:** EMIDEC ischemic scar and microvascular obstruction resulting from coronary occlusion serve as tissue characterization benchmarks and must not be conflated with non-ischemic, diffuse mid-wall fibrosis observed in alcoholic cardiomyopathy.
