# CRIP Cross-Platform Reproducibility Report: Segmentation Evaluation Artifacts & Test Portability

**Platform:** Cardiomyopathy Recovery Intelligence Platform (CRIP)  
**Location:** `/Users/cstever/Desktop/AlcoholicRecovery` (and Ubuntu / WSL2: `~/college/comp_sci_5542/AlcoholicRecovery`)  
**Package:** `src/crip/`  
**Auditor / Engineer:** Senior Clinical AI Software Engineer & ML Reproducibility Auditor  
**Date:** 2026-11-02  
**Baseline Test Status on Clean Checkout:** **174 passed, 4 skipped, 1 warning** (0 failures)  
**Full Integration Test Status with Assets:** **178 passed, 0 skipped, 1 warning** (0 failures)  

---

## 1. Problem Statement & Root Cause Analysis

### 1.1 Observed Symptom on Fresh Clone
When the repository was cloned into a fresh Linux / Ubuntu environment under Windows Subsystem for Linux (WSL2), running `pytest` yielded:
```text
171 passed, 5 failed, 1 warning
```
All five failures were caused by the absence of:
```text
data/processed/acdc/segmentation_evaluation.csv
```
The failures were concentrated in:
- `tests/test_segmentation_pipeline.py`:
  - `TestSegmentationInferenceEngine::test_cohort_evaluation_summary_exists`
- `tests/test_stage2_evaluation.py`:
  - `TestBlandAltmanAndMeasurementAgreement::test_real_acdc_20_test_cases_reproducibility`
  - `TestHausdorffDistanceEvaluation::test_hd95_summary_statistics_structure`
  - `TestHausdorffDistanceEvaluation::test_hd95_box_plot_creation`
  - `TestPosterFiguresAndStreamlitIntegration::test_poster_figures_export_and_manifest`

### 1.2 Actual Root Cause
1. **`.gitignore` Exclusions for Generated Data:**  
   Line 48 of `.gitignore` explicitly ignores `data/processed/`. Consequently, `data/processed/acdc/segmentation_evaluation.csv` and `data/processed/acdc/segmentation_evaluation_summary.json` were excluded from version control on the original development machine. This exclusion is standard research practice for generated clinical tables.
2. **Missing CLI Reproduction Entry Point:**  
   While `src/crip/evaluation/segmentation_metrics.py` implemented `run_cohort_segmentation_evaluation(...)`, it lacked a command-line interface (`__main__` entry point and argument parsing). A researcher on a fresh checkout had no direct documented CLI command to generate these evaluation artifacts.
3. **Coupling of Portable Unit Tests with Empirical Assets:**  
   Tests verifying HD95 dictionary schemas (`test_hd95_summary_statistics_structure`), Plotly trace creation (`test_hd95_box_plot_creation`), and poster figure exports (`test_poster_figures_export_and_manifest`) were hardcoded to read `data/processed/acdc/segmentation_evaluation.csv` from disk, rather than using synthetic test fixtures. Consequently, purely algorithmic unit tests failed when optional data assets were absent.

---

## 2. Missing Assets & Restoration Prerequisites

On a fresh checkout, the following items are intentionally excluded from Git per research governance and file size constraints:

| Asset | Local Path | Reason for Git Exclusion | Restoration Source / Prerequisite |
| :--- | :--- | :--- | :--- |
| **Raw ACDC 2017 Dataset** | `data/raw/acdc/training/` | Challenge data license terms; binary NIfTI files | Download from official challenge portal ([creatis.insa-lyon.fr](https://www.creatis.insa-lyon.fr/Challenge/acdc/)) as documented in README Section 12. |
| **Trained 2D U-Net Model Checkpoint** | `models/mri/best_unet_model.pt` | Binary weight file (~4.9 MB); excluded by `models/**/*.pt` | Trained via `python -m crip.mri.train` or restored from secure model repository. |
| **Held-Out Test Split Manifest** | `docs/acdc/dataset_split.json` | **TRACKED IN GIT** (Seed 42, 60/20/20 stratified split) | Available immediately upon clone. |
| **Per-Patient Evaluation CSV** | `data/processed/acdc/segmentation_evaluation.csv` | Generated data artifact; excluded by `data/processed/` | Regenerated via `python -m crip.evaluation.segmentation_metrics`. |
| **Cohort Summary JSON** | `data/processed/acdc/segmentation_evaluation_summary.json` | Generated summary artifact; excluded by `data/processed/` | Regenerated via `python -m crip.evaluation.segmentation_metrics`. |

---

## 3. Exact Reproduction Commands

### 3.1 Added CLI Entry Point
We upgraded [`src/crip/evaluation/segmentation_metrics.py`](file:///Users/cstever/Desktop/AlcoholicRecovery/src/crip/evaluation/segmentation_metrics.py) with a dedicated CLI interface supporting `--data-dir`, `--checkpoint`, `--split`, `--output-csv`, and `--output-summary`.

### 3.2 Command Execution
Once the ACDC training dataset and trained checkpoint are placed in their respective locations, run:

```bash
# Activate virtual environment
source .venv/bin/activate  # or venv/bin/activate

# Regenerate held-out test evaluation artifacts (n = 20 patients)
python -m crip.evaluation.segmentation_metrics --split test
```

#### Output Verification:
```text
=== Evaluation Complete ===
Split: test
Evaluated Patients: 20
LVEF MAE: 3.86%
Mean Foreground Dice: 0.839
Output CSV: data/processed/acdc/segmentation_evaluation.csv
Output Summary: data/processed/acdc/segmentation_evaluation_summary.json
```

### 3.3 Regenerating Poster Figures
```bash
python -c "from crip.evaluation.poster_figures import generate_all_poster_figures; generate_all_poster_figures()"
```

---

## 4. Test Suite Refactoring & Portability Design

The test suite was systematically restructured into two distinct categories:

### 4.1 Portable Unit Tests (Self-Contained via Synthetic Fixtures)
- Added `synthetic_evaluation_df` fixture in [`tests/test_stage2_evaluation.py`](file:///Users/cstever/Desktop/AlcoholicRecovery/tests/test_stage2_evaluation.py) generating a deterministic 20-row evaluation DataFrame.
- **`test_hd95_summary_statistics_structure`**: Verifies schema (`{"LV", "RV", "MYO"}`), units (`mm`), keys (`ed`, `es`, `pooled`), and IQR mathematical relationships using the synthetic fixture.
- **`test_hd95_box_plot_creation`**: Verifies Plotly trace generation (3 chamber traces) using the synthetic fixture.
- **`test_poster_figures_export_and_manifest`**: Verifies high-resolution export across PNG, SVG, PDF, and `figure_manifest.json` schema in an isolated `tmp_path` using the synthetic fixture.

### 4.2 Empirical Integration Tests (Explicit Skips on Missing Assets)
Tests that strictly assert empirical ground-truth numbers against real ACDC patients now check for asset existence and skip cleanly when assets are absent:
- **`test_real_acdc_20_test_cases_reproducibility`**: Validates exact empirical agreement benchmarks (LVEF MAE 3.86%, Bias -0.55%, LoA [-10.53, +9.43]%, LVEDV Bias +2.13 mL, LVESV Bias +2.35 mL). Skips with diagnostic message if `segmentation_evaluation.csv` is missing.
- **`test_real_acdc_hd95_reproducibility`**: Validates empirical HD95 medians (LV 2.91 mm, RV 7.16 mm, Myo 4.94 mm). Skips if `segmentation_evaluation.csv` is missing.
- **`test_real_acdc_poster_figures_export`**: Validates poster export against real empirical data. Skips if `segmentation_evaluation.csv` is missing.
- **`test_cohort_evaluation_summary_exists`** (in `tests/test_segmentation_pipeline.py`): Validates summary JSON fields. Skips if summary or CSV is missing.

---

## 5. Platform Differences: macOS (Apple Silicon) vs. Ubuntu Linux (WSL2)

| Dimension | macOS Development Environment | Ubuntu Linux under WSL2 |
| :--- | :--- | :--- |
| **OS Kernel / Arch** | Darwin 25.6.0 (`arm64`, Apple Silicon) | Linux 5.15+ / 6.x (`x86_64` or `arm64`) |
| **Python Runtime** | Python 3.13.15 (Homebrew/system framework) | Python 3.12.x / 3.11.x (`python3-venv`) |
| **PyTorch Acceleration** | `mps` (Metal Performance Shaders) | `cuda` (NVIDIA GPU pass-through) or `cpu` |
| **Filesystem Pathing** | Native APFS case-insensitive/preserving | Native ext4 (recommended: `~/...`); avoid `/mnt/c/` |
| **Matplotlib Cache** | Requires explicit `MPLCONFIGDIR=/tmp` | Standard user cache directory (`~/.cache/matplotlib`) |
| **PyArrow Serialization** | Native wheel | Requires pre-compiled `pyarrow` wheel or `libarrow` |

---

## 6. Verification Results

### 6.1 Clean Checkout Simulation (Without Evaluation Artifacts)
Simulated by removing `data/processed/acdc/segmentation_evaluation.csv` and `data/processed/acdc/segmentation_evaluation_summary.json`:
```bash
pytest -q -rs
```
**Result:**
```text
=========================== short test summary info ============================
SKIPPED [1] tests/test_segmentation_pipeline.py:375: Held-out segmentation evaluation artifacts ('segmentation_evaluation.csv' / 'segmentation_evaluation_summary.json') not found on this checkout. Run 'python -m crip.evaluation.segmentation_metrics' to generate.
SKIPPED [1] tests/test_stage2_evaluation.py:159: Real ACDC 20-case evaluation artifact ('data/processed/acdc/segmentation_evaluation.csv') not found. Run 'python -m crip.evaluation.segmentation_metrics' with raw ACDC data and checkpoint to generate.
SKIPPED [1] tests/test_stage2_evaluation.py:239: Real ACDC 20-case evaluation artifact ('data/processed/acdc/segmentation_evaluation.csv') not found. Run 'python -m crip.evaluation.segmentation_metrics' with raw ACDC data and checkpoint to generate.
SKIPPED [1] tests/test_stage2_evaluation.py:338: Real ACDC 20-case evaluation artifact ('data/processed/acdc/segmentation_evaluation.csv') not found. Run 'python -m crip.evaluation.segmentation_metrics' with raw ACDC data and checkpoint to generate.
================== 174 passed, 4 skipped, 1 warning in 19.88s ==================
```
- **Passed:** 174
- **Skipped:** 4 (explicit, informative integration skips)
- **Failed:** **0**

### 6.2 Full Empirical Integration Test Run (With Generated Artifacts)
With generated evaluation CSV and summary JSON present:
```bash
pytest -q
```
**Result:**
```text
======================= 178 passed, 1 warning in 20.70s ========================
```
- **Passed:** **178**
- **Skipped:** 0
- **Failed:** **0**
- **Warnings:** 1 (benign upstream MONAI deprecation warning)

---

## 7. Remaining Limitations

1. **ACDC Access Requirements:**  
   Because the ACDC challenge dataset is distributed under academic usage terms requiring user registration, raw imaging scans cannot be bundled in a public Git repository. Users on fresh machines must obtain the data following README Section 12 to run empirical evaluations.
2. **GPU Performance Scaling:**  
   On systems without an NVIDIA GPU (WSL2 CPU mode) or Apple Silicon MPS, running inference over the 20 test scans takes approximately 45–60 seconds on CPU compared to ~5 seconds on MPS/CUDA.

---

## 8. Git Status

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
	modified:   README.md
	modified:   src/crip/evaluation/segmentation_metrics.py
	modified:   tests/test_segmentation_pipeline.py
	modified:   tests/test_stage2_evaluation.py

Untracked files:
	docs/evaluation/wsl_reproducibility_report.md
```

In accordance with instructions:
- No automatic git commit or push has been performed.
- Reproduction commands, test refactoring, and documentation are complete.
