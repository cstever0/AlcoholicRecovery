# Joint Cine Training WSL Dataset Diagnosis & Preflight Hardening Report

**Platform:** Cardiomyopathy Recovery Intelligence Platform (CRIP)  
**Date:** November 2026  
**Status:** Resolved & Verified  
**Affected Platforms:** Ubuntu / Windows Subsystem for Linux (WSL), Linux, macOS  

---

## 1. Executive Summary

When running joint cine U-Net training on Ubuntu/WSL:
```bash
python -m crip.pipeline train --task cine --epochs 20
```
the pipeline logged:
```text
2026-11-02 20:07:42,238 [INFO] crip.pipeline.joint_dataset: Joint indexing complete: 404 total slices (404 ACDC, 0 M&Ms).
```
and abruptly terminated with:
```text
ZeroDivisionError: integer modulo by zero
  File "src/crip/pipeline/joint_dataset.py", line 276, in __iter__
    batch.append(mnms_pool[mnms_idx % len(mnms_pool)])
```

This report documents the root-cause diagnosis, the defensive engineering measures implemented, automated test coverage added, and instructions for reliable cross-platform execution on macOS and Ubuntu/WSL.

---

## 2. Root Cause Analysis

### 2.1 Git-Ignored Medical Imaging Repositories
Raw CMR datasets (`data/raw/acdc/`, `data/raw/mnms/`, `data/raw/emidec/`) are correctly excluded from version control in `.gitignore` due to file size and healthcare data governance. On a fresh clone or deployment inside an Ubuntu/WSL environment (`~/college/comp_sci_5542/AlcoholicRecovery`), if the M&Ms dataset was not yet downloaded or unpacked into `data/raw/mnms/`, the directory either was absent or contained 0 paired training cases.

### 2.2 Silent Exception Handling During Slice Indexing
In `JointCineSegmentationDataset._build_slice_index()`, attempts to load missing M&Ms cases caught exceptions and logged warnings:
```python
except Exception as e:
    logger.warning(f"Failed to load M&Ms case {cid}: {e}")
    continue
```
When all 60 training cases failed to load (e.g. because `data/raw/mnms` was empty), indexing concluded with `acdc_slice_count = 404` and `mnms_slice_count = 0` without halting.

### 2.3 Unvalidated Sampler Division
The dataset was then passed into `BalancedDatasetBatchSampler`. Because `mnms_indices = []`, `len(mnms_pool) == 0`. When constructing batches, the modulo operator:
```python
mnms_pool[mnms_idx % len(mnms_pool)]  # % 0
```
triggered an unhandled `ZeroDivisionError: integer modulo by zero`.

### 2.4 Hardcoded Relative Paths
`JointCineSegmentationDataset` and `CineSegmentationConfig` lacked configurable paths for the M&Ms root directory, making it difficult to point to alternative storage mounts or verify dataset availability prior to training.

---

## 3. Engineering Improvements & Defensive Hardening

### 3.1 Dataset Configuration Parameterization (`src/crip/pipeline/models_config.py`)
Extended `CineSegmentationConfig` with explicit data paths:
- `acdc_data_dir: str = "data/raw/acdc/training"`
- `mnms_data_dir: str = "data/raw/mnms"`

### 3.2 Dataset and Sampler Hardening (`src/crip/pipeline/joint_dataset.py`)
- Parameterized `JointCineSegmentationDataset` to accept `mnms_data_dir`.
- Added public slice count properties (`self.acdc_slice_count`, `self.mnms_slice_count`).
- Added rigorous pre-iteration guards to `BalancedDatasetBatchSampler.__init__` and `__iter__`:
  - If either pool is empty, raises an informative `ValueError` specifying slice counts and actionable resolution steps.
  - Completely prevents modulo-by-zero operations.

### 3.3 Strict Preflight Validation (`src/crip/pipeline/joint_trainer.py`)
Introduced `JointCineTrainer.validate_preflight(strict: bool) -> Dict[str, Any]`:
1. **Manifest Integrity:** Verifies `data/processed/unified/cohort_splits.json` exists.
2. **Holdout Quarantine:** Confirms zero overlap between held-out evaluation cases and training/validation sets.
3. **Physical File Verification:** Verifies `data/raw/acdc/training/` and `data/raw/mnms/` exist and contain the required paired cases.
4. **Actionable Errors:** If M&Ms cases are missing, outputs:
   `"Joint cine training requires annotated M&Ms training cases. No usable M&Ms slices were found. Verify data/raw/mnms/ and the unified cohort manifest."`
5. Integrated `validate_preflight(strict=True)` at the entry of `JointCineTrainer.train()`, ensuring training cannot start without required datasets.

### 3.4 CLI Preflight Audit & Dry-Run Mode (`src/crip/pipeline/cli.py`)
- Updated `python -m crip.pipeline train --task cine --dry-run` to run the preflight audit and display cohort availability.
- Exits with status code 0 if datasets are valid, or status code 1 with an error checklist if missing.
- Added optional `--acdc-dir` and `--mnms-dir` command-line overrides.

---

## 4. Automated Regression Tests

Added test class `TestWSLSafeguardsAndPreflight` to `tests/test_joint_cine_training.py`:

| Test Name | Verification Focus | Result |
| :--- | :--- | :---: |
| `test_sampler_empty_mnms_raises_value_error` | `BalancedDatasetBatchSampler` raises `ValueError` when M&Ms pool is empty | **PASSED** |
| `test_sampler_empty_acdc_raises_value_error` | `BalancedDatasetBatchSampler` raises `ValueError` when ACDC pool is empty | **PASSED** |
| `test_sampler_both_empty_raises_value_error` | `BalancedDatasetBatchSampler` raises `ValueError` when both pools are empty | **PASSED** |
| `test_preflight_audit_missing_mnms` | `validate_preflight` detects missing M&Ms directory and raises `RuntimeError` | **PASSED** |
| `test_preflight_audit_missing_acdc` | `validate_preflight` detects missing ACDC directory | **PASSED** |
| `test_preflight_audit_passes_with_real_data` | `validate_preflight` returns `is_valid: True` when real datasets are present | **PASSED** |
| `test_cli_dry_run_success` | CLI `--dry-run` exits with return code 0 on valid datasets | **PASSED** |
| `test_cli_dry_run_fails_with_missing_mnms` | CLI `--dry-run` exits with return code 1 when M&Ms is missing | **PASSED** |

---

## 5. Verification Output

### Dry-Run Verification (Real Datasets Present)
```text
$ python -m crip.pipeline train --task cine --dry-run
================================================================================
 CRIP MULTI-TASK MODEL TRAINING (CINE)
================================================================================

Task: CINE_VENTRICULAR_SEGMENTATION
  Architecture: UNet_2D (MONAI 2D U-Net)
  Target Classes: 4 (0: BG, 1: RV, 2: MYO, 3: LV)
  Training Cohort: 60 ACDC + 60 M&Ms = 120 patients
  Validation Cohort: 20 ACDC + 20 M&Ms = 40 patients
  Loss Function: CompoundDiceCELoss (Dice: 0.5, CE: 0.5)
  Normalization: percentile_zero_mean_unit_variance
  Target Checkpoint: models/multidataset/cine_joint_unet.pt

Executing Joint Cine Preflight Audit (Dry Run)...
  - Manifest Present: True
  - ACDC Training Cases: 60 / 60
  - ACDC Validation Cases: 20 / 20
  - M&Ms Training Cases: 60 / 60
  - M&Ms Validation Cases: 20 / 20
  - Holdout Quarantined: True
  - Target Checkpoint: models/multidataset/cine_joint_unet.pt

[DRY RUN MODE]: Preflight audit passed. Joint datasets and configuration are valid.
```

### Dry-Run Verification (Missing M&Ms Dataset on Fresh WSL Clone)
```text
$ python -m crip.pipeline train --task cine --dry-run --mnms-dir /tmp/empty_mnms
================================================================================
 CRIP MULTI-TASK MODEL TRAINING (CINE)
================================================================================
Executing Joint Cine Preflight Audit (Dry Run)...
  - Manifest Present: True
  - ACDC Training Cases: 60 / 60
  - ACDC Validation Cases: 20 / 20
  - M&Ms Training Cases: 0 / 60
  - M&Ms Validation Cases: 0 / 20
  - Holdout Quarantined: True
  - Target Checkpoint: models/multidataset/cine_joint_unet.pt

[PREFLIGHT CHECK FAILED]:
  * Joint cine training requires annotated M&Ms training cases. No usable M&Ms slices were found. Verify data/raw/mnms/ and the unified cohort manifest.
```
*(Exit code: 1)*

---

## 6. Checkpoint & Data Integrity Safeguards

- **No Retraining Executed:** Existing checkpoint `models/multidataset/cine_joint_unet.pt` (SHA-256: `fd4f7da4...`) was preserved unchanged.
- **Split Invariance:** Split assignments in `docs/acdc/dataset_split.json` and `data/processed/unified/cohort_splits.json` were strictly unaltered.
- **Holdout Isolation:** All 34 designated external M&Ms holdout cases remain quarantined with zero patient overlap.
