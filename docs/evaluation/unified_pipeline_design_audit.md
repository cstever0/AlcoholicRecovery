# CRIP Unified Multi-Dataset Pipeline Design and Leakage Audit

**Audit Date:** 2026-11-02  
**Environment:** macOS (Apple Silicon arm64) & Ubuntu / WSL2 Compatible  
**Python Runtime:** 3.13.15  
**Automated Test Baseline:** 255 passing tests (100% pass rate)

---

## 1. Audit Scope and Objectives

This audit evaluates the architectural design, cohort split manifests, empirical feature extraction, data leakage safeguards, and licensing provenance for the unified three-dataset CRIP pipeline:
- **ACDC 2017:** 100 Cine SSFP cases
- **M&Ms 2020:** 345 Cine SSFP cases (114 paired with expert masks)
- **EMIDEC 2020:** 150 LGE CMR cases (100 annotated cases)

---

## 2. Train/Test Data Leakage Audit

A comprehensive inspection of potential data leakage pathways was performed across data preparation, feature extraction, neural network training, and synthetic generation:

| Leakage Risk Category | Failure Mode / Vulnerability | Architectural Protection Implemented | Verification Status |
| :--- | :--- | :--- | :---: |
| **Cross-Partition Patient Leakage** | Same patient appearing in both training and test sets via slice-level splitting | Strict patient-level partitioning in `src/crip/pipeline/splits.py`. Zero subject overlap verified across all splits. | **VERIFIED (Zero Overlap)** |
| **ACDC Benchmark Regression** | Overwriting or shuffling the original 20-patient held-out test cohort | Hard-pinned to `docs/acdc/dataset_split.json`. Automated test confirms exact 20 patient IDs are preserved. | **VERIFIED (Inviolate)** |
| **M&Ms Evaluation Contamination** | Using all 114 external evaluation cases for joint training, invalidating external test validity | Partitioned into 80 Training Pool cases (Centers 1 & 2) and 34 untouched evaluation cases (Center 3). Historical 114-case baseline preserved as frozen reference. | **VERIFIED (Isolated)** |
| **Spatial / Normalization Leakage** | Computing normalization statistics across pooled train + test slices | Image intensity clipping and standard scaling are fitted per-slice/volume independently during inference. | **VERIFIED** |
| **Model Inference Error Contamination** | Blending model prediction errors into empirical feature priors | Feature bank draws exclusively from expert ground-truth annotations (`mask_provenance="expert_ground_truth"`). | **VERIFIED** |
| **Pathology Conflation Leakage** | Treating ischemic scar / MVO as alcoholic cardiomyopathy fibrosis | Strict semantic separation: static LGE scans cannot yield dynamic LVEF; CAD ischemic scar is explicitly marked. | **VERIFIED** |
| **Synthetic Profile Duplication** | Repeating RNG states across training and evaluation synthetic cohorts | Seeded generator (`seed=42`) with deterministic RNG streams and separate evaluation subsets. | **VERIFIED** |

---

## 3. Cohort Split Verification Summary

Persisted in `data/processed/unified/cohort_splits.json`:

### ACDC Cohort Split:
- **Total Patients:** 100
- **Training:** 60 patients (12 DCM, 12 HCM, 12 MINF, 12 NOR, 12 RV)
- **Validation:** 20 patients (4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV)
- **Held-out Test:** 20 patients (4 DCM, 4 HCM, 4 MINF, 4 NOR, 4 RV)
- **Patient Leakage:** $\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$

### M&Ms Cohort Split:
- **Total Paired Studies:** 114
- **Joint Training Candidate Pool:** 80 studies (Centers 1 & 2; Siemens and Philips scanners)
- **Untouched Held-Out Evaluation Cohort:** 34 studies (Center 3; GE and Canon scanners)
- **Historical Frozen-Model Benchmark:** Preserved across all 114 studies ($0.725$ Dice, $8.48\%$ LVEF MAE)
- **Patient Leakage:** $\text{Train Pool} \cap \text{Untouched Test} = \emptyset$

### EMIDEC Cohort Split:
- **Total Annotated Studies:** 100 (33 Normal Controls, 67 Pathological Infarction Cases)
- **Stratification:** Balanced on pathology group and microvascular obstruction (MVO) presence
- **60/20/20 Partition:**
  - Train ($N=60$): 20 Normal, 24 Path MVO+, 16 Path MVO-
  - Val ($N=20$): 6 Normal, 8 Path MVO+, 6 Path MVO-
  - Test ($N=20$): 7 Normal, 8 Path MVO+, 5 Path MVO-
- **5-Fold Cross-Validation:** 5 balanced folds with exactly 20 validation cases per fold (80 training / 20 validation)
- **Patient Leakage:** All folds and splits verified disjoint

---

## 4. Empirical Feature Bank Audit

Extracted to `data/processed/unified/empirical_feature_bank.csv` (314 records):

- **ACDC Cases:** 100 (Cine SSFP, 100% complete biventricular volumetrics)
- **M&Ms Cases:** 114 (Cine SSFP, 100% complete biventricular volumetrics, clinical metadata for age, sex, height, weight)
- **EMIDEC Cases:** 100 (Static LGE CMR, 100% complete tissue viability sizing, clinical demographics and cardiovascular risk factors)
- **Combined Empirical Metrics:**
  - Cine LVEDV: $161.3 \pm 64.4\text{ mL}$ ($N=214$)
  - Cine LVEF: $51.7 \pm 17.7\%$ ($N=214$)
  - Myocardial Mass: $121.4 \pm 38.3\text{ g}$ ($N=314$)
  - Pathological Infarct Burden: $18.71 \pm 9.46\%$ ($N=67$)
  - Pathological MVO Prevalence: $59.7\%$ ($40 / 67$ cases)

---

## 5. Licensing and Provenance Audit

| Dataset | Canonical Source | License | Authorized Use in CRIP |
| :--- | :--- | :--- | :--- |
| **ACDC 2017** | University Hospital of Dijon / CREATIS | Open Academic Benchmark / CC BY-NC-SA 4.0 | Non-commercial research, benchmark evaluation |
| **M&Ms 2020** | Multi-Centre Consortium / MICCAI 2020 | CC BY-NC-SA 4.0 | Non-commercial cross-vendor generalizability research |
| **EMIDEC 2020**| Le2i / University of Burgundy / MICCAI 2020 | CC BY-NC-SA 4.0 | Non-commercial tissue characterization & scar research |

---

## 6. Verification and Test Results

The unified architecture, schemas, split manifests, feature bank, and CLI were validated using deterministic automated tests:

```text
============================== test session starts ==============================
collected 255 items

tests/test_acdc_integration.py ....                                      [  1%]
tests/test_app_init.py .....                                             [  3%]
tests/test_cardiac_metrics.py ....                                       [  5%]
tests/test_dataset_loaders.py ..............                             [ 10%]
tests/test_dataset_registry.py .......................                   [ 19%]
tests/test_ehr_db.py ....                                                [ 21%]
tests/test_emidec_tissue_metrics.py ............                         [ 25%]
tests/test_external_mnms.py ............                                 [ 30%]
tests/test_inspect_acdc.py .....                                         [ 32%]
tests/test_milestone4_integration.py ........................            [ 41%]
tests/test_mri_loader.py ..............                                  [ 47%]
tests/test_pipeline_and_measurements.py ................                 [ 53%]
tests/test_recovery_modeling.py .........................                [ 63%]
tests/test_schemas.py ..........                                         [ 67%]
tests/test_segmentation_pipeline.py ......................               [ 76%]
tests/test_stage1_ui.py ........                                         [ 79%]
tests/test_stage2_evaluation.py .................                        [ 85%]
tests/test_synthetic_cohort.py ....................                      [ 93%]
tests/test_unified_pipeline.py ................                          [100%]

======================= 255 passed, 1 warning in 16.02s ========================
```

### Stop Conditions Compliance:
- **No models retrained.**
- **No checkpoint files overwritten** (`models/mri/best_unet_model.pt` intact).
- **Original ACDC and M&Ms evaluation benchmarks preserved without modification.**
- **No clinical variables fabricated.**
- **No claim of real recovery outcome validation.**
- **No git commits or pushes executed.**
