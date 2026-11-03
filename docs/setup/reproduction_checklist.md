# CRIP — Reproduction Checklist & Operational Verification Matrix

**Document Purpose:** This document provides an itemized, executable checklist for reproducing the Cardiomyopathy Recovery Intelligence Platform (CRIP). It details verification criteria, required file artifacts, row counts, expected checksums, and cross-platform operating notes for both reproduction pathways:
- **Path A:** Synthetic-Only Quick Start (Zero external downloads, no GPU required, ~3–5 minutes).
- **Path B:** Complete Multi-Dataset Research Reproduction (Full medical imaging pipeline across ACDC, M&Ms, and EMIDEC).

---

## 1. Environment & Platform Compatibility Matrix

| Operating System | Recommended Environment | Acceleration Backend | Hardware Requirements | Verified Status |
| :--- | :--- | :---: | :--- | :---: |
| **macOS (Apple Silicon)** | Native zsh (`arm64`) | Apple Silicon MPS (`mps`) | M1/M2/M3/M4, 8–16 GB RAM | **Primary Dev Environment** |
| **macOS (Intel)** | Native zsh (`x86_64`) | CPU fallback (`cpu`) | Intel Core i7/i9, 16 GB RAM | **Fully Supported** |
| **Ubuntu Linux (22.04 / 24.04)** | Native bash (`ext4`) | NVIDIA CUDA (`cuda`) / CPU | 8–16 GB RAM, CUDA 12+ (opt.) | **Fully Supported** |
| **Ubuntu on Windows (WSL2)** | Linux bash (`~/...`) | NVIDIA CUDA (`cuda`) / CPU | 16 GB RAM, WSL2 mirrored net | **Secondary Repro Environment** |
| **Native Windows 10/11** | PowerShell (`x64`) | CPU (`cpu`) | 8–16 GB RAM | *Experimental (Path A verified)* |

> [!WARNING]
> **WSL2 Storage Directive:** Always clone and execute inside the native Linux virtual disk (`/home/<user>/...` or `~/...`). Storing files on Windows NTFS mounts (`/mnt/c/...`) causes extreme 9p translation overhead (10× slowdown) and file-locking errors during multi-slice NIfTI I/O.

---

## 2. Path A — Synthetic-Only Quick Start Checklist

Use this checklist for rapid evaluation, classroom demonstration, or UI exploration without downloading real clinical MRI scans.

### Execution Steps:
- [ ] **Step A.1 — Clone Repository:**
  ```bash
  git clone https://github.com/cstever0/AlcoholicRecovery.git
  cd AlcoholicRecovery
  ```
- [ ] **Step A.2 — Setup Virtual Environment:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- [ ] **Step A.3 — Install Dependencies:**
  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .
  ```
- [ ] **Step A.4 — Environment Smoke Test:**
  ```bash
  python -c "import crip, streamlit, torch, sklearn, plotly; print('✓ CRIP Environment Verified')"
  ```
  *Verification Condition:* Outputs `✓ CRIP Environment Verified` without import errors.
- [ ] **Step A.5 — Generate Baseline Synthetic Cohort:**
  ```bash
  python -m crip.synthetic.cohort_generator --n-patients 3000 --seed 42 --abstinence-prob 0.70 --output-dir data/synthetic
  ```
  *Verification Condition:* `data/synthetic/synthetic_cardiomyopathy_cohort.csv` exists and contains 3,000 rows. `data/synthetic/ehr_synthetic.db` exists.
- [ ] **Step A.6 — Train Baseline Recovery Model:**
  ```bash
  python -m crip.modeling.train
  ```
  *Verification Condition:* `models/recovery/recovery_pipeline.joblib` exists. Test ROC-AUC reports **0.9565** on the 419 held-out test patients.
- [ ] **Step A.7 — Launch Streamlit Application:**
  ```bash
  python -m streamlit run src/crip/app/main.py
  ```
  *Verification Condition:* Browser opens `http://localhost:8501`.

### Path A Operational Verification:
| Application Page / Feature | Expected Behavior in Path A | Pass Criteria |
| :--- | :--- | :---: |
| **Page 0: Overview Dashboard** | System indicators active; raw MRI counts show 0 (unloaded). | [ ] Verified |
| **Page 1: Patient Profile & Cohort** | All 3,000 synthetic patient records interactive with full filtering. | [ ] Verified |
| **Page 2: Cardiac MRI Analysis** | Graceful notice: CMR scans omitted in Path A (app does NOT crash). | [ ] Verified |
| **Page 3: Recovery Estimation** | Calibrated recovery prediction meter, SHAP waterfall, What-If sliders, FHIR export fully operational. | [ ] Verified |
| **Page 4: Model Evaluation** | Synthetic recovery benchmarks (Experiment 1 & 2) and synthetic cohort explorer fully functional. | [ ] Verified |
| **Page 5: Cardiac Measurements** | Graceful notice: Physical scans omitted in Path A (app does NOT crash). | [ ] Verified |
| **Demonstration Cases (A, B, C)** | Interactive execution functional via CLI and UI. | [ ] Verified |

---

## 3. Path B — Complete Multi-Dataset Research Reproduction Checklist

Use this checklist for rigorous, complete end-to-end reproduction of deep learning segmentation, physical chamber volumetrics, multi-vendor generalizability, delayed-enhancement tissue metrics, and empirical recovery modeling.

### Phase 1: Environment & System Preparation
- [ ] **Step B.1.1 — Dedicated Reproduction Branch:**
  ```bash
  git checkout -b reproduction-run
  ```
- [ ] **Step B.1.2 — Environment Activation & Build Tools:**
  ```bash
  source .venv/bin/activate
  export MPLCONFIGDIR=/tmp/matplotlib
  ```

### Phase 2: Raw MRI Acquisition & Layout Verification
- [ ] **Step B.2.1 — ACDC 2017 Dataset:** 100 cases (`patient001`–`patient100`) extracted under `data/raw/acdc/training/`.
- [ ] **Step B.2.2 — M&Ms 2020 Dataset:** 114 paired cases (80 in `Training/Labeled/`, 34 in `Validation/`) under `data/raw/mnms/`.
- [ ] **Step B.2.3 — EMIDEC 2020 Dataset:** 100 annotated cases (`Case_001`–`Case_100` or `Case_N001`–`Case_P100`) under `data/raw/emidec/`.
- [ ] **Step B.2.4 — Windows/WSL NTFS Metadata Cleanup:**
  ```bash
  find data/raw -type f -name '*:Zone.Identifier' -print
  find data/raw -type f -name '*:Zone.Identifier' -delete
  ```
  *Verification Condition:* Zero `:Zone.Identifier` sidecar files remaining.
- [ ] **Step B.2.5 — Ingestion Integrity Audit:**
  ```bash
  python -m crip.pipeline audit
  ```
  *Verification Condition:* Reports all 3 datasets available (ACDC 100, M&Ms 114 paired, EMIDEC 100 annotated).

### Phase 3: Split Manifest Verification & Repair
- [ ] **Step B.3.1 — Cohort Split Synchronization:**
  ```bash
  python -m crip.pipeline prepare --repair-missing-splits
  ```
  *Verification Condition:* `data/processed/unified/cohort_splits.json` contains:
  - ACDC: 60 train, 20 val, 20 test.
  - M&Ms: 60 train, 20 val, 34 holdout.
  - EMIDEC: 60 train, 20 val, 20 test.

### Phase 4: Physical Cardiac MRI Preprocessing (Source Measurements)
- [ ] **Step B.4.1 — ACDC Biventricular Volumetric Extraction:**
  ```bash
  python -m crip.mri.pipeline --data-dir data/raw/acdc/training --output-dir data/processed/acdc
  ```
  *Target Artifact:* `data/processed/acdc/cardiac_measurements.csv` (100 rows).
- [ ] **Step B.4.2 — M&Ms Multi-Vendor Metric Extraction:**
  ```bash
  python -m crip.evaluation.external_mnms
  ```
  *Target Artifact:* `data/processed/mnms/external_evaluation.csv` (114 rows).
- [ ] **Step B.4.3 — EMIDEC Tissue Quantification:**
  ```bash
  python -m crip.emidec.tissue_metrics
  ```
  *Target Artifact:* `data/processed/emidec/tissue_measurements.csv` (100 rows).
- [ ] **Step B.4.4 — Preprocessing Verification Smoke Test:**
  ```bash
  python -c "
  import pandas as pd
  from pathlib import Path
  for p, n, count in [
      (Path('data/processed/acdc/cardiac_measurements.csv'), 'ACDC', 100),
      (Path('data/processed/mnms/external_evaluation.csv'), 'M&Ms', 114),
      (Path('data/processed/emidec/tissue_measurements.csv'), 'EMIDEC', 100),
  ]:
      assert p.exists(), f'Missing {p}'
      df = pd.read_csv(p)
      assert len(df) == count, f'{n} count mismatch: expected {count}, got {len(df)}'
      print(f'✓ {n:<8}: {len(df)} cases verified ({p})')
  print('✓ All 3 preprocessing source CSVs verified!')
  "
  ```
  *Verification Condition:* All three source CSVs pass row count validation.

### Phase 5: Unified Empirical Feature Bank & Synthetic Generation
- [ ] **Step B.5.1 — Compile Empirical Feature Bank:**
  ```bash
  python -m crip.pipeline features
  ```
  *Target Artifact:* `data/processed/unified/empirical_feature_bank.csv` (314 rows).
- [ ] **Step B.5.2 — Synthesize Empirically Informed Cohort:**
  ```bash
  python -m crip.pipeline synthesize --n-patients 3000 --empirical --seed 42 --export
  ```
  *Target Artifact:* `data/synthetic/synthetic_cardiomyopathy_cohort.csv` (3,000 rows with 75% ACM / 25% Ischemic mixture).

### Phase 6: Recovery Prediction Model Training
- [ ] **Step B.6.1 — Recovery Preflight Dry-Run:**
  ```bash
  python -m crip.pipeline train --task recovery --dry-run
  ```
  *Verification Condition:* Preflight reports all 5 prerequisite files present with zero errors.
- [ ] **Step B.6.2 — Multi-Dataset Recovery Benchmarking:**
  ```bash
  python -m crip.pipeline train --task recovery
  ```
  *Verification Condition:* Empirical Logistic Regression achieves ROC-AUC **0.9048**; Random Forest achieves **0.9052**. 24 figures exported to `docs/evaluation/multidataset/synthetic_integration/`.

### Phase 7: MRI Segmentation Model Training (or Checkpoint Verification)
- [ ] **Step B.7.1 — Baseline Cine U-Net:**
  ```bash
  # Check existing weights or train 25 epochs
  python src/crip/mri/train_segmentation.py --epochs 25
  ```
  *Target Artifact:* `models/mri/best_unet_model.pt` (SHA-256: `71ea5423...`).
- [ ] **Step B.7.2 — Joint Cine U-Net Preflight & Training:**
  ```bash
  python -m crip.pipeline train --task cine --dry-run
  python -m crip.pipeline train --task cine --epochs 20
  ```
  *Target Artifact:* `models/multidataset/cine_joint_unet.pt` (SHA-256: `fd4f7da4...`).
- [ ] **Step B.7.3 — Dedicated EMIDEC LGE U-Net Preflight & Training:**
  ```bash
  python -m crip.pipeline train --task lge --dry-run
  python -m crip.pipeline train --task lge --epochs 30
  ```
  *Target Artifact:* `models/multidataset/emidec_lge_unet.pt` (SHA-256: `630d637e...` or `d62fb350...`).
- [ ] **Step B.7.4 — Checkpoint Checksum Inspection:**
  ```bash
  shasum -a 256 models/mri/best_unet_model.pt models/multidataset/cine_joint_unet.pt models/multidataset/emidec_lge_unet.pt
  ```

### Phase 8: Held-Out Model Evaluation & Bland–Altman Inputs
- [ ] **Step B.8.1 — ACDC Held-Out Segmentation Evaluation (Mandatory for Bland–Altman):**
  ```bash
  python -m crip.evaluation.segmentation_metrics \
      --data-dir data/raw/acdc/training \
      --checkpoint models/mri/best_unet_model.pt \
      --split test \
      --output-csv data/processed/acdc/segmentation_evaluation.csv \
      --output-summary data/processed/acdc/segmentation_evaluation_summary.json
  ```
  *Verification Condition:* `data/processed/acdc/segmentation_evaluation.csv` contains 20 rows. Mean Dice is **0.839**, LVEF MAE is **3.86%**.
- [ ] **Step B.8.2 — Joint Cine Matched Evaluation:**
  ```bash
  python -m crip.pipeline evaluate
  ```
  *Verification Condition:* Evaluates 54 holdout cases (20 ACDC + 34 M&Ms).
- [ ] **Step B.8.3 — Held-Out Inference Smoke Test:**
  ```bash
  python -c "
  import torch
  from crip.mri.acdc_loader import ACDCDatasetLoader
  from crip.mri.segmentation_inference import predict_cardiac_frame_volume, build_unet_model

  loader = ACDCDatasetLoader('data/raw/acdc/training')
  patient = loader.load_patient('patient007')
  model = build_unet_model(in_channels=1, out_channels=4, device='cpu')
  chk = torch.load('models/mri/best_unet_model.pt', map_location='cpu', weights_only=False)
  model.load_state_dict(chk['model_state_dict'])
  pred = predict_cardiac_frame_volume(patient.frame_ed.image_data, model, device='cpu')
  print('✓ Held-out patient007 ED inference verified:', pred.shape)
  "
  ```

### Phase 9: Scientific Figure Generation & Export
- [ ] **Step B.9.1 — Poster Figures 1–8 (ACDC Volumetrics & Recovery):**
  ```bash
  python -m crip.evaluation.poster_figures
  ```
  *Target Directory:* `docs/evaluation/poster_figures/` (8 figures in PNG, SVG, PDF).
- [ ] **Step B.9.2 — M&Ms Multi-Vendor Figures 1–5:**
  ```bash
  python -m crip.evaluation.poster_figures_mnms
  ```
  *Target Directory:* `docs/evaluation/external_datasets/mnms/` (5 figures in PNG, SVG, PDF).
- [ ] **Step B.9.3 — EMIDEC Viability Figures 1–6:**
  ```bash
  python -m crip.evaluation.poster_figures_emidec
  ```
  *Target Directory:* `docs/evaluation/external_datasets/emidec/` (6 figures in PNG, SVG, PDF).

### Phase 10: Automated Testing & Streamlit Verification
- [ ] **Step B.10.1 — Complete Test Suite Execution:**
  ```bash
  MPLCONFIGDIR=/tmp/matplotlib pytest -q
  ```
  *Verification Condition:* **353 passed** in ~24s.
- [ ] **Step B.10.2 — Launch Streamlit Web Application:**
  ```bash
  python -m streamlit run src/crip/app/main.py
  # Under WSL2:
  python -m streamlit run src/crip/app/main.py --server.address 0.0.0.0 --server.port 8501
  ```
- [ ] **Step B.10.3 — Page-by-Page Interactive Verification:**
  - Page 0: All datasets green, all checkpoints verified.
  - Page 1: 3,000 synthetic patient cohort filters active.
  - Page 2 Tab 1: ACDC contour toggle & slice navigation.
  - Page 2 Tab 2: 4-panel cine comparison & disagreement map.
  - Page 2 Tab 3: EMIDEC 5-class LGE viewer & experimental warning.
  - Page 3: Recovery prediction meter, SHAP waterfall, What-If sliders.
  - Page 4: Real MRI benchmarks, Bland–Altman agreement plots, calibration curves (Exp 1 & 2), figure explorer.
  - Page 5: Cardiac chamber measurements & paired AI-vs-expert comparison table.

---

## 4. Itemized Verification Artifact Matrix

| Artifact Path | Producing Command / Script | Expected Row / Record Count | Critical Consumers |
| :--- | :--- | :---: | :--- |
| `data/processed/acdc/cardiac_measurements.csv` | `python -m crip.mri.pipeline ...` | 100 rows | Feature bank (`pipeline features`), Page 5 |
| `data/processed/mnms/external_evaluation.csv` | `python -m crip.evaluation.external_mnms` | 114 rows | Feature bank (`pipeline features`), Page 4 Tab 1 |
| `data/processed/emidec/tissue_measurements.csv` | `python -m crip.emidec.tissue_metrics` | 100 rows | Feature bank (`pipeline features`), Page 2 Tab 3 |
| `data/processed/unified/cohort_splits.json` | `python -m crip.pipeline prepare` | 314 study keys | Data loaders, training pipelines, Page 0 |
| `data/processed/unified/empirical_feature_bank.csv`| `python -m crip.pipeline features` | 314 rows | Synthetic generator (`pipeline synthesize`) |
| `data/synthetic/synthetic_cardiomyopathy_cohort.csv` | `cohort_generator` or `pipeline synthesize` | 3,000 rows | Page 1, Page 3, Page 4 Tab 4 |
| `models/recovery/recovery_pipeline.joblib` | `python -m crip.modeling.train` | Serialized model | Page 3 Recovery Estimator |
| `models/mri/best_unet_model.pt` | `python src/crip/mri/train_segmentation.py` | PyTorch checkpoint | Page 2 Tab 1, segmentation evaluation |
| `models/multidataset/cine_joint_unet.pt` | `python -m crip.pipeline train --task cine` | PyTorch checkpoint | Page 2 Tab 2 Cine Comparison |
| `models/multidataset/emidec_lge_unet.pt` | `python -m crip.pipeline train --task lge` | PyTorch checkpoint | Page 2 Tab 3 LGE Workspace |
| `data/processed/acdc/segmentation_evaluation.csv` | `python -m crip.evaluation.segmentation_metrics` | 20 rows | Page 4 Tab 2 (Bland–Altman), Poster figures |

---

## 5. Known Scientific & Operational Limitations

1. **Synthetic Outcomes:** Recovery endpoints ($\Delta\text{LVEF} \ge 10\%$) and PEth biomarker concentrations are generated from published empirical distributions; they are computational research estimates and do not represent prospectively collected clinical trial patients.
2. **ACDC Cohort Scope:** The ACDC cohort comprises DCM, HCM, MINF, and RV abnormalities; it does not contain confirmed alcoholic cardiomyopathy cases.
3. **EMIDEC Kinematic Invariant:** EMIDEC contains static delayed-enhancement CMR scans; computing dynamic ejection fraction (LVEF) from static LGE is strictly prohibited.
4. **CAD vs. ACM Pathology Boundary:** EMIDEC models acute ischemic myocardial infarction and microvascular obstruction resulting from coronary occlusion, whereas alcoholic cardiomyopathy presents with diffuse non-ischemic mid-wall fibrosis.
5. **Cross-Platform Numerical Variations:** Minor floating-point differences between Apple Silicon MPS, Linux NVIDIA CUDA, and CPU vectorization cause minor variations in neural network loss convergence and Matplotlib figure anti-aliasing.
