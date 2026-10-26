# Model Training & Evaluation Guide

This guide details the training workflows for both machine learning components in CRIP:
1. The **MONAI 2D U-Net** for automated cardiac MRI segmentation.
2. The **Scikit-learn Logistic Regression Pipeline** for 6-month recovery prediction.

---

## 1. Automated Cardiac MRI Segmentation (MONAI 2D U-Net)

### Architecture
- Model: MONAI `UNet` (2D multi-class segmentation).
- Spatial Dimensions: 2D (short-axis slices resized to $224 \times 224$ during training, restored to native spatial matrix size during inference).
- In Channels: 1 (grayscale CMR intensity normalized to $[0, 1]$ per volume).
- Out Channels: 4 (0: Background, 1: RV Cavity, 2: LV Myocardium, 3: LV Cavity).
- Loss Function: `CompoundDiceCELoss` (Dice Loss + Cross-Entropy Loss).

### Stratified Patient Splitting (No Slice Leakage)
To prevent data contamination across train and evaluation sets, partitioning is strictly patient-level:
- **Cohort:** 100 annotated ACDC training patients.
- **Split:** 60 Train (60%), 20 Validation (20%), 20 Held-Out Test (20%).
- **Stratification:** Exactly 12 train, 4 val, and 4 test patients for each of the 5 diagnostic groups (DCM, HCM, MINF, NOR, RV).
- **Manifest:** Saved at `data/processed/acdc/dataset_split.json`.

### Training Command
```bash
python src/crip/mri/train_segmentation.py --epochs 25 --batch-size 16 --lr 1e-3
```

- **Device Support:** Auto-selects `mps` on Apple Silicon, `cuda` on NVIDIA GPUs, or `cpu`.
- **Output Checkpoint:** `models/mri/best_unet_model.pt`
- **Output History:** `models/mri/training_history.json`

### Evaluation on Held-Out Test Patients
```bash
python -c "from crip.evaluation.segmentation_metrics import run_cohort_segmentation_evaluation; run_cohort_segmentation_evaluation()"
```

---

## 2. Recovery Prediction Pipeline (Scikit-Learn)

### Modeling Dataset & Leakage Prevention
- **Population:** 2,095 synthetic patients with sustained alcohol abstinence.
- **Feature Whitelist:** Strictly restricted to 20 baseline variables:
  - CMR Volumetrics: `baseline_lvef`, `edv_ml`, `esv_ml`, `stroke_volume_ml`, `rvedv_ml`, `rvef_percent`, `myocardial_mass_g`.
  - PEth Toxicology: `peth_value_ng_ml`, `peth_available`, `reported_drinks_per_week`, `time_since_last_drink_days`, `years_heavy_drinking`, `alcohol_history_available`.
  - Genetics: `ttn_classification`.
  - CMR Tissue Fibrosis: `lge_status`, `lge_pattern`, `lge_extent_percent`, `lge_available`.
  - Demographics: `age`, `biological_sex`.
- **Forbidden Columns:** All post-baseline variables (`followup_lvef`, `delta_lvef`, adverse events) are strictly excluded from feature inputs.

### Training & Serialization Command
```bash
python -m crip.modeling.train
```

### Feature Modality Ablation
To quantify the relative prognostic utility of imaging vs. toxicology vs. genetics:
```bash
python -m crip.modeling.train --run-ablation
```

**Serialized Artifacts:**
- `models/recovery/recovery_pipeline.joblib`
- `models/recovery/metadata.json`
- `models/recovery/feature_schema.json`
- `models/recovery/splits.json`
- `models/recovery/test_metrics.json`
