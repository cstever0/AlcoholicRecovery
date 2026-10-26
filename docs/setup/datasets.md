# ACDC Dataset Acquisition & Directory Organization Guide

This document describes the manual download, verification, and file layout of the **Automated Cardiac Diagnosis Challenge (ACDC 2017)** dataset for CRIP.

---

## 1. Challenge Overview & Licensing

The ACDC dataset was curated by the CREATIS laboratory, University of Lyon, and University Hospital of Dijon for the MICCAI 2017 challenge.

- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Restrictions:** Strictly non-commercial academic research use only.
- **Mandatory Citation:**
  > Bernard O, Lalande A, Zotti C, Cervenansky F, et al. *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?* **IEEE Transactions on Medical Imaging**, 2018; 37(11):2514-2525. DOI: 10.1109/TMI.2018.2837502.

---

## 2. Acquisition Steps

1. Create an account on the challenge web portal: [humanheart-project.creatis.insa-lyon.fr](https://humanheart-project.creatis.insa-lyon.fr).
2. Download the training dataset (`training.zip`, ~1.8 GB compressed) and testing dataset (`testing.zip`, ~900 MB compressed).
3. Unzip the archives into `data/raw/acdc/`.

---

## 3. Directory Layout Specification

Ensure the local path strictly matches:

```text
AlcoholicRecovery/
└── data/
    └── raw/
        └── acdc/
            ├── training/
            │   ├── patient001/
            │   │   ├── Info.cfg
            │   │   ├── patient001_4d.nii.gz
            │   │   ├── patient001_frame01.nii.gz
            │   │   ├── patient001_frame01_gt.nii.gz
            │   │   ├── patient001_frame12.nii.gz
            │   │   └── patient001_frame12_gt.nii.gz
            │   └── ... (patient001 to patient100)
            └── testing/
                ├── patient101/
                │   ├── Info.cfg
                │   ├── patient101_4d.nii.gz
                │   ├── patient101_frame01.nii.gz
                │   └── patient101_frame14.nii.gz
                └── ... (patient101 to patient150)
```

> [!NOTE]
> The **training** partition contains 100 patients with public expert segmentation masks (`_gt.nii.gz`) for End-Diastole (ED) and End-Systole (ES).
> The **testing** partition contains 50 patients; in the official challenge distribution, testing segmentation masks were retained by the organizers for online evaluation and are not public.

---

## 4. Verification CLI

Run the dataset validation tool from the project root:

```bash
python -m crip.mri.inspect_acdc --data-dir data/raw/acdc/training --output data/processed/acdc/acdc_validation_report.json
```

**Diagnostic Subgroup Distribution (100 Training Patients):**
- **DCM:** 20 patients (Dilated Cardiomyopathy)
- **HCM:** 20 patients (Hypertrophic Cardiomyopathy)
- **MINF:** 20 patients (Myocardial Infarction)
- **NOR:** 20 patients (Normal controls)
- **RV:** 20 patients (Right Ventricular abnormality)
