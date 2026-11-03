# CRIP — Final End-to-End Reproducibility Guide

**Cardiomyopathy Recovery Intelligence Platform (CRIP)**  
**Version:** 2.0 (Post Stage 6 Integration)  
**Supported Platforms:** macOS (Apple Silicon arm64 / Intel x86_64), Ubuntu Linux 22.04+ LTS, Ubuntu on WSL2

---

## 1. System Requirements & Hardware Prerequisites

| Resource | macOS (Apple Silicon) | Ubuntu Linux / WSL2 |
| :--- | :--- | :--- |
| **Operating System** | macOS 14.0+ (Sonoma, Sequoia) | Ubuntu 22.04 or 24.04 LTS |
| **Architecture** | Apple Silicon (`arm64`, M1/M2/M3/M4) | x86_64 or aarch64 |
| **Python Version** | Python 3.11, 3.12, or 3.13 | Python 3.11, 3.12, or 3.13 |
| **RAM** | Minimum 16 GB unified memory | Minimum 16 GB system RAM |
| **Disk Space** | ~25 GB free space for datasets and models | ~25 GB free space |
| **Acceleration** | Apple Metal Performance Shaders (MPS) or CPU | NVIDIA CUDA 12.x or CPU |

---

## 2. Cloning the Repository & Environment Setup

### 2.1 Clone Repository
```bash
git clone https://github.com/your-org/crip.git
cd crip
```

### 2.2 Create and Activate Python Virtual Environment

#### On macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

#### On Ubuntu Linux / WSL2:
```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip libgl1-mesa-glx
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 2.3 Install Dependencies
```bash
pip install -e ".[dev]"
```

*Verification:*
```bash
python -c "import torch, monai, SimpleITK, streamlit; print('Environment OK:', torch.__version__, monai.__version__)"
```

---

## 3. Dataset Acquisition and Directory Structure

Due to data sharing agreements and challenge licensing restrictions, raw medical imaging datasets are **not distributed in Git**. Authorized researchers must acquire the challenge archives and place them into the designated directory structure:

### 3.1 Placement Directory Structure
```text
data/raw/
├── acdc/
│   ├── training/
│   │   ├── patient001/
│   │   │   ├── Info.cfg
│   │   │   ├── patient001_4d.nii.gz
│   │   │   ├── patient001_frame01.nii.gz
│   │   │   ├── patient001_frame01_gt.nii.gz
│   │   │   ├── patient001_frame12.nii.gz
│   │   │   └── patient001_frame12_gt.nii.gz
│   │   └── ... (patient001 to patient100)
│   └── testing/
├── mnms/
│   ├── 211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
│   └── OpenDataset/
│       └── ... (114 multi-vendor cine cases)
└── emidec/
    └── ... (100 delayed-enhancement cases: Case_001 to Case_100)
```

---

## 4. Model Checkpoint Restoration & Integrity Verification

Because neural network checkpoints exceed version control limits, checkpoints must be restored from secure institutional object storage into `models/`.

### 4.1 Expected Checkpoint Paths and SHA-256 Checksums

```text
models/
├── mri/
│   └── best_unet_model.pt
│       SHA-256: 71ea54235803562acc67b8903deecaaf113d45a1c7b443a97ba9fbe6573072c9
│       Size: 4,953,057 bytes
│       Task: In-domain ACDC 4-class Cine U-Net
│
└── multidataset/
    ├── cine_joint_unet.pt
    │   SHA-256: fd4f7da40f756794c99d9e2cad1e06a080701e86b5386afcd686a24b5c12e50d
    │   Size: 1,657,627 bytes
    │   Task: Joint ACDC + M&Ms 4-class Cine U-Net
    │
    └── emidec_lge_unet.pt
        SHA-256: d62fb350fed747d7c69cd8bfe6a78c5cea71ba5857bc28d140ef5f464b588c56
        Size: 19,694,841 bytes
        Task: Experimental EMIDEC 5-class LGE U-Net
```

### 4.2 Automated Checksum Verification Script
```bash
python -c "
import hashlib
from pathlib import Path

expected = {
    'models/mri/best_unet_model.pt': '71ea54235803562acc67b8903deecaaf113d45a1c7b443a97ba9fbe6573072c9',
    'models/multidataset/cine_joint_unet.pt': 'fd4f7da40f756794c99d9e2cad1e06a080701e86b5386afcd686a24b5c12e50d',
    'models/multidataset/emidec_lge_unet.pt': 'd62fb350fed747d7c69cd8bfe6a78c5cea71ba5857bc28d140ef5f464b588c56',
}

for path_str, exp_hash in expected.items():
    p = Path(path_str)
    assert p.is_file(), f'Missing checkpoint: {path_str}'
    actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual_hash == exp_hash, f'Hash mismatch on {path_str}!'
    print(f'Verified: {path_str} (SHA256 matches)')
print('All checkpoints verified successfully.')
"
```

---

## 5. Automated Regression Test Execution

Run the complete 290-test test suite across all subsystems:

```bash
# macOS and Ubuntu/WSL:
MPLCONFIGDIR=/tmp/matplotlib pytest -v
```

Expected output:
```text
290 passed in ~18 seconds
```

---

## 6. End-to-End Pipeline Reproduction

Execute the full reproducible research pipeline using CRIP's unified CLI:

### 6.1 Audit Datasets & Partitions
```bash
python -m crip.pipeline audit
```
*Validates that ACDC (100), M&Ms (114), and EMIDEC (100) cases match the unified cohort split partitions ($N=314$).*

### 6.2 Extract Unified Empirical Feature Bank
```bash
python -m crip.pipeline extract-features
```
*Generates `data/processed/unified/empirical_feature_bank.csv` with exactly 314 records.*

### 6.3 Generate Synthetic Cohorts
```bash
# Generate Baseline literature-grounded cohort (N = 3,000):
python -m crip.synthetic.cohort_generator

# Generate Empirically informed multi-dataset cohort (N = 3,000):
python -m crip.pipeline generate-cohort --method copula
```

### 6.4 Reproduce Recovery Model Training & Benchmarks
```bash
# Train canonical sustained-abstinence benchmark models (Experiment 1):
python -m crip.modeling.train

# Train and evaluate dual all-comers benchmark models (Experiment 2):
python -m crip.pipeline train-recovery --all-comers
```

---

## 7. Launching the Multi-Dataset Streamlit Application

Launch the unified research dashboard:

```bash
python -m streamlit run src/crip/app/main.py
```

Open your browser to:
**`http://localhost:8501`**

### Workspace Navigation:
- **`🏠 0_Overview_Dashboard.py`:** Checkpoint status, dataset registry, and CLI reference.
- **`👤 1_Patient_Profile.py`:** Synthetic patient cohort explorer with multimodal parameter inspection.
- **`🫀 2_MRI_Analysis.py`:**
  - *Tab 1:* ACDC Cine analysis and biventricular volumetrics.
  - *Tab 2:* Cine model comparison (ACDC vs Joint Cine with spatial difference map).
  - *Tab 3:* Dedicated EMIDEC LGE tissue workspace (experimental notice, scar/no-reflow quantification).
- **`📈 3_Recovery_Estimate.py`:** 6-month simulated recovery prediction and waterfall feature attributions.
- **`🔬 4_Evaluation.py`:**
  - *Tab 1:* Real MRI segmentation benchmarks (in-domain, external M&Ms, matched comparison, EMIDEC LGE).
  - *Tab 2:* Bland–Altman measurement agreement (LVEF, LVEDV, LVESV, LVSV).
  - *Tab 3:* Recovery prediction benchmarks (Experiment 1 Sustained Abstinence vs Experiment 2 All-Comers).
  - *Tab 4:* Synthetic cohort explorer (Baseline vs Empirical generator distributions & invariants).
  - *Tab 5:* Scientific figure explorer with PNG, vector SVG, and PDF downloads.
  - *Tab 6:* Scientific audit and clinical governance declarations.
- **`📊 5_Cardiac_Measurements.py`:** Dedicated biventricular measurement analysis.

---

## 8. Cross-Platform Troubleshooting & Notes

1. **Headless Execution & Matplotlib Warnings:**
   - Always ensure `MPLCONFIGDIR=/tmp/matplotlib` is set when running in CI or headless environments to avoid GUI display errors.
2. **Apple Silicon PyTorch Memory Warning:**
   - On macOS with unified memory, MONAI operations default cleanly to CPU for deterministic, bit-level evaluation reproducibility.
3. **WSL2 File Permissions:**
   - Under WSL2, avoid placing the repository inside `/mnt/c/` to prevent NTFS filesystem latency. Store inside `/home/<user>/crip/` for native ext4 performance.
