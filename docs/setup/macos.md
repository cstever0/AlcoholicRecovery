# macOS Setup Guide for CRIP

This guide details configuration and execution of the Cardiomyopathy Recovery Intelligence Platform (CRIP) on macOS, covering both Apple Silicon (M1/M2/M3/M4) and Intel architectures.

---

## 1. System Requirements & Homebrew Installation

### Prerequisites
- macOS Monterey (12.0) or later (macOS Sonoma 14+ recommended).
- Apple Silicon (arm64) or Intel (x86_64).
- Minimum 8 GB RAM (16 GB recommended).
- Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Install Homebrew
If not already installed, set up Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## 2. Python Environment Setup

Install Python 3.11, 3.12, or 3.13 via Homebrew:
```bash
brew install python@3.12
```

Create an isolated virtual environment in the repository root:
```bash
cd AlcoholicRecovery
python3.12 -m venv .venv
source .venv/bin/activate
```

Confirm the active Python executable:
```bash
which python
# Output: /path/to/AlcoholicRecovery/.venv/bin/python
```

---

## 3. Dependency Installation

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 4. Verifying Apple Silicon Metal Performance Shaders (MPS)

On Apple Silicon Macs, PyTorch leverages the unified GPU memory architecture via Metal Performance Shaders (MPS):

```bash
python -c "import torch; print('MPS Built:', torch.backends.mps.is_built(), '| MPS Available:', torch.backends.mps.is_available())"
```

**Expected Output:**
```text
MPS Built: True | MPS Available: True
```

The CRIP U-Net segmentation trainer (`src/crip/mri/train_segmentation.py`) automatically selects `mps` when running on Apple Silicon.

---

## 5. Troubleshooting macOS Specifics

### Matplotlib Cache Directory Permission
In sandboxed or restricted terminal sessions, Matplotlib may warn that `~/.matplotlib` is not writable. Resolve by exporting:
```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

### SimpleITK / ITK NIfTI Scale Warnings
When loading ACDC NIfTI files, you may observe:
```text
WARNING: NiftiImageIO: patientXXX_frameYY.nii.gz has unexpected scales in sform
```
This is a standard ITK coordinate header notice and has zero impact on segmentation or volumetric calculation.
