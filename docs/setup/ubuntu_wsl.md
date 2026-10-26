# Ubuntu Linux & Windows Subsystem for Linux (WSL2) Setup Guide

This guide covers setting up CRIP on native Ubuntu Linux (22.04 LTS or 24.04 LTS) and Ubuntu running under WSL2 on Windows 10/11.

---

## 1. System Package Prerequisites

Update the APT package manager and install build essentials and Python headers:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git build-essential
```

---

## 2. WSL2 Specific Instructions

### A. Update WSL2 in Windows PowerShell
Run PowerShell as Administrator:
```powershell
wsl --update
```

### B. Filesystem Placement (Critical Performance Tip)
Always clone the repository into the **native Linux virtual ext4 filesystem** (e.g., `~/projects/AlcoholicRecovery` or `/home/<username>/AlcoholicRecovery`).
> [!WARNING]
> DO NOT clone or run CRIP under `/mnt/c/Users/...`. Accessing the Windows NTFS mount from WSL2 results in severe file I/O latency when processing NIfTI imaging slices and can trigger file-locking errors during virtual environment creation.

### C. Accessing the Streamlit UI from Windows
When you run Streamlit inside WSL2:
```bash
python -m streamlit run src/crip/app/main.py
```
Open your standard Windows web browser (Chrome, Edge, Firefox) and navigate to:
```
http://localhost:8501
```
WSL2 automatically bridges localhost ports to Windows. If port forwarding fails, bind to all interfaces:
```bash
python -m streamlit run src/crip/app/main.py --server.address 0.0.0.0
```

---

## 3. Virtual Environment & Dependency Installation

```bash
cd AlcoholicRecovery
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 4. Optional NVIDIA GPU Acceleration (CUDA)

If your system (or Windows host with WSL2 GPU pass-through) includes an NVIDIA GPU:

Verify CUDA support inside the virtual environment:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is available, CRIP automatically selects `cuda` for 2D U-Net training and inference.
If CUDA is not present, CRIP falls back to multi-threaded CPU execution without code changes.
