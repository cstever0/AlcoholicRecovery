# CRIP — Cine Model Comparison Patient Loading Bug Fix Report

**Platform:** Cardiomyopathy Recovery Intelligence Platform (CRIP)  
**Location:** `/Users/cstever/Desktop/AlcoholicRecovery`  
**Date:** 2026-11-02  
**Component:** `src/crip/app/components/cine_comparison.py` & `src/crip/mri/acdc_loader.py`  
**Status:** Resolved & Verified (308 Passing Automated Tests)  

---

## 1. Issue Summary

When navigating to **MRI Analysis → Tab 2: Cine Model Comparison (ACDC + M&Ms)** in the Streamlit application and selecting ACDC case `patient001`, the interface displayed the following error banner:

```text
Error loading patient001: 'int' object has no attribute 'upper'
```

---

## 2. Root Cause Analysis & Traceback

### 2.1 Traceback Reproduction
Direct reproduction in Python isolated the failure:

```python
from crip.mri.acdc_loader import ACDCDataLoader

loader = ACDCDataLoader()
meta = loader.get_patient_metadata("patient001")
frame_idx = meta.ed_frame  # int (e.g. 1)
loader.load_cardiac_frame("patient001", frame_idx)
```

```text
Traceback (most recent call last):
  File "<string>", line 6, in <module>
    loader.load_cardiac_frame('patient001', frame_idx)
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cstever/Desktop/AlcoholicRecovery/src/crip/mri/acdc_loader.py", line 220, in load_cardiac_frame
    phase_upper = phase.upper()
                  ^^^^^^^^^^^
AttributeError: 'int' object has no attribute 'upper'
```

### 2.2 Root Cause Details
1. **Failing Call Site:** `src/crip/app/components/cine_comparison.py`, line 170:
   ```python
   meta = acdc_loader.get_patient_metadata(study_id)
   frame_idx = meta.ed_frame if phase_key == "ED" else meta.es_frame
   frame = acdc_loader.load_cardiac_frame(study_id, frame_idx)
   ```
2. **Function Signature:** `ACDCDataLoader.load_cardiac_frame(self, patient_id: str, phase: str = "ED", load_mask: bool = True) -> ACDCCardiacFrame`
3. **Type Mismatch:** The caller passed `frame_idx` (an integer, e.g. `1` or `12`) as the second positional argument (`phase`), which expected a cardiac phase string (`"ED"` or `"ES"`). Inside `load_cardiac_frame`, calling `phase.upper()` on an integer raised `AttributeError: 'int' object has no attribute 'upper'`.
4. **Additional Discrepancies at Call Site:**
   - Attribute Name: The component attempted to read `frame.ground_truth_mask`, but `ACDCCardiacFrame` defines `mask_data`.
   - Dimension Ordering: SimpleITK loads ACDC volumes as `(slices, height, width)`, whereas the component and M&Ms loader expect `(height, width, slices)` for short-axis slice indexing (`img_3d[:, :, slice_idx]`).

---

## 3. Code Modifications

### 3.1 `src/crip/app/components/cine_comparison.py`
Corrected the call site to use explicit keyword arguments and standardized 3D volume dimensions:
- Replaced ambiguous positional call with `acdc_loader.load_cardiac_frame(study_id, phase=phase_key, load_mask=True)`.
- Added transposition from `(slices, height, width)` to `(height, width, slices)` via `np.transpose(..., (1, 2, 0))` to match M&Ms convention.
- Safely extracted `frame.mask_data` (and supported `frame.ground_truth_mask`).

```python
if is_acdc:
    frame = acdc_loader.load_cardiac_frame(study_id, phase=phase_key, load_mask=True)
    # ACDC arrays are stored as (slices, height, width).
    # Transpose to (height, width, slices) to standardize with M&Ms convention.
    img_3d = np.transpose(frame.image_data, (1, 2, 0))
    gt_mask_3d = (
        np.transpose(frame.mask_data, (1, 2, 0))
        if frame.mask_data is not None
        else None
    )
    spacing = frame.voxel_spacing
```

### 3.2 `src/crip/mri/acdc_loader.py`
Hardened `ACDCDataLoader.load_cardiac_frame` and `ACDCCardiacFrame`:
1. **Type & Enum Support:** Supported both string phases (`"ED"`, `"ES"`, `"End-Diastole"`, `"End-Systole"`) and `CardiacPhase` enum values (`CardiacPhase.ED`, `CardiacPhase.ES`).
2. **Explicit Type Validation:** Rejects non-string, non-enum arguments (such as integers) with an informative `TypeError`:
   ```python
   raise TypeError(
       f"Cardiac phase must be a string ('ED' or 'ES') or CardiacPhase enum, "
       f"got {type(phase).__name__}: {phase}"
   )
   ```
3. **Invalid String Rejection:** Rejects invalid phase strings with a descriptive `ValueError`.
4. **Backward Compatibility Alias:** Added `@property def ground_truth_mask` on `ACDCCardiacFrame` aliasing `self.mask_data`.

---

## 4. Verification and Regression Testing

### 4.1 Automated Test Suite (`tests/test_cine_comparison_loading.py`)
Added 12 targeted unit, integration, and Streamlit AppTest tests covering:
- ACDC `patient001` (training smoke case) loading in ED and ES.
- ACDC `patient007` (held-out test case) loading in ED and ES.
- `CardiacPhase` enum resolution and case-insensitive string parsing.
- Explicit rejection of integers (`TypeError`) and unknown phase strings (`ValueError`).
- 3D array transposition and slice dimension conformity with 2D U-Net inference (`predict_slice_mask`).
- M&Ms case loading, phase extraction, and canonical label remapping.
- Visual overlay generation (`create_cine_overlay`) and AI difference map (`create_difference_map`).
- Graceful degradation when model checkpoints or dataset paths are missing.
- Streamlit `AppTest` verifying interactive switching across patients (`patient001` $\to$ `patient007`), phases (`ED` $\to$ `ES`), and datasets (ACDC $\to$ M&Ms) without exceptions.

### 4.2 Full Repository Test Execution
```bash
MPLCONFIGDIR=/tmp/matplotlib pytest -q
```
**Result:** **308 passed, 1 warning in 21.12s** (100% pass rate across all 26 test modules).

### 4.3 Streamlit Interactive Verification
Programmatic execution via Streamlit's official `AppTest` test harness validated:
- `patient001` loads without errors.
- `patient007` loads without errors.
- Switching to `ES` phase re-renders contours and metrics without errors.
- Switching to M&Ms multi-vendor studies loads and remaps contours without errors.
- Zero `AttributeError` or `st.error` exceptions observed.

---

## 5. Constraints Compliance Check

- **Model Weights:** Untouched; no models were retrained.
- **Dataset Splits:** Preserved intact (`docs/acdc/dataset_split.json` and `cohort_splits.json`).
- **Evaluation Benchmarks:** Canonical metrics (e.g. LR 0.9565, Joint Cine 54-case evaluation) unchanged.
- **Git State:** Uncommitted changes confined to the fix, new regression tests, and documentation.
