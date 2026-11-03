"""M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Segmentation) Loader.

Discovers and parses 4D cine CMR short-axis acquisitions and multi-class
ground truth masks from the official M&Ms 2020 Challenge dataset:
- Scanners: Siemens, Philips, GE, Canon
- Centres: 4 clinical centres across Spain and Germany
- Pathologies: DCM, HCM, NOR, HHD, ARV, etc.

CRITICAL LABEL CONVENTION NOTICE:
In M&Ms ground-truth annotations:
  Label 1 = Left Ventricular (LV) Cavity
  Label 2 = Left Ventricular Myocardium
  Label 3 = Right Ventricular (RV) Cavity
This is inverted relative to ACDC (1=RV, 3=LV).
The loader supports optional automated mapping to the canonical ACDC convention
(`mapped_to_canonical=True`), while preserving raw labels in metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd

from crip.datasets.labels import (
    MNMSLabel,
    MNMS_LABEL_NAMES,
    map_mnms_to_canonical,
)
from crip.datasets.schemas import (
    CardiacPhase,
    CaseMetadata,
    DatasetStatus,
    ModalityType,
)

logger = logging.getLogger("crip.datasets.mnms_loader")

DEFAULT_MNMS_ROOT = Path("data/raw/mnms")
DEFAULT_MNMS_CSV = DEFAULT_MNMS_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"


class MNMSDataLoader:
    """Specialized data loader for the M&Ms 2020 Multi-Centre Multi-Vendor CMR dataset."""

    def __init__(
        self,
        root_dir: Optional[Union[Path, str]] = None,
        csv_path: Optional[Union[Path, str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else DEFAULT_MNMS_ROOT
        self.csv_path = Path(csv_path) if csv_path else (self.root_dir / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")

        self.split_dirs = [
            self.root_dir / "Training" / "Labeled",
            self.root_dir / "Training" / "Unlabeled",
            self.root_dir / "Validation",
            self.root_dir / "Testing",
        ]

        self._clinical_df: Optional[pd.DataFrame] = None
        self._cases: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def is_available(self) -> bool:
        """Check if any M&Ms case folders are present on disk."""
        resolved = self._resolve_split_dirs()
        return any(d.is_dir() and any(d.iterdir()) for d, _ in resolved)

    def _resolve_split_dirs(self) -> List[Tuple[Path, str]]:
        """Resolve split directories, supporting case-insensitive paths and OpenDataset subfolders."""
        search_roots = [self.root_dir]
        for sub in ["OpenDataset", "opendataset", "open_dataset"]:
            sub_p = self.root_dir / sub
            if sub_p.is_dir():
                search_roots.insert(0, sub_p)

        found_splits: List[Tuple[Path, str]] = []
        for root in search_roots:
            candidate_splits = [
                ("Training/Labeled", ["Training/Labeled", "training/labeled", "Training/labeled"]),
                ("Training/Unlabeled", ["Training/Unlabeled", "training/unlabeled", "Training/unlabeled"]),
                ("Validation", ["Validation", "validation"]),
                ("Testing", ["Testing", "testing"]),
            ]
            for canonical_name, variants in candidate_splits:
                for var in variants:
                    p = root / var
                    if p.is_dir():
                        found_splits.append((p, canonical_name))
                        break
            if found_splits:
                break

        if not found_splits:
            found_splits = [(d, str(d.relative_to(self.root_dir))) for d in self.split_dirs if d.is_dir()]
        return found_splits

    def _find_case_files(self, case_dir: Path, case_id: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Find image and GT mask files with case-insensitive and nested directory support."""
        direct_files = [p for p in case_dir.iterdir() if p.is_file() and not p.name.startswith(".")]

        nested_files: List[Path] = []
        for sub in case_dir.iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                nested_files.extend([p for p in sub.iterdir() if p.is_file() and not p.name.startswith(".")])

        candidate_files = direct_files + nested_files

        imgs: List[Path] = []
        gts: List[Path] = []

        for p in candidate_files:
            fname = p.name.lower()
            if ("_sa_gt.nii" in fname or "_sa_seg.nii" in fname or f"{case_id.lower()}_gt.nii" in fname):
                gts.append(p)
            elif "_sa.nii" in fname and "_gt" not in fname and "_seg" not in fname:
                imgs.append(p)
            elif fname in (f"{case_id.lower()}.nii.gz", f"{case_id.lower()}.nii"):
                imgs.append(p)

        img_path = imgs[0] if imgs else None
        mask_path = gts[0] if gts else None
        return img_path, mask_path

    def _load_clinical_df(self) -> pd.DataFrame:
        """Load and index clinical metadata CSV."""
        if self._clinical_df is not None:
            return self._clinical_df

        if self.csv_path.is_file():
            try:
                df = pd.read_csv(self.csv_path)
                if "External code" in df.columns:
                    df = df.set_index("External code", drop=False)
                self._clinical_df = df
                return df
            except Exception as e:
                logger.warning(f"Could not load M&Ms clinical CSV {self.csv_path}: {e}")

        self._clinical_df = pd.DataFrame()
        return self._clinical_df

    def discover_cases(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Discover all available cases across M&Ms split directories and correlate with metadata."""
        if self._cases is not None and not force_refresh:
            return self._cases

        df = self._load_clinical_df()
        cases: Dict[str, Dict[str, Any]] = {}

        resolved_splits = self._resolve_split_dirs()
        for s_dir, canonical_split in resolved_splits:
            if not s_dir.is_dir():
                continue

            for case_dir in sorted(s_dir.iterdir()):
                if not case_dir.is_dir():
                    continue

                cid = case_dir.name
                img_path, mask_path = self._find_case_files(case_dir, cid)

                # Match clinical info from CSV
                csv_info = {}
                if not df.empty and cid in df.index:
                    row = df.loc[cid]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    csv_info = row.to_dict()

                ed_val = csv_info.get("ED")
                es_val = csv_info.get("ES")
                ed_frame = int(ed_val) if pd.notna(ed_val) else None
                es_frame = int(es_val) if pd.notna(es_val) else None

                has_image = (img_path is not None)
                has_mask = (mask_path is not None)

                if has_image and has_mask:
                    if ed_frame is not None and es_frame is not None:
                        status = "ELIGIBLE_PAIRED"
                        rejection_reason = None
                    else:
                        status = "MISSING_ED_ES_METADATA"
                        rejection_reason = "Missing ED/ES temporal phase frame indices in clinical metadata"
                elif not has_image and has_mask:
                    status = "MISSING_IMAGE"
                    rejection_reason = "Ground-truth mask exists, but 4D cine MRI image is missing"
                elif has_image and not has_mask:
                    status = "MISSING_MASK"
                    rejection_reason = "4D cine MRI image exists, but ground-truth segmentation mask is missing"
                else:
                    status = "EMPTY_DIRECTORY"
                    rejection_reason = "Directory exists but contains no valid NIfTI image or mask files"

                cases[cid] = {
                    "case_id": cid,
                    "split_category": canonical_split,
                    "directory": case_dir,
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "has_image": has_image,
                    "has_mask": has_mask,
                    "status": status,
                    "rejection_reason": rejection_reason,
                    "csv_metadata": csv_info,
                    "vendor": csv_info.get("VendorName") or csv_info.get("Vendor"),
                    "ed_frame": ed_frame,
                    "es_frame": es_frame,
                    "diagnosis": csv_info.get("Diagnosis") or csv_info.get("Pathology"),
                }

        self._cases = cases
        logger.info(f"Discovered {len(cases)} M&Ms cases ({sum(1 for c in cases.values() if c['has_image'] and c['has_mask'])} with image+mask)")
        return cases

    def get_discovery_summary(self) -> Dict[str, Any]:
        """Produce itemized dataset discovery, pairing, and rejection statistics."""
        cases = self.discover_cases()
        summary: Dict[str, Any] = {
            "total_discovered": len(cases),
            "with_images": sum(1 for c in cases.values() if c["has_image"]),
            "with_masks": sum(1 for c in cases.values() if c["has_mask"]),
            "usable_pairs": sum(1 for c in cases.values() if c["status"] == "ELIGIBLE_PAIRED"),
            "missing_images": sum(1 for c in cases.values() if not c["has_image"]),
            "missing_masks": sum(1 for c in cases.values() if not c["has_mask"]),
            "empty_directories": sum(1 for c in cases.values() if c["status"] == "EMPTY_DIRECTORY"),
            "missing_ed_es": sum(1 for c in cases.values() if c["status"] == "MISSING_ED_ES_METADATA"),
            "rejection_breakdown": {},
        }
        for c in cases.values():
            st = c["status"]
            summary["rejection_breakdown"][st] = summary["rejection_breakdown"].get(st, 0) + 1
        return summary

    def get_case_ids(self, only_paired: bool = False) -> List[str]:
        """Return discovered case IDs. If only_paired=True, returns only cases with image AND mask."""
        cases = self.discover_cases()
        if only_paired:
            return sorted([cid for cid, v in cases.items() if v["has_image"] and v["has_mask"]])
        return sorted(list(cases.keys()))

    def get_case_metadata(self, case_id: str) -> CaseMetadata:
        """Construct standardized CaseMetadata for a specified M&Ms case."""
        cases = self.discover_cases()
        if case_id not in cases:
            raise KeyError(f"Case '{case_id}' not found in M&Ms dataset.")

        c_info = cases[case_id]
        img_path = c_info["image_path"]
        mask_path = c_info["mask_path"]
        csv_meta = c_info["csv_metadata"]

        shape: Tuple[int, ...] = ()
        zooms: Tuple[float, ...] = ()
        affine: List[List[float]] = []

        if img_path and img_path.is_file():
            img_nii = nib.load(img_path)
            shape = tuple(img_nii.shape)
            zooms = tuple(float(z) for z in img_nii.header.get_zooms())
            affine = img_nii.affine.tolist()
        elif mask_path and mask_path.is_file():
            mask_nii = nib.load(mask_path)
            shape = tuple(mask_nii.shape)
            zooms = tuple(float(z) for z in mask_nii.header.get_zooms())
            affine = mask_nii.affine.tolist()

        raw_labels: List[int] = []
        if mask_path and mask_path.is_file():
            mask_nii = nib.load(mask_path)
            raw_labels = sorted(np.unique(mask_nii.get_fdata()).astype(int).tolist())

        ed_idx = csv_meta.get("ED")
        es_idx = csv_meta.get("ES")

        return CaseMetadata(
            case_id=case_id,
            dataset_id="mnms",
            image_path=img_path if img_path else Path(""),
            mask_path=mask_path,
            has_mask=c_info["has_mask"],
            spatial_shape=shape,
            voxel_spacing_mm=zooms,
            affine=affine,
            phases_available=[CardiacPhase.FULL_CINE, CardiacPhase.ED, CardiacPhase.ES] if shape and len(shape) == 4 else [CardiacPhase.STATIC],
            ed_frame_index=int(ed_idx) if pd.notna(ed_idx) else None,
            es_frame_index=int(es_idx) if pd.notna(es_idx) else None,
            vendor=str(csv_meta.get("VendorName")) if pd.notna(csv_meta.get("VendorName")) else None,
            center=f"Centre_{csv_meta.get('Centre')}" if pd.notna(csv_meta.get("Centre")) else None,
            diagnosis=str(csv_meta.get("Pathology")) if pd.notna(csv_meta.get("Pathology")) else None,
            sex=str(csv_meta.get("Sex")).upper() if pd.notna(csv_meta.get("Sex")) else None,
            age=float(csv_meta.get("Age")) if pd.notna(csv_meta.get("Age")) else None,
            weight_kg=float(csv_meta.get("Weight")) if pd.notna(csv_meta.get("Weight")) else None,
            height_cm=float(csv_meta.get("Height")) if pd.notna(csv_meta.get("Height")) else None,
            raw_labels_present=raw_labels,
            clinical_metadata=csv_meta,
        )

    def load_cine_image(self, case_id: str) -> Tuple[np.ndarray, Tuple[float, ...], np.ndarray]:
        """Load 4D cine MRI volume (H, W, Slices, Frames), voxel spacing, and affine.

        Returns:
            Tuple of (cine_array_4d, voxel_spacing_mm, affine_4x4)
        """
        cases = self.discover_cases()
        if case_id not in cases or not cases[case_id]["image_path"]:
            raise FileNotFoundError(f"Cine image not found for M&Ms case '{case_id}'")

        nii = nib.load(cases[case_id]["image_path"])
        arr = np.asarray(nii.get_fdata(), dtype=np.float32)
        zooms = tuple(float(z) for z in nii.header.get_zooms())
        affine = nii.affine
        return arr, zooms, affine

    def load_cine_mask(
        self,
        case_id: str,
        mapped_to_canonical: bool = True,
    ) -> Optional[Tuple[np.ndarray, Tuple[float, ...], np.ndarray]]:
        """Load 4D ground-truth mask volume with optional canonical label remapping.

        Args:
            case_id: M&Ms case identifier (e.g. 'A0S9V9').
            mapped_to_canonical: If True, maps M&Ms raw labels (1=LV, 2=Myo, 3=RV) to
                ACDC canonical conventions (3=LV, 2=Myo, 1=RV). If False, returns raw labels.

        Returns:
            Tuple of (mask_array_4d, voxel_spacing_mm, affine_4x4) or None if no mask exists.
        """
        cases = self.discover_cases()
        if case_id not in cases or not cases[case_id]["mask_path"]:
            return None

        nii = nib.load(cases[case_id]["mask_path"])
        arr = np.asarray(nii.get_fdata(), dtype=np.int64)
        if mapped_to_canonical:
            arr = map_mnms_to_canonical(arr)
        zooms = tuple(float(z) for z in nii.header.get_zooms())
        affine = nii.affine
        return arr, zooms, affine

    def load_phase_frames(
        self,
        case_id: str,
        mapped_to_canonical: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract individual 3D End-Diastole (ED) and End-Systole (ES) image and mask volumes.

        Returns:
            Dictionary containing:
              - 'ED': {'image': 3D array, 'mask': Optional 3D array, 'frame_index': int}
              - 'ES': {'image': 3D array, 'mask': Optional 3D array, 'frame_index': int}
              - 'voxel_spacing_mm': tuple
              - 'affine': 4x4 array
        """
        meta = self.get_case_metadata(case_id)
        img_4d, zooms, affine = self.load_cine_image(case_id)
        mask_res = self.load_cine_mask(case_id, mapped_to_canonical=mapped_to_canonical)
        mask_4d = mask_res[0] if mask_res else None

        n_frames = img_4d.shape[-1] if len(img_4d.shape) == 4 else 1

        # Use CSV indices or detect annotated frames in mask
        ed_idx = meta.ed_frame_index
        es_idx = meta.es_frame_index

        if (ed_idx is None or es_idx is None) and mask_4d is not None:
            # Detect annotated frames
            annotated = [t for t in range(n_frames) if np.any(mask_4d[:, :, :, t] > 0)]
            if len(annotated) >= 2:
                ed_idx = min(annotated)
                es_idx = max(annotated)
            elif len(annotated) == 1:
                ed_idx = annotated[0]
                es_idx = annotated[0]

        ed_idx = ed_idx if ed_idx is not None and ed_idx < n_frames else 0
        es_idx = es_idx if es_idx is not None and es_idx < n_frames else min(1, n_frames - 1)

        ed_img = img_4d[:, :, :, ed_idx] if len(img_4d.shape) == 4 else img_4d
        es_img = img_4d[:, :, :, es_idx] if len(img_4d.shape) == 4 else img_4d

        ed_mask = mask_4d[:, :, :, ed_idx] if mask_4d is not None and len(mask_4d.shape) == 4 else None
        es_mask = mask_4d[:, :, :, es_idx] if mask_4d is not None and len(mask_4d.shape) == 4 else None

        return {
            "ED": {"image": ed_img, "mask": ed_mask, "frame_index": ed_idx},
            "ES": {"image": es_img, "mask": es_mask, "frame_index": es_idx},
            "voxel_spacing_mm": zooms[:3],
            "affine": affine,
        }

    def load_case_cine(
        self,
        case_id: str,
        remap_to_canonical: bool = True,
    ) -> Dict[str, Any]:
        """Load complete 4D cine data for an M&Ms case.

        Returns:
            Dict containing case_id, image (4D array), mask (4D array or None),
            voxel_spacing, and affine.
        """
        img_4d, zooms, affine = self.load_cine_image(case_id)
        mask_res = self.load_cine_mask(case_id, mapped_to_canonical=remap_to_canonical)
        mask_4d = mask_res[0] if mask_res is not None else None
        return {
            "case_id": case_id,
            "image": img_4d,
            "mask": mask_4d,
            "voxel_spacing": zooms,
            "affine": affine,
        }

    def extract_phase_volumes(
        self,
        case_id: str,
        phase: CardiacPhase,
        remap_to_canonical: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Extract a single 3D volume (image, mask, affine) for ED or ES phase.

        Args:
            case_id: M&Ms case ID.
            phase: CardiacPhase.ED or CardiacPhase.ES.
            remap_to_canonical: Whether to map mask labels to ACDC convention.

        Returns:
            Tuple of (volume_3d, mask_3d, affine_4x4).
        """
        if phase not in (CardiacPhase.ED, CardiacPhase.ES):
            raise ValueError(f"Only ED and ES phases are supported for phase extraction, got {phase}")

        phase_key = "ED" if phase == CardiacPhase.ED else "ES"
        phases = self.load_phase_frames(case_id, mapped_to_canonical=remap_to_canonical)
        entry = phases[phase_key]
        return entry["image"], entry["mask"], phases["affine"]

    def validate_case_alignment(self, case_id: str) -> bool:
        """Verify spatial shapes, slice count, and affine match between cine image and mask."""
        cases = self.discover_cases()
        if case_id not in cases:
            return False

        c = cases[case_id]
        if not c["has_image"] or not c["has_mask"]:
            return True  # Cannot check unpaired case

        img_nii = nib.load(c["image_path"])
        mask_nii = nib.load(c["mask_path"])

        if img_nii.shape != mask_nii.shape:
            logger.error(f"Shape mismatch in {case_id}: img {img_nii.shape} vs mask {mask_nii.shape}")
            return False

        if not np.allclose(img_nii.affine, mask_nii.affine, atol=1e-3):
            logger.error(f"Affine mismatch in {case_id}")
            return False

        return True
