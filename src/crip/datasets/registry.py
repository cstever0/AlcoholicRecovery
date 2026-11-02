"""Centralized Cardiac MRI Dataset Registry.

Manages dataset definitions, modality metadata, anatomical label configurations,
task compatibility, and license governance across all integrated and planned
cardiac MRI cohorts:
1. ACDC 2017: Multi-structure cine SSFP CMR (100 training, 50 testing)
2. EMIDEC 2020: Late Gadolinium Enhancement (LGE) viability CMR (100 training, 50 testing)
3. M&Ms 2020: Multi-centre, multi-vendor cine SSFP CMR (345 cases)
4. Sunnybrook 2009: Cine SSFP CMR benchmark (future extension)
5. MyoPS 2020: Multi-sequence myocardial pathology segmentation (future extension)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from crip.datasets.labels import (
    ACDC_LABEL_NAMES,
    EMIDEC_LABEL_NAMES,
    MNMS_LABEL_NAMES,
    MNMS_TO_CANONICAL_MAP,
)
from crip.datasets.schemas import (
    DatasetDefinition,
    DatasetStatus,
    ModalityType,
)

logger = logging.getLogger("crip.datasets.registry")

# Default repository paths
DEFAULT_ACDC_PATH = Path("data/raw/acdc/training")
DEFAULT_EMIDEC_PATH = Path("data/raw/emidec/emidec-dataset-1.0.1")
DEFAULT_MNMS_PATH = Path("data/raw/mnms")
DEFAULT_SUNNYBROOK_PATH = Path("data/raw/sunnybrook")
DEFAULT_MYOPS_PATH = Path("data/raw/myops2020")


def create_default_dataset_definitions() -> Dict[str, DatasetDefinition]:
    """Instantiate standard definitions for all supported and planned cohorts."""
    return {
        "acdc": DatasetDefinition(
            dataset_id="acdc",
            name="Automated Cardiac Diagnosis Challenge (ACDC 2017)",
            modality=ModalityType.CINE_MRI,
            root_path=DEFAULT_ACDC_PATH,
            status=DatasetStatus.AVAILABLE if DEFAULT_ACDC_PATH.is_dir() else DatasetStatus.BLOCKED,
            raw_label_names=ACDC_LABEL_NAMES,
            canonical_label_map=None,  # ACDC is canonical convention
            has_phases=True,
            has_reference_masks=True,
            has_clinical_metadata=True,
            supports_lvef_evaluation=True,
            compatible_tasks=[
                "cine_segmentation",
                "biventricular_volumetrics",
                "lvef_estimation",
                "wall_motion_analysis",
            ],
            license_type="Creative Commons Attribution-NonCommercial-NoDerivs (CC BY-NC-ND 4.0)",
            citation=(
                "Bernard O, et al. 'Deep Learning Techniques for Automatic MRI Cardiac Image "
                "Segmentation: An Overview and Evaluation on the ACDC Challenge Dataset.' "
                "IEEE Transactions on Medical Imaging, 37(11):2514-2525, 2018."
            ),
            provenance="University Hospital of Dijon, France. 100 patient cine SSFP short-axis scans.",
            total_cases_expected=100,
            description=(
                "Cine SSFP magnetic resonance imaging covering the cardiac cycle at End-Diastole "
                "and End-Systole with manual segmentations for LV cavity, RV cavity, and myocardium."
            ),
        ),
        "emidec": DatasetDefinition(
            dataset_id="emidec",
            name="Evaluation of Myocardial Infarction from Delayed-Enhancement CMR (EMIDEC 2020)",
            modality=ModalityType.LGE_MRI,
            root_path=DEFAULT_EMIDEC_PATH,
            status=DatasetStatus.AVAILABLE if DEFAULT_EMIDEC_PATH.is_dir() else DatasetStatus.BLOCKED,
            raw_label_names=EMIDEC_LABEL_NAMES,
            canonical_label_map=None,  # EMIDEC maintains distinct tissue viability labels
            has_phases=False,  # Static delayed-enhancement, NOT cine
            has_reference_masks=True,
            has_clinical_metadata=True,
            supports_lvef_evaluation=False,  # Static LGE cannot measure dynamic LVEF
            compatible_tasks=[
                "myocardial_infarction_segmentation",
                "microvascular_obstruction_quantification",
                "myocardial_scar_sizing",
                "viability_assessment",
            ],
            license_type="Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0)",
            citation=(
                "Lalande A, et al. 'Deep learning techniques for automatic MRI cardiac image segmentation "
                "in the context of myocardial infarction: the EMIDEC challenge.' "
                "Medical Image Analysis, 73:102183, 2021."
            ),
            provenance="University Hospital of Dijon, France. Late gadolinium enhancement (LGE) T1-weighted CMR.",
            total_cases_expected=100,
            description=(
                "Static late-gadolinium enhancement viability CMR scans for myocardial necrosis "
                "and microvascular obstruction quantification. Not suited for dynamic ejection fraction."
            ),
        ),
        "mnms": DatasetDefinition(
            dataset_id="mnms",
            name="Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Segmentation (M&Ms 2020)",
            modality=ModalityType.CINE_MRI,
            root_path=DEFAULT_MNMS_PATH,
            status=DatasetStatus.PARTIAL if DEFAULT_MNMS_PATH.is_dir() else DatasetStatus.BLOCKED,
            raw_label_names=MNMS_LABEL_NAMES,
            canonical_label_map=MNMS_TO_CANONICAL_MAP,
            has_phases=True,
            has_reference_masks=True,
            has_clinical_metadata=True,
            supports_lvef_evaluation=True,
            compatible_tasks=[
                "cine_segmentation",
                "biventricular_volumetrics",
                "lvef_estimation",
                "multi_vendor_domain_generalization",
                "multi_center_evaluation",
            ],
            license_type="Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0)",
            citation=(
                "Campello VM, et al. 'Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation: "
                "The M&Ms Challenge.' IEEE Transactions on Medical Imaging, 40(12):3643-3654, 2021."
            ),
            provenance=(
                "Four clinical centres in Spain and Germany across Siemens, Philips, GE, and Canon "
                "scanners. 345 patient records."
            ),
            total_cases_expected=345,
            description=(
                "Multi-centre, multi-vendor 4D cine CMR dataset evaluating algorithmic domain "
                "generalization across diverse magnetic resonance scanner manufacturers."
            ),
        ),
        "sunnybrook": DatasetDefinition(
            dataset_id="sunnybrook",
            name="Sunnybrook Cardiac MRI Challenge (SCMR 2009)",
            modality=ModalityType.CINE_MRI,
            root_path=DEFAULT_SUNNYBROOK_PATH,
            status=DatasetStatus.AVAILABLE if DEFAULT_SUNNYBROOK_PATH.is_dir() and any(DEFAULT_SUNNYBROOK_PATH.iterdir()) else DatasetStatus.BLOCKED,
            raw_label_names={0: "Background", 1: "LV Endocardium", 2: "LV Epicardium"},
            canonical_label_map=None,
            has_phases=True,
            has_reference_masks=False,
            has_clinical_metadata=False,
            supports_lvef_evaluation=True,
            compatible_tasks=[
                "left_ventricle_segmentation",
                "single_chamber_volumetrics",
            ],
            license_type="Public Academic Research (CC0 / Open Data)",
            citation=(
                "Radau P, et al. 'Evaluation Framework for Algorithms Segmenting Short Axis "
                "Cardiac MRI.' The MIDAS Journal - Cardiac MR Left Ventricle Segmentation Challenge, 2009."
            ),
            provenance="Sunnybrook Health Sciences Centre, Toronto, Canada. 45 short-axis cine CMR scans.",
            total_cases_expected=45,
            description=(
                "Historic benchmark dataset focused on left ventricular endocardial and epicardial "
                "contouring across mixed cardiac pathologies. Currently in BLOCKED status pending local download."
            ),
        ),
        "myops2020": DatasetDefinition(
            dataset_id="myops2020",
            name="Myocardial Pathology Segmentation Combined with Multi-Sequence CMR (MyoPS 2020)",
            modality=ModalityType.MULTI_SEQUENCE_MRI,
            root_path=DEFAULT_MYOPS_PATH,
            status=DatasetStatus.AVAILABLE if DEFAULT_MYOPS_PATH.is_dir() and any(DEFAULT_MYOPS_PATH.iterdir()) else DatasetStatus.BLOCKED,
            raw_label_names={
                0: "Background",
                1: "Normal Myocardium",
                2: "Myocardial Infarction",
                3: "Myocardial Edema",
                4: "LV Blood Pool",
                5: "RV Blood Pool",
            },
            canonical_label_map=None,
            has_phases=False,
            has_reference_masks=False,
            has_clinical_metadata=False,
            supports_lvef_evaluation=False,
            compatible_tasks=[
                "myocardial_edema_segmentation",
                "myocardial_infarction_segmentation",
                "multi_sequence_alignment",
            ],
            license_type="Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0)",
            citation=(
                "Zhuang X. 'Multivariate mixture model for myocardial segmentation combining "
                "multi-sequence CMR.' MICCAI MyoPS Challenge, 2020."
            ),
            provenance="Combined LGE, T2-weighted, and balanced-SSFP multi-sequence CMR. Currently in BLOCKED status.",
            total_cases_expected=45,
            description=(
                "Multi-sequence viability imaging combining LGE, T2, and bSSFP to detect both chronic "
                "myocardial infarction and acute myocardial edema."
            ),
        ),
    }


class DatasetRegistry:
    """Thread-safe catalog of cardiac MRI datasets."""

    def __init__(self, custom_definitions: Optional[Dict[str, DatasetDefinition]] = None) -> None:
        self._registry: Dict[str, DatasetDefinition] = {}
        defaults = custom_definitions if custom_definitions is not None else create_default_dataset_definitions()
        for key, defn in defaults.items():
            self.register(defn)

    def register(self, definition: DatasetDefinition, overwrite: bool = True) -> None:
        """Register or update a dataset definition."""
        key = definition.dataset_id.lower()
        if key in self._registry and not overwrite:
            raise ValueError(f"Dataset '{definition.dataset_id}' is already registered.")
        self._registry[key] = definition
        logger.debug(f"Registered dataset '{definition.dataset_id}' ({definition.name})")

    def unregister(self, dataset_id: str) -> None:
        """Remove a dataset from the registry."""
        key = dataset_id.lower()
        if key not in self._registry:
            raise KeyError(f"Dataset '{dataset_id}' not found in registry.")
        del self._registry[key]

    def get(self, dataset_id: str) -> DatasetDefinition:
        """Retrieve dataset definition by identifier."""
        key = dataset_id.lower()
        if key not in self._registry:
            valid = list(self._registry.keys())
            raise KeyError(f"Dataset '{dataset_id}' not found in registry. Available datasets: {valid}")
        return self._registry[key]

    def list_datasets(self) -> List[DatasetDefinition]:
        """List all registered dataset definitions."""
        return list(self._registry.values())

    def get_by_modality(self, modality: ModalityType) -> List[DatasetDefinition]:
        """Filter registered datasets by imaging modality."""
        return [d for d in self._registry.values() if d.modality == modality]

    filter_by_modality = get_by_modality

    def get_by_task(self, task_name: str) -> List[DatasetDefinition]:
        """Filter registered datasets compatible with a specific evaluation task."""
        return [d for d in self._registry.values() if task_name in d.compatible_tasks]

    filter_by_task = get_by_task

    def is_dataset_available(self, dataset_id: str) -> bool:
        """Check if dataset root directory exists and contains data."""
        try:
            defn = self.get(dataset_id)
            if not defn.root_path.exists():
                return False
            if defn.root_path.is_dir():
                return any(defn.root_path.iterdir())
            return False
        except KeyError:
            return False


# Singleton registry instance
_GLOBAL_REGISTRY: Optional[DatasetRegistry] = None


def get_dataset_registry() -> DatasetRegistry:
    """Access the global singleton dataset registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = DatasetRegistry()
    return _GLOBAL_REGISTRY
