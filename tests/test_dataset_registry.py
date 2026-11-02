"""Tests for CRIP Multi-Dataset Registry, Schemas, and Label Definitions.

Validates dataset registration, querying, modality filtering, task compatibility,
and label mapping between multi-center datasets (ACDC, M&Ms, EMIDEC).
"""

from pathlib import Path
import numpy as np
import pytest

from crip.datasets.labels import (
    ACDC_LABELS,
    CANONICAL_TO_MNMS_MAP,
    EMIDEC_LABELS,
    MNMS_LABELS,
    MNMS_TO_CANONICAL_MAP,
    ACDCLabel,
    EMIDECLabel,
    MNMSLabel,
    map_canonical_to_mnms,
    map_mnms_to_canonical,
    validate_label_set,
)
from crip.datasets.registry import DatasetRegistry, get_dataset_registry
from crip.datasets.schemas import (
    CardiacPhase,
    CaseMetadata,
    DatasetDefinition,
    DatasetStatus,
    ModalityType,
)


class TestDatasetSchemas:
    """Test core dataset schemas and enums."""

    def test_modality_type_enum(self):
        assert ModalityType.CINE_MRI.value == "Cine SSFP Dynamic CMR"
        assert ModalityType.LGE_MRI.value == "Late Gadolinium Enhancement (LGE) Viability CMR"
        assert ModalityType.MULTI_SEQUENCE_MRI.value == "Multi-Sequence Multi-Contrast CMR"

    def test_cardiac_phase_enum(self):
        assert CardiacPhase.ED.value == "End-Diastole"
        assert CardiacPhase.ES.value == "End-Systole"
        assert CardiacPhase.STATIC.value == "Static Acquisition (Non-cine)"
        assert CardiacPhase.FULL_CINE.value == "Full Cardiac Cycle (4D Cine)"

    def test_dataset_status_enum(self):
        assert DatasetStatus.AVAILABLE.value == "AVAILABLE"
        assert DatasetStatus.PARTIAL.value == "PARTIAL"
        assert DatasetStatus.BLOCKED.value == "BLOCKED"

    def test_case_metadata_defaults(self):
        meta = CaseMetadata(
            case_id="TEST_001",
            dataset_id="test_ds",
            image_path=Path("/path/to/img.nii.gz"),
        )
        assert meta.case_id == "TEST_001"
        assert meta.dataset_id == "test_ds"
        assert meta.mask_path is None
        assert meta.has_mask is False
        assert meta.clinical_metadata == {}
        assert meta.spatial_shape == ()


class TestDatasetRegistry:
    """Test registry initialization, pre-registration, and query methods."""

    def test_registry_singleton(self):
        reg1 = get_dataset_registry()
        reg2 = get_dataset_registry()
        assert reg1 is reg2

    def test_default_registered_datasets(self):
        registry = DatasetRegistry()
        datasets = {d.dataset_id: d for d in registry.list_datasets()}
        expected = {"acdc", "emidec", "mnms", "sunnybrook", "myops2020"}
        assert expected.issubset(set(datasets.keys()))

    def test_get_registered_dataset(self):
        registry = DatasetRegistry()
        acdc = registry.get("acdc")
        assert acdc.dataset_id == "acdc"
        assert acdc.modality == ModalityType.CINE_MRI
        assert acdc.has_reference_masks is True
        assert acdc.supports_lvef_evaluation is True
        assert "cine_segmentation" in acdc.compatible_tasks

    def test_emidec_registry_entry(self):
        registry = DatasetRegistry()
        emidec = registry.get("emidec")
        assert emidec.dataset_id == "emidec"
        assert emidec.modality == ModalityType.LGE_MRI
        assert "myocardial_infarction_segmentation" in emidec.compatible_tasks
        assert "viability_assessment" in emidec.compatible_tasks
        # EMIDEC does not support dynamic ejection fraction
        assert emidec.supports_lvef_evaluation is False
        assert "CC BY-NC-SA 4.0" in emidec.license_type

    def test_mnms_registry_entry(self):
        registry = DatasetRegistry()
        mnms = registry.get("mnms")
        assert mnms.dataset_id == "mnms"
        assert mnms.modality == ModalityType.CINE_MRI
        assert "cine_segmentation" in mnms.compatible_tasks
        assert mnms.supports_lvef_evaluation is True
        assert "CC BY-NC-SA 4.0" in mnms.license_type

    def test_blocked_datasets(self):
        registry = DatasetRegistry()
        sunnybrook = registry.get("sunnybrook")
        assert sunnybrook.status == DatasetStatus.BLOCKED

        myops = registry.get("myops2020")
        assert myops.status == DatasetStatus.BLOCKED

    def test_get_nonexistent_dataset_raises(self):
        registry = DatasetRegistry()
        with pytest.raises(KeyError, match="not found in registry"):
            registry.get("non_existent_dataset")

    def test_filter_by_modality(self):
        registry = DatasetRegistry()
        cine_datasets = registry.filter_by_modality(ModalityType.CINE_MRI)
        cine_ids = {d.dataset_id for d in cine_datasets}
        assert "acdc" in cine_ids
        assert "mnms" in cine_ids
        assert "emidec" not in cine_ids

        lge_datasets = registry.filter_by_modality(ModalityType.LGE_MRI)
        lge_ids = {d.dataset_id for d in lge_datasets}
        assert "emidec" in lge_ids
        assert "acdc" not in lge_ids

    def test_filter_by_task(self):
        registry = DatasetRegistry()
        cine_seg_datasets = registry.filter_by_task("cine_segmentation")
        cine_ids = {d.dataset_id for d in cine_seg_datasets}
        assert "acdc" in cine_ids
        assert "mnms" in cine_ids
        assert "emidec" not in cine_ids

        scar_datasets = registry.filter_by_task("viability_assessment")
        scar_ids = {d.dataset_id for d in scar_datasets}
        assert "emidec" in scar_ids
        assert "acdc" not in scar_ids

    def test_register_duplicate_raises(self):
        registry = DatasetRegistry()
        dup = DatasetDefinition(
            dataset_id="acdc",
            name="ACDC Duplicate",
            modality=ModalityType.CINE_MRI,
            root_path=Path("data/raw/acdc"),
            status=DatasetStatus.AVAILABLE,
            raw_label_names={0: "BG"},
        )
        with pytest.raises(ValueError, match="already registered"):
            registry.register(dup, overwrite=False)

    def test_register_and_unregister(self):
        registry = DatasetRegistry()
        custom = DatasetDefinition(
            dataset_id="custom_study",
            name="Custom Study",
            modality=ModalityType.CINE_MRI,
            root_path=Path("data/raw/custom"),
            status=DatasetStatus.AVAILABLE,
            raw_label_names={0: "BG"},
        )
        registry.register(custom)
        assert registry.get("custom_study").name == "Custom Study"
        registry.unregister("custom_study")
        with pytest.raises(KeyError):
            registry.get("custom_study")


class TestLabelDefinitionsAndMapping:
    """Test label definitions, conversions, and clinical isolation."""

    def test_acdc_label_values(self):
        assert ACDCLabel.BACKGROUND == 0
        assert ACDCLabel.RV == 1
        assert ACDCLabel.RV_CAVITY == 1
        assert ACDCLabel.MYOCARDIUM == 2
        assert ACDCLabel.LV == 3
        assert ACDCLabel.LV_CAVITY == 3
        assert ACDC_LABELS[1] == "Right Ventricular Cavity"
        assert ACDC_LABELS[3] == "Left Ventricular Cavity"

    def test_mnms_label_values(self):
        assert MNMSLabel.BACKGROUND == 0
        assert MNMSLabel.LV == 1
        assert MNMSLabel.LV_CAVITY == 1
        assert MNMSLabel.MYOCARDIUM == 2
        assert MNMSLabel.RV == 3
        assert MNMSLabel.RV_CAVITY == 3
        assert MNMS_LABELS[1] == "Left Ventricular Cavity"
        assert MNMS_LABELS[3] == "Right Ventricular Cavity"

    def test_emidec_label_values(self):
        assert EMIDECLabel.BACKGROUND == 0
        assert EMIDECLabel.LV_CAVITY == 1
        assert EMIDECLabel.NORMAL_MYOCARDIUM == 2
        assert EMIDECLabel.MYOCARDIAL_INFARCTION == 3
        assert EMIDECLabel.NO_REFLOW == 4
        assert EMIDEC_LABELS[3] == "Myocardial Infarction (Scar / Hyperenhancement)"
        assert EMIDECLabel.MYOCARDIAL_INFARCTION.name == "MYOCARDIAL_INFARCTION"
        assert ACDCLabel.LV.name in ("LV", "LV_CAVITY")

    def test_mnms_to_canonical_mapping(self):
        # M&Ms raw: 0=BG, 1=LV, 2=MYO, 3=RV
        # Canonical ACDC: 0=BG, 1=RV, 2=MYO, 3=LV
        raw_mnms = np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ], dtype=np.int16)

        canonical = map_mnms_to_canonical(raw_mnms)
        # Expected:
        # [0, 3, 2, 1]
        # [1, 2, 3, 0]
        assert canonical[0, 0] == 0
        assert canonical[0, 1] == 3  # LV -> 3
        assert canonical[0, 2] == 2  # MYO -> 2
        assert canonical[0, 3] == 1  # RV -> 1
        assert canonical[1, 0] == 1  # RV -> 1
        assert canonical[1, 1] == 2  # MYO -> 2
        assert canonical[1, 2] == 3  # LV -> 3
        assert canonical[1, 3] == 0  # BG -> 0

    def test_canonical_to_mnms_mapping(self):
        canonical = np.array([
            [0, 1, 2, 3],
        ], dtype=np.int16)
        # Canonical: 0=BG, 1=RV, 2=MYO, 3=LV
        # M&Ms raw: 0=BG, 1=LV, 2=MYO, 3=RV
        mnms = map_canonical_to_mnms(canonical)
        assert mnms[0, 0] == 0
        assert mnms[0, 1] == 3  # RV -> 3
        assert mnms[0, 2] == 2  # MYO -> 2
        assert mnms[0, 3] == 1  # LV -> 1

    def test_mapping_round_trip(self):
        original = np.random.choice([0, 1, 2, 3], size=(16, 16, 8)).astype(np.int16)
        mapped = map_mnms_to_canonical(original)
        recovered = map_canonical_to_mnms(mapped)
        np.testing.assert_array_equal(original, recovered)

    def test_validate_label_set(self):
        valid_acdc = np.array([0, 1, 2, 3])
        assert validate_label_set(valid_acdc, {0, 1, 2, 3}) is True

        invalid_acdc = np.array([0, 1, 2, 4])
        assert validate_label_set(invalid_acdc, {0, 1, 2, 3}) is False

    def test_map_mnms_unknown_label_preservation(self):
        raw = np.array([0, 1, 2, 3, 99], dtype=np.int16)
        canonical = map_mnms_to_canonical(raw)
        assert canonical[-1] == 99  # Preserves unmapped labels
