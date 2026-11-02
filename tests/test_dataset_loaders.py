"""Unit and integration tests for EMIDEC and M&Ms dataset loaders.

Includes synthetic mock tests for reproducible unit verification and
real-data integration tests conditional on local raw data availability.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import pytest

from crip.datasets.emidec_loader import EMIDECDataLoader
from crip.datasets.labels import ACDCLabel, EMIDECLabel, MNMSLabel
from crip.datasets.mnms_loader import MNMSDataLoader
from crip.datasets.schemas import CardiacPhase


# ==============================================================================
# Synthetic Unit Tests (Never skip, run anywhere)
# ==============================================================================


@pytest.fixture
def synthetic_emidec_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic EMIDEC directory structure."""
    ds_dir = tmp_path / "emidec-dataset-1.0.1"
    case_dir = ds_dir / "Case_N001"
    img_dir = case_dir / "Images"
    cnt_dir = case_dir / "Contours"
    img_dir.mkdir(parents=True)
    cnt_dir.mkdir(parents=True)

    # 3D synthetic volume (32, 32, 4)
    affine = np.diag([1.5, 1.5, 5.0, 1.0])
    img_data = (np.random.rand(32, 32, 4) * 500).astype(np.float32)
    img_nii = nib.Nifti1Image(img_data, affine)
    nib.save(img_nii, str(img_dir / "Case_N001.nii.gz"))

    # 3D synthetic contour
    cnt_data = np.zeros((32, 32, 4), dtype=np.int16)
    cnt_data[10:20, 10:20, :] = EMIDECLabel.LV_CAVITY.value
    cnt_data[8:22, 8:22, :] = np.where(
        cnt_data[8:22, 8:22, :] == 0,
        EMIDECLabel.NORMAL_MYOCARDIUM.value,
        cnt_data[8:22, 8:22, :],
    )
    cnt_nii = nib.Nifti1Image(cnt_data, affine)
    nib.save(cnt_nii, str(cnt_dir / "Case_N001.nii.gz"))

    # Clinical text file
    clinical_file = ds_dir / "Case N001.txt"
    clinical_file.write_text(
        "Patient ID: N001\n"
        "Sex: M\n"
        "Age: 55\n"
        "Weight: 75\n"
        "Height: 178\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def synthetic_mnms_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic M&Ms directory structure."""
    case_dir = tmp_path / "Training" / "Labeled" / "MMS001"
    case_dir.mkdir(parents=True)

    # 4D synthetic cine: (32, 32, 4, 8)
    affine = np.diag([1.25, 1.25, 8.0, 1.0])
    cine_data = (np.random.rand(32, 32, 4, 8) * 400).astype(np.float32)
    cine_nii = nib.Nifti1Image(cine_data, affine)
    nib.save(cine_nii, str(case_dir / "MMS001_sa.nii.gz"))

    # 4D synthetic mask with labels at frame 0 (ED) and frame 4 (ES)
    mask_data = np.zeros((32, 32, 4, 8), dtype=np.int16)
    # Raw M&Ms: 1 = LV, 2 = MYO, 3 = RV
    mask_data[10:20, 10:20, :, 0] = MNMSLabel.LV.value
    mask_data[5:10, 10:20, :, 0] = MNMSLabel.RV.value
    mask_data[8:22, 8:22, :, 0] = np.where(
        mask_data[8:22, 8:22, :, 0] == 0,
        MNMSLabel.MYOCARDIUM.value,
        mask_data[8:22, 8:22, :, 0],
    )

    mask_data[12:18, 12:18, :, 4] = MNMSLabel.LV.value
    mask_data[6:10, 12:18, :, 4] = MNMSLabel.RV.value
    mask_data[10:20, 10:20, :, 4] = np.where(
        mask_data[10:20, 10:20, :, 4] == 0,
        MNMSLabel.MYOCARDIUM.value,
        mask_data[10:20, 10:20, :, 4],
    )

    mask_nii = nib.Nifti1Image(mask_data, affine)
    nib.save(mask_nii, str(case_dir / "MMS001_sa_gt.nii.gz"))

    # Clinical CSV
    csv_file = tmp_path / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    csv_file.write_text(
        "External code,VendorName,Vendor,Centre,ED,ES,Diagnosis\n"
        "MMS001,Siemens,A,1,0,4,NOR\n",
        encoding="utf-8",
    )

    return tmp_path


class TestSyntheticEMIDECLoader:
    """Synthetic unit tests for EMIDEC loader."""

    def test_discover_cases(self, synthetic_emidec_dir: Path):
        loader = EMIDECDataLoader(root_dir=synthetic_emidec_dir)
        cases = loader.discover_cases()
        assert "Case_N001" in cases
        assert cases["Case_N001"]["split"] == "train"
        assert cases["Case_N001"]["has_contour"] is True
        assert cases["Case_N001"]["has_clinical"] is True

    def test_load_case_data(self, synthetic_emidec_dir: Path):
        loader = EMIDECDataLoader(root_dir=synthetic_emidec_dir)
        case_data = loader.load_case("Case_N001")

        assert case_data["case_id"] == "Case_N001"
        assert case_data["image"].shape == (32, 32, 4)
        assert case_data["contour"].shape == (32, 32, 4)
        assert case_data["has_contour"] is True
        assert tuple(round(v, 2) for v in case_data["voxel_spacing"][:3]) == (1.5, 1.5, 5.0)

        # Check metadata
        meta = case_data["clinical_metadata"]
        assert meta.get("age") == 55.0
        assert meta.get("sex") == "M"

    def test_load_nonexistent_case_raises(self, synthetic_emidec_dir: Path):
        loader = EMIDECDataLoader(root_dir=synthetic_emidec_dir)
        with pytest.raises(KeyError, match="not found"):
            loader.load_case("Case_Missing")

    def test_case_metadata_object(self, synthetic_emidec_dir: Path):
        loader = EMIDECDataLoader(root_dir=synthetic_emidec_dir)
        meta_obj = loader.get_case_metadata("Case_N001")
        assert meta_obj.dataset_id == "emidec"
        assert meta_obj.case_id == "Case_N001"
        assert meta_obj.phases_available == [CardiacPhase.STATIC]
        assert meta_obj.spatial_shape == (32, 32, 4)


class TestSyntheticMNMSLoader:
    """Synthetic unit tests for M&Ms loader."""

    def test_discover_cases(self, synthetic_mnms_dir: Path):
        loader = MNMSDataLoader(root_dir=synthetic_mnms_dir)
        cases = loader.discover_cases()
        assert "MMS001" in cases
        assert cases["MMS001"]["vendor"] == "Siemens"
        assert cases["MMS001"]["ed_frame"] == 0
        assert cases["MMS001"]["es_frame"] == 4
        assert cases["MMS001"]["diagnosis"] == "NOR"

    def test_load_case_cine_canonical_remapping(self, synthetic_mnms_dir: Path):
        loader = MNMSDataLoader(root_dir=synthetic_mnms_dir)

        # Default remap_to_canonical = True
        case_data = loader.load_case_cine("MMS001", remap_to_canonical=True)
        assert case_data["image"].shape == (32, 32, 4, 8)
        assert case_data["mask"].shape == (32, 32, 4, 8)

        # Canonical mapping checks:
        # In synthetic fixture:
        # [10:20, 10:20, :, 0] was set to MNMSLabel.LV (1) -> should map to ACDCLabel.LV (3)
        # [5:10, 10:20, :, 0] was set to MNMSLabel.RV (3) -> should map to ACDCLabel.RV (1)
        # [8:22, 8:22, :, 0] non-zero was set to MNMSLabel.MYO (2) -> should remain ACDCLabel.MYO (2)
        ed_mask = case_data["mask"][:, :, :, 0]
        assert ed_mask[15, 15, 0] == ACDCLabel.LV.value
        assert ed_mask[7, 15, 0] == ACDCLabel.RV.value

    def test_load_case_cine_raw_labels(self, synthetic_mnms_dir: Path):
        loader = MNMSDataLoader(root_dir=synthetic_mnms_dir)

        case_data = loader.load_case_cine("MMS001", remap_to_canonical=False)
        raw_mask = case_data["mask"][:, :, :, 0]
        # In raw M&Ms: 1 is LV, 3 is RV
        assert raw_mask[15, 15, 0] == MNMSLabel.LV.value
        assert raw_mask[7, 15, 0] == MNMSLabel.RV.value

    def test_extract_phase_volumes(self, synthetic_mnms_dir: Path):
        loader = MNMSDataLoader(root_dir=synthetic_mnms_dir)

        ed_vol, ed_mask, affine = loader.extract_phase_volumes("MMS001", CardiacPhase.ED)
        assert ed_vol.shape == (32, 32, 4)
        assert ed_mask.shape == (32, 32, 4)
        assert affine.shape == (4, 4)

        es_vol, es_mask, _ = loader.extract_phase_volumes("MMS001", CardiacPhase.ES)
        assert es_vol.shape == (32, 32, 4)
        assert es_mask.shape == (32, 32, 4)

    def test_extract_invalid_phase_raises(self, synthetic_mnms_dir: Path):
        loader = MNMSDataLoader(root_dir=synthetic_mnms_dir)
        with pytest.raises(ValueError, match="Only ED and ES"):
            loader.extract_phase_volumes("MMS001", CardiacPhase.STATIC)


# ==============================================================================
# Real Data Integration Tests (Run if local datasets exist)
# ==============================================================================

REAL_EMIDEC_PATH = Path("data/raw/emidec/emidec-dataset-1.0.1")
REAL_MNMS_PATH = Path("data/raw/mnms")


@pytest.mark.skipif(
    not REAL_EMIDEC_PATH.exists(),
    reason="EMIDEC raw data directory not found",
)
class TestRealEMIDECIntegration:
    """Integration test against actual EMIDEC dataset on disk."""

    def test_emidec_real_cases_discovered(self):
        loader = EMIDECDataLoader()
        cases = loader.discover_cases()
        assert len(cases) == 150
        labeled = [c for c in cases.values() if c["has_contour"]]
        assert len(labeled) == 100

    def test_emidec_load_real_normal_case(self):
        loader = EMIDECDataLoader()
        cases = loader.discover_cases()
        normal_ids = [cid for cid, c in cases.items() if c["pathology"] == "Normal"]
        assert len(normal_ids) > 0
        first_norm = normal_ids[0]
        case_data = loader.load_case(first_norm)
        assert case_data["has_contour"] is True
        assert len(case_data["image"].shape) == 3
        assert case_data["image"].shape == case_data["contour"].shape
        assert case_data["pathology"] == "Normal"
        assert "age" in case_data["clinical_metadata"]

    def test_emidec_load_real_pathological_case(self):
        loader = EMIDECDataLoader()
        case_data = loader.load_case("Case_P001")
        assert case_data["has_contour"] is True
        assert "Infarction" in case_data["pathology"]
        contour_labels = set(np.unique(case_data["contour"]))
        # Should contain infarction (3) or no-reflow (4) in addition to normal tissue
        assert bool(contour_labels.intersection({EMIDECLabel.MYOCARDIAL_INFARCTION.value, EMIDECLabel.NO_REFLOW.value}))


@pytest.mark.skipif(
    not (REAL_MNMS_PATH / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv").exists(),
    reason="M&Ms raw data not found",
)
class TestRealMNMSIntegration:
    """Integration test against actual M&Ms dataset on disk."""

    def test_mnms_real_cases_discovered(self):
        loader = MNMSDataLoader()
        cases = loader.discover_cases()
        assert len(cases) == 345
        paired = [c for c in cases.values() if c["has_image"] and c["has_mask"]]
        assert len(paired) == 114

    def test_mnms_load_real_case(self):
        loader = MNMSDataLoader()
        # Find first case with both image and mask
        cases = loader.discover_cases()
        paired_ids = [cid for cid, c in cases.items() if c["has_image"] and c["has_mask"]]
        assert len(paired_ids) > 0

        first_id = paired_ids[0]
        case_data = loader.load_case_cine(first_id, remap_to_canonical=True)
        assert len(case_data["image"].shape) == 4
        assert len(case_data["mask"].shape) == 4
        assert case_data["image"].shape == case_data["mask"].shape

        # Verify ED / ES phase extraction
        ed_vol, ed_mask, _ = loader.extract_phase_volumes(first_id, CardiacPhase.ED)
        assert len(ed_vol.shape) == 3
        assert ed_vol.shape == ed_mask.shape
