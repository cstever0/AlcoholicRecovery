"""Test configuration and shared fixtures for CRIP unit tests."""

from datetime import date
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from crip.synthetic.schemas import (
    FollowUpOutcomes,
    LGEStatus,
    PatientProfile,
    PEthMeasurement,
    PEthRiskTier,
    RecoveryCategory,
    Sex,
    TTNVariant,
)
from crip.ehr.db import SyntheticEHRDatabase


@pytest.fixture
def synthetic_acdc_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic ACDC dataset directory with valid structure."""
    dataset_dir = tmp_path / "acdc_synthetic"
    p1_dir = dataset_dir / "patient001"
    p1_dir.mkdir(parents=True)

    info_content = (
        "ED: 1\n"
        "ES: 6\n"
        "Group: DCM\n"
        "Height: 180.0\n"
        "NbFrame: 10\n"
        "Weight: 80.0\n"
    )
    (p1_dir / "Info.cfg").write_text(info_content, encoding="utf-8")

    affine = np.diag([1.5, 1.5, 10.0, 1.0])
    img_ed_data = np.ones((16, 16, 4), dtype=np.float32) * 100.0
    img_es_data = np.ones((16, 16, 4), dtype=np.float32) * 80.0

    mask_ed_data = np.zeros((16, 16, 4), dtype=np.uint8)
    mask_ed_data[4:8, 4:8, :] = 1
    mask_ed_data[8:12, 4:8, :] = 2
    mask_ed_data[8:12, 8:12, :] = 3

    mask_es_data = np.zeros((16, 16, 4), dtype=np.uint8)
    mask_es_data[5:7, 5:7, :] = 1
    mask_es_data[8:11, 5:8, :] = 2
    mask_es_data[8:11, 8:11, :] = 3

    nib.save(nib.Nifti1Image(img_ed_data, affine), str(p1_dir / "patient001_frame01.nii.gz"))
    nib.save(nib.Nifti1Image(mask_ed_data, affine), str(p1_dir / "patient001_frame01_gt.nii.gz"))
    nib.save(nib.Nifti1Image(img_es_data, affine), str(p1_dir / "patient001_frame06.nii.gz"))
    nib.save(nib.Nifti1Image(mask_es_data, affine), str(p1_dir / "patient001_frame06_gt.nii.gz"))

    cine_data = np.ones((16, 16, 4, 10), dtype=np.float32)
    nib.save(nib.Nifti1Image(cine_data, affine), str(p1_dir / "patient001_4d.nii.gz"))

    return dataset_dir


@pytest.fixture
def valid_peth_measurement() -> PEthMeasurement:
    """Fixture providing a typical heavy-consumption PEth measurement."""
    return PEthMeasurement(
        value_ng_ml=350.0,
        measurement_date=date(2025, 3, 1),
    )


@pytest.fixture
def valid_follow_up_outcomes() -> FollowUpOutcomes:
    """Fixture providing a completed 6-month recovery outcome."""
    return FollowUpOutcomes(
        follow_up_date=date(2025, 9, 1),
        follow_up_lvef=52.0,
        recovery_category=RecoveryCategory.COMPLETE_RECOVERY,
        adverse_cardiac_events=[],
        sustained_abstinence_confirmed=True,
    )


@pytest.fixture
def valid_patient_profile(valid_peth_measurement, valid_follow_up_outcomes) -> PatientProfile:
    """Fixture providing a fully valid synthetic patient profile."""
    # EDV = 200, ESV = 140 -> SV = 60 -> LVEF = 30%
    return PatientProfile(
        patient_id="CRIP-SYN-TEST01",
        age=50,
        biological_sex=Sex.MALE,
        baseline_lvef=30.0,
        edv_ml=200.0,
        esv_ml=140.0,
        peth=valid_peth_measurement,
        ttn_variant=TTNVariant.NOT_DETECTED,
        lge_status=LGEStatus.NEGATIVE,
        follow_up=valid_follow_up_outcomes,
    )


@pytest.fixture
def temp_ehr_db(tmp_path) -> SyntheticEHRDatabase:
    """Fixture providing a temporary SQLite database."""
    db_file = tmp_path / "test_synthetic_ehr.db"
    return SyntheticEHRDatabase(db_path=db_file)
