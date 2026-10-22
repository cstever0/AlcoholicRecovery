"""Test configuration and shared fixtures for CRIP unit tests."""

from datetime import date
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
