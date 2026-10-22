"""Unit tests for SQLite synthetic EHR database."""

from datetime import date
from crip.ehr.db import SyntheticEHRDatabase, seed_reference_synthetic_cohort
from crip.synthetic.schemas import (
    FollowUpOutcomes,
    LGEStatus,
    PatientProfile,
    PEthMeasurement,
    RecoveryCategory,
    Sex,
    TTNVariant,
)


class TestSyntheticEHRDatabase:
    """Tests for SQLite synthetic EHR operations."""

    def test_database_initialization(self, temp_ehr_db):
        assert temp_ehr_db.count_patients() == 0

    def test_insert_and_get_patient(self, temp_ehr_db, valid_patient_profile):
        temp_ehr_db.insert_patient(valid_patient_profile)
        assert temp_ehr_db.count_patients() == 1

        retrieved = temp_ehr_db.get_patient("CRIP-SYN-TEST01")
        assert retrieved is not None
        assert retrieved.patient_id == "CRIP-SYN-TEST01"
        assert retrieved.age == 50
        assert retrieved.baseline_lvef == 30.0
        assert retrieved.edv_ml == 200.0
        assert retrieved.esv_ml == 140.0
        assert retrieved.peth.value_ng_ml == 350.0
        assert retrieved.ttn_variant == TTNVariant.NOT_DETECTED
        assert retrieved.lge_status == LGEStatus.NEGATIVE
        assert retrieved.follow_up.recovery_category == RecoveryCategory.COMPLETE_RECOVERY

    def test_get_nonexistent_patient_returns_none(self, temp_ehr_db):
        assert temp_ehr_db.get_patient("CRIP-NONEXISTENT") is None

    def test_seed_reference_cohort(self, temp_ehr_db):
        seed_reference_synthetic_cohort(temp_ehr_db)
        assert temp_ehr_db.count_patients() == 3

        # Calling seed again should be idempotent
        seed_reference_synthetic_cohort(temp_ehr_db)
        assert temp_ehr_db.count_patients() == 3

        patients = temp_ehr_db.list_patients()
        assert len(patients) == 3
        assert {p.patient_id for p in patients} == {"CRIP-SYN-001", "CRIP-SYN-002", "CRIP-SYN-003"}
