"""Unit tests for synthetic patient data schemas and validation rules."""

from datetime import date
import pytest
from pydantic import ValidationError

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


class TestPEthMeasurement:
    """Tests for PEth biomarker schema and risk stratification."""

    def test_peth_abstinent_tier(self):
        m = PEthMeasurement(value_ng_ml=12.5, measurement_date=date(2025, 1, 1))
        assert m.risk_tier == PEthRiskTier.ABSTINENT_OR_LIGHT

    def test_peth_moderate_tier(self):
        m = PEthMeasurement(value_ng_ml=110.0, measurement_date=date(2025, 1, 1))
        assert m.risk_tier == PEthRiskTier.SIGNIFICANT_MODERATE

    def test_peth_heavy_tier(self):
        m = PEthMeasurement(value_ng_ml=450.0, measurement_date=date(2025, 1, 1))
        assert m.risk_tier == PEthRiskTier.HEAVY_CHRONIC

    def test_negative_peth_rejected(self):
        with pytest.raises(ValidationError):
            PEthMeasurement(value_ng_ml=-5.0, measurement_date=date(2025, 1, 1))


class TestPatientProfile:
    """Tests for synthetic patient profile model validation."""

    def test_valid_patient_creation(self, valid_patient_profile):
        assert valid_patient_profile.patient_id == "CRIP-SYN-TEST01"
        assert valid_patient_profile.stroke_volume_ml == 60.0
        assert valid_patient_profile.volume_derived_lvef == 30.0

    def test_esv_greater_than_edv_rejected(self, valid_peth_measurement):
        with pytest.raises(ValidationError) as excinfo:
            PatientProfile(
                patient_id="CRIP-SYN-INVALID",
                age=45,
                baseline_lvef=30.0,
                edv_ml=150.0,
                esv_ml=180.0,  # Invalid: ESV > EDV
                peth=valid_peth_measurement,
            )
        assert "strictly less than end-diastolic volume" in str(excinfo.value)

    def test_baseline_lvef_mismatch_rejected(self, valid_peth_measurement):
        # EDV = 200, ESV = 140 -> derived LVEF is 30.0%
        # baseline_lvef = 45.0% deviates by 15% (> 5% tolerance)
        with pytest.raises(ValidationError) as excinfo:
            PatientProfile(
                patient_id="CRIP-SYN-INVALID",
                age=45,
                baseline_lvef=45.0,
                edv_ml=200.0,
                esv_ml=140.0,
                peth=valid_peth_measurement,
            )
        assert "deviates by >5%" in str(excinfo.value)

    def test_age_out_of_bounds_rejected(self, valid_peth_measurement):
        with pytest.raises(ValidationError):
            PatientProfile(
                patient_id="CRIP-SYN-INVALID",
                age=15,  # Under 18
                baseline_lvef=30.0,
                edv_ml=200.0,
                esv_ml=140.0,
                peth=valid_peth_measurement,
            )

    def test_whitespace_patient_id_rejected(self, valid_peth_measurement):
        with pytest.raises(ValidationError):
            PatientProfile(
                patient_id="   ",
                age=45,
                baseline_lvef=30.0,
                edv_ml=200.0,
                esv_ml=140.0,
                peth=valid_peth_measurement,
            )

    def test_serialization_and_deserialization(self, valid_patient_profile):
        json_data = valid_patient_profile.model_dump_json()
        reconstructed = PatientProfile.model_validate_json(json_data)
        assert reconstructed.patient_id == valid_patient_profile.patient_id
        assert reconstructed.baseline_lvef == valid_patient_profile.baseline_lvef
        assert reconstructed.ttn_variant == valid_patient_profile.ttn_variant
        assert reconstructed.lge_status == valid_patient_profile.lge_status
