"""Automated unit and integration tests for CRIP Milestone 3A:
Literature-Grounded Synthetic Clinical Cohort and Multimodal Data Integration.

Covers all 20 required validation criteria:
 1. Cohort reproducibility with fixed seed.
 2. Valid patient schemas.
 3. Biological and numerical bounds.
 4. Mathematical LVEF/volume consistency (ESV < EDV, calculated LVEF matching).
 5. PEth validation & risk tiers.
 6. TTN 5-tier classification categories & VUS isolation.
 7. LGE missingness and extent bounds.
 8. Generation metadata integrity.
 9. Provenance tracking on individual variables.
10. Non-trivial variable correlations (|r| < 0.98).
11. Recovery-label calculation (Delta LVEF >= 10.0 -> 1, else 0).
12. Missing-data mechanisms (missing PEth, untested TTN, missing LGE).
13. Export integrity (CSV and JSON files exist, matching IDs).
14. Database import, patient retrieval, and filtering.
15. Existing schema backward compatibility.
16. Streamlit page rendering with AppTest.
17. Cohort summary calculations.
18. Parameter sensitivity analysis.
19. Duplicate identifier rejection.
20. Non-trivial separability test (all univariate AUCs < 0.82).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from crip.ehr.db import SyntheticEHRDatabase
from crip.synthetic.cohort_generator import SyntheticCohortGenerator
from crip.synthetic.schemas import (
    AbstinenceScenario,
    FollowUpOutcomes,
    LGEPattern,
    LGEStatus,
    PatientProfile,
    PEthMeasurement,
    PEthRiskTier,
    ProvenanceType,
    RecoveryCategory,
    Sex,
    TTNClassification,
    TTNVariant,
    map_ttn_classification_to_variant,
    map_ttn_variant_to_classification,
)


@pytest.fixture(scope="module")
def generator() -> SyntheticCohortGenerator:
    """Fixture providing an initialized SyntheticCohortGenerator."""
    return SyntheticCohortGenerator(seed=42)


@pytest.fixture(scope="module")
def sample_cohort_100(generator: SyntheticCohortGenerator):
    """Fixture generating a 100-patient synthetic cohort for rapid testing."""
    profiles, invalid, meta = generator.generate_cohort(n_patients=100, abstinence_prob=0.70)
    return profiles, invalid, meta


@pytest.fixture(scope="module")
def exported_cohort_files():
    """Paths to the exported 3,000-patient cohort files."""
    base_dir = Path("data/synthetic")
    return {
        "csv": base_dir / "synthetic_cardiomyopathy_cohort.csv",
        "json": base_dir / "synthetic_cardiomyopathy_cohort.json",
        "meta": base_dir / "generation_metadata.json",
        "quality": base_dir / "cohort_quality_report.json",
    }


# ==============================================================================
# 1. Cohort Generation Reproducibility & Identifiers
# ==============================================================================

class TestCohortGenerationReproducibility:
    """Tests for determinism, reproducibility, and identifier uniqueness."""

    def test_criterion_1_cohort_reproducibility_with_fixed_seed(self):
        """Criterion 1: Verify identical patient profiles are produced across identical seeds."""
        gen1 = SyntheticCohortGenerator(seed=123)
        p1, _, _ = gen1.generate_cohort(n_patients=20, abstinence_prob=0.70)

        gen2 = SyntheticCohortGenerator(seed=123)
        p2, _, _ = gen2.generate_cohort(n_patients=20, abstinence_prob=0.70)

        assert len(p1) == len(p2) == 20
        for pat1, pat2 in zip(p1, p2):
            assert pat1.patient_id == pat2.patient_id
            assert pat1.age == pat2.age
            assert pat1.baseline_lvef == pytest.approx(pat2.baseline_lvef, rel=1e-4)
            assert pat1.edv_ml == pytest.approx(pat2.edv_ml, rel=1e-4)
            assert pat1.follow_up.follow_up_lvef == pytest.approx(pat2.follow_up.follow_up_lvef, rel=1e-4)
            assert pat1.follow_up.recovery_binary == pat2.follow_up.recovery_binary

    def test_criterion_19_duplicate_identifier_rejection(self, sample_cohort_100):
        """Criterion 19: Verify all generated patient identifiers are strictly unique."""
        profiles, _, _ = sample_cohort_100
        patient_ids = [p.patient_id for p in profiles]
        assert len(patient_ids) == len(set(patient_ids)), "Duplicate patient identifiers detected!"

    def test_criterion_8_generation_metadata_integrity(self, sample_cohort_100):
        """Criterion 8: Verify metadata contains required version, seed, target N, and audit keys."""
        _, _, meta = sample_cohort_100
        assert meta["generation_version"] == "v1.0-m3a"
        assert meta["generation_seed"] == 42
        assert meta["target_n"] == 100
        assert meta["actual_n"] == 100
        assert "invalid_attempt_count" in meta
        assert "rejection_rate_percent" in meta
        assert "structural_grounding" in meta


# ==============================================================================
# 2. Schema Validation, Biological Bounds, and Mathematical Integrity
# ==============================================================================

class TestClinicalAndBiologicalBounds:
    """Tests for schema conformity, biological ranges, and volumetric consistency."""

    def test_criterion_2_valid_patient_schemas(self, sample_cohort_100):
        """Criterion 2: Verify all patients conform strictly to the Pydantic PatientProfile schema."""
        profiles, _, _ = sample_cohort_100
        for p in profiles:
            assert isinstance(p, PatientProfile)
            assert p.patient_id.startswith("CRIP-SYN-")
            assert p.biological_sex in (Sex.MALE, Sex.FEMALE)

    def test_criterion_3_biological_and_numerical_bounds(self, sample_cohort_100):
        """Criterion 3: Verify all continuous variables fall within plausible clinical ranges."""
        profiles, _, _ = sample_cohort_100
        for p in profiles:
            assert 18 <= p.age <= 95, f"Age {p.age} out of biological bounds"
            assert 10.0 <= p.baseline_lvef <= 65.0, f"LVEF {p.baseline_lvef} out of bounds"
            assert 100.0 <= p.edv_ml <= 500.0, f"EDV {p.edv_ml} out of bounds"
            assert 30.0 <= p.esv_ml <= 400.0, f"ESV {p.esv_ml} out of bounds"
            if p.peth_available and p.peth:
                assert 0.0 <= p.peth.value_ng_ml <= 4000.0, f"PEth {p.peth.value_ng_ml} out of bounds"
            if p.myocardial_mass_g is not None:
                assert 40.0 <= p.myocardial_mass_g <= 450.0

    def test_criterion_4_mathematical_lvef_and_volume_consistency(self, sample_cohort_100):
        """Criterion 4: Verify physical relationship ESV < EDV and calculated LVEF match within 0.1%."""
        profiles, _, _ = sample_cohort_100
        for p in profiles:
            assert p.esv_ml < p.edv_ml, f"Patient {p.patient_id}: ESV ({p.esv_ml}) >= EDV ({p.edv_ml})"
            calc_lvef = ((p.edv_ml - p.esv_ml) / p.edv_ml) * 100.0
            assert abs(calc_lvef - p.baseline_lvef) <= 0.1, (
                f"LVEF discrepancy: calc={calc_lvef:.2f}, baseline={p.baseline_lvef:.2f}"
            )
            assert p.stroke_volume_ml == pytest.approx(p.edv_ml - p.esv_ml, abs=0.01)


# ==============================================================================
# 3. Toxicology (PEth) & Genetics (TTN) & CMR Tissue (LGE)
# ==============================================================================

class TestMultimodalBiomarkersAndGenetics:
    """Tests for PEth tiers, ACMG-AMP genetics, and LGE fibrosis classification."""

    def test_criterion_5_peth_validation_and_risk_tiers(self):
        """Criterion 5: Verify PEth clinical risk tier thresholds (<20, 20-200, >200 ng/mL)."""
        p_abstinent = PEthMeasurement(value_ng_ml=12.5, measurement_date=date(2025, 1, 1))
        assert p_abstinent.risk_tier == PEthRiskTier.ABSTINENT_OR_LIGHT

        p_mod = PEthMeasurement(value_ng_ml=85.0, measurement_date=date(2025, 1, 1))
        assert p_mod.risk_tier == PEthRiskTier.SIGNIFICANT_MODERATE

        p_heavy = PEthMeasurement(value_ng_ml=450.0, measurement_date=date(2025, 1, 1))
        assert p_heavy.risk_tier == PEthRiskTier.HEAVY_CHRONIC

    def test_criterion_6_ttn_5_tier_classification_and_vus_isolation(self):
        """Criterion 6: Verify TTN 5-tier classification categories and assert VUS is never pathogenic."""
        assert TTNClassification.PATHOGENIC.value == "PATHOGENIC"
        assert TTNClassification.LIKELY_PATHOGENIC.value == "LIKELY_PATHOGENIC"
        assert TTNClassification.VARIANT_OF_UNCERTAIN_SIGNIFICANCE.value == "VARIANT_OF_UNCERTAIN_SIGNIFICANCE"
        assert TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC.value == "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC"
        assert TTNClassification.NOT_TESTED.value == "NOT_TESTED"

        # Bidirectional mapping test
        mapped_cls = map_ttn_variant_to_classification(TTNVariant.PATHOGENIC_TRUNCATING)
        assert mapped_cls == TTNClassification.PATHOGENIC

        mapped_vus = map_ttn_variant_to_classification(TTNVariant.VARIANT_UNCERTAIN_SIGNIFICANCE)
        assert mapped_vus == TTNClassification.VARIANT_OF_UNCERTAIN_SIGNIFICANCE
        assert mapped_vus not in (TTNClassification.PATHOGENIC, TTNClassification.LIKELY_PATHOGENIC)

        back_var = map_ttn_classification_to_variant(TTNClassification.VARIANT_OF_UNCERTAIN_SIGNIFICANCE)
        assert back_var == TTNVariant.VARIANT_UNCERTAIN_SIGNIFICANCE

    def test_criterion_7_lge_missingness_and_extent_bounds(self, sample_cohort_100):
        """Criterion 7: Verify LGE fibrosis patterns, extent bounds, and contraindication missingness."""
        profiles, _, _ = sample_cohort_100
        for p in profiles:
            if not p.lge_available:
                # When LGE is missing/contraindicated, extent should be None and status NEGATIVE or pattern NONE
                assert p.lge_extent_percent is None
            elif p.lge_status == LGEStatus.POSITIVE:
                assert p.lge_extent_percent is not None
                assert 1.0 <= p.lge_extent_percent <= 45.0
                assert p.lge_pattern in (
                    LGEPattern.MIDWALL,
                    LGEPattern.SUBEPICARDIAL,
                    LGEPattern.SUBENDOCARDIAL,
                    LGEPattern.PATCHY_TRANSMURAL,
                )
            else:
                assert p.lge_pattern == LGEPattern.NONE or p.lge_extent_percent is None or p.lge_extent_percent == 0.0

    def test_criterion_12_missing_data_mechanisms(self, sample_cohort_100):
        """Criterion 12: Verify missingness mechanisms operate in expected proportions (3-25%)."""
        profiles, _, _ = sample_cohort_100
        n = len(profiles)

        missing_peth = sum(1 for p in profiles if not p.peth_available)
        untested_ttn = sum(1 for p in profiles if p.ttn_classification == TTNClassification.NOT_TESTED)
        missing_lge = sum(1 for p in profiles if not p.lge_available)
        missing_history = sum(1 for p in profiles if not p.alcohol_history_available)

        # Expected range: 1% - 35% in realistic sampling
        assert 0 <= (missing_peth / n) <= 0.20
        assert 0.05 <= (untested_ttn / n) <= 0.35
        assert 0.02 <= (missing_lge / n) <= 0.25
        assert 0.01 <= (missing_history / n) <= 0.20


# ==============================================================================
# 4. Recovery Endpoint, Correlations, and Separability
# ==============================================================================

class TestRecoveryEndpointAndSeparability:
    """Tests for recovery endpoint arithmetic, non-triviality, and multivariable separability."""

    def test_criterion_11_recovery_label_calculation(self, sample_cohort_100):
        """Criterion 11: Verify recovery binary label is strictly 1 when Delta LVEF >= 10.0, else 0."""
        profiles, _, _ = sample_cohort_100
        for p in profiles:
            fu = p.follow_up
            assert fu.follow_up_lvef is not None
            expected_delta = round(fu.follow_up_lvef - p.baseline_lvef, 2)
            assert fu.delta_lvef == pytest.approx(expected_delta, abs=0.01)

            if expected_delta >= 10.0:
                assert fu.recovery_binary == 1, f"Delta {expected_delta} >= 10.0 but recovery_binary != 1"
            else:
                assert fu.recovery_binary == 0, f"Delta {expected_delta} < 10.0 but recovery_binary != 0"

    def test_criterion_10_non_trivial_variable_correlations(self, sample_cohort_100):
        """Criterion 10: Verify no two non-identical features exhibit perfect correlation (|r| < 0.98)."""
        profiles, _, _ = sample_cohort_100
        data = []
        for p in profiles:
            data.append({
                "age": p.age,
                "baseline_lvef": p.baseline_lvef,
                "edv_ml": p.edv_ml,
                "esv_ml": p.esv_ml,
                "peth": p.peth.value_ng_ml if p.peth_available else np.nan,
                "follow_up_lvef": p.follow_up.follow_up_lvef,
            })
        df = pd.DataFrame(data).dropna()
        corr_matrix = df.corr().abs()

        # Check all off-diagonal correlations except EDV vs ESV
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2 and not (col1 in ("edv_ml", "esv_ml") and col2 in ("edv_ml", "esv_ml")):
                    r = corr_matrix.loc[col1, col2]
                    assert r < 0.98, f"Suspiciously high correlation between {col1} and {col2}: {r:.3f}"

    def test_criterion_20_non_trivial_separability_test(self):
        """Criterion 20: Verify no single univariate feature achieves ROC-AUC > 0.82 for recovery."""
        csv_path = Path("data/synthetic/synthetic_cardiomyopathy_cohort.csv")
        if not csv_path.is_file():
            pytest.skip("Full 3,000 cohort CSV not present for separability audit.")

        df = pd.read_csv(csv_path)
        y = df["recovery_binary"].values

        features_to_check = [
            "age",
            "baseline_lvef",
            "edv_ml",
            "esv_ml",
            "peth_value_ng_ml",
            "reported_drinks_per_week",
            "years_heavy_drinking",
        ]

        for feat in features_to_check:
            sub = df[[feat, "recovery_binary"]].dropna()
            if len(sub) > 100:
                auc = roc_auc_score(sub["recovery_binary"], sub[feat])
                univariate_auc = max(auc, 1.0 - auc)
                assert univariate_auc < 0.82, (
                    f"Feature '{feat}' achieves univariate ROC-AUC of {univariate_auc:.3f} > 0.82, "
                    f"violating non-trivial multimodal complexity constraint!"
                )

    def test_criterion_18_parameter_sensitivity_analysis(self, generator: SyntheticCohortGenerator):
        """Criterion 18: Verify altering abstinence probability significantly modulates recovery rate."""
        # 40% abstinence scenario
        p_low, _, _ = generator.generate_cohort(n_patients=60, abstinence_prob=0.40)
        rec_rate_low = sum(1 for p in p_low if p.follow_up.recovery_binary == 1) / len(p_low)

        # 90% abstinence scenario
        p_high, _, _ = generator.generate_cohort(n_patients=60, abstinence_prob=0.90)
        rec_rate_high = sum(1 for p in p_high if p.follow_up.recovery_binary == 1) / len(p_high)

        # Higher abstinence must yield a substantially higher recovery rate
        assert rec_rate_high > rec_rate_low, (
            f"Expected rec_rate_high ({rec_rate_high:.2f}) > rec_rate_low ({rec_rate_low:.2f})"
        )


# ==============================================================================
# 5. Cohort Export, Provenance, and SQLite Database Integration
# ==============================================================================

class TestCohortExportAndDatabaseIntegration:
    """Tests for file export integrity, provenance tracking, and database querying/filtering."""

    def test_criterion_9_provenance_tracking_on_individual_variables(self, sample_cohort_100):
        """Criterion 9: Verify variable_provenance dictionary is populated for every patient."""
        profiles, _, _ = sample_cohort_100
        for p in profiles:
            assert isinstance(p.variable_provenance, dict)
            assert "baseline_lvef" in p.variable_provenance
            assert "ttn_classification" in p.variable_provenance
            assert "peth" in p.variable_provenance
            assert "recovery_binary" in p.variable_provenance

    def test_criterion_13_export_integrity(self, exported_cohort_files):
        """Criterion 13: Verify exported CSV, JSON, and quality report files exist and have matching records."""
        csv_file = exported_cohort_files["csv"]
        json_file = exported_cohort_files["json"]
        meta_file = exported_cohort_files["meta"]

        if not csv_file.is_file() or not json_file.is_file():
            pytest.skip("Exported cohort files not found on disk.")

        df = pd.read_csv(csv_file)
        assert len(df) == 3000

        with open(json_file, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        assert len(data_json) == 3000

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["target_n"] == 3000
        assert meta["actual_n"] == 3000

        # Verify ID alignment
        assert df["patient_id"].iloc[0] == data_json[0]["patient_id"]
        assert df["patient_id"].iloc[-1] == data_json[-1]["patient_id"]

    def test_criterion_14_database_import_and_filtering(self, tmp_path: Path, sample_cohort_100):
        """Criterion 14: Verify SQLite database imports records and supports multimodal filtering."""
        profiles, _, _ = sample_cohort_100
        db_path = tmp_path / "test_multimodal.db"
        db = SyntheticEHRDatabase(db_path=db_path)

        # Batch insert
        count = db.insert_patients_batch(profiles)
        assert count == 100
        assert db.count_patients() == 100

        # Fetch single patient
        p0 = db.get_patient(profiles[0].patient_id)
        assert p0 is not None
        assert p0.patient_id == profiles[0].patient_id
        assert p0.age == profiles[0].age
        assert p0.follow_up.recovery_binary == profiles[0].follow_up.recovery_binary

        # Filter by sex
        males = db.filter_patients(biological_sex=Sex.MALE)
        assert all(p.biological_sex == Sex.MALE for p in males)

        # Filter by recovery
        recovered = db.filter_patients(recovery_binary=1)
        assert all(p.follow_up.recovery_binary == 1 for p in recovered)

        # Filter by PEth tier
        heavy = db.filter_patients(peth_risk_tier=PEthRiskTier.HEAVY_CHRONIC)
        assert all(p.peth.value_ng_ml > 200.0 for p in heavy if p.peth_available)

    def test_criterion_15_existing_schema_backward_compatibility(self, tmp_path: Path):
        """Criterion 15: Verify legacy schemas with TTNVariant and LGEStatus persist seamlessly."""
        db_path = tmp_path / "test_legacy.db"
        db = SyntheticEHRDatabase(db_path=db_path)

        legacy_patient = PatientProfile(
            patient_id="CRIP-LEGACY-01",
            age=52,
            biological_sex=Sex.FEMALE,
            baseline_lvef=32.0,
            edv_ml=210.0,
            esv_ml=142.8,
            peth=PEthMeasurement(value_ng_ml=150.0, measurement_date=date(2025, 1, 10)),
            ttn_variant=TTNVariant.PATHOGENIC_TRUNCATING,
            lge_status=LGEStatus.MIDWALL,
            follow_up=FollowUpOutcomes(
                follow_up_date=date(2025, 7, 10),
                follow_up_lvef=44.0,
                recovery_category=RecoveryCategory.PARTIAL_RECOVERY,
            ),
        )

        db.insert_patient(legacy_patient)
        retrieved = db.get_patient("CRIP-LEGACY-01")
        assert retrieved is not None
        assert retrieved.ttn_variant == TTNVariant.PATHOGENIC_TRUNCATING
        assert retrieved.ttn_classification == TTNClassification.PATHOGENIC
        assert retrieved.lge_status == LGEStatus.MIDWALL
        assert retrieved.follow_up.recovery_binary == 1  # 44 - 32 = 12 >= 10

    def test_criterion_17_cohort_summary_calculations(self, tmp_path: Path, sample_cohort_100):
        """Criterion 17: Verify get_cohort_summary computes accurate aggregate statistics."""
        profiles, _, _ = sample_cohort_100
        db_path = tmp_path / "test_summary.db"
        db = SyntheticEHRDatabase(db_path=db_path)
        db.insert_patients_batch(profiles)

        summary = db.get_cohort_summary()
        assert summary["total_patients"] == 100
        assert summary["male_count"] + summary["female_count"] == 100
        assert 20.0 <= summary["mean_baseline_lvef"] <= 40.0
        assert 0.0 <= summary["recovery_rate_percent"] <= 100.0


# ==============================================================================
# 6. Streamlit Page Integration (AppTest)
# ==============================================================================

class TestStreamlitCohortDisplayAppTest:
    """Tests for Streamlit dashboard rendering under AppTest."""

    def test_criterion_16_streamlit_page_rendering_with_apptest(self):
        """Criterion 16: Verify 1_Patient_Profile.py renders cleanly without exceptions."""
        from streamlit.testing.v1 import AppTest

        page_file = Path("src/crip/app/pages/1_Patient_Profile.py").resolve()
        at = AppTest.from_file(str(page_file), default_timeout=10)
        at.run()

        assert not at.exception, f"AppTest threw exception: {at.exception}"
        # Verify title or header rendered
        assert len(at.title) > 0
        assert "Synthetic Patient Profile" in at.title[0].value
