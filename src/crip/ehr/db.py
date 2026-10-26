"""Local SQLite synthetic EHR database manager for CRIP.

DISCLAIMER:
Provides local simulated EHR storage for synthetic patient records only.
No live EHR integrations, cloud connections, or real patient data are permitted.
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

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
    map_ttn_variant_to_classification,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/synthetic/ehr_synthetic.db")


class SyntheticEHRDatabase:
    """Manages local SQLite database operations for simulated synthetic patient records."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize database tables for synthetic patients and handle column migrations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS synthetic_patients (
                    patient_id TEXT PRIMARY KEY,
                    age INTEGER NOT NULL,
                    biological_sex TEXT NOT NULL,
                    baseline_lvef REAL NOT NULL,
                    edv_ml REAL NOT NULL,
                    esv_ml REAL NOT NULL,
                    peth_value_ng_ml REAL,
                    peth_measurement_date TEXT,
                    ttn_variant TEXT NOT NULL,
                    lge_status TEXT NOT NULL,
                    follow_up_date TEXT,
                    follow_up_lvef REAL,
                    recovery_category TEXT NOT NULL,
                    adverse_cardiac_events TEXT,
                    sustained_abstinence_confirmed INTEGER,
                    rvedv_ml REAL,
                    rvesv_ml REAL,
                    rvef_percent REAL,
                    myocardial_mass_g REAL,
                    reported_drinks_per_week REAL,
                    time_since_last_drink_days INTEGER,
                    years_heavy_drinking REAL,
                    alcohol_history_available INTEGER DEFAULT 1,
                    peth_available INTEGER DEFAULT 1,
                    ttn_classification TEXT,
                    ttn_provenance TEXT,
                    lge_pattern TEXT,
                    lge_extent_percent REAL,
                    lge_available INTEGER DEFAULT 1,
                    mri_source TEXT,
                    measurement_date TEXT,
                    provenance TEXT,
                    generation_version TEXT,
                    generation_seed INTEGER,
                    variable_provenance TEXT,
                    delta_lvef REAL,
                    recovery_binary INTEGER,
                    assumed_abstinence_scenario TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Auto-migrate: check existing columns in case table was created with an older schema
            cursor.execute("PRAGMA table_info(synthetic_patients)")
            existing_cols = {row["name"] for row in cursor.fetchall()}

            multimodal_columns = [
                ("rvedv_ml", "REAL"),
                ("rvesv_ml", "REAL"),
                ("rvef_percent", "REAL"),
                ("myocardial_mass_g", "REAL"),
                ("reported_drinks_per_week", "REAL"),
                ("time_since_last_drink_days", "INTEGER"),
                ("years_heavy_drinking", "REAL"),
                ("alcohol_history_available", "INTEGER DEFAULT 1"),
                ("peth_available", "INTEGER DEFAULT 1"),
                ("ttn_classification", "TEXT"),
                ("ttn_provenance", "TEXT"),
                ("lge_pattern", "TEXT"),
                ("lge_extent_percent", "REAL"),
                ("lge_available", "INTEGER DEFAULT 1"),
                ("mri_source", "TEXT"),
                ("measurement_date", "TEXT"),
                ("provenance", "TEXT"),
                ("generation_version", "TEXT"),
                ("generation_seed", "INTEGER"),
                ("variable_provenance", "TEXT"),
                ("delta_lvef", "REAL"),
                ("recovery_binary", "INTEGER"),
                ("assumed_abstinence_scenario", "TEXT"),
            ]

            for col_name, col_type in multimodal_columns:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE synthetic_patients ADD COLUMN {col_name} {col_type};")

            conn.commit()

    def insert_patient(self, patient: PatientProfile) -> None:
        """Insert or replace a single synthetic patient record."""
        self.insert_patients_batch([patient])

    def insert_patients_batch(self, patients: List[PatientProfile]) -> int:
        """Batch insert or replace synthetic patient records efficiently in a single transaction."""
        if not patients:
            return 0

        rows = []
        for p in patients:
            peth_val = p.peth.value_ng_ml if p.peth else None
            peth_dt = p.peth.measurement_date.isoformat() if p.peth and p.peth.measurement_date else None
            ttn_class_val = p.ttn_classification.value if p.ttn_classification else map_ttn_variant_to_classification(p.ttn_variant).value
            lge_pat_val = p.lge_pattern.value if p.lge_pattern else ("MIDWALL" if p.lge_status == LGEStatus.POSITIVE else "NONE")
            var_prov_json = json.dumps(p.variable_provenance) if p.variable_provenance else None
            adv_events_json = json.dumps(p.follow_up.adverse_cardiac_events) if p.follow_up.adverse_cardiac_events else json.dumps([])
            sustained = 1 if p.follow_up.sustained_abstinence_confirmed else (0 if p.follow_up.sustained_abstinence_confirmed is False else None)

            rows.append((
                p.patient_id,
                p.age,
                p.biological_sex.value,
                p.baseline_lvef,
                p.edv_ml,
                p.esv_ml,
                peth_val,
                peth_dt,
                p.ttn_variant.value,
                p.lge_status.value,
                p.follow_up.follow_up_date.isoformat() if p.follow_up.follow_up_date else None,
                p.follow_up.follow_up_lvef,
                p.follow_up.recovery_category.value,
                adv_events_json,
                sustained,
                p.rvedv_ml,
                p.rvesv_ml,
                p.rvef_percent,
                p.myocardial_mass_g,
                p.reported_drinks_per_week,
                p.time_since_last_drink_days,
                p.years_heavy_drinking,
                1 if p.alcohol_history_available else 0,
                1 if p.peth_available else 0,
                ttn_class_val,
                p.ttn_provenance,
                lge_pat_val,
                p.lge_extent_percent,
                1 if p.lge_available else 0,
                p.mri_source,
                p.measurement_date.isoformat() if p.measurement_date else None,
                p.provenance.value if hasattr(p.provenance, "value") else str(p.provenance),
                p.generation_version,
                p.generation_seed,
                var_prov_json,
                p.delta_lvef,
                p.recovery_binary,
                p.follow_up.assumed_abstinence_scenario.value if p.follow_up.assumed_abstinence_scenario else None,
            ))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO synthetic_patients (
                    patient_id, age, biological_sex, baseline_lvef, edv_ml, esv_ml,
                    peth_value_ng_ml, peth_measurement_date, ttn_variant, lge_status,
                    follow_up_date, follow_up_lvef, recovery_category,
                    adverse_cardiac_events, sustained_abstinence_confirmed,
                    rvedv_ml, rvesv_ml, rvef_percent, myocardial_mass_g,
                    reported_drinks_per_week, time_since_last_drink_days, years_heavy_drinking,
                    alcohol_history_available, peth_available, ttn_classification, ttn_provenance,
                    lge_pattern, lge_extent_percent, lge_available, mri_source,
                    measurement_date, provenance, generation_version, generation_seed,
                    variable_provenance, delta_lvef, recovery_binary, assumed_abstinence_scenario
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                rows,
            )
            conn.commit()

        return len(rows)

    def import_cohort_from_json(self, json_path: Path | str) -> int:
        """Import synthetic patient cohort directly from exported JSON file."""
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cohort JSON not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        profiles: List[PatientProfile] = []
        for item in raw_data:
            profiles.append(PatientProfile.model_validate(item))

        count = self.insert_patients_batch(profiles)
        logger.info(f"Imported {count} patients into SQLite database from {path}")
        return count

    def import_cohort_from_csv(self, csv_path: Path | str) -> int:
        """Import synthetic patient cohort from exported CSV file."""
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cohort CSV not found: {path}")

        profiles: List[PatientProfile] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_id = row["patient_id"]
                age = int(row["age"])
                sex = Sex(row["biological_sex"])
                bl_lvef = float(row["baseline_lvef"])
                edv = float(row["edv_ml"])
                esv = float(row["esv_ml"])

                # PEth
                peth_val_str = row.get("peth_value_ng_ml")
                peth_avail = row.get("peth_available", "True").lower() in ("true", "1")
                if peth_val_str and peth_val_str.strip():
                    peth_val = float(peth_val_str)
                    peth_date_str = row.get("peth_measurement_date")
                    peth_dt = date.fromisoformat(peth_date_str) if peth_date_str else date(2025, 1, 15)
                else:
                    peth_val = 0.0
                    peth_dt = date(2025, 1, 15)
                peth_obj = PEthMeasurement(value_ng_ml=peth_val, measurement_date=peth_dt)

                # TTN
                ttn_class_str = row.get("ttn_classification", "NO_PATHOGENIC_OR_LIKELY_PATHOGENIC")
                try:
                    ttn_class = TTNClassification(ttn_class_str)
                except ValueError:
                    ttn_class = TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC

                ttn_leg_str = row.get("ttn_variant_legacy") or row.get("ttn_variant", "NOT_DETECTED")
                try:
                    ttn_legacy = TTNVariant(ttn_leg_str)
                except ValueError:
                    ttn_legacy = TTNVariant.NOT_DETECTED

                # LGE
                lge_stat = LGEStatus(row.get("lge_status", "NEGATIVE"))
                lge_pat_str = row.get("lge_pattern", "NONE")
                try:
                    lge_pat = LGEPattern(lge_pat_str)
                except ValueError:
                    lge_pat = LGEPattern.NONE

                lge_ext_str = row.get("lge_extent_percent")
                lge_extent = float(lge_ext_str) if lge_ext_str and lge_ext_str.strip() else None
                lge_avail = row.get("lge_available", "True").lower() in ("true", "1")

                # Volumetrics
                rvedv = float(row["rvedv_ml"]) if row.get("rvedv_ml") else None
                rvesv = float(row["rvesv_ml"]) if row.get("rvesv_ml") else None
                rvef = float(row["rvef_percent"]) if row.get("rvef_percent") else None
                myo_mass = float(row["myocardial_mass_g"]) if row.get("myocardial_mass_g") else None

                # Alcohol history
                weekly_drinks = float(row["reported_drinks_per_week"]) if row.get("reported_drinks_per_week") else None
                time_drink = int(row["time_since_last_drink_days"]) if row.get("time_since_last_drink_days") else None
                yrs_heavy = float(row["years_heavy_drinking"]) if row.get("years_heavy_drinking") else None
                alc_avail = row.get("alcohol_history_available", "True").lower() in ("true", "1")

                # Follow up
                fu_date_str = row.get("follow_up_date")
                fu_date = date.fromisoformat(fu_date_str) if fu_date_str and fu_date_str.strip() else None
                fu_lvef = float(row["follow_up_lvef"]) if row.get("follow_up_lvef") else None
                rec_cat = RecoveryCategory(row.get("recovery_category", "PENDING_EVALUATION"))
                delta_lvef = float(row["delta_lvef"]) if row.get("delta_lvef") else (round(fu_lvef - bl_lvef, 2) if fu_lvef is not None else None)
                rec_bin = int(row["recovery_binary"]) if row.get("recovery_binary") is not None and row["recovery_binary"] != "" else (1 if delta_lvef is not None and delta_lvef >= 10.0 else 0 if delta_lvef is not None else None)

                adv_str = row.get("adverse_cardiac_events", "NONE")
                adv_events = [x.strip() for x in adv_str.split(";") if x.strip() and x.strip() != "NONE"]

                sust_val = row.get("sustained_abstinence_confirmed")
                sustained = None
                if sust_val is not None and sust_val != "":
                    sustained = sust_val.lower() in ("true", "1")

                scen_str = row.get("assumed_abstinence_scenario", "SUSTAINED_ABSTINENCE")
                try:
                    scen = AbstinenceScenario(scen_str)
                except ValueError:
                    scen = AbstinenceScenario.SUSTAINED_ABSTINENCE

                fu_obj = FollowUpOutcomes(
                    follow_up_date=fu_date,
                    follow_up_lvef=fu_lvef,
                    recovery_category=rec_cat,
                    recovery_binary=rec_bin,
                    delta_lvef=delta_lvef,
                    adverse_cardiac_events=adv_events,
                    sustained_abstinence_confirmed=sustained,
                    assumed_abstinence_scenario=scen,
                )

                mri_source = row.get("mri_source", "SYNTHETIC_SIMULATED")
                m_date_str = row.get("measurement_date")
                m_date = date.fromisoformat(m_date_str) if m_date_str and m_date_str.strip() else None

                prov_str = row.get("provenance", "SYNTHETIC_SIMULATED")
                try:
                    prov = ProvenanceType(prov_str)
                except ValueError:
                    prov = ProvenanceType.SYNTHETIC_SIMULATED

                seed_str = row.get("generation_seed")
                gen_seed = int(seed_str) if seed_str and seed_str.strip() else None

                profile = PatientProfile(
                    patient_id=p_id,
                    age=age,
                    biological_sex=sex,
                    baseline_lvef=bl_lvef,
                    edv_ml=edv,
                    esv_ml=esv,
                    peth=peth_obj,
                    ttn_variant=ttn_legacy,
                    ttn_classification=ttn_class,
                    ttn_provenance=row.get("ttn_provenance", "ClinVar / ACMG-AMP 2015 Tiered Assessment"),
                    lge_status=lge_stat,
                    lge_pattern=lge_pat,
                    lge_extent_percent=lge_extent,
                    lge_available=lge_avail,
                    rvedv_ml=rvedv,
                    rvesv_ml=rvesv,
                    rvef_percent=rvef,
                    myocardial_mass_g=myo_mass,
                    reported_drinks_per_week=weekly_drinks,
                    time_since_last_drink_days=time_drink,
                    years_heavy_drinking=yrs_heavy,
                    alcohol_history_available=alc_avail,
                    peth_available=peth_avail,
                    mri_source=mri_source,
                    measurement_date=m_date,
                    provenance=prov,
                    generation_version=row.get("generation_version", "v1.0-m3a"),
                    generation_seed=gen_seed,
                    follow_up=fu_obj,
                )
                profiles.append(profile)

        count = self.insert_patients_batch(profiles)
        logger.info(f"Imported {count} patients into SQLite database from CSV {path}")
        return count

    def get_patient(self, patient_id: str) -> Optional[PatientProfile]:
        """Fetch a single synthetic patient by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM synthetic_patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_patient(row)

    def list_patients(self, limit: Optional[int] = None) -> List[PatientProfile]:
        """List synthetic patient records ordered by patient ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = "SELECT * FROM synthetic_patients ORDER BY patient_id ASC"
            if limit:
                sql += f" LIMIT {int(limit)}"
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [self._row_to_patient(r) for r in rows]

    def count_patients(self) -> int:
        """Count total synthetic patients in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM synthetic_patients")
            return cursor.fetchone()[0]

    def filter_patients(
        self,
        biological_sex: Optional[Union[Sex, str]] = None,
        ttn_classification: Optional[Union[TTNClassification, str]] = None,
        ttn_variant: Optional[Union[TTNVariant, str]] = None,
        lge_status: Optional[Union[LGEStatus, str]] = None,
        lge_pattern: Optional[Union[LGEPattern, str]] = None,
        recovery_binary: Optional[int] = None,
        recovery_category: Optional[Union[RecoveryCategory, str]] = None,
        peth_risk_tier: Optional[Union[PEthRiskTier, str]] = None,
        min_lvef: Optional[float] = None,
        max_lvef: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[PatientProfile]:
        """Filter patients by clinical, genetic, and imaging criteria."""
        conditions: List[str] = []
        params: List[Any] = []

        if biological_sex:
            val = biological_sex.value if hasattr(biological_sex, "value") else str(biological_sex)
            conditions.append("biological_sex = ?")
            params.append(val)

        if ttn_classification:
            val = ttn_classification.value if hasattr(ttn_classification, "value") else str(ttn_classification)
            conditions.append("ttn_classification = ?")
            params.append(val)

        if ttn_variant:
            val = ttn_variant.value if hasattr(ttn_variant, "value") else str(ttn_variant)
            conditions.append("ttn_variant = ?")
            params.append(val)

        if lge_status:
            val = lge_status.value if hasattr(lge_status, "value") else str(lge_status)
            conditions.append("lge_status = ?")
            params.append(val)

        if lge_pattern:
            val = lge_pattern.value if hasattr(lge_pattern, "value") else str(lge_pattern)
            conditions.append("lge_pattern = ?")
            params.append(val)

        if recovery_binary is not None:
            conditions.append("recovery_binary = ?")
            params.append(int(recovery_binary))

        if recovery_category:
            val = recovery_category.value if hasattr(recovery_category, "value") else str(recovery_category)
            conditions.append("recovery_category = ?")
            params.append(val)

        if min_lvef is not None:
            conditions.append("baseline_lvef >= ?")
            params.append(float(min_lvef))

        if max_lvef is not None:
            conditions.append("baseline_lvef <= ?")
            params.append(float(max_lvef))

        if peth_risk_tier:
            val = peth_risk_tier.value if hasattr(peth_risk_tier, "value") else str(peth_risk_tier)
            if val == PEthRiskTier.ABSTINENT_OR_LIGHT.value:
                conditions.append("peth_value_ng_ml < 20.0")
            elif val == PEthRiskTier.SIGNIFICANT_MODERATE.value:
                conditions.append("peth_value_ng_ml >= 20.0 AND peth_value_ng_ml <= 200.0")
            elif val == PEthRiskTier.HEAVY_CHRONIC.value:
                conditions.append("peth_value_ng_ml > 200.0")

        query = "SELECT * FROM synthetic_patients"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY patient_id ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_patient(r) for r in rows]

    def get_cohort_summary(self) -> Dict[str, Any]:
        """Aggregate summary statistics for synthetic patients stored in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_count,
                    AVG(age) as mean_age,
                    AVG(baseline_lvef) as mean_baseline_lvef,
                    AVG(edv_ml) as mean_edv,
                    AVG(esv_ml) as mean_esv,
                    AVG(follow_up_lvef) as mean_follow_up_lvef,
                    AVG(delta_lvef) as mean_delta_lvef,
                    SUM(CASE WHEN recovery_binary = 1 THEN 1 ELSE 0 END) as recovery_count,
                    SUM(CASE WHEN biological_sex = 'MALE' THEN 1 ELSE 0 END) as male_count,
                    SUM(CASE WHEN biological_sex = 'FEMALE' THEN 1 ELSE 0 END) as female_count,
                    SUM(CASE WHEN lge_status = 'POSITIVE' THEN 1 ELSE 0 END) as lge_positive_count,
                    SUM(CASE WHEN ttn_classification IN ('PATHOGENIC', 'LIKELY_PATHOGENIC') THEN 1 ELSE 0 END) as ttn_pathogenic_count,
                    SUM(CASE WHEN peth_available = 1 THEN 1 ELSE 0 END) as peth_available_count
                FROM synthetic_patients
                """
            )
            row = cursor.fetchone()
            total = row["total_count"] or 0
            rec_cnt = row["recovery_count"] or 0
            rec_rate = round((rec_cnt / total) * 100.0, 2) if total > 0 else 0.0

            return {
                "total_patients": total,
                "male_count": row["male_count"] or 0,
                "female_count": row["female_count"] or 0,
                "mean_age": round(row["mean_age"], 1) if row["mean_age"] is not None else None,
                "mean_baseline_lvef": round(row["mean_baseline_lvef"], 1) if row["mean_baseline_lvef"] is not None else None,
                "mean_edv_ml": round(row["mean_edv"], 1) if row["mean_edv"] is not None else None,
                "mean_esv_ml": round(row["mean_esv"], 1) if row["mean_esv"] is not None else None,
                "mean_follow_up_lvef": round(row["mean_follow_up_lvef"], 1) if row["mean_follow_up_lvef"] is not None else None,
                "mean_delta_lvef": round(row["mean_delta_lvef"], 1) if row["mean_delta_lvef"] is not None else None,
                "recovery_count": rec_cnt,
                "recovery_rate_percent": rec_rate,
                "lge_positive_count": row["lge_positive_count"] or 0,
                "ttn_pathogenic_count": row["ttn_pathogenic_count"] or 0,
                "peth_available_count": row["peth_available_count"] or 0,
            }

    def _row_to_patient(self, row: sqlite3.Row) -> PatientProfile:
        row_keys = row.keys()

        adverse_events = []
        if "adverse_cardiac_events" in row_keys and row["adverse_cardiac_events"]:
            try:
                adverse_events = json.loads(row["adverse_cardiac_events"])
            except json.JSONDecodeError:
                adverse_events = [x.strip() for x in str(row["adverse_cardiac_events"]).split(";") if x.strip() and x.strip() != "NONE"]

        follow_up_date = None
        if "follow_up_date" in row_keys and row["follow_up_date"]:
            follow_up_date = date.fromisoformat(row["follow_up_date"])

        sustained = None
        if "sustained_abstinence_confirmed" in row_keys and row["sustained_abstinence_confirmed"] is not None:
            sustained = bool(row["sustained_abstinence_confirmed"])

        bl_lvef = row["baseline_lvef"]
        fu_lvef = row["follow_up_lvef"] if "follow_up_lvef" in row_keys else None

        delta_lvef = None
        if "delta_lvef" in row_keys and row["delta_lvef"] is not None:
            delta_lvef = float(row["delta_lvef"])
        elif fu_lvef is not None:
            delta_lvef = round(fu_lvef - bl_lvef, 2)

        recovery_binary = None
        if "recovery_binary" in row_keys and row["recovery_binary"] is not None:
            recovery_binary = int(row["recovery_binary"])
        elif delta_lvef is not None:
            recovery_binary = 1 if delta_lvef >= 10.0 else 0

        abstinence_scenario = AbstinenceScenario.SUSTAINED_ABSTINENCE
        if "assumed_abstinence_scenario" in row_keys and row["assumed_abstinence_scenario"]:
            try:
                abstinence_scenario = AbstinenceScenario(row["assumed_abstinence_scenario"])
            except ValueError:
                abstinence_scenario = AbstinenceScenario.SUSTAINED_ABSTINENCE

        follow_up = FollowUpOutcomes(
            follow_up_date=follow_up_date,
            follow_up_lvef=fu_lvef,
            recovery_category=RecoveryCategory(row["recovery_category"]),
            recovery_binary=recovery_binary,
            delta_lvef=delta_lvef,
            adverse_cardiac_events=adverse_events,
            sustained_abstinence_confirmed=sustained,
            assumed_abstinence_scenario=abstinence_scenario,
        )

        peth_val = row["peth_value_ng_ml"] if "peth_value_ng_ml" in row_keys and row["peth_value_ng_ml"] is not None else 0.0
        peth_dt_str = row["peth_measurement_date"] if "peth_measurement_date" in row_keys and row["peth_measurement_date"] else None
        peth_dt = date.fromisoformat(peth_dt_str) if peth_dt_str else date(2025, 1, 15)

        peth = PEthMeasurement(
            value_ng_ml=peth_val,
            measurement_date=peth_dt,
        )

        # TTN
        ttn_legacy = TTNVariant(row["ttn_variant"])
        ttn_class = None
        if "ttn_classification" in row_keys and row["ttn_classification"]:
            try:
                ttn_class = TTNClassification(row["ttn_classification"])
            except ValueError:
                ttn_class = map_ttn_variant_to_classification(ttn_legacy)
        else:
            ttn_class = map_ttn_variant_to_classification(ttn_legacy)

        # LGE
        lge_stat = LGEStatus(row["lge_status"])
        lge_pat = LGEPattern.NONE
        if "lge_pattern" in row_keys and row["lge_pattern"]:
            try:
                lge_pat = LGEPattern(row["lge_pattern"])
            except ValueError:
                lge_pat = LGEPattern.MIDWALL if lge_stat == LGEStatus.POSITIVE else LGEPattern.NONE
        else:
            lge_pat = LGEPattern.MIDWALL if lge_stat == LGEStatus.POSITIVE else LGEPattern.NONE

        var_prov = {}
        if "variable_provenance" in row_keys and row["variable_provenance"]:
            try:
                var_prov = json.loads(row["variable_provenance"])
            except json.JSONDecodeError:
                var_prov = {}

        meas_date = None
        if "measurement_date" in row_keys and row["measurement_date"]:
            meas_date = date.fromisoformat(row["measurement_date"])

        prov_val = ProvenanceType.SYNTHETIC_SIMULATED
        if "provenance" in row_keys and row["provenance"]:
            try:
                prov_val = ProvenanceType(row["provenance"])
            except ValueError:
                prov_val = ProvenanceType.SYNTHETIC_SIMULATED

        return PatientProfile(
            patient_id=row["patient_id"],
            age=row["age"],
            biological_sex=Sex(row["biological_sex"]),
            baseline_lvef=row["baseline_lvef"],
            edv_ml=row["edv_ml"],
            esv_ml=row["esv_ml"],
            peth=peth,
            ttn_variant=ttn_legacy,
            ttn_classification=ttn_class,
            ttn_provenance=row["ttn_provenance"] if "ttn_provenance" in row_keys and row["ttn_provenance"] else "ClinVar / ACMG-AMP 2015 Tiered Assessment",
            lge_status=lge_stat,
            lge_pattern=lge_pat,
            lge_extent_percent=float(row["lge_extent_percent"]) if "lge_extent_percent" in row_keys and row["lge_extent_percent"] is not None else None,
            lge_available=bool(row["lge_available"]) if "lge_available" in row_keys and row["lge_available"] is not None else True,
            rvedv_ml=float(row["rvedv_ml"]) if "rvedv_ml" in row_keys and row["rvedv_ml"] is not None else None,
            rvesv_ml=float(row["rvesv_ml"]) if "rvesv_ml" in row_keys and row["rvesv_ml"] is not None else None,
            rvef_percent=float(row["rvef_percent"]) if "rvef_percent" in row_keys and row["rvef_percent"] is not None else None,
            myocardial_mass_g=float(row["myocardial_mass_g"]) if "myocardial_mass_g" in row_keys and row["myocardial_mass_g"] is not None else None,
            reported_drinks_per_week=float(row["reported_drinks_per_week"]) if "reported_drinks_per_week" in row_keys and row["reported_drinks_per_week"] is not None else None,
            time_since_last_drink_days=int(row["time_since_last_drink_days"]) if "time_since_last_drink_days" in row_keys and row["time_since_last_drink_days"] is not None else None,
            years_heavy_drinking=float(row["years_heavy_drinking"]) if "years_heavy_drinking" in row_keys and row["years_heavy_drinking"] is not None else None,
            alcohol_history_available=bool(row["alcohol_history_available"]) if "alcohol_history_available" in row_keys and row["alcohol_history_available"] is not None else True,
            peth_available=bool(row["peth_available"]) if "peth_available" in row_keys and row["peth_available"] is not None else True,
            mri_source=row["mri_source"] if "mri_source" in row_keys and row["mri_source"] else "SYNTHETIC_SIMULATED",
            measurement_date=meas_date,
            provenance=prov_val,
            generation_version=row["generation_version"] if "generation_version" in row_keys and row["generation_version"] else "v1.0-m3a",
            generation_seed=row["generation_seed"] if "generation_seed" in row_keys and row["generation_seed"] is not None else None,
            variable_provenance=var_prov,
            follow_up=follow_up,
        )


def seed_reference_synthetic_cohort(db: SyntheticEHRDatabase) -> None:
    """Seed a small initial set of illustrative synthetic cases if database is empty."""
    if db.count_patients() > 0:
        return

    sample_patients = [
        PatientProfile(
            patient_id="CRIP-SYN-001",
            age=48,
            biological_sex=Sex.MALE,
            baseline_lvef=28.0,
            edv_ml=250.0,
            esv_ml=180.0,
            peth=PEthMeasurement(value_ng_ml=412.0, measurement_date=date(2025, 1, 15)),
            ttn_variant=TTNVariant.NOT_DETECTED,
            ttn_classification=TTNClassification.NO_PATHOGENIC_OR_LIKELY_PATHOGENIC,
            lge_status=LGEStatus.NEGATIVE,
            lge_pattern=LGEPattern.NONE,
            follow_up=FollowUpOutcomes(
                follow_up_date=date(2025, 7, 15),
                follow_up_lvef=52.0,
                delta_lvef=24.0,
                recovery_binary=1,
                recovery_category=RecoveryCategory.COMPLETE_RECOVERY,
                adverse_cardiac_events=[],
                sustained_abstinence_confirmed=True,
                assumed_abstinence_scenario=AbstinenceScenario.SUSTAINED_ABSTINENCE,
            ),
        ),
        PatientProfile(
            patient_id="CRIP-SYN-002",
            age=54,
            biological_sex=Sex.MALE,
            baseline_lvef=32.0,
            edv_ml=220.0,
            esv_ml=150.0,
            peth=PEthMeasurement(value_ng_ml=280.0, measurement_date=date(2025, 2, 1)),
            ttn_variant=TTNVariant.PATHOGENIC_TRUNCATING,
            ttn_classification=TTNClassification.PATHOGENIC,
            lge_status=LGEStatus.MIDWALL,
            lge_pattern=LGEPattern.MIDWALL,
            lge_extent_percent=8.5,
            follow_up=FollowUpOutcomes(
                follow_up_date=date(2025, 8, 1),
                follow_up_lvef=36.0,
                delta_lvef=4.0,
                recovery_binary=0,
                recovery_category=RecoveryCategory.STABLE_NON_RECOVERY,
                adverse_cardiac_events=["Heart failure exacerbation"],
                sustained_abstinence_confirmed=False,
                assumed_abstinence_scenario=AbstinenceScenario.ONGOING_HEAVY_CONSUMPTION,
            ),
        ),
        PatientProfile(
            patient_id="CRIP-SYN-003",
            age=42,
            biological_sex=Sex.FEMALE,
            baseline_lvef=35.0,
            edv_ml=200.0,
            esv_ml=130.0,
            peth=PEthMeasurement(value_ng_ml=145.0, measurement_date=date(2025, 2, 20)),
            ttn_variant=TTNVariant.VARIANT_UNCERTAIN_SIGNIFICANCE,
            ttn_classification=TTNClassification.VARIANT_OF_UNCERTAIN_SIGNIFICANCE,
            lge_status=LGEStatus.NEGATIVE,
            lge_pattern=LGEPattern.NONE,
            follow_up=FollowUpOutcomes(
                follow_up_date=date(2025, 8, 20),
                follow_up_lvef=47.0,
                delta_lvef=12.0,
                recovery_binary=1,
                recovery_category=RecoveryCategory.PARTIAL_RECOVERY,
                adverse_cardiac_events=[],
                sustained_abstinence_confirmed=True,
                assumed_abstinence_scenario=AbstinenceScenario.SUSTAINED_ABSTINENCE,
            ),
        ),
    ]

    for p in sample_patients:
        db.insert_patient(p)
