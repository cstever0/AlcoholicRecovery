"""Local SQLite synthetic EHR database manager for CRIP.

DISCLAIMER:
Provides local simulated EHR storage for synthetic patient records only.
No live EHR integrations, cloud connections, or real patient data are permitted.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator, List, Optional

from crip.synthetic.schemas import (
    FollowUpOutcomes,
    LGEStatus,
    PatientProfile,
    PEthMeasurement,
    RecoveryCategory,
    Sex,
    TTNVariant,
)


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
        """Initialize database tables for synthetic patients."""
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
                    peth_value_ng_ml REAL NOT NULL,
                    peth_measurement_date TEXT NOT NULL,
                    ttn_variant TEXT NOT NULL,
                    lge_status TEXT NOT NULL,
                    follow_up_date TEXT,
                    follow_up_lvef REAL,
                    recovery_category TEXT NOT NULL,
                    adverse_cardiac_events TEXT,
                    sustained_abstinence_confirmed INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def insert_patient(self, patient: PatientProfile) -> None:
        """Insert or replace a synthetic patient record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO synthetic_patients (
                    patient_id, age, biological_sex, baseline_lvef, edv_ml, esv_ml,
                    peth_value_ng_ml, peth_measurement_date, ttn_variant, lge_status,
                    follow_up_date, follow_up_lvef, recovery_category,
                    adverse_cardiac_events, sustained_abstinence_confirmed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient.patient_id,
                    patient.age,
                    patient.biological_sex.value,
                    patient.baseline_lvef,
                    patient.edv_ml,
                    patient.esv_ml,
                    patient.peth.value_ng_ml,
                    patient.peth.measurement_date.isoformat(),
                    patient.ttn_variant.value,
                    patient.lge_status.value,
                    patient.follow_up.follow_up_date.isoformat() if patient.follow_up.follow_up_date else None,
                    patient.follow_up.follow_up_lvef,
                    patient.follow_up.recovery_category.value,
                    json.dumps(patient.follow_up.adverse_cardiac_events),
                    1 if patient.follow_up.sustained_abstinence_confirmed else (0 if patient.follow_up.sustained_abstinence_confirmed is False else None),
                ),
            )
            conn.commit()

    def get_patient(self, patient_id: str) -> Optional[PatientProfile]:
        """Fetch a single synthetic patient by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM synthetic_patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_patient(row)

    def list_patients(self) -> List[PatientProfile]:
        """List all synthetic patient records ordered by patient ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM synthetic_patients ORDER BY patient_id ASC")
            rows = cursor.fetchall()
            return [self._row_to_patient(r) for r in rows]

    def count_patients(self) -> int:
        """Count total synthetic patients in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM synthetic_patients")
            return cursor.fetchone()[0]

    def _row_to_patient(self, row: sqlite3.Row) -> PatientProfile:
        adverse_events = []
        if row["adverse_cardiac_events"]:
            try:
                adverse_events = json.loads(row["adverse_cardiac_events"])
            except json.JSONDecodeError:
                adverse_events = []

        follow_up_date = None
        if row["follow_up_date"]:
            follow_up_date = date.fromisoformat(row["follow_up_date"])

        sustained = None
        if row["sustained_abstinence_confirmed"] is not None:
            sustained = bool(row["sustained_abstinence_confirmed"])

        follow_up = FollowUpOutcomes(
            follow_up_date=follow_up_date,
            follow_up_lvef=row["follow_up_lvef"],
            recovery_category=RecoveryCategory(row["recovery_category"]),
            adverse_cardiac_events=adverse_events,
            sustained_abstinence_confirmed=sustained,
        )

        peth = PEthMeasurement(
            value_ng_ml=row["peth_value_ng_ml"],
            measurement_date=date.fromisoformat(row["peth_measurement_date"]),
        )

        return PatientProfile(
            patient_id=row["patient_id"],
            age=row["age"],
            biological_sex=Sex(row["biological_sex"]),
            baseline_lvef=row["baseline_lvef"],
            edv_ml=row["edv_ml"],
            esv_ml=row["esv_ml"],
            peth=peth,
            ttn_variant=TTNVariant(row["ttn_variant"]),
            lge_status=LGEStatus(row["lge_status"]),
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
            lge_status=LGEStatus.NEGATIVE,
            follow_up=FollowUpOutcomes(
                follow_up_date=date(2025, 7, 15),
                follow_up_lvef=52.0,
                recovery_category=RecoveryCategory.COMPLETE_RECOVERY,
                adverse_cardiac_events=[],
                sustained_abstinence_confirmed=True,
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
            lge_status=LGEStatus.MIDWALL,
            follow_up=FollowUpOutcomes(
                follow_up_date=date(2025, 8, 1),
                follow_up_lvef=36.0,
                recovery_category=RecoveryCategory.STABLE_NON_RECOVERY,
                adverse_cardiac_events=["Heart failure exacerbation"],
                sustained_abstinence_confirmed=False,
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
            lge_status=LGEStatus.NEGATIVE,
            follow_up=FollowUpOutcomes(
                follow_up_date=date(2025, 8, 20),
                follow_up_lvef=47.0,
                recovery_category=RecoveryCategory.PARTIAL_RECOVERY,
                adverse_cardiac_events=[],
                sustained_abstinence_confirmed=True,
            ),
        ),
    ]

    for p in sample_patients:
        db.insert_patient(p)
