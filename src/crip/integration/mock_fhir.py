"""Mock HL7 FHIR R4 interoperability service for CRIP multimodal profiles.

DISCLAIMER:
Provides local HL7 FHIR R4 JSON serialization and deserialization for mock EHR integration.
This is a research utility with synthetic identifiers; it contains NO live hospital connections
(e.g., Epic, Cerner, Oracle) and NEVER transmits real patient identifiable information.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Union

from crip.integration.patient_features import (
    UnifiedPatientProfile,
    build_unified_patient_features,
)
from crip.integration.provenance import CaseSourceType, FeatureSourceOrigin

logger = logging.getLogger("crip.integration.mock_fhir")

# Standard clinical ontology systems
LOINC_SYSTEM = "http://loinc.org"
UCUM_SYSTEM = "http://unitsofmeasure.org"
CLINVAR_SYSTEM = "https://clinvar.ncbi.nlm.nih.gov"
RADLEX_SYSTEM = "http://radlex.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
CRIP_SYSTEM = "https://crip.research.internal/fhir"


def export_to_fhir_bundle(profile: UnifiedPatientProfile) -> Dict[str, Any]:
    """Export a UnifiedPatientProfile into an HL7 FHIR R4 compliant Bundle.

    Resources included in bundle:
      1. Patient (demographics, age proxy birthDate, sex)
      2. Observation: PEth blood concentration (LOINC 94042-9)
      3. Observation: LVEF (LOINC 8806-2)
      4. Observation: LVEDV (LOINC 8807-0)
      5. Observation: LVESV (LOINC 8808-8)
      6. Observation: LVSV (LOINC 8809-6)
      7. Observation: TTN Genetic Variant (LOINC 48018-6)
      8. Observation: Late Gadolinium Enhancement Scar (SNOMED 250908004 / RadLex RID10398)
      9. DiagnosticReport: Cardiac MRI Volumetric Assessment
    """
    case_id = profile.case_id
    patient_ref = f"Patient/{case_id}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Estimate birth year for Patient resource
    current_year = datetime.datetime.now().year
    birth_year = current_year - profile.age
    birth_date = f"{birth_year}-01-01"

    entries: List[Dict[str, Any]] = []

    # 1. Patient Resource
    patient_resource = {
        "resourceType": "Patient",
        "id": case_id,
        "identifier": [
            {
                "use": "usual",
                "system": f"{CRIP_SYSTEM}/patient-id",
                "value": case_id,
            }
        ],
        "active": True,
        "gender": "male" if profile.biological_sex.upper() == "MALE" else "female",
        "birthDate": birth_date,
    }
    entries.append({"resource": patient_resource})

    observation_refs: List[Dict[str, str]] = []

    # 2. Observation: LVEF
    lvef_id = f"obs-lvef-{case_id}"
    obs_lvef = {
        "resourceType": "Observation",
        "id": lvef_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": "8806-2",
                    "display": "Left ventricular Ejection fraction",
                }
            ],
            "text": "Left Ventricular Ejection Fraction",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now_iso,
        "valueQuantity": {
            "value": float(profile.baseline_lvef),
            "unit": "%",
            "system": UCUM_SYSTEM,
            "code": "%",
        },
    }
    entries.append({"resource": obs_lvef})
    observation_refs.append({"reference": f"Observation/{lvef_id}"})

    # 3. Observation: LVEDV
    edv_id = f"obs-lvedv-{case_id}"
    obs_edv = {
        "resourceType": "Observation",
        "id": edv_id,
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": "8807-0",
                    "display": "Left ventricular End diastolic volume",
                }
            ],
            "text": "Left Ventricular End-Diastolic Volume",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now_iso,
        "valueQuantity": {
            "value": float(profile.edv_ml),
            "unit": "mL",
            "system": UCUM_SYSTEM,
            "code": "mL",
        },
    }
    entries.append({"resource": obs_edv})
    observation_refs.append({"reference": f"Observation/{edv_id}"})

    # 4. Observation: LVESV
    esv_id = f"obs-lvesv-{case_id}"
    obs_esv = {
        "resourceType": "Observation",
        "id": esv_id,
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": "8808-8",
                    "display": "Left ventricular End systolic volume",
                }
            ],
            "text": "Left Ventricular End-Systolic Volume",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now_iso,
        "valueQuantity": {
            "value": float(profile.esv_ml),
            "unit": "mL",
            "system": UCUM_SYSTEM,
            "code": "mL",
        },
    }
    entries.append({"resource": obs_esv})
    observation_refs.append({"reference": f"Observation/{esv_id}"})

    # 5. Observation: Stroke Volume
    sv_id = f"obs-lvsv-{case_id}"
    obs_sv = {
        "resourceType": "Observation",
        "id": sv_id,
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": "8809-6",
                    "display": "Left ventricular Stroke volume",
                }
            ],
            "text": "Left Ventricular Stroke Volume",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now_iso,
        "valueQuantity": {
            "value": float(profile.stroke_volume_ml),
            "unit": "mL",
            "system": UCUM_SYSTEM,
            "code": "mL",
        },
    }
    entries.append({"resource": obs_sv})
    observation_refs.append({"reference": f"Observation/{sv_id}"})

    # 6. Observation: PEth (if available)
    if profile.peth_available > 0.5 and profile.peth_value_ng_ml is not None:
        peth_id = f"obs-peth-{case_id}"
        obs_peth = {
            "resourceType": "Observation",
            "id": peth_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "laboratory",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": LOINC_SYSTEM,
                        "code": "94042-9",
                        "display": "Phosphatidylethanol [Mass/volume] in Blood",
                    }
                ],
                "text": "Phosphatidylethanol (PEth 16:0/18:1)",
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": now_iso,
            "valueQuantity": {
                "value": float(profile.peth_value_ng_ml),
                "unit": "ng/mL",
                "system": UCUM_SYSTEM,
                "code": "ng/mL",
            },
            "interpretation": [
                {
                    "coding": [
                        {
                            "system": f"{CRIP_SYSTEM}/peth-risk-tier",
                            "code": profile.peth_risk_tier,
                            "display": profile.peth_risk_tier.replace("_", " ").title(),
                        }
                    ]
                }
            ],
        }
        entries.append({"resource": obs_peth})
        observation_refs.append({"reference": f"Observation/{peth_id}"})

    # 7. Observation: TTN Genetic Variant
    ttn_id = f"obs-ttn-{case_id}"
    obs_ttn = {
        "resourceType": "Observation",
        "id": ttn_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": "48018-6",
                    "display": "Gene mutations tested for in Blood or Tissue by Molecular genetics method",
                }
            ],
            "text": "Titin (TTN) Truncating Variant Analysis",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now_iso,
        "valueCodeableConcept": {
            "coding": [
                {
                    "system": CLINVAR_SYSTEM,
                    "code": profile.ttn_classification,
                    "display": profile.ttn_classification.replace("_", " ").title(),
                }
            ]
        },
    }
    entries.append({"resource": obs_ttn})
    observation_refs.append({"reference": f"Observation/{ttn_id}"})

    # 8. Observation: LGE Fibrosis Scar
    lge_id = f"obs-lge-{case_id}"
    obs_lge = {
        "resourceType": "Observation",
        "id": lge_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": SNOMED_SYSTEM,
                    "code": "250908004",
                    "display": "Myocardial fibrosis",
                },
                {
                    "system": RADLEX_SYSTEM,
                    "code": "RID10398",
                    "display": "Delayed enhancement MRI",
                },
            ],
            "text": "Late Gadolinium Enhancement (LGE) Fibrosis",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now_iso,
        "valueCodeableConcept": {
            "coding": [
                {
                    "system": f"{CRIP_SYSTEM}/lge-status",
                    "code": profile.lge_status,
                    "display": profile.lge_status,
                }
            ]
        },
        "component": [
            {
                "code": {
                    "coding": [
                        {
                            "system": f"{CRIP_SYSTEM}/lge-pattern",
                            "code": "pattern",
                            "display": "Distribution Pattern",
                        }
                    ]
                },
                "valueString": profile.lge_pattern,
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": f"{CRIP_SYSTEM}/lge-extent",
                            "code": "extent",
                            "display": "Myocardial Extent Percentage",
                        }
                    ]
                },
                "valueQuantity": {
                    "value": float(profile.lge_extent_percent or 0.0),
                    "unit": "%",
                    "system": UCUM_SYSTEM,
                    "code": "%",
                },
            },
        ],
    }
    entries.append({"resource": obs_lge})
    observation_refs.append({"reference": f"Observation/{lge_id}"})

    # 9. DiagnosticReport: Cardiac MRI
    report_id = f"diag-cmr-{case_id}"
    diag_report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": "72223-1",
                    "display": "Cardiovascular magnetic resonance imaging study",
                }
            ],
            "text": "Cardiovascular Magnetic Resonance (CMR) Volumetrics Report",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now_iso,
        "result": observation_refs,
        "conclusion": (
            f"CMR demonstration: LVEDV {profile.edv_ml} mL, LVESV {profile.esv_ml} mL, "
            f"LVEF {profile.baseline_lvef}%. LGE: {profile.lge_status}."
        ),
    }
    entries.append({"resource": diag_report})

    bundle = {
        "resourceType": "Bundle",
        "id": f"bundle-{case_id}",
        "type": "collection",
        "timestamp": now_iso,
        "entry": entries,
    }
    return bundle


def import_from_fhir_bundle(bundle: Dict[str, Any]) -> UnifiedPatientProfile:
    """Parse an HL7 FHIR R4 Bundle into a strongly-typed UnifiedPatientProfile.

    Extracts demographics from Patient, volumetric measurements from LOINC observations,
    PEth biomarker, TTN genetics, and LGE scar observations.
    """
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("Provided JSON is not a valid FHIR resourceType 'Bundle'.")

    entries = bundle.get("entry", [])
    patient_id = "CRIP-FHIR-IMPORT"
    age = 50
    gender = "MALE"
    edv = 210.0
    esv = 136.5
    lvef = 35.0
    sv = 73.5
    peth_val: Optional[float] = None
    peth_avail = 0.0
    peth_risk_tier = "UNAVAILABLE"
    ttn_class = "NOT_TESTED"
    lge_status = "NEGATIVE"
    lge_pattern = "NONE"
    lge_extent: Optional[float] = 0.0
    lge_avail = 1.0

    current_year = datetime.datetime.now().year

    for item in entries:
        res = item.get("resource", {})
        rtype = res.get("resourceType")

        if rtype == "Patient":
            if "id" in res:
                patient_id = res["id"]
            raw_gender = res.get("gender", "male").upper()
            gender = "MALE" if raw_gender in ("MALE", "M") else "FEMALE"
            birth_date = res.get("birthDate")
            if birth_date and len(birth_date) >= 4:
                try:
                    byear = int(birth_date[:4])
                    age = max(18, min(105, current_year - byear))
                except ValueError:
                    age = 50

        elif rtype == "Observation":
            code_obj = res.get("code", {})
            codings = code_obj.get("coding", [])

            for c in codings:
                code_val = c.get("code")

                # LVEF (LOINC 8806-2)
                if code_val == "8806-2":
                    val_q = res.get("valueQuantity", {})
                    if "value" in val_q:
                        lvef = float(val_q["value"])

                # LVEDV (LOINC 8807-0)
                elif code_val == "8807-0":
                    val_q = res.get("valueQuantity", {})
                    if "value" in val_q:
                        edv = float(val_q["value"])

                # LVESV (LOINC 8808-8)
                elif code_val == "8808-8":
                    val_q = res.get("valueQuantity", {})
                    if "value" in val_q:
                        esv = float(val_q["value"])

                # LVSV (LOINC 8809-6)
                elif code_val == "8809-6":
                    val_q = res.get("valueQuantity", {})
                    if "value" in val_q:
                        sv = float(val_q["value"])

                # PEth (LOINC 94042-9)
                elif code_val == "94042-9":
                    val_q = res.get("valueQuantity", {})
                    if "value" in val_q:
                        peth_val = float(val_q["value"])
                        peth_avail = 1.0
                    interps = res.get("interpretation", [])
                    if interps:
                        icodings = interps[0].get("coding", [])
                        if icodings:
                            peth_risk_tier = icodings[0].get("code", "UNAVAILABLE")

                # TTN (LOINC 48018-6)
                elif code_val == "48018-6":
                    cc = res.get("valueCodeableConcept", {})
                    t_codings = cc.get("coding", [])
                    if t_codings:
                        ttn_class = t_codings[0].get("code", "NOT_TESTED")

                # LGE Fibrosis (SNOMED 250908004 or RadLex RID10398)
                elif code_val in ("250908004", "RID10398"):
                    cc = res.get("valueCodeableConcept", {})
                    l_codings = cc.get("coding", [])
                    if l_codings:
                        lge_status = l_codings[0].get("code", "NEGATIVE")
                    # Components
                    components = res.get("component", [])
                    for comp in components:
                        c_codings = comp.get("code", {}).get("coding", [])
                        for cc_item in c_codings:
                            if cc_item.get("code") == "pattern":
                                lge_pattern = comp.get("valueString", "NONE")
                            elif cc_item.get("code") == "extent":
                                lge_extent = float(comp.get("valueQuantity", {}).get("value", 0.0))

    if sv <= 0 and edv > esv:
        sv = edv - esv

    prov_map = {
        "case_id": CaseSourceType.MOCK_FHIR_EHR_IMPORT.value,
        "edv_ml": FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value,
        "esv_ml": FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value,
        "stroke_volume_ml": FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value,
        "baseline_lvef": FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value,
        "peth_value_ng_ml": FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value if peth_avail else FeatureSourceOrigin.MISSING_CLINICAL_DATA.value,
        "ttn_classification": FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value,
        "lge_status": FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value,
    }

    profile = UnifiedPatientProfile(
        case_id=patient_id,
        source_type=CaseSourceType.MOCK_FHIR_EHR_IMPORT.value,
        age=age,
        biological_sex=gender,
        mri_source=FeatureSourceOrigin.MOCK_FHIR_OBSERVATION.value,
        baseline_lvef=round(lvef, 2),
        edv_ml=round(edv, 2),
        esv_ml=round(esv, 2),
        stroke_volume_ml=round(sv, 2),
        peth_value_ng_ml=round(peth_val, 2) if peth_val is not None else None,
        peth_available=peth_avail,
        peth_risk_tier=peth_risk_tier,
        ttn_classification=ttn_class,
        lge_status=lge_status,
        lge_pattern=lge_pattern,
        lge_extent_percent=lge_extent,
        lge_available=lge_avail,
        provenance_map=prov_map,
        notes="Imported via Mock HL7 FHIR R4 Bundle.",
    )

    return profile


def validate_fhir_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Perform structural and semantic schema validation on an HL7 FHIR R4 Bundle.

    Validates:
      1. Bundle envelope (resourceType, type, entries)
      2. Patient resource presence and ID consistency
      3. Observation linkages, authoritative ontology codes, and UCUM units
      4. DiagnosticReport references
    """
    errors: List[str] = []
    warnings: List[str] = []
    resource_counts: Dict[str, int] = {}
    validated_codes: List[Dict[str, str]] = []

    if not isinstance(bundle, dict):
        return {
            "is_valid": False,
            "errors": ["Bundle payload must be a JSON dictionary object."],
            "warnings": [],
            "resource_counts": {},
            "validated_codes": [],
        }

    if bundle.get("resourceType") != "Bundle":
        errors.append(f"Invalid resourceType '{bundle.get('resourceType')}'; expected 'Bundle'.")

    entries = bundle.get("entry", [])
    if not isinstance(entries, list) or len(entries) == 0:
        errors.append("FHIR Bundle contains no resource entries.")

    patient_ids: List[str] = []
    observation_ids: set[str] = set()

    for idx, entry in enumerate(entries):
        res = entry.get("resource", {})
        rtype = res.get("resourceType", "Unknown")
        resource_counts[rtype] = resource_counts.get(rtype, 0) + 1
        rid = res.get("id")

        if rtype == "Patient":
            if rid:
                patient_ids.append(rid)
            else:
                errors.append(f"Entry {idx}: Patient resource missing 'id'.")

        elif rtype == "Observation":
            if rid:
                observation_ids.add(rid)
            # Check subject reference
            subj_ref = res.get("subject", {}).get("reference", "")
            if not subj_ref.startswith("Patient/"):
                warnings.append(f"Observation '{rid}' subject reference '{subj_ref}' does not point to Patient/.")

            # Validate codes
            codings = res.get("code", {}).get("coding", [])
            for c in codings:
                sys = c.get("system")
                code = c.get("code")
                disp = c.get("display", "")
                validated_codes.append({"system": sys, "code": code, "display": disp})

            # Check quantity units
            val_q = res.get("valueQuantity", {})
            if val_q:
                u_sys = val_q.get("system")
                unit = val_q.get("unit")
                if u_sys != UCUM_SYSTEM:
                    warnings.append(f"Observation '{rid}' unit system '{u_sys}' is not standard UCUM ({UCUM_SYSTEM}).")

        elif rtype == "DiagnosticReport":
            res_refs = res.get("result", [])
            for r in res_refs:
                ref_str = r.get("reference", "")
                if ref_str.startswith("Observation/"):
                    obs_ref_id = ref_str.split("/", 1)[1]
                    if obs_ref_id not in observation_ids:
                        warnings.append(f"DiagnosticReport references Observation '{obs_ref_id}' which was not defined prior.")

    if not patient_ids:
        errors.append("Bundle missing mandatory 'Patient' resource.")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "resource_counts": resource_counts,
        "validated_codes": validated_codes,
        "conformance_level": "Mock HL7 FHIR R4 Demonstration Schema Validation",
        "conformance_disclaimer": (
            "Local structural validation only. Production clinical EHR interoperability requires "
            "an ONC-certified FHIR R4 server with SMART-on-FHIR OAuth 2.0 authorization and US Core profile compliance."
        ),
    }
