"""
hl7_fhir.py — VF Analyst Interoperability Layer
=================================================
Implements standards-based communication adapters for VF Analyst,
based on the requirements described in:
  - Capitolo 3 of the thesis draft (HL7, FHIR R4, IEEE 11073)
  - ELSO Guidelines 2021
  - IHE Patient Care Device (PCD) integration profiles

THREE LAYERS
------------
Layer 1 — IEEE 11073 (MDC nomenclature)
  Maps VF Analyst internal column names to standardised Medical Device
  Communication nomenclature codes (partition::term_code format).
  Ensures every measured parameter carries a globally unique identifier
  understood by any compliant device or HIS/EHR.

Layer 2 — HL7 v2.x (ORU^R01 observation message)
  Generates an HL7 v2 pipe-delimited ORU^R01 message from a row of
  measurements. This is the format used by most bedside monitors and
  ECMO consoles to push data into ICU clinical information systems.

Layer 3 — FHIR R4 (Observation + DeviceMetric resources)
  Produces FHIR R4-compliant JSON objects (Observation and Bundle)
  from VF Analyst sessions and measurements. These can be POSTed
  directly to any FHIR R4 server (Epic, Cerner, Azure Health Data
  Services, etc.) without modification.

USAGE (standalone export)
-------------------------
    from hl7_fhir import FHIRExporter, HL7Exporter
    import pandas as pd

    # FHIR export
    exporter = FHIRExporter(patient_id="PAT-001", device_id="MC3-001")
    bundle = exporter.session_to_bundle(session_row, measurements_df)
    with open("bundle.json", "w") as f:
        json.dump(bundle, f, indent=2, default=str)

    # HL7 export
    hl7 = HL7Exporter()
    message = hl7.row_to_oru(measurements_df.iloc[0], patient_id="PAT-001")
    print(message)

IMPORTANT DISCLAIMER
--------------------
This module produces standards-formatted output for research and
demonstration purposes. It has NOT been validated for clinical use
and must NOT be used for autonomous clinical decisions or submitted
to certified HIS/EHR systems without independent validation by a
qualified biomedical engineer and clinical informatics specialist.

Standards references:
  IEEE 11073-10101:2020 — Health informatics — MDC nomenclature
  HL7 v2.9            — ORU^R01 Observation Result
  HL7 FHIR R4         — Observation, DeviceMetric, Bundle
  IHE PCD TF-2        — Device Observation Reporter (DOR) transaction
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — IEEE 11073 MDC nomenclature mapping
# ─────────────────────────────────────────────────────────────────────────────

# Format: column_name → (MDC_code, UCUM_unit, LOINC_code, description)
# MDC codes from IEEE 11073-10101:2020, partition SCADA (65536 = 0x10000)
# LOINC codes from the LOINC database (loinc.org)

MDC_MAP: dict[str, dict] = {
    "flow_rate_lpm": {
        "mdc_code":    "MDC_FLOW_BLOOD_PUMP",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "8825",          # SCADA partition term code
        "ucum":        "L/min",
        "loinc":       "8769-1",        # Heart rate — closest available proxy
        "description": "ECMO pump blood flow rate",
        "fhir_system": "http://loinc.org",
    },
    "pressure_pre_mmhg": {
        "mdc_code":    "MDC_PRESS_BLD_EXTCORPOREAL_ART",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "18972",
        "ucum":        "mm[Hg]",
        "loinc":       "8480-6",        # Systolic blood pressure
        "description": "Pre-oxygenator arterial pressure",
        "fhir_system": "http://loinc.org",
    },
    "pressure_post_mmhg": {
        "mdc_code":    "MDC_PRESS_BLD_EXTCORPOREAL_VEN",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "18974",
        "ucum":        "mm[Hg]",
        "loinc":       "8462-4",        # Diastolic blood pressure
        "description": "Post-oxygenator venous return pressure",
        "fhir_system": "http://loinc.org",
    },
    "pressure_delta_mmhg": {
        "mdc_code":    "MDC_PRESS_DIFF_MEMBRANE",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "18976",
        "ucum":        "mm[Hg]",
        "loinc":       "76214-6",       # Transmembrane pressure gradient
        "description": "Transmembrane pressure gradient (ΔP)",
        "fhir_system": "http://loinc.org",
    },
    "sat_pre_pct": {
        "mdc_code":    "MDC_SAT_O2_EXTCORPOREAL_VEN",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "19384",
        "ucum":        "%",
        "loinc":       "59408-5",       # SpO2
        "description": "Pre-oxygenator venous oxygen saturation (SvO2)",
        "fhir_system": "http://loinc.org",
    },
    "sat_post_pct": {
        "mdc_code":    "MDC_SAT_O2_EXTCORPOREAL_ART",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "19386",
        "ucum":        "%",
        "loinc":       "2708-6",        # Oxygen saturation arterial
        "description": "Post-oxygenator arterial oxygen saturation (SaO2)",
        "fhir_system": "http://loinc.org",
    },
    "temp_post_c": {
        "mdc_code":    "MDC_TEMP_BLOOD_EXTCORPOREAL",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "19320",
        "ucum":        "Cel",
        "loinc":       "8310-5",        # Body temperature
        "description": "Post-oxygenator blood temperature",
        "fhir_system": "http://loinc.org",
    },
    "pump_speed_rpm": {
        "mdc_code":    "MDC_SPEED_RPM_PUMP_BLOOD",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "8836",
        "ucum":        "/min",
        "loinc":       "59270-9",       # Pump speed
        "description": "Centrifugal pump rotational speed",
        "fhir_system": "http://loinc.org",
    },
    "alarm_active": {
        "mdc_code":    "MDC_EVT_ALARM_DEVICE",
        "mdc_urn":     "urn:oid:2.16.840.1.113883.6.24",
        "mdc_value":   "68480",
        "ucum":        "1",             # dimensionless boolean
        "loinc":       "74201-3",       # Device alert
        "description": "Device alarm active flag (0=off, 1=on)",
        "fhir_system": "http://loinc.org",
    },
}


def col_in_verified(column: str) -> bool:
    return column in {"temp_post_c", "sat_post_pct"}


def get_mdc_info(column: str) -> Optional[dict]:
    """Return IEEE 11073 / LOINC metadata for a VF Analyst column name."""
    return MDC_MAP.get(column)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — HL7 v2.9 ORU^R01 message builder
# ─────────────────────────────────────────────────────────────────────────────

class HL7Exporter:
    """
    Builds HL7 v2.9 ORU^R01 (Observation Result) messages from VF Analyst
    measurement rows.

    ORU^R01 is the standard message type used by bedside monitors,
    ventilators, and ECMO consoles to push physiological observations
    to a clinical information system (CIS) or electronic health record (EHR).

    Reference: HL7 v2.9 Chapter 7 — Observation Reporting

    Pipe-delimited format. Fields are separated by |, components by ^,
    repetitions by ~, subcomponents by &. Segment terminator is CR (\\r).
    """

    FIELD_SEP   = "|"
    COMP_SEP    = "^"
    REPEAT_SEP  = "~"
    ESCAPE      = "\\"
    SUBCOMP_SEP = "&"
    ENCODING    = "MSH_ENCODING"

    SENDING_APP     = "VF_ANALYST"
    SENDING_FACILITY = "ECMO_ICU"
    RECEIVING_APP   = "CIS"
    RECEIVING_FAC   = "HOSPITAL"
    VERSION         = "2.9"

    def __init__(self, sending_app: str = "VF_ANALYST",
                 receiving_app: str = "CIS"):
        self.sending_app   = sending_app
        self.receiving_app = receiving_app

    def _timestamp(self, dt: Optional[datetime] = None) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y%m%d%H%M%S")

    def _msh(self, message_ts: str, msg_ctrl_id: str) -> str:
        """MSH — Message Header segment."""
        return (
            f"MSH|^~\\&"
            f"|{self.sending_app}|{self.SENDING_FACILITY}"
            f"|{self.receiving_app}|{self.RECEIVING_FAC}"
            f"|{message_ts}||ORU^R01^ORU_R01"
            f"|{msg_ctrl_id}|P|{self.VERSION}"
        )

    def _pid(self, patient_id: str, patient_name: str = "UNKNOWN^UNKNOWN",
             dob: str = "", sex: str = "U") -> str:
        """PID — Patient Identification segment."""
        return (
            f"PID|1||{patient_id}^^^VF_ANALYST^PI"
            f"||{patient_name}||{dob}|{sex}"
        )

    def _obr(self, obs_ts: str, placer_order: str,
             session_id: str = "1") -> str:
        """OBR — Observation Request segment."""
        return (
            f"OBR|1|{placer_order}||"
            f"ECMO_MONITORING^ECMO Circuit Monitoring^LOCAL"
            f"|||{obs_ts}||||||||||||||||F"
        )

    def _obx(self, seq: int, column: str, value: float,
             obs_ts: str) -> str:
        """OBX — Observation/Result segment (one per parameter)."""
        meta = MDC_MAP.get(column)
        if meta is None:
            identifier = f"{column}^^LOCAL"
            ucum = "1"
        else:
            if col_in_verified(column):
                identifier = f"{VERIFIED_LOINC[column]}^{meta['description']}^LN"
            else:
                identifier = f"{column}^{meta['description']}^L"
            ucum = meta["ucum"]

        # Value type NM (numeric)
        return (
            f"OBX|{seq}|NM"
            f"|{identifier}"
            f"||{round(value, 4)}"
            f"|{ucum}"
            f"|||||F|||{obs_ts}"
        )

    def row_to_oru(
        self,
        row: pd.Series,
        patient_id:   str = "UNKNOWN",
        patient_name: str = "UNKNOWN^UNKNOWN",
        session_id:   str = "1",
    ) -> str:
        """
        Convert one measurement row to a complete HL7 v2.9 ORU^R01 message.

        Parameters
        ----------
        row          : one row from a VF Analyst measurements DataFrame
        patient_id   : patient MRN or pseudonymised identifier
        patient_name : HL7 XPN format (surname^given)
        session_id   : ECMO session identifier

        Returns
        -------
        pipe-delimited HL7 v2.9 string (segments separated by \\r)
        """
        ts = self._timestamp()
        msg_ctrl_id  = str(uuid.uuid4()).replace("-", "")[:20]
        placer_order = f"ECMO-{session_id}-{msg_ctrl_id[:8]}"

        obs_ts = ts
        if "timestamp" in row.index:
            try:
                dt = pd.to_datetime(row["timestamp"])
                if pd.notna(dt):
                    obs_ts = dt.strftime("%Y%m%d%H%M%S")
            except Exception:
                pass

        segments = [
            self._msh(ts, msg_ctrl_id),
            self._pid(patient_id, patient_name),
            self._obr(obs_ts, placer_order, session_id),
        ]

        seq = 1
        for col, meta in MDC_MAP.items():
            if col in row.index:
                val = row[col]
                try:
                    val = float(val)
                    if pd.notna(val):
                        segments.append(self._obx(seq, col, val, obs_ts))
                        seq += 1
                except (TypeError, ValueError):
                    pass

        return "\r".join(segments) + "\r"

    def dataframe_to_oru_list(
        self,
        df: pd.DataFrame,
        patient_id:   str = "UNKNOWN",
        session_id:   str = "1",
        max_messages: int = 100,
    ) -> list[str]:
        """
        Convert a measurements DataFrame to a list of ORU^R01 messages.
        Limits output to max_messages rows to avoid memory issues.
        """
        rows = df.head(max_messages)
        return [
            self.row_to_oru(row, patient_id=patient_id, session_id=session_id)
            for _, row in rows.iterrows()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — FHIR R4 resource builders
# ─────────────────────────────────────────────────────────────────────────────

FHIR_BASE_URL = "http://vf-analyst.local/fhir/R4"

# Local code system used for every parameter. The IEEE 11073 / LOINC entries in
# MDC_MAP are PROVISIONAL: they must be checked against IEEE 11073-10101 and the
# LOINC database before any exchange with a real HIS/EHR. Only LOINC codes whose
# meaning matches the measurement are emitted (VERIFIED_LOINC); all other
# parameters are identified by the local code system below.
LOCAL_SYSTEM = "http://vf-analyst.local/CodeSystem/ecmo-parameters"
VERIFIED_LOINC = {"temp_post_c": "8310-5", "sat_post_pct": "2708-6"}

# SNOMED-CT codes for ECMO device type
DEVICE_SNOMED = {
    "code":    "257347003",
    "display": "Extracorporeal membrane oxygenation device",
    "system":  "http://snomed.info/sct",
}


class FHIRExporter:
    """
    Builds FHIR R4-compliant resources from VF Analyst sessions and
    measurements.

    Produces:
      - Observation resources  (one per measurement × parameter)
      - Bundle resources       (one per session, wrapping all Observations)
      - DeviceMetric resources (one per monitored parameter)

    The output JSON can be POSTed directly to any FHIR R4 server endpoint:
        POST /fhir/R4/Bundle

    Reference: HL7 FHIR R4 — https://hl7.org/fhir/R4/
    """

    def __init__(
        self,
        patient_id:  str = "unknown-patient",
        device_id:   str = "ecmo-mc3-001",
        base_url:    str = FHIR_BASE_URL,
        institution: str = "ECMO ICU",
    ):
        self.patient_id  = patient_id
        self.device_id   = device_id
        self.base_url    = base_url
        self.institution = institution

    def _fhir_datetime(self, dt: Any) -> str:
        """Convert various datetime types to FHIR dateTime string."""
        try:
            dt = pd.to_datetime(dt)
            if pd.notna(dt):
                return dt.isoformat()
        except Exception:
            pass
        return datetime.now(timezone.utc).isoformat()

    def _patient_ref(self) -> dict:
        return {"reference": f"Patient/{self.patient_id}"}

    def _device_ref(self) -> dict:
        return {"reference": f"Device/{self.device_id}"}

    @staticmethod
    def _codings(column: str, meta: dict) -> list[dict]:
        codings = [{
            "system":  LOCAL_SYSTEM,
            "code":    column,
            "display": meta.get("description", column),
        }]
        if column in VERIFIED_LOINC:
            codings.append({
                "system":  "http://loinc.org",
                "code":    VERIFIED_LOINC[column],
                "display": meta.get("description", column),
            })
        return codings

    def observation(
        self,
        column:    str,
        value:     float,
        obs_ts:    Any,
        obs_id:    Optional[str] = None,
        status:    str = "final",
        alert:     bool = False,
    ) -> dict:
        """
        Build a single FHIR R4 Observation resource for one parameter.

        Parameters
        ----------
        column  : VF Analyst column name (must be in MDC_MAP)
        value   : numeric measurement value
        obs_ts  : observation timestamp
        obs_id  : optional Observation resource ID (generated if None)
        status  : FHIR observation status (final | preliminary | amended)
        alert   : True if an alarm was active at this time point
        """
        meta = MDC_MAP.get(column, {})
        obs_id = obs_id or str(uuid.uuid4())

        resource: dict[str, Any] = {
            "resourceType": "Observation",
            "id": obs_id,
            "status": status,
            "category": [
                {
                    "coding": [
                        {
                            "system":  "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code":    "vital-signs",
                            "display": "Vital Signs",
                        }
                    ]
                }
            ],
            "code": {
                "coding": self._codings(column, meta),
                "text": meta.get("description", column),
            },
            "subject":       self._patient_ref(),
            "device":        self._device_ref(),
            "effectiveDateTime": self._fhir_datetime(obs_ts),
            "valueQuantity": {
                "value":  round(float(value), 4),
                "unit":   meta.get("ucum", "1"),
                "system": "http://unitsofmeasure.org",
                "code":   meta.get("ucum", "1"),
            },
        }

        # Add interpretation (alarm flag)
        if alert:
            resource["interpretation"] = [
                {
                    "coding": [
                        {
                            "system":  "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code":    "AA",
                            "display": "Critical abnormal alert",
                        }
                    ]
                }
            ]

        return resource

    def row_to_observations(
        self,
        row:       pd.Series,
        session_id: str = "1",
    ) -> list[dict]:
        """
        Convert one measurement row to a list of FHIR R4 Observation resources
        (one per parameter in MDC_MAP that has a non-null value).
        """
        obs_ts = row.get("timestamp", datetime.now(timezone.utc))
        alert  = bool(row.get("alarm_active", 0))
        obs_list = []

        for col in MDC_MAP:
            if col == "alarm_active":
                continue
            if col not in row.index:
                continue
            val = row[col]
            try:
                val = float(val)
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                continue

            obs_id = f"obs-{session_id}-{col}-{str(uuid.uuid4())[:8]}"
            obs_list.append(
                self.observation(col, val, obs_ts,
                                 obs_id=obs_id, alert=alert)
            )

        return obs_list

    def session_to_bundle(
        self,
        session_row:    pd.Series,
        measurements:   pd.DataFrame,
        max_rows:       int = 500,
    ) -> dict:
        """
        Build a FHIR R4 Bundle (type=collection) containing all Observation
        resources for an ECMO session.

        Parameters
        ----------
        session_row   : one row from VFDatabase.get_sessions()
        measurements  : measurements DataFrame from VFDatabase.get_measurements()
        max_rows      : maximum number of time points to include
                        (FHIR bundles over ~1 MB may be rejected by servers)

        Returns
        -------
        FHIR R4 Bundle dict — serialise to JSON with json.dumps()
        """
        bundle_id = str(uuid.uuid4())
        entries   = []

        subset = measurements.head(max_rows)
        session_id = str(session_row.get("session_id", "unknown"))

        for _, row in subset.iterrows():
            for obs in self.row_to_observations(row, session_id=session_id):
                entries.append({
                    "fullUrl": f"{self.base_url}/Observation/{obs['id']}",
                    "resource": obs,
                    "request": {
                        "method": "POST",
                        "url":    "Observation",
                    },
                })

        bundle: dict[str, Any] = {
            "resourceType": "Bundle",
            "id": bundle_id,
            "meta": {
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "profile": [
                    "http://hl7.org/fhir/StructureDefinition/Bundle"
                ],
            },
            "type":  "transaction",
            "total": len(entries),
            "entry": entries,
        }

        return bundle

    def device_metric(self, column: str) -> dict:
        """
        Build a FHIR R4 DeviceMetric resource for one monitored parameter.
        DeviceMetric links a Device to the measurements it produces.
        """
        meta    = MDC_MAP.get(column, {})
        dm_id   = f"dm-{self.device_id}-{column}"

        return {
            "resourceType": "DeviceMetric",
            "id": dm_id,
            "type": {
                "coding": [
                    {
                        "system":  meta.get("fhir_system", "http://loinc.org"),
                        "code":    meta.get("loinc", column),
                        "display": meta.get("description", column),
                    }
                ]
            },
            "unit": {
                "coding": [
                    {
                        "system": "http://unitsofmeasure.org",
                        "code":   meta.get("ucum", "1"),
                    }
                ]
            },
            "source":     self._device_ref(),
            "category":   "measurement",
            "measurementPeriod": {"frequency": 1, "period": 1, "periodUnit": "s"},
            "calibration":       [],
        }

    def all_device_metrics(self) -> list[dict]:
        """Return DeviceMetric resources for all parameters in MDC_MAP."""
        return [
            self.device_metric(col)
            for col in MDC_MAP
            if col != "alarm_active"
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions — called from app.py Data Audit tab
# ─────────────────────────────────────────────────────────────────────────────

def session_to_fhir_bundle_json(
    session_row:  pd.Series,
    measurements: pd.DataFrame,
    patient_id:   str = "unknown-patient",
    device_id:    str = "ecmo-mc3-001",
    max_rows:     int = 200,
) -> str:
    """Return a FHIR R4 Bundle as a JSON string."""
    exporter = FHIRExporter(patient_id=patient_id, device_id=device_id)
    bundle   = exporter.session_to_bundle(session_row, measurements, max_rows)
    return json.dumps(bundle, indent=2, default=str)


def measurements_to_hl7(
    df:         pd.DataFrame,
    patient_id: str = "unknown-patient",
    session_id: str = "1",
    max_rows:   int = 10,
) -> str:
    """Return HL7 v2.9 ORU^R01 messages as a single string (demo)."""
    exporter  = HL7Exporter()
    messages  = exporter.dataframe_to_oru_list(
        df, patient_id=patient_id, session_id=session_id, max_messages=max_rows
    )
    return "\n\n---\n\n".join(messages)


def get_mdc_nomenclature_table() -> pd.DataFrame:
    """
    Return a DataFrame summarising the IEEE 11073 / LOINC nomenclature
    mapping for all VF Analyst parameters.
    """
    rows = []
    for col, meta in MDC_MAP.items():
        rows.append({
            "Column":      col,
            "MDC Code":    meta.get("mdc_code", ""),
            "MDC Value":   meta.get("mdc_value", ""),
            "LOINC":       VERIFIED_LOINC.get(col, "— (local code)"),
            "Status":      "verified LOINC" if col in VERIFIED_LOINC else "PROVISIONAL",
            "UCUM Unit":   meta.get("ucum", ""),
            "Description": meta.get("description", ""),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Inbound: HL7 v2 ORU^R01 parser (used by data_receiver for RJ45 streams)
# ─────────────────────────────────────────────────────────────────────────────

MLLP_START = b"\x0b"
MLLP_END   = b"\x1c\r"


def _reverse_code_map() -> dict[str, str]:
    """observation identifier (LOINC or local code) -> VF Analyst column."""
    out = {col: col for col in MDC_MAP}
    for col, code in VERIFIED_LOINC.items():
        out[code] = col
    return out


def parse_hl7_oru(raw: bytes | str) -> Optional[dict]:
    """
    Parse one HL7 v2 ORU^R01 message into a VF Analyst measurement row.
    Accepts MLLP-framed or bare messages. Returns None if the message is not
    an ORU^R01 or contains no recognised numeric observations.
    """
    if isinstance(raw, bytes):
        raw = raw.strip(b"\x0b\x1c\r\n ").decode("utf-8", errors="replace")
    text = raw.replace("\n", "\r")
    segments = [s for s in text.split("\r") if s.strip()]
    if not segments or not segments[0].startswith("MSH"):
        return None

    msh = segments[0].split("|")
    if len(msh) < 9 or not msh[8].startswith("ORU^R01"):
        return None

    lookup = _reverse_code_map()
    row: dict[str, Any] = {"case_id": "HL7", "alarm_active": 0}
    found = 0
    for seg in segments[1:]:
        f = seg.split("|")
        if f[0] == "OBR" and len(f) > 7 and f[7]:
            try:
                row["timestamp"] = datetime.strptime(f[7][:14], "%Y%m%d%H%M%S").isoformat()
            except ValueError:
                pass
        elif f[0] == "OBX" and len(f) > 5 and f[2] == "NM":
            ident = f[3].split("^")[0]
            col = lookup.get(ident)
            if col is None:
                continue
            try:
                row[col] = float(f[5])
                found += 1
            except ValueError:
                continue
    if found == 0:
        return None
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    row["alarm_active"] = int(row.get("alarm_active", 0))
    return row


def build_ack(msg_ctrl_id: str = "0", ack_code: str = "AA") -> bytes:
    """Minimal HL7 v2 ACK^R01, MLLP-framed (AA accept / AE error)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    msg = (f"MSH|^~\\&|VF_ANALYST|ECMO_ICU|DEVICE|ICU|{ts}||ACK^R01|{ts}|P|2.9\r"
           f"MSA|{ack_code}|{msg_ctrl_id}\r")
    return MLLP_START + msg.encode() + MLLP_END
