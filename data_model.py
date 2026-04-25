"""
data_model.py — VF Analyst Platform
======================================
Defines the normalized internal schema and provides a lightweight SQLite
persistence layer.

Schema design decisions:
- One table per logical concept (sessions, measurements, alarms)
- Immutable raw data stored alongside cleaned values for traceability
- Extensible: add new device types as new rows in the 'device_type' column
- SQLite used for the demo; swap for PostgreSQL by changing the engine URL

Tables
------
sessions        : one row per uploaded file / clinical case
measurements    : time-series measurements (one row per timestamp)
alarm_events    : subset of measurements where alarm_active = True
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default in-memory database (replaced with a file path for persistence)
DEFAULT_DB = ":memory:"

# ---------------------------------------------------------------------------
# DDL — table definitions
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      TEXT    NOT NULL,
    device_type  TEXT    DEFAULT 'unknown',    -- e.g. 'rotaflow', 'cardiohelp'
    center_id    TEXT    DEFAULT 'unknown',    -- clinical center identifier
    filename     TEXT,
    row_count    INTEGER,
    time_start   TEXT,
    time_end     TEXT,
    provenance   TEXT,                         -- JSON blob: cleaning audit trail
    uploaded_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS measurements (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER REFERENCES sessions(session_id),
    timestamp               TEXT,
    case_id                 TEXT,
    pressure_pre_mmhg       REAL,
    pressure_post_mmhg      REAL,
    pressure_delta_mmhg     REAL,
    sat_pre_pct             REAL,
    sat_post_pct            REAL,
    temp_post_c             REAL,
    flow_rate_lpm           REAL,
    pump_speed_rpm          REAL,
    alarm_active            INTEGER,           -- 0/1 boolean
    alarm_bubble            TEXT,
    technical_fault         REAL
);

CREATE TABLE IF NOT EXISTS alarm_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(session_id),
    timestamp       TEXT,
    case_id         TEXT,
    alarm_type      TEXT,                      -- 'overall', 'bubble', 'technical'
    alarm_value     TEXT                       -- raw value at alarm time
);

CREATE INDEX IF NOT EXISTS idx_meas_session   ON measurements(session_id);
CREATE INDEX IF NOT EXISTS idx_meas_timestamp ON measurements(timestamp);
CREATE INDEX IF NOT EXISTS idx_alarm_session  ON alarm_events(session_id);
"""

# Measurement columns that map directly to the DB table
MEAS_COLS = [
    "timestamp", "case_id",
    "pressure_pre_mmhg", "pressure_post_mmhg", "pressure_delta_mmhg",
    "sat_pre_pct", "sat_post_pct", "temp_post_c",
    "flow_rate_lpm", "pump_speed_rpm",
    "alarm_active", "alarm_bubble", "technical_fault",
]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class VFDatabase:
    """
    Thin wrapper around an SQLite connection.
    Use VFDatabase(':memory:') for the demo or a file path for persistence.
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("VFDatabase initialised at '%s'", db_path)

    def _init_schema(self) -> None:
        self.conn.executescript(DDL)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert_session(
        self,
        case_id: str,
        filename: str,
        row_count: int,
        time_start: Optional[str],
        time_end: Optional[str],
        provenance: dict,
        device_type: str = "unknown",
        center_id: str = "unknown",
    ) -> int:
        """Insert a session record and return the new session_id."""
        cur = self.conn.execute(
            """
            INSERT INTO sessions
                (case_id, device_type, center_id, filename, row_count,
                 time_start, time_end, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id, device_type, center_id, filename, row_count,
                time_start, time_end, json.dumps(provenance, default=str),
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def insert_measurements(self, df: pd.DataFrame, session_id: int) -> int:
        """Bulk-insert measurement rows. Returns number of rows inserted."""
        cols_present = [c for c in MEAS_COLS if c in df.columns]
        subset = df[cols_present].copy()
        subset["session_id"] = session_id

        # SQLite wants strings for timestamps
        if "timestamp" in subset.columns:
            subset["timestamp"] = subset["timestamp"].astype(str)
        if "alarm_active" in subset.columns:
            subset["alarm_active"] = subset["alarm_active"].astype(int)

        subset.to_sql("measurements", self.conn, if_exists="append", index=False)
        self.conn.commit()
        return len(subset)

    def insert_alarm_events(self, df: pd.DataFrame, session_id: int) -> int:
        """Extract and store rows where any alarm was active."""
        alarm_rows = []

        if "alarm_active" in df.columns:
            active = df[df["alarm_active"] == True]
            for _, row in active.iterrows():
                alarm_rows.append({
                    "session_id": session_id,
                    "timestamp": str(row.get("timestamp", "")),
                    "case_id": row.get("case_id", ""),
                    "alarm_type": "overall",
                    "alarm_value": "Alarm",
                })

        if "alarm_bubble" in df.columns:
            bubble = df[df["alarm_bubble"].isin(["A", "E"])]
            for _, row in bubble.iterrows():
                alarm_rows.append({
                    "session_id": session_id,
                    "timestamp": str(row.get("timestamp", "")),
                    "case_id": row.get("case_id", ""),
                    "alarm_type": "bubble",
                    "alarm_value": str(row["alarm_bubble"]),
                })

        if alarm_rows:
            pd.DataFrame(alarm_rows).to_sql(
                "alarm_events", self.conn, if_exists="append", index=False
            )
            self.conn.commit()

        return len(alarm_rows)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_sessions(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM sessions ORDER BY uploaded_at DESC", self.conn)

    def get_measurements(self, session_id: Optional[int] = None) -> pd.DataFrame:
        if session_id is not None:
            df = pd.read_sql(
                "SELECT * FROM measurements WHERE session_id = ? ORDER BY timestamp",
                self.conn,
                params=(session_id,),
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM measurements ORDER BY timestamp", self.conn
            )

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df

    def get_measurements_multi(self, session_ids: list[int]) -> pd.DataFrame:
        """
        Return measurements for multiple sessions, tagged with session_id.
        Useful for comparative analysis and cohort-level views.
        """
        if not session_ids:
            return pd.DataFrame()
        placeholders = ",".join("?" * len(session_ids))
        df = pd.read_sql(
            f"SELECT * FROM measurements WHERE session_id IN ({placeholders}) ORDER BY session_id, timestamp",
            self.conn,
            params=session_ids,
        )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def get_alarm_events(self, session_id: Optional[int] = None) -> pd.DataFrame:
        if session_id is not None:
            return pd.read_sql(
                "SELECT * FROM alarm_events WHERE session_id = ? ORDER BY timestamp",
                self.conn,
                params=(session_id,),
            )
        return pd.read_sql(
            "SELECT * FROM alarm_events ORDER BY timestamp", self.conn
        )

    def get_alarm_events_multi(self, session_ids: list[int]) -> pd.DataFrame:
        """Return alarm events for multiple sessions."""
        if not session_ids:
            return pd.DataFrame()
        placeholders = ",".join("?" * len(session_ids))
        return pd.read_sql(
            f"SELECT * FROM alarm_events WHERE session_id IN ({placeholders}) ORDER BY session_id, timestamp",
            self.conn,
            params=session_ids,
        )

    def close(self) -> None:
        self.conn.close()
