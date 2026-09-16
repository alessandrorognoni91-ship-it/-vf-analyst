"""
data_receiver.py — VF Analyst Live Data Receiver
==================================================
Handles two real-time data ingestion modes:

  MODE 1 — TCP/IP Live Stream (RJ45)
    Connects to the VitalFlow MC3 device network interface.
    Receives data packets, parses them, writes to SQLite every N seconds.
    Runs as a background thread alongside the Streamlit dashboard.

  MODE 2 — USB Auto-Watcher
    Monitors a directory (e.g. /media/usb or a local folder) for new CSV files.
    When a new file appears (e.g. from a USB stick), it is automatically
    ingested — no manual upload needed.
    Useful when live TCP is not available.

USAGE
-----
Both modes run as background threads. Start them from app.py:

    from data_receiver import start_live_receiver, start_usb_watcher

    # Mode 1: TCP live
    start_live_receiver(db, host="192.168.1.100", port=8080)

    # Mode 2: USB auto-watch
    start_usb_watcher(db, watch_dir="/media/usb")

IMPORTANT: The TCP receiver uses a SIMULATED data format.
When you obtain the actual VitalFlow network protocol documentation,
replace the _parse_packet() function with the correct parser.
Contact Medtronic/MC3 technical support for the network interface spec.

Author: VF Analyst project
"""

from __future__ import annotations

import io
import json
import logging
import os
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import load_csv
from data_cleaning import clean
from data_model import VFDatabase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared live data buffer
# ─────────────────────────────────────────────────────────────────────────────

class LiveBuffer:
    """
    Thread-safe circular buffer for live measurements.
    Holds the last N seconds of data in memory for real-time display.
    Also flushes periodically to the SQLite database for persistence.
    """

    MAX_ROWS  = 3600   # keep last 1 hour at 1 Hz
    FLUSH_SEC = 10     # write to DB every 10 seconds

    def __init__(self, db: VFDatabase, session_id: int):
        self.db         = db
        self.session_id = session_id
        self._lock      = threading.Lock()
        self._rows:  list[dict] = []
        self._last_flush = time.time()

    def push(self, row: dict) -> None:
        """Add one measurement row. Thread-safe."""
        row.setdefault("timestamp", datetime.utcnow().isoformat())
        with self._lock:
            self._rows.append(row)
            if len(self._rows) > self.MAX_ROWS:
                self._rows = self._rows[-self.MAX_ROWS:]

        # Flush to DB periodically
        if time.time() - self._last_flush >= self.FLUSH_SEC:
            self.flush_to_db()

    def get_dataframe(self, last_n: int = 300) -> pd.DataFrame:
        """Return last N rows as a DataFrame for display. Thread-safe."""
        with self._lock:
            rows = self._rows[-last_n:]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def flush_to_db(self) -> None:
        """Persist buffered rows to SQLite. Called automatically."""
        with self._lock:
            if not self._rows:
                return
            rows_to_flush = self._rows.copy()

        try:
            df = pd.DataFrame(rows_to_flush)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            self.db.insert_measurements(df, self.session_id)
            self._last_flush = time.time()
            logger.debug("Flushed %d rows to DB (session %d)",
                         len(rows_to_flush), self.session_id)
        except Exception as exc:
            logger.warning("DB flush failed: %s", exc)

    @property
    def row_count(self) -> int:
        with self._lock:
            return len(self._rows)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — TCP/IP Live Receiver
# ─────────────────────────────────────────────────────────────────────────────

# ── Packet parser ─────────────────────────────────────────────────────────────
# REPLACE THIS FUNCTION when you have the actual VitalFlow protocol spec.
# Current implementation accepts two formats:
#   A) JSON line: {"flow": 3.5, "pressure_delta": 28.0, "rpm": 3200, ...}
#   B) Simulated binary struct (for testing without the device)

def _parse_packet(data: bytes) -> Optional[dict]:
    """
    Parse one data packet from the VitalFlow network interface.

    TODO: Replace with the actual VitalFlow/MC3 protocol parser.
    Contact Medtronic technical support for the network interface specification.

    Currently handles:
    - JSON lines (most common in modern medical devices)
    - Raw binary struct (8 floats: flow, p_pre, p_post, p_delta,
                          sat_pre, sat_post, temp, rpm)
    """
    raw = data.strip()
    if not raw:
        return None

    # ── Try JSON first ────────────────────────────────────────────────────────
    try:
        obj = json.loads(raw)
        return {
            "timestamp":           obj.get("ts", datetime.utcnow().isoformat()),
            "case_id":             str(obj.get("case_id", "LIVE")),
            "flow_rate_lpm":       float(obj.get("flow",           obj.get("flow_rate_lpm",       0))),
            "pressure_pre_mmhg":   float(obj.get("p_pre",          obj.get("pressure_pre_mmhg",   0))),
            "pressure_post_mmhg":  float(obj.get("p_post",         obj.get("pressure_post_mmhg",  0))),
            "pressure_delta_mmhg": float(obj.get("p_delta",        obj.get("pressure_delta_mmhg", 0))),
            "sat_pre_pct":         float(obj.get("sat_pre",        obj.get("sat_pre_pct",         0))),
            "sat_post_pct":        float(obj.get("sat_post",       obj.get("sat_post_pct",        0))),
            "temp_post_c":         float(obj.get("temp",           obj.get("temp_post_c",         37.0))),
            "pump_speed_rpm":      float(obj.get("rpm",            obj.get("pump_speed_rpm",      0))),
            "alarm_active":        int(obj.get("alarm", 0)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # ── Try binary struct (8 × float32 = 32 bytes) ────────────────────────────
    if len(raw) >= 32:
        try:
            vals = struct.unpack(">8f", raw[:32])
            return {
                "timestamp":           datetime.utcnow().isoformat(),
                "case_id":             "LIVE",
                "flow_rate_lpm":       vals[0],
                "pressure_pre_mmhg":   vals[1],
                "pressure_post_mmhg":  vals[2],
                "pressure_delta_mmhg": vals[3],
                "sat_pre_pct":         vals[4],
                "sat_post_pct":        vals[5],
                "temp_post_c":         vals[6],
                "pump_speed_rpm":      vals[7],
                "alarm_active":        0,
            }
        except struct.error:
            pass

    logger.debug("Unparseable packet (%d bytes): %s…", len(raw), raw[:40])
    return None


class TCPReceiver(threading.Thread):
    """
    Background thread that maintains a TCP connection to the VitalFlow device
    and streams measurements into a LiveBuffer.

    The thread reconnects automatically on connection loss (with exponential
    backoff), so it is robust to brief network interruptions.
    """

    RECONNECT_DELAY_SEC = 5
    MAX_RECONNECT_DELAY = 60
    READ_TIMEOUT_SEC    = 5
    LINE_DELIMITER      = b"\n"   # adjust if device uses \r\n or fixed-length frames

    def __init__(
        self,
        buffer:  LiveBuffer,
        host:    str,
        port:    int,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True, name="TCPReceiver")
        self.buffer     = buffer
        self.host       = host
        self.port       = port
        self._stop      = stop_event
        self.connected  = False
        self.error_msg: str = ""
        self.packets_received = 0

    def run(self) -> None:
        delay = self.RECONNECT_DELAY_SEC
        while not self._stop.is_set():
            try:
                logger.info("Connecting to %s:%d…", self.host, self.port)
                with socket.create_connection(
                    (self.host, self.port), timeout=self.READ_TIMEOUT_SEC
                ) as sock:
                    sock.settimeout(self.READ_TIMEOUT_SEC)
                    self.connected = True
                    self.error_msg = ""
                    delay          = self.RECONNECT_DELAY_SEC   # reset backoff
                    logger.info("Connected to VitalFlow at %s:%d", self.host, self.port)

                    remainder = b""
                    while not self._stop.is_set():
                        try:
                            chunk = sock.recv(4096)
                        except socket.timeout:
                            continue
                        if not chunk:
                            raise ConnectionResetError("Device closed connection")

                        remainder += chunk
                        while self.LINE_DELIMITER in remainder:
                            line, remainder = remainder.split(self.LINE_DELIMITER, 1)
                            row = _parse_packet(line)
                            if row:
                                self.buffer.push(row)
                                self.packets_received += 1

            except (ConnectionRefusedError, ConnectionResetError, OSError) as exc:
                self.connected = False
                self.error_msg = str(exc)
                logger.warning("TCP error: %s — retry in %ds", exc, delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.MAX_RECONNECT_DELAY)

        self.buffer.flush_to_db()
        logger.info("TCPReceiver stopped")


# ── Public function: start live receiver ──────────────────────────────────────

# Global references (one per process)
_tcp_receiver:   Optional[TCPReceiver]   = None
_live_buffer:    Optional[LiveBuffer]    = None
_tcp_stop_event: Optional[threading.Event] = None


def start_live_receiver(
    db:   VFDatabase,
    host: str = "192.168.1.100",
    port: int = 8080,
) -> tuple[LiveBuffer, TCPReceiver]:
    """
    Start the TCP receiver in a background thread.

    Parameters
    ----------
    db   : VFDatabase instance (shared with the Streamlit app)
    host : IP address of the VitalFlow device on the local network
    port : TCP port (check VitalFlow documentation; commonly 8080 or 9000)

    Returns
    -------
    (buffer, receiver) — the buffer is used by the dashboard to read live data
    """
    global _tcp_receiver, _live_buffer, _tcp_stop_event

    if _tcp_receiver and _tcp_receiver.is_alive():
        logger.info("TCP receiver already running")
        return _live_buffer, _tcp_receiver

    # Create a live session in the DB
    session_id = db.insert_session(
        case_id="LIVE",
        filename=f"live_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        row_count=0,
        time_start=datetime.utcnow().isoformat(),
        time_end=None,
        provenance={"mode": "live_tcp", "host": host, "port": port},
    )

    _live_buffer    = LiveBuffer(db, session_id)
    _tcp_stop_event = threading.Event()
    _tcp_receiver   = TCPReceiver(_live_buffer, host, port, _tcp_stop_event)
    _tcp_receiver.start()

    logger.info("Live TCP receiver started → %s:%d (session %d)", host, port, session_id)
    return _live_buffer, _tcp_receiver


def stop_live_receiver() -> None:
    """Gracefully stop the TCP receiver thread."""
    global _tcp_stop_event
    if _tcp_stop_event:
        _tcp_stop_event.set()


def get_live_buffer() -> Optional[LiveBuffer]:
    """Return the active live buffer, or None if not started."""
    return _live_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Simulated live stream (for demo / testing without device)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedReceiver(threading.Thread):
    """
    Generates realistic simulated ECMO data for demo purposes.
    Mirrors the same LiveBuffer interface as TCPReceiver.

    Simulates:
    - Normal operation with physiological noise
    - Occasional alarm events (random spikes)
    - Gradual oxygenator pressure drift (simulated thrombosis)
    """

    def __init__(self, buffer: LiveBuffer, stop_event: threading.Event,
                 hz: float = 1.0):
        super().__init__(daemon=True, name="SimulatedReceiver")
        self.buffer     = buffer
        self._stop      = stop_event
        self._interval  = 1.0 / hz
        self.connected  = True   # always "connected" in sim mode
        self.error_msg  = ""
        self.packets_received = 0

        # Simulation state
        self._t              = 0.0
        self._drift_pressure = 0.0   # simulates gradual TMP rise

    def _next_row(self) -> dict:
        t = self._t
        self._t += self._interval

        # Gradual pressure drift (simulates thrombosis developing over ~30 min)
        self._drift_pressure += 0.03

        flow      = max(0.5, 3.5 + 0.3 * np.sin(t / 120) + np.random.normal(0, 0.08))
        p_delta   = max(0, 28 + self._drift_pressure + 5 * np.sin(t / 60)
                        + np.random.normal(0, 2.0))
        p_pre     = max(0, 200 + 10 * np.sin(t / 90) + np.random.normal(0, 5))
        p_post    = max(0, p_pre - p_delta)
        sat_pre   = max(40, min(100, 68 + 3 * np.sin(t / 200) + np.random.normal(0, 1.5)))
        sat_post  = max(40, min(100, 97 + np.random.normal(0, 0.8)))
        temp      = max(33, min(41, 37.0 + 0.2 * np.sin(t / 300) + np.random.normal(0, 0.1)))
        rpm       = max(1000, 3200 + 50 * np.sin(t / 180) + np.random.normal(0, 30))
        alarm     = int(p_delta > 60 or flow < 1.5 or sat_pre < 62)

        return {
            "timestamp":           datetime.utcnow().isoformat(),
            "case_id":             "SIM",
            "flow_rate_lpm":       round(flow, 3),
            "pressure_pre_mmhg":   round(p_pre, 1),
            "pressure_post_mmhg":  round(p_post, 1),
            "pressure_delta_mmhg": round(p_delta, 1),
            "sat_pre_pct":         round(sat_pre, 1),
            "sat_post_pct":        round(sat_post, 1),
            "temp_post_c":         round(temp, 2),
            "pump_speed_rpm":      round(rpm, 0),
            "alarm_active":        alarm,
        }

    def run(self) -> None:
        logger.info("Simulated receiver started (1 Hz)")
        while not self._stop.is_set():
            row = self._next_row()
            self.buffer.push(row)
            self.packets_received += 1
            self._stop.wait(self._interval)
        self.buffer.flush_to_db()
        logger.info("Simulated receiver stopped")


def start_simulated_receiver(db: VFDatabase) -> tuple[LiveBuffer, SimulatedReceiver]:
    """
    Start a simulated data stream for demo / testing without a real device.
    Produces realistic ECMO data including a gradual pressure drift that
    will trigger the oxygenator thrombosis alert after ~20 minutes.
    """
    global _tcp_receiver, _live_buffer, _tcp_stop_event

    if _tcp_receiver and _tcp_receiver.is_alive():
        return _live_buffer, _tcp_receiver

    session_id = db.insert_session(
        case_id="SIM",
        filename=f"simulated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        row_count=0,
        time_start=datetime.utcnow().isoformat(),
        time_end=None,
        provenance={"mode": "simulated"},
    )

    _live_buffer    = LiveBuffer(db, session_id)
    _tcp_stop_event = threading.Event()
    _tcp_receiver   = SimulatedReceiver(_live_buffer, _tcp_stop_event)
    _tcp_receiver.start()

    logger.info("Simulated receiver started (session %d)", session_id)
    return _live_buffer, _tcp_receiver


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — USB Auto-Watcher
# ─────────────────────────────────────────────────────────────────────────────

class USBWatcher(threading.Thread):
    """
    Watches a directory for new CSV files and auto-ingests them.

    On Raspberry Pi: set watch_dir to "/media/usb" or "/media/pi"
    (wherever the OS mounts USB sticks).

    On any OS: set watch_dir to any folder — drop a CSV there
    and it will be ingested automatically within poll_interval seconds.
    """

    def __init__(
        self,
        db:            VFDatabase,
        watch_dir:     str,
        stop_event:    threading.Event,
        poll_interval: float = 5.0,
        on_new_file=None,   # optional callback(session_id, filename)
    ):
        super().__init__(daemon=True, name="USBWatcher")
        self.db            = db
        self.watch_dir     = Path(watch_dir)
        self._stop         = stop_event
        self._interval     = poll_interval
        self.on_new_file   = on_new_file
        self._seen:  set   = set()
        self.files_ingested = 0
        self.last_error    = ""

    def run(self) -> None:
        logger.info("USB watcher started → %s", self.watch_dir)

        # Pre-populate seen set (don't re-ingest existing files on startup)
        if self.watch_dir.exists():
            self._seen = {
                f.name for f in self.watch_dir.glob("*.csv")
            }

        while not self._stop.is_set():
            self._stop.wait(self._interval)
            self._scan()

        logger.info("USB watcher stopped")

    def _scan(self) -> None:
        if not self.watch_dir.exists():
            return

        for csv_path in sorted(self.watch_dir.glob("*.csv")):
            if csv_path.name in self._seen:
                continue

            logger.info("New CSV detected: %s", csv_path.name)
            try:
                sid = self._ingest(csv_path)
                self._seen.add(csv_path.name)
                self.files_ingested += 1
                self.last_error = ""
                if self.on_new_file:
                    self.on_new_file(sid, csv_path.name)
                logger.info("Auto-ingested %s → session %d", csv_path.name, sid)
            except Exception as exc:
                self.last_error = f"{csv_path.name}: {exc}"
                logger.warning("Auto-ingest failed for %s: %s", csv_path.name, exc)

    def _ingest(self, path: Path) -> int:
        raw_df, _ = load_csv(str(path))
        clean_df, provenance = clean(raw_df)

        ts_ok      = "timestamp" in clean_df.columns and clean_df["timestamp"].notna().any()
        time_start = str(clean_df["timestamp"].min()) if ts_ok else None
        time_end   = str(clean_df["timestamp"].max()) if ts_ok else None
        case_id    = clean_df["case_id"].iloc[0] if "case_id" in clean_df.columns else "USB"

        sid = self.db.insert_session(
            case_id=case_id,
            filename=path.name,
            row_count=len(clean_df),
            time_start=time_start,
            time_end=time_end,
            provenance={**provenance, "source": "usb_auto"},
        )
        self.db.insert_measurements(clean_df, sid)
        self.db.insert_alarm_events(clean_df, sid)
        return sid


# Global USB watcher reference
_usb_watcher:     Optional[USBWatcher]    = None
_usb_stop_event:  Optional[threading.Event] = None


def start_usb_watcher(
    db:         VFDatabase,
    watch_dir:  str = "/media/usb",
    on_new_file = None,
) -> USBWatcher:
    """
    Start the USB auto-watcher in a background thread.

    Parameters
    ----------
    db         : shared VFDatabase instance
    watch_dir  : path to monitor for new CSV files
                 Raspberry Pi default: "/media/usb" or "/media/pi"
                 Development default: "./usb_drop" (create this folder locally)
    on_new_file: optional callback called with (session_id, filename)
                 when a new file is ingested; used to trigger Streamlit rerun

    Returns
    -------
    USBWatcher thread instance
    """
    global _usb_watcher, _usb_stop_event

    if _usb_watcher and _usb_watcher.is_alive():
        logger.info("USB watcher already running")
        return _usb_watcher

    Path(watch_dir).mkdir(parents=True, exist_ok=True)
    _usb_stop_event = threading.Event()
    _usb_watcher    = USBWatcher(db, watch_dir, _usb_stop_event,
                                  on_new_file=on_new_file)
    _usb_watcher.start()
    logger.info("USB watcher started → %s", watch_dir)
    return _usb_watcher


def stop_usb_watcher() -> None:
    global _usb_stop_event
    if _usb_stop_event:
        _usb_stop_event.set()


def get_usb_watcher() -> Optional[USBWatcher]:
    return _usb_watcher
