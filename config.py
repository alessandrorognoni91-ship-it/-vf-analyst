"""
config.py — VF Analyst Platform Configuration
=============================================
Central place for column mappings, sentinel codes, and device profiles.
Add new device formats here without touching the pipeline code.
"""

# ---------------------------------------------------------------------------
# Canonical (internal) column names → normalized schema field names
# ---------------------------------------------------------------------------
# Each entry: raw_column_name → canonical_name
# Add alternate device column names for the same concept to the same canonical key.
COLUMN_MAP: dict[str, str] = {
    # Timestamps
    "Timestamp": "timestamp",

    # Status flags
    "Overall alarm status": "alarm_overall",
    "Bubble alarm status": "alarm_bubble",
    "Technical Fault": "technical_fault",
    "Case Number": "case_id",

    # Measurements — pressures
    "Pre-oxygenator pressure mmHg": "pressure_pre_mmhg",
    "Post-oxygenator pressure mmHg": "pressure_post_mmhg",
    "Oxygenator pressure delta mmHg": "pressure_delta_mmhg",

    # Measurements — saturations
    "Pre-oxygenator saturation %": "sat_pre_pct",
    "Post-oxygenator saturation %": "sat_post_pct",

    # Measurements — temperature / flow / pump
    "Post-oxygenator temperature C": "temp_post_c",
    "Flow rate L/min": "flow_rate_lpm",
    "Measured pump speed PRM": "pump_speed_rpm",

    # Alarm limits — flow
    "Flow alarm low limit L/min": "alarm_flow_low_lpm",
    "Flow alarm high limit L/min": "alarm_flow_high_lpm",

    # Alarm limits — pre-oxygenator pressure
    "Pre-oxygenator pressure alarm low limit mmHg": "alarm_pressure_pre_low",
    "Pre-oxygenator pressure alarm high limit mmHg": "alarm_pressure_pre_high",

    # Alarm limits — post-oxygenator pressure
    "Post-oxygenator pressure alarm low limit mmHg": "alarm_pressure_post_low",
    "Post-oxygenator pressure alarm high limit mmHg": "alarm_pressure_post_high",

    # Alarm limits — delta pressure
    "Oxygenator pressure delta alarm low limit mmHg": "alarm_pressure_delta_low",
    "Oxygenator pressure delta alarm high limit mmHg": "alarm_pressure_delta_high",

    # Alarm limits — saturations
    "Pre-oxygenator saturation alarm low limit %": "alarm_sat_pre_low",
    "Pre-oxygenator saturation alarm high limit %": "alarm_sat_pre_high",
    "Post-oxygenator saturation alarm low limit %": "alarm_sat_post_low",

    # Alarm limits — temperature
    "Post-oxygenator temperature alarm low limit C": "alarm_temp_post_low",
    "Post-oxygenator temperature alarm high limit C": "alarm_temp_post_high",
}

# ---------------------------------------------------------------------------
# Sentinel string values that appear in measurement columns
# ---------------------------------------------------------------------------
# "--" → sensor not connected / value unavailable  → mapped to NaN
# "Lo" → value is below the measurable range       → mapped to NaN (flagged)
# "Hi" → value is above the measurable range       → mapped to NaN (flagged)
# "A"  → alarm limit set to Auto                   → mapped to NaN
# "D"  → alarm limit disabled                      → mapped to NaN
# "E"  → bubble alarm: error / sensor not present  → mapped to NaN
MEASUREMENT_SENTINELS: set[str] = {"--", "Lo", "Hi", "A", "D", "E"}

# Which canonical columns are continuous measurement values
MEASUREMENT_COLS: list[str] = [
    "pressure_pre_mmhg",
    "pressure_post_mmhg",
    "pressure_delta_mmhg",
    "sat_pre_pct",
    "sat_post_pct",
    "temp_post_c",
    "flow_rate_lpm",
    "pump_speed_rpm",
]

# Columns that are alarm-limit configuration (numeric, but less often plotted)
ALARM_LIMIT_COLS: list[str] = [
    "alarm_flow_low_lpm",
    "alarm_flow_high_lpm",
    "alarm_pressure_pre_low",
    "alarm_pressure_pre_high",
    "alarm_pressure_post_low",
    "alarm_pressure_post_high",
    "alarm_pressure_delta_low",
    "alarm_pressure_delta_high",
    "alarm_sat_pre_low",
    "alarm_sat_pre_high",
    "alarm_sat_post_low",
    "alarm_temp_post_low",
    "alarm_temp_post_high",
]

# ---------------------------------------------------------------------------
# Timestamp formats to try (in order) during parsing
# ---------------------------------------------------------------------------
TIMESTAMP_FORMATS: list[str] = [
    "%Y.%m.%d %H:%M:%S.%f",   # 2026.01.31 15:33:56.553  ← device format
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y%m%d%H%M%S",
]

# ---------------------------------------------------------------------------
# Human-readable labels for the dashboard (canonical_name → display label)
# ---------------------------------------------------------------------------
DISPLAY_LABELS: dict[str, str] = {
    "pressure_pre_mmhg":    "Pre-Oxygenator Pressure (mmHg)",
    "pressure_post_mmhg":   "Post-Oxygenator Pressure (mmHg)",
    "pressure_delta_mmhg":  "Oxygenator ΔPressure (mmHg)",
    "sat_pre_pct":          "Pre-Oxygenator SatO₂ (%)",
    "sat_post_pct":         "Post-Oxygenator SatO₂ (%)",
    "temp_post_c":          "Post-Oxygenator Temperature (°C)",
    "flow_rate_lpm":        "Flow Rate (L/min)",
    "pump_speed_rpm":       "Pump Speed (RPM)",
}
