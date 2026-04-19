"""
data_cleaning.py — VF Analyst Platform
========================================
Transforms the raw (column-mapped) DataFrame into a clean, typed DataFrame
ready for storage and visualization.

Traceability: every transformation is logged and summarised in a provenance dict
so clinicians / engineers can audit what was changed and why.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

from config import (
    MEASUREMENT_COLS,
    ALARM_LIMIT_COLS,
    MEASUREMENT_SENTINELS,
)
from data_loader import try_parse_timestamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply a deterministic, auditable cleaning pipeline to the raw mapped DataFrame.

    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned, typed DataFrame.
    provenance : dict
        Summary of every transformation applied (for traceability).
    """
    df = df.copy()
    provenance: dict[str, Any] = {}

    df, provenance["timestamp"] = _clean_timestamp(df)
    df, provenance["measurements"] = _clean_measurements(df)
    df, provenance["alarm_limits"] = _clean_alarm_limits(df)
    df, provenance["status_flags"] = _clean_status_flags(df)
    df, provenance["case_id"] = _clean_case_id(df)
    df, provenance["dedup"] = _deduplicate(df)

    provenance["final_shape"] = {"rows": len(df), "cols": len(df.columns)}
    return df, provenance


# ---------------------------------------------------------------------------
# Step-by-step cleaners
# ---------------------------------------------------------------------------

def _clean_timestamp(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prov: dict[str, Any] = {}

    if "timestamp" not in df.columns:
        prov["status"] = "column_missing"
        df["timestamp"] = pd.NaT
        return df, prov

    raw_nulls = df["timestamp"].isna().sum()
    df["timestamp"] = try_parse_timestamp(df["timestamp"])
    parsed_nulls = df["timestamp"].isna().sum()

    prov["status"] = "parsed"
    prov["unparseable_rows"] = int(parsed_nulls - raw_nulls)

    # Sort chronologically so time-series plots are correct
    df = df.sort_values("timestamp").reset_index(drop=True)
    prov["sorted"] = True

    return df, prov


def _clean_measurements(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Convert measurement columns to float64.
    Sentinel strings (--  Lo  Hi  A  D  E) → NaN.
    Records a per-column count of sentinels encountered for traceability.
    """
    prov: dict[str, dict] = {}

    for col in MEASUREMENT_COLS:
        if col not in df.columns:
            logger.debug("Measurement column '%s' not in DataFrame — skipping.", col)
            continue

        col_prov: dict[str, Any] = {}

        # Count sentinel occurrences before replacing
        sentinel_mask = df[col].isin(MEASUREMENT_SENTINELS)
        col_prov["sentinel_count"] = int(sentinel_mask.sum())
        col_prov["sentinels_found"] = (
            df.loc[sentinel_mask, col].value_counts().to_dict()
        )

        # Replace sentinels with NaN then coerce to float
        df[col] = df[col].replace(list(MEASUREMENT_SENTINELS), np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")

        col_prov["total_nan_after"] = int(df[col].isna().sum())
        prov[col] = col_prov

    return df, prov


def _clean_alarm_limits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Alarm limit columns use the same sentinel codes as measurements.
    Convert to float; preserve NaN for Auto/Disabled limits.
    """
    prov: dict[str, int] = {}

    for col in ALARM_LIMIT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].replace(list(MEASUREMENT_SENTINELS), np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        prov[col] = int(df[col].isna().sum())

    return df, prov


def _clean_status_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical alarm status columns as boolean / category.

    - alarm_overall : 'Alarm' → True, NaN → False (no alarm active)
    - alarm_bubble  : keep as category (E=error, A=alarm, D=disabled, --=ok)
    - technical_fault: coerce to numeric; NaN = no fault
    """
    prov: dict[str, Any] = {}

    if "alarm_overall" in df.columns:
        df["alarm_overall"] = df["alarm_overall"].fillna("").str.strip()
        df["alarm_active"] = df["alarm_overall"] == "Alarm"
        df.drop(columns=["alarm_overall"], inplace=True)
        prov["alarm_active_true_count"] = int(df["alarm_active"].sum())

    if "alarm_bubble" in df.columns:
        df["alarm_bubble"] = df["alarm_bubble"].replace("--", "OK").astype("category")
        prov["alarm_bubble_distribution"] = df["alarm_bubble"].value_counts().to_dict()

    if "technical_fault" in df.columns:
        df["technical_fault"] = pd.to_numeric(df["technical_fault"], errors="coerce")
        prov["technical_fault_non_null"] = int(df["technical_fault"].notna().sum())

    return df, prov


def _clean_case_id(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prov: dict[str, Any] = {}

    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str).str.strip()
        prov["unique_cases"] = df["case_id"].unique().tolist()
    else:
        df["case_id"] = "unknown"
        prov["note"] = "case_id defaulted to 'unknown'"

    return df, prov


def _deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicate rows. In biomedical logs duplicates can arise
    from USB export glitches or repeated file uploads.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df.reset_index(drop=True), {"duplicates_removed": removed}


# ---------------------------------------------------------------------------
# Utility: summarize a clean DataFrame for the dashboard
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a flat dict of key summary statistics for display."""
    summary: dict[str, Any] = {
        "total_records": len(df),
        "case_ids": df["case_id"].unique().tolist() if "case_id" in df.columns else [],
    }

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        summary["time_start"] = str(df["timestamp"].min())
        summary["time_end"] = str(df["timestamp"].max())
        duration = df["timestamp"].max() - df["timestamp"].min()
        summary["duration_minutes"] = round(duration.total_seconds() / 60, 1)

    if "alarm_active" in df.columns:
        summary["alarm_events"] = int(df["alarm_active"].sum())

    for col in MEASUREMENT_COLS:
        if col in df.columns and df[col].notna().any():
            summary[f"{col}_mean"] = round(df[col].mean(), 2)
            summary[f"{col}_min"] = round(df[col].min(), 2)
            summary[f"{col}_max"] = round(df[col].max(), 2)

    return summary
