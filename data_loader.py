"""
data_loader.py — VF Analyst Platform
=====================================
Responsible for ingesting CSV files from biomedical devices.

Design principles:
- Never mutate the raw DataFrame; return a copy
- Emit structured warnings (not exceptions) for recoverable issues
- Support heterogeneous column names via config.COLUMN_MAP
- Extensible: add new device formats to config.COLUMN_MAP without changing this file
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from config import COLUMN_MAP, TIMESTAMP_FORMATS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(source: Union[str, Path, io.BytesIO, io.StringIO]) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a CSV file from a file path or a file-like object (Streamlit upload).

    Returns
    -------
    df : pd.DataFrame
        Raw DataFrame with original column names preserved.
    warnings : list[str]
        Human-readable messages about issues detected during loading.
    """
    warnings: list[str] = []

    # --- 1. Read raw bytes ---------------------------------------------------
    try:
        if isinstance(source, (str, Path)):
            raw = pd.read_csv(source, dtype=str, keep_default_na=False)
        else:
            # Streamlit UploadedFile or any file-like
            raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Cannot parse CSV: {exc}") from exc

    # --- 2. Strip whitespace from all string values --------------------------
    raw = raw.apply(lambda col: col.str.strip() if col.dtype == object else col)
    raw.columns = [c.strip() for c in raw.columns]

    # --- 3. Map column names -------------------------------------------------
    raw, unmapped = _map_columns(raw)
    if unmapped:
        warnings.append(
            f"Unrecognised columns (kept as-is): {', '.join(unmapped)}"
        )

    # --- 4. Detect timestamp column ------------------------------------------
    if "timestamp" not in raw.columns:
        warnings.append("No 'Timestamp' column found. Time-series plots will be unavailable.")

    # --- 5. Detect case / device id ------------------------------------------
    if "case_id" not in raw.columns:
        warnings.append("No 'Case Number' column found. Defaulting case_id to 'unknown'.")
        raw["case_id"] = "unknown"

    logger.info("Loaded %d rows, %d columns", len(raw), len(raw.columns))
    return raw, warnings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Rename raw device column names to canonical names defined in config.COLUMN_MAP.
    Columns not found in the map are left unchanged and listed as unmapped.
    """
    rename_dict: dict[str, str] = {}
    unmapped: list[str] = []

    for col in df.columns:
        canonical = COLUMN_MAP.get(col)
        if canonical:
            rename_dict[col] = canonical
        else:
            unmapped.append(col)

    return df.rename(columns=rename_dict), unmapped


def try_parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Attempt to parse a timestamp column using known device formats.
    Falls back to pandas' inference if none match exactly.
    Returns a Series of datetime64 (NaT for unparseable rows).
    """
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            logger.info("Timestamp parsed with format: %s", fmt)
            return parsed
        except (ValueError, TypeError):
            continue

    # Last resort: let pandas infer
    logger.warning("Falling back to pandas auto-inference for timestamps.")
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
