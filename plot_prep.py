"""
plot_prep.py - shared, UI-independent data preparation for CSV and Live views.

Both the CSV dashboard and the Live monitor call the SAME functions here, so a
given dataset is cut, windowed and evaluated identically whatever its source.

  prepare_timeseries   sort by time, drop rows without a timestamp
  find_recordings      split a log into continuous recordings (gap detection)
  insert_gap_breaks    add NaN rows at gaps so lines never join across a gap
  window_options /
  apply_window         the same "time window" choices for CSV and Live
  normalize_rate       bring any sampling rate to the device-log rate (1/min)
                       so clinical rules mean the same thing on both sources
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE_RATE = "1min"          # sampling rate the clinical rules are defined on
RULE_RATE_THRESHOLD_S = 45  # data finer than this is resampled to RULE_RATE
MIN_GAP_S = 600             # a pause longer than 10 min starts a new recording
GAP_FACTOR = 10             # ... or 10x the typical sampling interval


def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted copy with valid timestamps only."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame() if df is None else df.iloc[0:0].copy()
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def median_interval_s(df: pd.DataFrame) -> float:
    if df is None or len(df) < 2:
        return 60.0
    d = df["timestamp"].diff().dt.total_seconds().dropna()
    d = d[d > 0]
    return float(d.median()) if len(d) else 60.0


def gap_threshold_s(df: pd.DataFrame) -> float:
    return max(MIN_GAP_S, GAP_FACTOR * median_interval_s(df))


def find_recordings(df: pd.DataFrame) -> list[dict]:
    """Continuous recordings: runs of samples separated by less than the gap threshold."""
    if df is None or df.empty:
        return []
    gap = df["timestamp"].diff().dt.total_seconds().fillna(0)
    run = (gap > gap_threshold_s(df)).cumsum()
    recs = []
    for k, part in df.groupby(run, sort=True):
        recs.append({
            "start": part["timestamp"].iloc[0],
            "end": part["timestamp"].iloc[-1],
            "n": int(len(part)),
        })
    return recs


def recording_label(i: int, rec: dict) -> str:
    mins = (rec["end"] - rec["start"]).total_seconds() / 60
    dur = f"{mins:.0f} min" if mins < 180 else f"{mins/60:.1f} h"
    return f"Recording {i + 1} · {rec['start']:%d %b %H:%M} → {rec['end']:%d %b %H:%M} · {dur}"


def gap_bounds(recs: list[dict]) -> list[tuple]:
    """(gap_start, gap_end) between consecutive recordings."""
    return [(recs[i]["end"], recs[i + 1]["start"]) for i in range(len(recs) - 1)]


def insert_gap_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """Insert one all-NaN row inside every gap so line traces do not bridge it."""
    if df is None or len(df) < 2:
        return df
    gap = df["timestamp"].diff().dt.total_seconds()
    idx = gap.index[gap > gap_threshold_s(df)]
    if len(idx) == 0:
        return df
    rows = []
    for i in idx:
        r = {c: np.nan for c in df.columns}
        r["timestamp"] = df.loc[i, "timestamp"] - pd.Timedelta(milliseconds=1)
        rows.append(r)
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def window_options(span_s: float, live: bool) -> list[tuple[str, float | None]]:
    """Same choices for both sources; only options shorter than the data are offered."""
    opts: list[tuple[str, float | None]] = []
    for label, s in [("Last 5 min", 300), ("Last 15 min", 900), ("Last 30 min", 1800),
                     ("Last 1 h", 3600), ("Last 6 h", 21600), ("Last 24 h", 86400)]:
        if s < span_s * 0.95:
            opts.append((label, s))
    whole = ("Whole recording" if not live else "Everything buffered", None)
    return opts + [whole]


def apply_window(df: pd.DataFrame, seconds: float | None) -> pd.DataFrame:
    if seconds is None or df is None or df.empty:
        return df
    return df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(seconds=seconds)]


def normalize_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring data to the device-log rate (1 sample/min) when it is finer.

    Clinical rules count SAMPLES (e.g. 'flow < 1.5 for 5 samples'). A 1 Hz live
    stream would otherwise be judged on 5 seconds where the CSV log is judged on
    5 minutes. Resampling live data to 1-minute medians makes both identical.
    Data already at ~1/min is returned unchanged.
    """
    d = prepare_timeseries(df)
    if d.empty or median_interval_s(d) >= RULE_RATE_THRESHOLD_S:
        return d
    num = d.select_dtypes("number").columns.tolist()
    agg = {c: "median" for c in num}
    if "alarm_active" in agg:
        agg["alarm_active"] = "max"
    for c in d.columns:
        if c not in agg and c != "timestamp":
            agg[c] = "first"
    r = d.set_index("timestamp").resample(RULE_RATE).agg(agg)
    r = r.dropna(how="all", subset=num)          # no filler rows across long gaps
    return r.reset_index()
