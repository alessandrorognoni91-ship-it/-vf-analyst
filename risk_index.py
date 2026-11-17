"""
risk_index.py - three headline indicators for the clinician cockpit.

UI independent; works identically on live data and on a CSV from the console
(both are brought to 1 sample/min by plot_prep.normalize_rate first).

  clot         oxygenator / circuit clot index (0-100)
  cavitation   pump suction / cavitation index (0-100, INDIRECT)
  neuro        brain-injury risk estimate from published registry odds ratios

HONEST LIMITS (also shown in the app):
* The 0-100 values are transparent heuristic INDICES built on the trends that
  ELSO (Circuit guideline) and Zakhary et al. (Crit Care 2020) describe. They
  are not probabilities and not validated.
* Cavitation cannot be measured directly: the log has no inlet (negative)
  pressure. The index uses flow-per-RPM loss and flow instability.
* Stroke risk cannot be derived from the device log. The neuro estimate starts
  from registry incidences and multiplies adjusted odds ratios of the factors
  the clinician ticks (Kalra et al., ELSO registry, peripheral VA-ECMO with
  cardiogenic shock, n=9,807; 2023 preprint, not peer reviewed). It does not
  apply to VV-ECMO, children or other populations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_rules import THRESHOLDS as TH

LEVELS = [(33.0, "LOW"), (66.0, "MODERATE"), (101.0, "HIGH")]


def level(score: float) -> str:
    if score is None or score != score:
        return "n/a"
    for hi, name in LEVELS:
        if score < hi:
            return name
    return "HIGH"


def _running(d: pd.DataFrame) -> pd.Series:
    if "pump_speed_rpm" in d.columns:
        return d["pump_speed_rpm"].fillna(0) > TH["pump_idle_rpm"]
    return pd.Series(True, index=d.index)


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(0, 1).fillna(0)


def clot_components(d: pd.DataFrame) -> pd.DataFrame:
    """Three clot-related signals, each 0-1."""
    out = pd.DataFrame(index=d.index)
    run = _running(d)
    if "pressure_delta_mmhg" in d.columns and "flow_rate_lpm" in d.columns:
        dp, flow = d["pressure_delta_mmhg"], d["flow_rate_lpm"]
        res = (dp / flow.where(flow >= TH["resistance_min_flow_lpm"])).where(run)
        cur = res.rolling(10, min_periods=5).median()
        rise = cur / cur.shift(60) - 1.0
        # +30 % vs 1 h earlier (rule threshold) -> 0.5 ; +60 % -> 1.0
        out["Membrane resistance (dP/flow) rising"] = _clip01(rise / (2 * (TH["resistance_rise_ratio"] - 1)))
        dpm = dp.rolling(10, min_periods=5).median()
        out["Membrane dP high"] = _clip01((dpm - 0.5 * TH["delta_high_mmhg"]) / (0.5 * TH["delta_high_mmhg"]))
    if "flow_rate_lpm" in d.columns:
        low = ((d["flow_rate_lpm"] < TH["flow_low_lpm"]) & run).astype(float)
        out["Low flow (stasis)"] = _clip01(low.rolling(30, min_periods=10).mean()) * 0.6
    return out


def cavitation_components(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if "pump_speed_rpm" in d.columns and "flow_rate_lpm" in d.columns:
        rpm, flow = d["pump_speed_rpm"], d["flow_rate_lpm"]
        ratio = (flow / rpm).where((rpm > TH["pump_idle_rpm"]) & (flow > 0.3))
        cur = ratio.rolling(10, min_periods=5).median()
        base = ratio.rolling(60, min_periods=30).median().shift(10)
        drop = 1.0 - cur / base
        # -25 % flow per RPM (rule threshold) -> 0.5 ; -50 % -> 1.0
        out["Flow per RPM falling"] = _clip01(drop / (2 * (1 - TH["flow_per_rpm_drop"])))
    return out


def instability_components(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if "flow_rate_lpm" in d.columns:
        s = d["flow_rate_lpm"].where(_running(d))
        m = s.rolling(30, min_periods=15).mean()
        cv = s.rolling(30, min_periods=15).std() / m.where(m > 0.5)
        out["Flow variability (CV)"] = _clip01(cv / 0.30)          # 15 % CV -> 0.5
    return out


def gas_components(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if "sat_post_pct" in d.columns:
        s = d["sat_post_pct"].where(_running(d)).rolling(5, min_periods=3).median()
        out["Post-oxygenator SatO2 low"] = _clip01((100.0 - s) / (2 * (100.0 - TH["sat_post_low_pct"])))
    return out


def lowflow_components(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if "flow_rate_lpm" in d.columns:
        f = d["flow_rate_lpm"].where(_running(d)).rolling(5, min_periods=3).median()
        lim = TH["flow_low_lpm"]
        out["Blood flow low"] = _clip01((2 * lim - f) / (2 * lim))   # at the limit -> 0.5
    return out


def temperature_components(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if "temp_post_c" in d.columns:
        t = d["temp_post_c"].rolling(5, min_periods=3).median()
        dev = (TH["temp_low_c"] - t).clip(lower=0) + (t - TH["temp_high_c"]).clip(lower=0)
        out["Temperature out of range"] = _clip01(dev / 2.0)         # 1 C outside -> 0.5
    return out


def alarm_components(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if "alarm_active" in d.columns:
        frac = d["alarm_active"].fillna(0).rolling(30, min_periods=10).mean()
        out["Device alarms active"] = _clip01(frac / 0.5)            # alarm half the time -> 1
    return out


def _summarise(comp: pd.DataFrame) -> dict:
    if comp.empty:
        return {"available": False}
    score = (100 * comp.max(axis=1)).round(0)
    last = float(score.iloc[-1])
    prev = score.iloc[-30:-15]
    recent = score.iloc[-15:]
    delta = float(recent.mean() - prev.mean()) if len(prev) >= 5 else 0.0
    trend = "rising" if delta > 8 else "falling" if delta < -8 else "steady"
    row = comp.iloc[-1]
    driver = row.idxmax() if row.max() > 0.05 else "No sign of concern"
    peak_i = int(np.argmax(score.values))
    return {
        "available": True,
        "score": last, "level": level(last), "trend": trend, "driver": driver,
        "peak": float(score.max()), "peak_level": level(float(score.max())),
        "peak_time": comp.index[peak_i],
        "pct_high": float((score >= LEVELS[1][0]).mean() * 100),
        "series": score,
    }


INDICATORS = [
    # key, title, component function
    ("clot", "Clot", clot_components),
    ("cavitation", "Cavitation / suction", cavitation_components),
    ("gas", "Gas exchange", gas_components),
    ("lowflow", "Low flow", lowflow_components),
    ("instability", "Flow instability", instability_components),
    ("temperature", "Temperature", temperature_components),
    ("alarms", "Device alarms", alarm_components),
]


def assess(d: pd.DataFrame) -> dict:
    """d: normalised (1/min) dataframe with a 'timestamp' column. Returns one summary per indicator."""
    d = d.reset_index(drop=True)
    out = {}
    for key, title, fn in INDICATORS:
        r = _summarise(fn(d))
        r["title"] = title
        if r.get("available") and "timestamp" in d.columns:
            r["peak_time"] = d["timestamp"].iloc[r["peak_time"]]
            r["series_time"] = d["timestamp"]
        out[key] = r
    return out


# ── Neuro estimate (registry odds ratios) ────────────────────────────────────
# Kalra A. et al., ELSO Registry, peripheral VA-ECMO, cardiogenic shock, n=9,807
# (Research Square preprint 2023, PMID 38045281; NOT peer reviewed).
NEURO = {
    "abi":      {"label": "Any acute brain injury", "base": 0.111,
                 "or": {"low_pp": 1.25, "hemolysis": 1.81, "arrest": 2.05,
                        "arrhythmia": 1.41, "rrt": 1.41}},
    "ischemia": {"label": "CNS ischaemia", "base": 0.062,
                 "or": {"low_pp": 1.26, "arrest": 2.21, "arrhythmia": 1.55}},
    "ich":      {"label": "Intracranial haemorrhage", "base": 0.032,
                 "or": {"hemolysis": 1.80, "rrt": 1.73, "arrest": 1.41,
                        "circuit_failure": 1.58}},
}
FACTOR_LABELS = {
    "low_pp": "Arterial pulse pressure <= 10 mmHg",
    "hemolysis": "Haemolysis (raised plasma free Hb)",
    "arrest": "Cardiac arrest before ECMO",
    "arrhythmia": "Arrhythmia",
    "rrt": "Renal replacement therapy",
    "circuit_failure": "Circuit failure / clot (set automatically from the clot index)",
}


def neuro_estimate(factors: dict) -> dict:
    """Odds-ratio adjustment of the registry base rate. factors: {name: bool}."""
    out = {}
    for key, m in NEURO.items():
        odds = m["base"] / (1 - m["base"])
        used = []
        for f, orr in m["or"].items():
            if factors.get(f):
                odds *= orr
                used.append(f)
        p = odds / (1 + odds)
        out[key] = {"label": m["label"], "base": m["base"], "p": p,
                    "ratio": p / m["base"], "used": used}
    r = max(out[k]["ratio"] for k in NEURO)   # worst relative increase over any outcome
    out["level"] = "BASELINE" if r < 1.05 else "LOW" if r < 1.5 else "MODERATE" if r < 2.5 else "HIGH"
    return out
