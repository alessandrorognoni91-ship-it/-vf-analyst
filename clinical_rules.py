"""
clinical_rules.py — VF Analyst Early Warning System
=====================================================
Evidence-based clinical algorithms for ECMO monitoring.

This module implements two layers of predictive rules:

  LAYER 1 — Fixed thresholds (rule-based)
    Simple if/then rules derived from published clinical guidelines.
    No training data required. Fully transparent and auditable.

  LAYER 2 — Composite clinical scores
    Multi-parameter scores adapted from peer-reviewed literature.
    Coefficients are fixed and published; no ML involved.

Each function returns a pd.Series or pd.DataFrame that can be merged
directly into the clean measurements DataFrame produced by data_cleaning.py.

All thresholds carry an explicit literature reference so clinicians and
engineers can trace every alert back to its source.

IMPORTANT DISCLAIMER
--------------------
This module is intended for RESEARCH AND DEMONSTRATION purposes only.
It has NOT been validated for clinical use and must NOT be used to make
autonomous clinical decisions. All outputs are decision-support signals
that must be interpreted by a qualified clinician.

References (checked against the sources, Nov 2026)
----------
[1] ELSO Guidelines for Adult and Pediatric Membrane Oxygenation Circuits.
    https://elso.org (oxygenator dP rise "regardless of flow" = clot indicator;
    post-membrane PaO2 should exceed ~300 mmHg at FdO2 1.0; no numeric dP,
    low-flow, pump-RPM or temperature limits are given).
[2] Zakhary B. et al. "How I approach membrane lung dysfunction in patients
    receiving ECMO." Crit Care 2020;24:671 (follow dP and dP/flow trend; post-
    membrane PO2 < 200 mmHg suggests failure; rising pump speed for stable flow
    is a surrogate of rising resistance; the authors state no cut-off defines
    dysfunction - values come from expert experience).
[3] Butt W. et al. J Extra Corpor Technol 2024;56:20-29 (narrative review of
    oxygenator/circuit change-out markers; none sufficient on its own).
[4] ELSO VA-ECMO interim guidelines (Lorusso et al.) / ELSO General guidelines:
    flow targets are patient-specific and judged on perfusion markers.

NOT included: the SAVE score (Schmidt et al. Eur Heart J 2015). Almost all its
inputs are pre-ECMO clinical/laboratory values that the device log does not
contain, so no valid score can be computed from it. Use the official calculator.

All numeric thresholds in THRESHOLDS are HEURISTIC, configurable defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# Single source of truth for the alert thresholds. The rule functions below use
# these as defaults and the dashboard (KPI cards, live and CSV) reads the same
# values, so a number shown next to a reading always matches the rule that fires.
THRESHOLDS = {
    # HEURISTIC defaults. ELSO / Zakhary 2020 / Butt 2024 give no numeric cut-off
    # for membrane dP, low flow or pump efficiency: set them per patient/centre.
    "delta_high_mmhg":        60.0,   # sustained membrane dP limit (device/centre specific)
    "resistance_rise_ratio":   1.30,  # dP/flow up 30 % vs 1 h earlier
    "resistance_min_flow_lpm": 1.0,   # resistance only computed above this flow
    "resistance_min_rise":     2.0,   # ... and by at least 2 mmHg per L/min
    "flow_low_lpm":            1.5,   # adult low-flow flag (patient-specific in practice)
    "sat_post_low_pct":       95.0,   # post-membrane SaO2 (PaO2 > 300 at FdO2 1.0 -> ~100 %)
    "pump_idle_rpm":         500.0,   # below this the pump is considered stopped
    "flow_per_rpm_drop":       0.75,  # flow/RPM below 75 % of previous hour
    "temp_low_c":             35.0,
    "temp_high_c":            38.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Result data class — returned by every rule function
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    """
    Standardised output for every clinical rule.

    Attributes
    ----------
    name        : Short machine-readable rule identifier
    label       : Human-readable name shown in the dashboard
    triggered   : Boolean Series — True where the rule fires
    severity    : 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
    description : One-sentence clinical meaning of this alert
    reference   : Literature source for the threshold used
    details     : Optional Series with numeric context (e.g. rate of rise)
    """
    name:        str
    label:       str
    triggered:   pd.Series
    severity:    str
    description: str
    reference:   str
    details:     Optional[pd.Series] = field(default=None)

    def summary(self) -> dict:
        """Return a plain dict suitable for display in Streamlit."""
        n = int(self.triggered.sum())
        pct = round(n / max(len(self.triggered), 1) * 100, 1)
        return {
            "rule":        self.name,
            "label":       self.label,
            "severity":    self.severity,
            "events":      n,
            "rate_pct":    pct,
            "description": self.description,
            "reference":   self.reference,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Fixed threshold rules
# ─────────────────────────────────────────────────────────────────────────────

def _sustained(cond: pd.Series, n: int) -> pd.Series:
    """True where `cond` held for the last n samples (NaN counts as False)."""
    c = cond.fillna(False).astype(int)
    return c.rolling(n, min_periods=n).sum() >= n


def _pump_running(df: pd.DataFrame) -> pd.Series:
    """Pump on = RPM above idle. Without an RPM column assume it is running."""
    if "pump_speed_rpm" in df.columns:
        return df["pump_speed_rpm"].fillna(0) > THRESHOLDS["pump_idle_rpm"]
    return pd.Series(True, index=df.index)


def rule_oxygenator_thrombosis_risk(
    df: pd.DataFrame,
    delta_threshold_mmhg: float = THRESHOLDS["delta_high_mmhg"],
    rise_ratio: float = THRESHOLDS["resistance_rise_ratio"],
    min_flow_lpm: float = THRESHOLDS["resistance_min_flow_lpm"],
    ref_lag_samples: int = 60,
    smooth: int = 10,
) -> RuleResult:
    """
    Oxygenator resistance trend (possible clot formation in the membrane).

    The membrane pressure drop (dP = P_pre - P_post) depends on blood flow, so
    an absolute dP alone is ambiguous. Both ELSO and Zakhary et al. ask for
    the TREND: ELSO calls a dP rise "regardless of flow" a good indicator of
    oxygenator clot; Zakhary et al. follow dP / blood flow (membrane
    resistance). Neither gives a numeric cut-off, so the numbers below are
    configurable HEURISTIC defaults, not guideline values.

      A) resistance R = dP / flow (flow >= min_flow) rose by more than
         `rise_ratio` (default +30 %) versus ~1 h earlier  AND by >= 2 mmHg per L/min
      B) sustained dP above a device/centre-specific absolute limit
         (default 60 mmHg, median of the last samples so one spike cannot fire it)
    """
    col = "pressure_delta_mmhg"
    if col not in df.columns or df[col].isna().all():
        return RuleResult(
            name="oxygenator_thrombosis", label="Oxygenator resistance rise",
            triggered=pd.Series(False, index=df.index), severity="HIGH",
            description="Column pressure_delta_mmhg not available.",
            reference="ELSO Circuit Guidelines; Zakhary et al. Crit Care 2020;24:671",
        )
    dp = df[col]
    sustained_abs = dp.rolling(smooth, min_periods=smooth).median() > delta_threshold_mmhg

    rising = pd.Series(False, index=df.index)
    resistance = pd.Series(np.nan, index=df.index)
    if "flow_rate_lpm" in df.columns:
        flow = df["flow_rate_lpm"]
        resistance = (dp / flow.where(flow >= min_flow_lpm)).where(_pump_running(df))
        cur = resistance.rolling(smooth, min_periods=max(3, smooth // 2)).median()
        ref = cur.shift(ref_lag_samples)
        rising = (cur > ref * rise_ratio) & ((cur - ref) >= THRESHOLDS["resistance_min_rise"])

    return RuleResult(
        name="oxygenator_thrombosis",
        label="Oxygenator resistance rise",
        triggered=(rising | sustained_abs).fillna(False),
        severity="HIGH",
        description=(
            f"Membrane resistance (dP/flow) up > {(rise_ratio - 1) * 100:.0f}% vs 1 h earlier, "
            f"or dP > {delta_threshold_mmhg:g} mmHg sustained. Possible clot in the membrane. "
            "Heuristic limits: ELSO gives no numeric cut-off."
        ),
        reference="ELSO Circuit Guidelines (dP rise = clot indicator); Zakhary et al. Crit Care 2020;24:671",
        details=resistance.rename("dp_per_flow_mmhg_per_lpm"),
    )


def rule_hypoperfusion(
    df: pd.DataFrame,
    flow_threshold_lpm: float = THRESHOLDS["flow_low_lpm"],
    consecutive_samples: int = 5,
) -> RuleResult:
    """
    Low circuit flow while the pump is running.

    Flow adequacy is patient-specific (body size, VA vs VV, weaning) and is
    judged on perfusion markers (SvO2, lactate, end-organ function), not on a
    flow number; ELSO gives no universal minimum. Low flow also raises circuit
    clotting risk. The 1.5 L/min default is a configurable heuristic for adults;
    set it to the patient's target. Pump-off periods and missing data are ignored.
    """
    col = "flow_rate_lpm"
    if col not in df.columns or df[col].isna().all():
        return RuleResult(
            name="hypoperfusion", label="Low circuit flow",
            triggered=pd.Series(False, index=df.index), severity="HIGH",
            description="Column flow_rate_lpm not available.",
            reference="ELSO guidelines (flow targets are patient-specific)",
        )
    low = (df[col] < flow_threshold_lpm) & _pump_running(df)
    return RuleResult(
        name="hypoperfusion",
        label="Low circuit flow",
        triggered=_sustained(low, consecutive_samples),
        severity="HIGH",
        description=(
            f"Blood flow < {flow_threshold_lpm:g} L/min for >= {consecutive_samples} readings "
            "with the pump running. Check perfusion markers; low flow also favours clotting. "
            "Threshold is a heuristic, not an ELSO value."
        ),
        reference="ELSO general/adult guidelines: flow targets are patient-specific",
        details=df[col].rename("flow_rate_lpm"),
    )


def rule_oxygenator_failure(
    df: pd.DataFrame,
    sat_post_threshold_pct: float = THRESHOLDS["sat_post_low_pct"],
    sustained_samples: int = 5,
) -> RuleResult:
    """
    Reduced oxygen transfer: post-oxygenator SaO2 below ~95 % while running.

    Blood leaving a working membrane lung at high FdO2 is ~100 % saturated
    (ELSO: post-membrane PaO2 should exceed ~300 mmHg at FdO2 1.0; Zakhary et
    al.: < 200 mmHg suggests a failing membrane). Saturation is a coarse
    surrogate for PaO2, so a drop below 95 % is already a late sign; it is
    NOT a diagnosis (needs VO2 / blood gas, and depends on FdO2 and flow).
    """
    col = "sat_post_pct"
    if col not in df.columns or df[col].isna().all():
        return RuleResult(
            name="oxygenator_failure", label="Reduced oxygenator gas exchange",
            triggered=pd.Series(False, index=df.index), severity="HIGH",
            description="Column sat_post_pct not available.",
            reference="ELSO Circuit Guidelines; Zakhary et al. Crit Care 2020",
        )
    low = (df[col] < sat_post_threshold_pct) & _pump_running(df)
    return RuleResult(
        name="oxygenator_failure",
        label="Reduced oxygenator gas exchange",
        triggered=_sustained(low, sustained_samples),
        severity="HIGH",
        description=(
            f"Post-oxygenator SatO2 < {sat_post_threshold_pct:g}% for >= {sustained_samples} "
            "readings while the pump runs. Confirm with a post-membrane blood gas "
            "(ELSO: PaO2 > 300 mmHg at FdO2 1.0; < 200 suggests failure)."
        ),
        reference="ELSO Circuit Guidelines; Zakhary et al. Crit Care 2020;24:671",
        details=df[col].rename("sat_post_pct"),
    )


def rule_pump_cavitation_risk(
    df: pd.DataFrame,
    drop_ratio: float = THRESHOLDS["flow_per_rpm_drop"],
    window: int = 10,
    ref_window: int = 60,
) -> RuleResult:
    """
    Pump efficiency loss: less flow than before at the same pump speed.

    The device log has no inlet (negative) pressure, so cavitation/chatter
    cannot be detected directly (ELSO gives no numeric limit either). What
    the log does show is flow per RPM: Zakhary et al. note that a rising pump
    speed needed to hold flow is a surrogate for rising resistance, and
    falling flow at constant RPM is typical of drainage problems
    (hypovolaemia, kinking, cannula position). Rule: flow/RPM below `drop_ratio`
    (default 75 %) of its own median over the previous hour.
    """
    if "pump_speed_rpm" not in df.columns or "flow_rate_lpm" not in df.columns:
        return RuleResult(
            name="pump_cavitation", label="Pump efficiency loss",
            triggered=pd.Series(False, index=df.index), severity="MODERATE",
            description="Required columns (pump_speed_rpm, flow_rate_lpm) not available.",
            reference="Zakhary et al. Crit Care 2020",
        )
    rpm, flow = df["pump_speed_rpm"], df["flow_rate_lpm"]
    ratio = (flow / rpm).where((rpm > THRESHOLDS["pump_idle_rpm"]) & (flow > 0.3))
    cur = ratio.rolling(window, min_periods=max(3, window // 2)).median()
    base = ratio.rolling(ref_window, min_periods=ref_window // 2).median().shift(window)
    return RuleResult(
        name="pump_cavitation",
        label="Pump efficiency loss",
        triggered=(cur < base * drop_ratio).fillna(False),
        severity="MODERATE",
        description=(
            f"Flow per RPM fell below {drop_ratio * 100:.0f}% of the previous hour. Possible "
            "drainage limitation (volume, kinking, cannula position) or rising circuit "
            "resistance. Cavitation itself needs inlet pressure, not in the log."
        ),
        reference="Zakhary et al. Crit Care 2020;24:671; ELSO Circuit Guidelines (no numeric limit)",
        details=ratio.rename("flow_per_rpm"),
    )


def rule_circuit_instability(
    df: pd.DataFrame,
    flow_cv_threshold: float = 0.15,
    window: int = 30,
) -> RuleResult:
    """
    Unstable blood flow: coefficient of variation of flow over `window` samples.

    The earlier version also flagged dP variability; dP is intrinsically noisy
    (CV > 20 % in 64 % of a real case log) so it carried no information and
    was removed. This is a generic signal-quality indicator, not a guideline
    criterion; it is a pointer to check for drainage problems/chatter.
    """
    if "flow_rate_lpm" not in df.columns or df["flow_rate_lpm"].isna().all():
        return RuleResult(
            name="circuit_instability", label="Unstable flow",
            triggered=pd.Series(False, index=df.index), severity="MODERATE",
            description="Column flow_rate_lpm not available.", reference="Monitoring heuristic",
        )
    s = df["flow_rate_lpm"].where(_pump_running(df))
    mean = s.rolling(window, min_periods=window // 2).mean()
    cv = s.rolling(window, min_periods=window // 2).std() / mean.where(mean > 0.5)
    return RuleResult(
        name="circuit_instability",
        label="Unstable flow",
        triggered=(cv > flow_cv_threshold).fillna(False),
        severity="MODERATE",
        description=(
            f"Flow variability (CV) > {flow_cv_threshold * 100:.0f}% over {window} readings. "
            "Check drainage, cannula position, volume status. Heuristic signal-quality flag."
        ),
        reference="Monitoring heuristic (not a guideline criterion)",
        details=cv.rename("flow_cv"),
    )


def rule_thermal_alert(
    df: pd.DataFrame,
    temp_high_c: float = THRESHOLDS["temp_high_c"],
    temp_low_c:  float = THRESHOLDS["temp_low_c"],
    sustained_samples: int = 5,
) -> RuleResult:
    """
    Detect abnormal post-oxygenator blood temperature.

    Fires when temperature exits the range [35.0 °C, 38.5 °C]
    for ≥ 5 consecutive samples.

    Clinical meaning:
      - Temperature > 38.5 °C: Possible circuit-induced inflammation,
        haemolysis, or infection. Also increases metabolic demand.
      - Temperature < 35.0 °C: Risk of coagulopathy and cardiac
        arrhythmia (hypothermia-induced).

    Reference: ELSO thermal management recommendations;
    standard ECMO perfusion practice.
    """
    col = "temp_post_c"
    if col not in df.columns or df[col].isna().all():
        empty = pd.Series(False, index=df.index)
        return RuleResult(
            name="thermal_alert",
            label="Blood temperature alert",
            triggered=empty,
            severity="LOW",
            description="Column temp_post_c not available.",
            reference="No ELSO numeric target; context-dependent",
        )

    series      = df[col].ffill()
    out_of_range = (series > temp_high_c) | (series < temp_low_c)
    rolling      = out_of_range.rolling(sustained_samples, min_periods=sustained_samples).sum()
    triggered    = rolling >= sustained_samples

    return RuleResult(
        name="thermal_alert",
        label="Blood temperature alert",
        triggered=triggered,
        severity="LOW",
        description=(
            f"Post-oxygenator temperature outside [{temp_low_c}–{temp_high_c} °C] "
            f"for ≥ {sustained_samples} readings. "
            "Risk of coagulopathy (low) or haemolysis/inflammation (high)."
        ),
        reference="No ELSO numeric target; context-dependent (e.g. intentional hypothermia after arrest)",
        details=series.rename("temp_post_c"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate runner — applies all rules to a DataFrame in one call
# ─────────────────────────────────────────────────────────────────────────────

def run_all_rules(
    df: pd.DataFrame,
    save_kwargs: Optional[dict] = None,  # kept for compatibility; ignored (SAVE removed)
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run all Layer 1 rules and optionally the SAVE score on a clean DataFrame.

    Parameters
    ----------
    df           : Clean measurements DataFrame from data_cleaning.clean()
    save_kwargs  : Optional dict of patient parameters for score_save_adapted()
                   e.g. {"age": 55, "weight_kg": 75, "acute_myocarditis": False}

    Returns
    -------
    enriched_df  : Original DataFrame with additional columns:
                     alert_<rule_name>  (bool)  — rule triggered
                     alert_any          (bool)  — any rule triggered
                     save_score_partial (float) — partial SAVE score
                     save_risk_level    (str)   — LOW/MODERATE/HIGH/CRITICAL
                     save_survival_est  (str)   — estimated survival %
    summaries    : List of dicts from RuleResult.summary() for dashboard display
    """
    df_out    = df.copy()
    summaries = []

    # ── Layer 1 rules ─────────────────────────────────────────────────────────
    rules = [
        rule_oxygenator_thrombosis_risk(df_out),
        rule_hypoperfusion(df_out),
        rule_oxygenator_failure(df_out),
        rule_pump_cavitation_risk(df_out),
        rule_circuit_instability(df_out),
        rule_thermal_alert(df_out),
    ]

    for rule in rules:
        col = f"alert_{rule.name}"
        df_out[col] = rule.triggered.astype(int)
        summaries.append(rule.summary())

    # Aggregate: any rule triggered
    alert_cols     = [f"alert_{r.name}" for r in rules]
    df_out["alert_any"] = df_out[alert_cols].any(axis=1).astype(int)

    return df_out, summaries


# ─────────────────────────────────────────────────────────────────────────────
# Severity colour mapping (for dashboard use)
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "LOW":      "#F0A500",   # amber
    "MODERATE": "#E8392A",   # red-orange
    "HIGH":     "#C0392B",   # red
    "CRITICAL": "#7B0000",   # dark red
}

SEVERITY_ORDER = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def severity_rank(s: str) -> int:
    """Return numeric rank for sorting (0 = lowest)."""
    try:
        return SEVERITY_ORDER.index(s)
    except ValueError:
        return -1
