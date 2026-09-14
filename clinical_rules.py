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

References
----------
[1] Monitoring the ECMO - PMC/NIH; Vercaemst L. et al.
    ASAIO J. 2010;56(3):159-66.
[2] ELSO Guidelines for Adult VV-ECMO. Tonna et al. 2021.
    https://www.elso.org
[3] ELSO Interim Guidelines for VA-ECMO. Lorusso et al. 2021.
    https://www.elso.org
[4] Schmidt M. et al. "Predicting survival after ECMO for refractory
    cardiogenic shock: the SAVE-score."
    Eur Heart J. 2015;36(33):2246-56. PMID: 26033984.
[5] Anticoagulation-free VV ECMO trial. ClinicalTrials.gov NCT04273607.
    Transmembrane pressure threshold > 10 mmHg/L/min.
[6] Impact of Pulse Pressure on Acute Brain Injury in VA-ECMO.
    ELSO Registry analysis. PMID: 38024227.
[7] Zachary B, Vercaemst L, et al. "How I approach membrane lung
    dysfunction." Critical Care 2020;24:671.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


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

def rule_oxygenator_thrombosis_risk(
    df: pd.DataFrame,
    delta_threshold_mmhg: float = 60.0,
    rate_threshold_mmhg_per_min: float = 20.0,
    window_minutes: int = 60,
    samples_per_minute: float = 1.0,
) -> RuleResult:
    """
    Detect risk of oxygenator thrombosis via transmembrane pressure (TMP) rise.

    Two complementary sub-rules (either triggers the alert):
      A) Absolute TMP > 60 mmHg  [Ref 1, Zachary 2020]
      B) Rate of rise > 20 mmHg in 60 minutes  [Ref 1]

    Clinical meaning: Rising TMP at constant flow indicates increasing
    resistance in the oxygenator, which is the earliest measurable sign
    of clot formation in the membrane fibres.

    Parameters
    ----------
    delta_threshold_mmhg        : Absolute TMP threshold [mmHg]
    rate_threshold_mmhg_per_min : Rate-of-rise threshold [mmHg/min]
    window_minutes              : Rolling window for rate calculation
    samples_per_minute          : Approximate sampling frequency
    """
    col = "pressure_delta_mmhg"
    if col not in df.columns or df[col].isna().all():
        empty = pd.Series(False, index=df.index)
        return RuleResult(
            name="oxygenator_thrombosis",
            label="Oxygenator thrombosis risk",
            triggered=empty,
            severity="CRITICAL",
            description="Column pressure_delta_mmhg not available.",
            reference="Vercaemst 2010 / ELSO 2021",
        )

    series = df[col].ffill().fillna(0)

    # Sub-rule A: absolute threshold
    above_absolute = series > delta_threshold_mmhg

    # Sub-rule B: rate of rise over the rolling window
    window_samples = max(1, int(window_minutes * samples_per_minute))
    rate_of_rise   = series.diff(window_samples) / window_minutes  # mmHg/min
    rising_fast    = rate_of_rise > rate_threshold_mmhg_per_min

    triggered = above_absolute | rising_fast

    return RuleResult(
        name="oxygenator_thrombosis",
        label="Oxygenator thrombosis risk",
        triggered=triggered,
        severity="CRITICAL",
        description=(
            f"TMP > {delta_threshold_mmhg} mmHg OR rising > "
            f"{rate_threshold_mmhg_per_min} mmHg/min over {window_minutes} min. "
            "Suggests clot formation in the oxygenator membrane."
        ),
        reference="Vercaemst L. et al. ASAIO J 2010; Zachary et al. Crit Care 2020",
        details=rate_of_rise.rename("tmp_rate_mmhg_per_min"),
    )


def rule_hypoperfusion(
    df: pd.DataFrame,
    flow_threshold_lpm: float = 1.5,
    consecutive_samples: int = 5,
) -> RuleResult:
    """
    Detect critical hypoperfusion from low ECMO blood flow.

    Threshold: flow < 1.5 L/min for ≥ 5 consecutive samples.

    Clinical meaning: Flow < 1.5 L/min indicates systemic perfusion
    insufficient to meet metabolic demands at rest, especially in adult
    patients. Sustained low flow raises risk of end-organ ischaemia.

    Reference: ELSO Red Book 5th ed.; ELSO VV-ECMO Guidelines 2021
    (minimum flow 240 ml/m²/min for gas exchange).
    """
    col = "flow_rate_lpm"
    if col not in df.columns or df[col].isna().all():
        empty = pd.Series(False, index=df.index)
        return RuleResult(
            name="hypoperfusion",
            label="Critical hypoperfusion",
            triggered=empty,
            severity="HIGH",
            description="Column flow_rate_lpm not available.",
            reference="ELSO Guidelines 2021",
        )

    series   = df[col].ffill().fillna(0)
    low_flow = series < flow_threshold_lpm

    # Must be sustained for N consecutive samples
    rolling_sum = low_flow.rolling(consecutive_samples, min_periods=consecutive_samples).sum()
    triggered   = rolling_sum >= consecutive_samples

    return RuleResult(
        name="hypoperfusion",
        label="Critical hypoperfusion",
        triggered=triggered,
        severity="HIGH",
        description=(
            f"Blood flow < {flow_threshold_lpm} L/min for "
            f"≥ {consecutive_samples} consecutive readings. "
            "Risk of end-organ ischaemia."
        ),
        reference="ELSO Red Book 5th ed.; ELSO VV-ECMO Guidelines 2021",
        details=series.rename("flow_rate_lpm"),
    )


def rule_oxygenator_failure(
    df: pd.DataFrame,
    sat_post_threshold_pct: float = 75.0,
    sustained_samples: int = 10,
) -> RuleResult:
    """
    Detect progressive oxygenator failure via post-membrane saturation drop.

    Threshold: post-oxygenator SatO₂ < 75% for ≥ 10 consecutive samples.

    Clinical meaning: Post-membrane saturation reflects the gas-exchange
    efficiency of the oxygenator. A sustained drop below 75% at constant
    sweep gas settings indicates membrane deterioration or clotting.

    Reference: ELSO Guidelines; Zachary et al. Critical Care 2020;24:671.
    Also: ClinicalTrials NCT04273607 — membrane PaO2/FiO2 < 200 mmHg
    as dysfunction criterion.
    """
    col = "sat_post_pct"
    if col not in df.columns or df[col].isna().all():
        empty = pd.Series(False, index=df.index)
        return RuleResult(
            name="oxygenator_failure",
            label="Oxygenator gas-exchange failure",
            triggered=empty,
            severity="HIGH",
            description="Column sat_post_pct not available.",
            reference="Zachary et al. Crit Care 2020",
        )

    series    = df[col].ffill().fillna(100)
    low_sat   = series < sat_post_threshold_pct
    rolling   = low_sat.rolling(sustained_samples, min_periods=sustained_samples).sum()
    triggered = rolling >= sustained_samples

    return RuleResult(
        name="oxygenator_failure",
        label="Oxygenator gas-exchange failure",
        triggered=triggered,
        severity="HIGH",
        description=(
            f"Post-oxygenator SatO₂ < {sat_post_threshold_pct}% "
            f"for ≥ {sustained_samples} consecutive readings. "
            "Progressive oxygenator membrane deterioration or clotting."
        ),
        reference="Zachary et al. Critical Care 2020;24:671; NCT04273607",
        details=series.rename("sat_post_pct"),
    )


def rule_pump_cavitation_risk(
    df: pd.DataFrame,
    rpm_threshold: float = 4500.0,
    flow_ratio_threshold: float = 0.80,
    window: int = 10,
) -> RuleResult:
    """
    Detect pump cavitation risk: high RPM with disproportionately low flow.

    Rule fires when:
      - Pump speed > 4500 RPM (high centrifugal stress)  AND
      - Flow / expected_flow_at_rpm < 0.80
        where expected_flow is estimated from RPM (0.001 L/min per RPM)

    Clinical meaning: If the pump spins fast but delivers less flow than
    expected, the likely causes are: venous drainage obstruction (suction
    event / chattering), hypovolaemia, or cannula malposition.
    Sustained cavitation can cause haemolysis.

    Reference: ELSO Circuit Guidelines 2022 (Gajkowski et al.);
    ELSO Red Book — centrifugal pump management.
    """
    rpm_col  = "pump_speed_rpm"
    flow_col = "flow_rate_lpm"

    if rpm_col not in df.columns or flow_col not in df.columns:
        empty = pd.Series(False, index=df.index)
        return RuleResult(
            name="pump_cavitation",
            label="Pump cavitation / suction event risk",
            triggered=empty,
            severity="MODERATE",
            description="Required columns (pump_speed_rpm, flow_rate_lpm) not available.",
            reference="ELSO Circuit Guidelines 2022",
        )

    rpm  = df[rpm_col].ffill().fillna(0)
    flow = df[flow_col].ffill().fillna(0)

    # Simple linear estimate: ~0.001 L/min per RPM (order-of-magnitude only)
    expected_flow = rpm * 0.001
    safe_expected = expected_flow.replace(0, np.nan)
    flow_ratio    = flow / safe_expected

    high_rpm        = rpm > rpm_threshold
    low_flow_ratio  = flow_ratio < flow_ratio_threshold

    # Smooth both conditions over a short window to reduce noise
    high_rpm_smooth = high_rpm.rolling(window, min_periods=1).mean() > 0.5
    low_flow_smooth = low_flow_ratio.rolling(window, min_periods=1).mean() > 0.5

    triggered = high_rpm_smooth & low_flow_smooth

    return RuleResult(
        name="pump_cavitation",
        label="Pump cavitation / suction event risk",
        triggered=triggered,
        severity="MODERATE",
        description=(
            f"Pump speed > {rpm_threshold:.0f} RPM with flow efficiency "
            f"< {flow_ratio_threshold*100:.0f}% of expected. "
            "Possible suction event, hypovolaemia, or cannula obstruction."
        ),
        reference="ELSO Circuit Guidelines 2022; Gajkowski et al. ASAIO J 2022",
        details=flow_ratio.rename("flow_efficiency_ratio"),
    )


def rule_circuit_instability(
    df: pd.DataFrame,
    flow_cv_threshold: float = 0.15,
    pressure_cv_threshold: float = 0.20,
    window: int = 30,
) -> RuleResult:
    """
    Detect overall circuit instability via coefficient of variation (CV)
    of flow rate and pressure delta over a rolling window.

    Rule fires when either:
      - CV(flow_rate) > 15%  over last N samples  OR
      - CV(pressure_delta) > 20%  over last N samples

    Clinical meaning: High variability in flow or pressure at constant
    RPM settings suggests an unstable circuit — possible causes include
    recirculation, cardiac arrhythmia interference, or clot fragments
    transiently obstructing the circuit.

    Reference: Clinical monitoring practice; ELSO Red Book 5th ed. Ch 5.
    """
    results = pd.Series(False, index=df.index)
    details = pd.Series(0.0, index=df.index)

    for col, thresh in [
        ("flow_rate_lpm", flow_cv_threshold),
        ("pressure_delta_mmhg", pressure_cv_threshold),
    ]:
        if col not in df.columns or df[col].isna().all():
            continue
        s    = df[col].ffill()
        mean = s.rolling(window, min_periods=window // 2).mean()
        std  = s.rolling(window, min_periods=window // 2).std()
        cv   = (std / mean.replace(0, np.nan)).fillna(0)
        results = results | (cv > thresh)
        details = details.combine(cv, max)

    return RuleResult(
        name="circuit_instability",
        label="Circuit instability",
        triggered=results,
        severity="MODERATE",
        description=(
            f"High variability in flow (CV > {flow_cv_threshold*100:.0f}%) "
            f"or pressure delta (CV > {pressure_cv_threshold*100:.0f}%) "
            f"over {window} samples. Possible recirculation or clot fragments."
        ),
        reference="ELSO Red Book 5th ed. Ch 5; clinical monitoring practice",
        details=details.rename("max_cv"),
    )


def rule_thermal_alert(
    df: pd.DataFrame,
    temp_high_c: float = 38.5,
    temp_low_c:  float = 35.0,
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
            reference="ELSO thermal management guidelines",
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
        reference="ELSO thermal management recommendations",
        details=series.rename("temp_post_c"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Composite clinical score (SAVE-adapted)
# ─────────────────────────────────────────────────────────────────────────────

def score_save_adapted(
    df: pd.DataFrame,
    age: Optional[float] = None,
    weight_kg: Optional[float] = None,
    acute_myocarditis: bool = False,
    vt_vf: bool = False,
    post_cardiac_surgery: bool = False,
) -> pd.DataFrame:
    """
    Compute a simplified, ECMO-device-observable adaptation of the SAVE score.

    IMPORTANT: This is a *partial* SAVE score using only the parameters
    available from the VitalFlow device log. Several SAVE variables
    (lactate, creatinine, bicarbonate) require laboratory data not present
    in the device CSV. The result is an *indicative* risk estimate, not
    the full validated score.

    Full SAVE score reference:
    Schmidt M. et al. Eur Heart J. 2015;36(33):2246-56. PMID: 26033984.
    Calculator: www.save-score.com

    Score interpretation (from Schmidt 2015):
      > 5   → LOW risk    (survival ~75%)
      1–5   → MODERATE risk (survival ~58%)
      -4–0  → HIGH risk   (survival ~42%)
      < -4  → CRITICAL risk (survival ~18%)

    Parameters
    ----------
    df                  : Clean measurements DataFrame (one session)
    age                 : Patient age in years (optional — improves score)
    weight_kg           : Patient weight in kg (optional)
    acute_myocarditis   : Whether the indication was acute myocarditis
    vt_vf               : Whether the indication was refractory VT/VF
    post_cardiac_surgery: Whether the indication was post-cardiac surgery

    Returns
    -------
    pd.DataFrame with columns:
      save_score_partial  : Numeric score (sum of available components)
      save_risk_level     : 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
      save_survival_est   : Estimated survival % (from Schmidt 2015 table)
      save_components_n   : Number of score components computed
      save_note           : Warning about partial computation
    """
    n = len(df)
    scores = pd.DataFrame(index=df.index)
    scores["save_score_partial"] = 0.0
    component_count = 0

    # ── Component 1: Age (Schmidt 2015 Table 2) ──────────────────────────────
    if age is not None:
        if age < 18:
            age_pts = 7
        elif age <= 38:
            age_pts = 7
        elif age <= 52:
            age_pts = 4
        elif age <= 62:
            age_pts = 3
        else:
            age_pts = 0
        scores["save_score_partial"] += age_pts
        component_count += 1

    # ── Component 2: Weight (Schmidt 2015) ───────────────────────────────────
    if weight_kg is not None:
        if weight_kg < 65:
            weight_pts = 1
        elif weight_kg <= 89:
            weight_pts = 2
        else:
            weight_pts = 0
        scores["save_score_partial"] += weight_pts
        component_count += 1

    # ── Component 3: Diagnosis bonus (Schmidt 2015) ──────────────────────────
    if acute_myocarditis:
        scores["save_score_partial"] += 3
        component_count += 1
    if vt_vf:
        scores["save_score_partial"] += 2
        component_count += 1
    if post_cardiac_surgery:
        scores["save_score_partial"] -= 3
        component_count += 1

    # ── Component 4: Pre-ECMO flow instability (device-observable proxy) ─────
    # Proxy for haemodynamic instability: using rolling CV of flow rate
    # This is NOT in the original SAVE score — it's a device-observable proxy
    # for the "pre-ECMO cardiac arrest" and "acute MI" components.
    if "flow_rate_lpm" in df.columns:
        flow = df["flow_rate_lpm"].ffill()
        flow_cv = flow.rolling(30, min_periods=10).std() / flow.rolling(30, min_periods=10).mean().replace(0, np.nan)
        # High CV (>25%) → subtract 1 point (proxy for haemodynamic instability)
        scores["save_score_partial"] -= (flow_cv > 0.25).astype(int)
        component_count += 1

    # ── Component 5: Oxygenator pressure proxy for organ perfusion ───────────
    if "pressure_delta_mmhg" in df.columns:
        p_delta = df["pressure_delta_mmhg"].ffill()
        # TMP > 50 mmHg → subtract 1 (proxy for circuit stress)
        scores["save_score_partial"] -= (p_delta > 50).astype(int)
        component_count += 1

    scores["save_components_n"] = component_count

    # ── Risk level classification (Schmidt 2015 Table 3) ─────────────────────
    def classify(s: float) -> str:
        if s > 5:    return "LOW"
        elif s >= 1: return "MODERATE"
        elif s >= -4: return "HIGH"
        else:         return "CRITICAL"

    def survival_est(s: float) -> str:
        if s > 5:    return "~75%"
        elif s >= 1: return "~58%"
        elif s >= -4: return "~42%"
        else:         return "~18%"

    scores["save_risk_level"]    = scores["save_score_partial"].map(classify)
    scores["save_survival_est"]  = scores["save_score_partial"].map(survival_est)
    scores["save_note"] = (
        f"PARTIAL score ({component_count} of 13 SAVE components). "
        "Laboratory values (lactate, creatinine, bicarbonate) not available "
        "from device log. Use full SAVE calculator at www.save-score.com "
        "for clinical decisions."
    )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate runner — applies all rules to a DataFrame in one call
# ─────────────────────────────────────────────────────────────────────────────

def run_all_rules(
    df: pd.DataFrame,
    save_kwargs: Optional[dict] = None,
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

    # ── Layer 2: SAVE score (optional) ────────────────────────────────────────
    save_kw = save_kwargs or {}
    save_df = score_save_adapted(df_out, **save_kw)
    for col in save_df.columns:
        df_out[col] = save_df[col].values

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
