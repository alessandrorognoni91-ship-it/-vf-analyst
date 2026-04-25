"""
app.py — VF Analyst Clinical Dashboard  v3.0
==============================================
UX/UI Refinement pass:
  - Top-of-page case navigator with Prev/Next buttons
  - Enriched case header card (duration, alarm summary, time range)
  - All charts: larger fonts, proper axis labels with units, titles
  - Consistent color semantics: blue=measurement, red=alarm, green=ok
  - Clinical interpretation note above every chart
  - Overview tab restructured: critical KPIs → trend sparklines → table
  - Vital Trends: sensible defaults (flow+pressure), alarm shading bands
  - Alarm tab: timeline as primary view, details in expander
  - No functional/data-processing changes from v2
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import MEASUREMENT_COLS, DISPLAY_LABELS
from data_loader import load_csv
from data_cleaning import clean, compute_summary
from data_model import VFDatabase
from ml_model import VFModel

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VF Analyst · Clinical Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens  (single source of truth for all colours & chart styling)
# ─────────────────────────────────────────────────────────────────────────────

C = {
    # Brand
    "primary":    "#1B4F72",   # navy — headings, strong text
    "accent":     "#2E86C1",   # mid-blue — all measurement lines
    "accent_fill":"rgba(46,134,193,0.10)",

    # Semantic
    "alarm":      "#C0392B",   # red — alarms ONLY
    "alarm_fill": "rgba(192,57,43,0.10)",
    "warn":       "#CA6F1E",   # amber — moderate risk / warnings
    "ok":         "#1E8449",   # green — normal / no-alarm

    # Layout
    "subtle":     "#7F8C8D",   # grey — secondary text
    "grid":       "#EDF0F2",   # very light — chart gridlines
    "border":     "#D5D8DC",   # card borders
    "bg_card":    "#F8F9FA",   # card backgrounds
}

# Distinct colours for multi-session overlays — blue first, then diverging
SESSION_COLORS = ["#2E86C1", "#1E8449", "#8E44AD", "#CA6F1E", "#C0392B"]

# Shared chart layout — applied to every figure via _base_fig()
# Larger fonts, more breathing room, clean white background
CHART_FONT = dict(family="'Segoe UI', Arial, sans-serif", size=14, color=C["primary"])
CHART_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=CHART_FONT,
    title_font=dict(family="'Segoe UI', Arial, sans-serif", size=16, color=C["primary"]),
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(
        orientation="h",
        y=-0.18, x=0,
        font=dict(size=13),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
    ),
)

# Safe minimal style applied to ALL figures (including px.imshow, px.box etc.)
# Do NOT add font/title_font here — px figures manage those internally.
CHART_BASE_SUB = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=40, t=60, b=60),
)

MAX_COMPARE = 5

# Clinical labels for chart Y-axes (include units explicitly)
YAXIS_LABELS: dict[str, str] = {
    "flow_rate_lpm":       "Blood Flow (L/min)",
    "pump_speed_rpm":      "Pump Speed (RPM)",
    "pressure_pre_mmhg":   "Pressure — Pre-oxygenator (mmHg)",
    "pressure_post_mmhg":  "Pressure — Post-oxygenator (mmHg)",
    "pressure_delta_mmhg": "Oxygenator ΔPressure (mmHg)",
    "sat_pre_pct":         "Pre-oxygenator SatO₂ (%)",
    "sat_post_pct":        "Post-oxygenator SatO₂ (%)",
    "temp_post_c":         "Post-oxygenator Temperature (°C)",
}

# Default parameters shown in Vital Trends (most clinically relevant first)
DEFAULT_TREND_PARAMS = ["flow_rate_lpm", "pressure_delta_mmhg", "sat_pre_pct", "pump_speed_rpm"]


# ─────────────────────────────────────────────────────────────────────────────
# CSS  — refined for clarity, readability, and clinical tone
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(f"""
<style>
/* Global typography */
html, body, [class*="css"] {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 15px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {C['primary']};
}}
section[data-testid="stSidebar"] * {{ color: #ECF0F1 !important; }}
section[data-testid="stSidebar"] hr {{ border-color: {C['accent']} !important; opacity:0.35; }}
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stSelectbox label {{
    font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase;
    color: #BDC3C7 !important;
}}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    border-bottom: 2px solid {C['border']};
    margin-bottom: 8px;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.88rem; font-weight: 500;
    color: {C['subtle']}; padding: 9px 20px;
    border-radius: 6px 6px 0 0;
    transition: background 0.15s;
}}
.stTabs [aria-selected="true"] {{
    background: #EAF2FB; color: {C['primary']} !important;
    font-weight: 700; border-bottom: 3px solid {C['accent']};
}}

/* ── KPI metric cards ── */
div[data-testid="metric-container"] {{
    background: {C['bg_card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 18px 20px 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
div[data-testid="metric-container"] label {{
    font-size: 0.70rem !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {C['subtle']} !important;
    font-weight: 600;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 2.1rem !important;
    font-weight: 700;
    color: {C['primary']} !important;
    line-height: 1.1;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.82rem !important;
}}

/* ── Case header card ── */
.case-header {{
    background: linear-gradient(135deg, {C['primary']} 0%, #2471A3 100%);
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 24px;
    color: white;
}}
.case-header-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
}}
.case-header-sub {{
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
}}

/* ── Case navigator (top of page) ── */
.nav-bar {{
    background: {C['bg_card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 10px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}}

/* ── Alarm badges ── */
.alarm-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    background: #FADBD8; color: {C['alarm']};
    border: 1px solid #E74C3C;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.82rem; font-weight: 700;
}}
.ok-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    background: #D5F5E3; color: {C['ok']};
    border: 1px solid #27AE60;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.82rem; font-weight: 700;
}}
.info-pill {{
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(255,255,255,0.15);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.80rem; color: rgba(255,255,255,0.9);
}}

/* ── Chart interpretation note ── */
.chart-note {{
    background: #EBF5FB;
    border-left: 4px solid {C['accent']};
    padding: 9px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.87rem;
    color: {C['primary']};
    margin-bottom: 12px;
    line-height: 1.55;
}}

/* ── Section headers ── */
h1 {{ color: {C['primary']}; font-size: 1.7rem; font-weight: 700; }}
h2 {{
    color: {C['primary']}; font-size: 1.25rem; font-weight: 600;
    border-bottom: 1px solid {C['border']};
    padding-bottom: 6px; margin-top: 1.5rem; margin-bottom: 0.8rem;
}}
h3 {{ color: {C['accent']}; font-size: 1.05rem; font-weight: 600; }}

/* ── Alarm highlight row in tables ── */
.alarm-row {{ background-color: #FADBD8 !important; }}

/* ── Expander polish ── */
details > summary {{
    font-size: 0.88rem; color: {C['subtle']};
    letter-spacing: 0.02em; cursor: pointer;
}}

/* ── Risk level colors ── */
.risk-high     {{ color: {C['alarm']}; font-weight: 700; }}
.risk-moderate {{ color: {C['warn']};  font-weight: 700; }}
.risk-low      {{ color: {C['ok']};    font-weight: 700; }}

/* ── Divider ── */
.section-divider {{
    border: none; border-top: 1px solid {C['border']};
    margin: 1.5rem 0;
}}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _note(text: str) -> None:
    """Render a clinical interpretation note above a chart."""
    st.markdown(f'<div class="chart-note">ℹ️&nbsp; {text}</div>', unsafe_allow_html=True)


def _alarm_rate(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total > 0 else "—"


def _session_label(row: pd.Series) -> str:
    return f"Case {row['case_id']}  ·  {row['filename']}"


def _base_fig(title: str = "", xaxis_title: str = "Time", yaxis_title: str = "", **kw) -> go.Figure:
    """
    Every chart in the app goes through this constructor.
    Guarantees: consistent font sizes, axis labels, grid style, margins.
    """
    fig = go.Figure()
    layout = {
        **CHART_BASE,
        "title": dict(text=title, font=dict(size=17, color=C["primary"]), x=0, xanchor="left"),
        "xaxis": dict(
            title=dict(text=xaxis_title, font=dict(size=14), standoff=12),
            showgrid=True, gridcolor=C["grid"], gridwidth=1,
            zeroline=False, tickfont=dict(size=13),
        ),
        "yaxis": dict(
            title=dict(text=yaxis_title, font=dict(size=14), standoff=12),
            showgrid=True, gridcolor=C["grid"], gridwidth=1,
            zeroline=False, tickfont=dict(size=13),
        ),
        **kw,
    }
    fig.update_layout(**layout)
    return fig


def _measurement_trace(fig, x, y, name: str, color: str = None, row=None, col=None) -> None:
    """Add a styled measurement line trace. Blue = measurement (always)."""
    kwargs = dict(
        x=x, y=y, mode="lines", name=name,
        line=dict(width=2.2, color=color or C["accent"]),
        connectgaps=False,
    )
    if row is not None:
        fig.add_trace(go.Scatter(**kwargs), row=row, col=col)
    else:
        fig.add_trace(go.Scatter(**kwargs))


def _alarm_trace(fig, x, y, name: str = "Alarm active", show_legend: bool = True,
                 row=None, col=None) -> None:
    """Add a styled alarm marker trace. Red = alarms (always)."""
    kwargs = dict(
        x=x, y=y, mode="markers", name=name,
        marker=dict(
            color=C["alarm"], size=7,
            symbol="circle", line=dict(width=1.5, color="white"),
        ),
        showlegend=show_legend,
    )
    if row is not None:
        fig.add_trace(go.Scatter(**kwargs), row=row, col=col)
    else:
        fig.add_trace(go.Scatter(**kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# Database & state
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_db() -> VFDatabase:
    return VFDatabase(":memory:")


def _init_state() -> None:
    defaults = {
        "db":                _get_db(),
        "active_session":    None,
        "selected_sessions": [],
        "vf_model":          None,
        "model_trained":     False,
        "pred_result":       None,
        "loaded_filenames":  set(),   # tracks filenames already ingested this session
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (upload + comparison session selector + filters)
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(db: VFDatabase) -> tuple[Optional[int], list[int], dict]:
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:18px 0 10px;">
  <span style="font-size:2.2rem;">🫀</span><br>
  <span style="font-size:1.2rem;font-weight:700;letter-spacing:.08em;">VF ANALYST</span><br>
  <span style="font-size:.70rem;opacity:.55;letter-spacing:.10em;">ECMO · PERFUSION · VAD</span>
</div>""", unsafe_allow_html=True)
        st.divider()

        # ── Upload ──────────────────────────────────────────────────────────
        st.markdown("**Upload Examination Files**")
        st.caption("Upload one or more CSV files from the device.")
        uploaded_files = st.file_uploader(
            "Drop files here", type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            for f in uploaded_files:
                _process_upload(db, f)

        # ── Sessions ────────────────────────────────────────────────────────
        st.divider()
        sessions_df = db.get_sessions()
        if sessions_df.empty:
            st.info("No files loaded yet.")
            return None, [], {}

        id_to_row = {int(r["session_id"]): r for _, r in sessions_df.iterrows()}
        all_ids   = list(id_to_row.keys())

        # Validate: if stored session no longer exists (e.g. page reload),
        # fall back to first — but NEVER auto-switch just because a new file
        # was uploaded.
        if st.session_state.active_session not in all_ids:
            st.session_state.active_session = all_ids[0]

        # Active session selector — shows the user's current selection.
        # Changing this dropdown immediately updates the dashboard.
        st.markdown("**Active Case**")
        active_id = st.selectbox(
            "Active session",
            options=all_ids,
            format_func=lambda k: _session_label(id_to_row[k]),
            index=all_ids.index(st.session_state.active_session),
            label_visibility="collapsed",
            key="sidebar_active_sel",
        )
        # Write back only if the user explicitly changed the dropdown
        if active_id != st.session_state.active_session:
            st.session_state.active_session = active_id

        # Comparison sessions
        st.divider()
        st.markdown("**Cases for Comparison**")
        st.caption(f"Up to {MAX_COMPARE} cases — used in Comparison & Cohort tabs.")
        prev_sel = [i for i in st.session_state.selected_sessions if i in all_ids]
        selected_ids = st.multiselect(
            "Compare", options=all_ids,
            default=prev_sel or all_ids[:min(2, len(all_ids))],
            format_func=lambda k: _session_label(id_to_row[k]),
            max_selections=MAX_COMPARE,
            label_visibility="collapsed",
        )
        st.session_state.selected_sessions = selected_ids

        # ── Filters ─────────────────────────────────────────────────────────
        filters: dict = {}
        meas_df = db.get_measurements(active_id)

        if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
            st.divider()
            st.markdown("**Time Window**")
            t_min = meas_df["timestamp"].min().to_pydatetime()
            t_max = meas_df["timestamp"].max().to_pydatetime()
            time_range = st.slider(
                "Time", min_value=t_min, max_value=t_max,
                value=(t_min, t_max), format="DD/MM HH:mm",
                label_visibility="collapsed",
            )
            filters["time_range"] = time_range

        if "alarm_active" in meas_df.columns:
            st.divider()
            filters["alarm_only"] = st.toggle(
                "Alarm periods only", value=False,
                help="Show only time points with an active alarm.",
            )

        st.divider()
        st.markdown(
            f'<div style="font-size:.73rem;opacity:.55;line-height:1.9;">'
            f'📁 {len(all_ids)} case(s) loaded</div>',
            unsafe_allow_html=True,
        )

    return active_id, selected_ids, filters


def _process_upload(db: VFDatabase, uploaded) -> None:
    # Guard 1: fast in-memory check (avoids DB query on every Streamlit rerun)
    if uploaded.name in st.session_state.loaded_filenames:
        return

    # Guard 2: DB-level check (handles page reload where session state is lost)
    sessions_df = db.get_sessions()
    if not sessions_df.empty and uploaded.name in sessions_df["filename"].values:
        st.session_state.loaded_filenames.add(uploaded.name)
        return

    # Remember if the user already had a case open before this upload.
    # If yes, we preserve their selection after the upload completes.
    had_active = st.session_state.active_session is not None

    with st.sidebar:
        with st.status(f"Loading {uploaded.name}…", expanded=False) as status:
            try:
                raw_df, warnings = load_csv(uploaded)
                clean_df, provenance = clean(raw_df)
                ts_ok      = "timestamp" in clean_df.columns and clean_df["timestamp"].notna().any()
                time_start = str(clean_df["timestamp"].min()) if ts_ok else None
                time_end   = str(clean_df["timestamp"].max()) if ts_ok else None
                case_id    = clean_df["case_id"].iloc[0] if "case_id" in clean_df.columns else "unknown"
                sid = db.insert_session(
                    case_id=case_id, filename=uploaded.name,
                    row_count=len(clean_df),
                    time_start=time_start, time_end=time_end,
                    provenance=provenance,
                )
                db.insert_measurements(clean_df, sid)
                db.insert_alarm_events(clean_df, sid)

                # Only auto-switch to the new file on the very first upload.
                # After that the user controls navigation themselves.
                if not had_active:
                    st.session_state.active_session = sid

                # Record as loaded so re-runs don't re-ingest
                st.session_state.loaded_filenames.add(uploaded.name)

                for w in warnings:
                    st.warning(w)
                status.update(label=f"✅ {uploaded.name}", state="complete")
            except Exception as exc:
                status.update(label=f"❌ {uploaded.name}", state="error")
                st.error(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Landing page
# ─────────────────────────────────────────────────────────────────────────────

def _render_landing() -> None:
    st.markdown("""
<div style="max-width:600px;margin:80px auto;text-align:center;">
  <div style="font-size:3.2rem;margin-bottom:14px;">🫀</div>
  <h1 style="font-size:1.9rem;margin-bottom:8px;">VF Analyst</h1>
  <p style="color:#7F8C8D;font-size:1rem;margin-bottom:36px;">
    Clinical dashboard for ECMO &amp; VAD perfusion monitoring
  </p>
  <div style="background:#EAF2FB;border:1px solid #AED6F1;border-radius:14px;
              padding:26px 32px;text-align:left;">
    <p style="font-weight:700;margin-bottom:10px;color:#1B4F72;">To get started:</p>
    <ol style="color:#2C3E50;line-height:2.2;font-size:.95rem;margin:0;">
      <li>Export the CSV log from the ECMO / VAD device via USB.</li>
      <li>Click <strong>Upload Examination Files</strong> in the left panel.</li>
      <li>You can load <strong>multiple cases</strong> to enable comparison.</li>
    </ol>
  </div>
  <p style="color:#BDC3C7;font-size:.75rem;margin-top:28px;">
    Compatible with Xenios · Maquet · Getinge device exports
  </p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP CASE NAVIGATOR  (NEW in v3 — sits above all tabs)
# ─────────────────────────────────────────────────────────────────────────────

def _render_case_navigator(
    db: VFDatabase,
    active_id: int,
    sessions_df: pd.DataFrame,
    all_ids: list[int],
) -> int:
    """
    Displays a slim read-only header bar showing the active case number,
    its position in the loaded list, duration, and alarm status.
    Navigation is done exclusively via the sidebar dropdown.
    """
    id_to_row   = {int(r["session_id"]): r for _, r in sessions_df.iterrows()}
    current_idx = all_ids.index(active_id) if active_id in all_ids else 0
    current_row = id_to_row[active_id]

    meas_df     = db.get_measurements(active_id)
    alarm_count = int(meas_df["alarm_active"].sum()) if "alarm_active" in meas_df.columns else 0
    total       = len(meas_df)

    dur_txt = "—"
    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        delta   = meas_df["timestamp"].max() - meas_df["timestamp"].min()
        dur_txt = f"{int(delta.total_seconds() / 60)} min"

    if alarm_count > 0:
        badge = (
            f'<span class="alarm-badge">⚠ {alarm_count:,} alarms '
            f'({_alarm_rate(alarm_count, total)})</span>'
        )
    else:
        badge = '<span class="ok-badge">✓ No alarms</span>'

    st.markdown(
        f'''<div style="display:flex;align-items:center;gap:18px;
                      padding:10px 20px;margin-bottom:14px;
                      background:{C["bg_card"]};border:1px solid {C["border"]};
                      border-radius:10px;">
          <span style="font-size:1.1rem;font-weight:700;color:{C["primary"]};">
            Case {current_row["case_id"]}
          </span>
          <span style="font-size:.82rem;color:{C["subtle"]};">
            {current_idx + 1} of {len(all_ids)} cases
          </span>
          <span style="font-size:.82rem;color:{C["subtle"]};">⏱ {dur_txt}</span>
          {badge}
        </div>''',
        unsafe_allow_html=True,
    )

    return active_id



# ─────────────────────────────────────────────────────────────────────────────
# CASE HEADER CARD  (NEW in v3 — replaces old inline header)
# ─────────────────────────────────────────────────────────────────────────────

def _render_case_header(
    meas_df: pd.DataFrame,
    srow: pd.Series,
    alarm_count: int,
) -> None:
    """
    Full-width navy gradient card showing:
    Case ID  |  filename  |  time range  |  duration  |  alarm summary
    """
    case_id  = srow["case_id"]
    filename = srow["filename"]

    # Time range
    time_start = time_end = dur_txt = "—"
    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        ts_min  = meas_df["timestamp"].min()
        ts_max  = meas_df["timestamp"].max()
        time_start = ts_min.strftime("%d %b %Y  %H:%M")
        time_end   = ts_max.strftime("%H:%M")
        dur_min    = int((ts_max - ts_min).total_seconds() / 60)
        dur_txt    = f"{dur_min} min"

    total = len(meas_df)
    alarm_pct = _alarm_rate(alarm_count, total)

    if alarm_count > 0:
        alarm_html = (
            f'<span class="info-pill" style="background:rgba(192,57,43,0.25);color:#FADBD8;">'
            f'⚠ {alarm_count:,} alarms · {alarm_pct}</span>'
        )
    else:
        alarm_html = (
            '<span class="info-pill" style="background:rgba(39,174,96,0.25);color:#D5F5E3;">'
            '✓ No alarms</span>'
        )

    st.markdown(f"""
<div class="case-header">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px;">
    <div>
      <p class="case-header-title">Case {case_id} — ECMO Session</p>
      <p class="case-header-sub">{filename}</p>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
      <span class="info-pill">📅 {time_start} – {time_end}</span>
      <span class="info-pill">⏱ {dur_txt}</span>
      <span class="info-pill">📊 {total:,} records</span>
      {alarm_html}
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard router
# ─────────────────────────────────────────────────────────────────────────────

def _render_dashboard(
    db: VFDatabase,
    active_id: int,
    selected_ids: list[int],
    filters: dict,
) -> None:
    sessions_df = db.get_sessions()
    all_ids     = list(sessions_df["session_id"].astype(int))

    # ── Top case navigator ────────────────────────────────────────────────────
    active_id = _render_case_navigator(db, active_id, sessions_df, all_ids)

    # Load data for the active session
    meas_df  = db.get_measurements(active_id)
    alarm_df = db.get_alarm_events(active_id)

    # Apply filters
    if "time_range" in filters and "timestamp" in meas_df.columns:
        t0, t1 = filters["time_range"]
        meas_df = meas_df[
            (meas_df["timestamp"] >= pd.Timestamp(t0)) &
            (meas_df["timestamp"] <= pd.Timestamp(t1))
        ]
        if not alarm_df.empty:
            ts = pd.to_datetime(alarm_df["timestamp"], errors="coerce")
            alarm_df = alarm_df[(ts >= pd.Timestamp(t0)) & (ts <= pd.Timestamp(t1))]
    if filters.get("alarm_only") and "alarm_active" in meas_df.columns:
        meas_df = meas_df[meas_df["alarm_active"] == 1]

    # ── Case header card ──────────────────────────────────────────────────────
    srow        = sessions_df[sessions_df["session_id"] == active_id].iloc[0]
    alarm_count = int(meas_df["alarm_active"].sum()) if "alarm_active" in meas_df.columns else 0
    _render_case_header(meas_df, srow, alarm_count)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    (tab_ov, tab_tr, tab_dist,
     tab_al, tab_cmp, tab_coh,
     tab_risk, tab_audit) = st.tabs([
        "📋 Overview",
        "📈 Vital Trends",
        "📊 Distributions",
        "🚨 Alarms",
        "⚖️ Comparison",
        "🏥 Cohort",
        "🧠 Risk Prediction",
        "🔍 Data Audit",
    ])

    with tab_ov:    _tab_overview(meas_df, alarm_df)
    with tab_tr:    _tab_trends(meas_df, alarm_df)
    with tab_dist:  _tab_distributions(meas_df)
    with tab_al:    _tab_alarms(alarm_df, meas_df)
    with tab_cmp:   _tab_compare(db, selected_ids, sessions_df)
    with tab_coh:   _tab_cohort(db, selected_ids, sessions_df)
    with tab_risk:  _tab_risk(db, selected_ids, active_id, sessions_df)
    with tab_audit: _tab_audit(db, active_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Overview  (v3 — restructured layout, better KPIs, cleaner sparklines)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_overview(meas_df: pd.DataFrame, alarm_df: pd.DataFrame) -> None:
    summary     = compute_summary(meas_df)
    alarm_count = summary.get("alarm_events", 0)
    total       = summary.get("total_records", 1)

    # ── Row 1: 4 KPI cards ────────────────────────────────────────────────────
    st.markdown("## Session at a Glance")
    k = st.columns(4)
    k[0].metric(
        "Session Duration",
        f"{int(summary['duration_minutes'])} min" if summary.get("duration_minutes") else "—",
        help="Total time from first to last recorded measurement.",
    )
    k[1].metric(
        "Total Measurements",
        f"{summary.get('total_records', 0):,}",
        help="Number of data points recorded by the device.",
    )
    k[2].metric(
        "Alarm Events",
        f"{alarm_count:,}",
        delta=f"{_alarm_rate(alarm_count, total)} of session" if alarm_count else "None recorded",
        delta_color="inverse" if alarm_count else "off",
        help="Number of time points with at least one active device alarm.",
    )
    avg_flow = summary.get("flow_rate_lpm_mean")
    k[3].metric(
        "Average Blood Flow",
        f"{avg_flow:.2f} L/min" if avg_flow else "—",
        help="Mean blood flow rate recorded over the entire session.",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Row 2: Priority sparkline charts (2-column layout) ────────────────────
    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        st.markdown("## Primary Vitals")
        _note(
            "The two most clinically critical parameters: blood flow rate and oxygenator "
            "pressure difference. Red circles mark moments when a device alarm was active."
        )

        col_l, col_r = st.columns(2)

        for container, col, chart_title, y_unit in [
            (col_l, "flow_rate_lpm",       "Blood Flow Rate Over Time",          "Blood Flow (L/min)"),
            (col_r, "pressure_delta_mmhg",  "Oxygenator Pressure Difference",     "ΔPressure (mmHg)"),
        ]:
            if col not in meas_df.columns:
                continue
            with container:
                fig = _base_fig(
                    title=chart_title,
                    xaxis_title="Time",
                    yaxis_title=y_unit,
                    height=300,
                )
                _measurement_trace(fig, meas_df["timestamp"], meas_df[col], name=y_unit)
                if "alarm_active" in meas_df.columns:
                    alm = meas_df[meas_df["alarm_active"] == 1]
                    if not alm.empty:
                        _alarm_trace(fig, alm["timestamp"], alm[col], show_legend=True)
                fig.update_layout(margin=dict(l=55, r=20, t=55, b=55))
                st.plotly_chart(fig, use_container_width=True)

        # Secondary pair
        col_l2, col_r2 = st.columns(2)
        for container, col, chart_title, y_unit in [
            (col_l2, "sat_pre_pct",    "Pre-oxygenator Oxygen Saturation", "SatO₂ (%)"),
            (col_r2, "pump_speed_rpm", "Pump Speed Over Time",             "Pump Speed (RPM)"),
        ]:
            if col not in meas_df.columns:
                continue
            with container:
                fig = _base_fig(
                    title=chart_title,
                    xaxis_title="Time",
                    yaxis_title=y_unit,
                    height=280,
                )
                _measurement_trace(fig, meas_df["timestamp"], meas_df[col], name=y_unit)
                if "alarm_active" in meas_df.columns:
                    alm = meas_df[meas_df["alarm_active"] == 1]
                    if not alm.empty:
                        _alarm_trace(fig, alm["timestamp"], alm[col], show_legend=False)
                fig.update_layout(margin=dict(l=55, r=20, t=50, b=55),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Statistics table ───────────────────────────────────────────────
    with st.expander("📋 Full Measurement Statistics Table", expanded=False):
        st.caption(
            "Mean, minimum, maximum, and standard deviation for each parameter. "
            "Data Coverage shows what fraction of records had a valid sensor signal."
        )
        stat_rows = []
        for c in MEASUREMENT_COLS:
            if c not in meas_df.columns or meas_df[c].dropna().empty:
                continue
            s = meas_df[c].dropna()
            stat_rows.append({
                "Parameter":     DISPLAY_LABELS.get(c, c),
                "Mean":          f"{s.mean():.1f}",
                "Min":           f"{s.min():.1f}",
                "Max":           f"{s.max():.1f}",
                "Std Dev":       f"{s.std():.1f}",
                "Data Coverage": f"{100 * len(s) / len(meas_df):.0f}%",
            })
        if stat_rows:
            st.dataframe(
                pd.DataFrame(stat_rows).set_index("Parameter"),
                use_container_width=True,
                height=min(80 + 38 * len(stat_rows), 420),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Vital Trends  (v3 — better defaults, alarm shading, axis units)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_trends(meas_df: pd.DataFrame, alarm_df: pd.DataFrame) -> None:
    st.markdown("## Vital Trends Over Time")

    if "timestamp" not in meas_df.columns or meas_df["timestamp"].isna().all():
        st.warning("No time information available for this session.")
        return

    available = [c for c in MEASUREMENT_COLS if c in meas_df.columns and meas_df[c].notna().any()]
    if not available:
        st.warning("No measurement data available.")
        return

    _note(
        "Each panel shows how a parameter changed over the session. "
        "Gaps in the line indicate periods when the sensor signal was unavailable. "
        "Red circles mark time points when a device alarm was active."
    )

    label_to_col = {DISPLAY_LABELS.get(c, c): c for c in available}

    # Smart defaults: pick the most clinically relevant available parameters
    default_labels = [
        DISPLAY_LABELS.get(c, c) for c in DEFAULT_TREND_PARAMS if c in available
    ][:4] or list(label_to_col.keys())[:3]

    selected_labels = st.multiselect(
        "Parameters to display  (select 1–4 for best readability)",
        options=list(label_to_col.keys()),
        default=default_labels,
        help="Choose parameters to display as time-series plots.",
    )

    if not selected_labels:
        st.info("Select at least one parameter above.")
        return

    selected_cols = [label_to_col[l] for l in selected_labels]
    n = len(selected_cols)

    # Build shared-x subplot figure
    subplot_titles = [
        f"{DISPLAY_LABELS.get(c, c)}  —  {YAXIS_LABELS.get(c, '')}"
        for c in selected_cols
    ]
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=max(0.06, min(0.14, 0.5 / n)),
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=15, color=C["primary"])
        ann.x    = 0
        ann.xanchor = "left"

    show_alarm_legend = True
    for i, col in enumerate(selected_cols, start=1):
        _measurement_trace(fig, meas_df["timestamp"], meas_df[col],
                           name=DISPLAY_LABELS.get(col, col), row=i, col=1)
        if "alarm_active" in meas_df.columns:
            alm = meas_df[meas_df["alarm_active"] == 1]
            if not alm.empty:
                _alarm_trace(fig, alm["timestamp"], alm[col],
                             show_legend=show_alarm_legend, row=i, col=1)
                show_alarm_legend = False  # show legend entry only once

        # Y-axis label per subplot
        y_label = YAXIS_LABELS.get(col, DISPLAY_LABELS.get(col, col))
        fig.update_yaxes(
            title_text=y_label,
            title_font=dict(size=13),
            tickfont=dict(size=12),
            showgrid=True, gridcolor=C["grid"],
            row=i, col=1,
        )

    fig.update_xaxes(
        showgrid=True, gridcolor=C["grid"],
        tickfont=dict(size=12),
    )
    fig.update_xaxes(title_text="Date / Time", title_font=dict(size=13), row=n, col=1)

    fig.update_layout(
        **CHART_BASE_SUB,
        height=max(280, 230 * n),
        legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Alarm limit bands — advanced, hidden by default
    with st.expander("⚙️ Advanced — Show configured alarm limit bands", expanded=False):
        st.caption(
            "Dashed red lines mark the alarm thresholds configured on the device. "
            "Readings outside these lines triggered an alarm."
        )
        _render_limits_overlay(meas_df, selected_cols)


def _render_limits_overlay(meas_df: pd.DataFrame, cols: list[str]) -> None:
    LIMIT_PAIRS = {
        "flow_rate_lpm":     ("alarm_flow_low_lpm",       "alarm_flow_high_lpm"),
        "pressure_pre_mmhg": ("alarm_pressure_pre_low",   "alarm_pressure_pre_high"),
        "pressure_post_mmhg":("alarm_pressure_post_low",  "alarm_pressure_post_high"),
        "sat_pre_pct":       ("alarm_sat_pre_low",        "alarm_sat_pre_high"),
        "temp_post_c":       ("alarm_temp_post_low",      "alarm_temp_post_high"),
    }
    for col in cols:
        if col not in LIMIT_PAIRS or col not in meas_df.columns:
            continue
        low_col, high_col = LIMIT_PAIRS[col]
        label = DISPLAY_LABELS.get(col, col)
        y_label = YAXIS_LABELS.get(col, label)

        fig = _base_fig(
            title=f"{label} — with alarm limits",
            xaxis_title="Time", yaxis_title=y_label,
            height=280,
        )
        _measurement_trace(fig, meas_df["timestamp"], meas_df[col], name="Measured")

        for lim_col, lim_name in [(low_col, "Low limit"), (high_col, "High limit")]:
            if lim_col in meas_df.columns and meas_df[lim_col].notna().any():
                lim_val = meas_df[lim_col].median()
                fig.add_hline(
                    y=lim_val, line_dash="dot",
                    line_color=C["alarm"], line_width=1.5,
                    annotation_text=f"{lim_name}: {lim_val:.1f}",
                    annotation_position="right",
                    annotation_font=dict(size=12, color=C["alarm"]),
                )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Distributions  (v3 — larger fonts, better axis labels)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_distributions(meas_df: pd.DataFrame) -> None:
    st.markdown("## Value Distributions")
    _note(
        "These charts show how frequently each parameter was recorded at different levels "
        "during the session. The dashed line marks the session average. "
        "A narrow distribution indicates a stable, well-controlled parameter."
    )

    available = [c for c in MEASUREMENT_COLS if c in meas_df.columns and meas_df[c].notna().any()]
    if not available:
        st.warning("No measurement data available.")
        return

    c1, c2 = st.columns(2)
    for idx, col in enumerate(available):
        values = meas_df[col].dropna()
        label  = DISPLAY_LABELS.get(col, col)
        y_lbl  = YAXIS_LABELS.get(col, label)
        mean_v = values.mean()

        fig = _base_fig(
            title=f"Distribution — {label}",
            xaxis_title=y_lbl,
            yaxis_title="Number of Readings",
            height=300,
        )
        fig.add_trace(go.Histogram(
            x=values, nbinsx=30,
            marker_color=C["accent"], marker_line_color="white",
            marker_line_width=0.5, opacity=0.85,
            name=label,
        ))
        fig.add_vline(
            x=mean_v, line_dash="dash",
            line_color=C["primary"], line_width=2,
            annotation_text=f"Avg: {mean_v:.1f}",
            annotation_position="top right",
            annotation_font=dict(size=13, color=C["primary"]),
        )
        fig.update_layout(showlegend=False, margin=dict(l=55, r=20, t=55, b=60))

        (c1 if idx % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)

    with st.expander("📐 Advanced — Parameter Correlation Matrix", expanded=False):
        st.caption(
            "Shows how strongly each pair of parameters moves together. "
            "Values near +1 mean they rise and fall together; near −1 means opposite; "
            "near 0 means no consistent relationship."
        )
        corr_cols = [c for c in available if meas_df[c].nunique() > 1]
        if len(corr_cols) >= 2:
            corr   = meas_df[corr_cols].corr()
            labels = [DISPLAY_LABELS.get(c, c) for c in corr_cols]
            fig_c  = px.imshow(
                corr.values, x=labels, y=labels,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                text_auto=".2f", aspect="auto",
            )
            fig_c.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=60, r=40, t=60, b=60),
                height=460,
            )
            fig_c.update_traces(textfont_size=12)
            st.plotly_chart(fig_c, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Alarms  (v3 — timeline primary, clear markers, details in expander)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_alarms(alarm_df: pd.DataFrame, meas_df: pd.DataFrame) -> None:
    st.markdown("## Alarm Review")

    alarm_count = len(alarm_df)
    total       = len(meas_df)

    if alarm_df.empty:
        st.success(
            "✅ **No alarm events were recorded** during the selected time window. "
            "All device parameters remained within configured limits."
        )
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Alarm Events", f"{alarm_count:,}",
              help="Number of time points where at least one alarm was active.")
    k2.metric("Alarm Coverage", _alarm_rate(alarm_count, total),
              help="Fraction of the session during which an alarm was active.")
    if "alarm_type" in alarm_df.columns:
        k3.metric("Alarm Categories", alarm_df["alarm_type"].nunique())
    k4.metric("Non-alarm Periods",
              _alarm_rate(total - alarm_count, total),
              help="Fraction of the session with no active alarm.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Primary: alarm timeline overlaid on flow rate ─────────────────────────
    st.markdown("### Alarm Timeline")
    _note(
        "Blood flow rate over time, with red circles marking every moment a device alarm was active. "
        "Clusters of red circles indicate periods of sustained alarm activity requiring review."
    )

    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        fig = _base_fig(
            title="Blood Flow Rate — with Alarm Events",
            xaxis_title="Date / Time",
            yaxis_title="Blood Flow (L/min)",
            height=320,
        )
        if "flow_rate_lpm" in meas_df.columns:
            _measurement_trace(fig, meas_df["timestamp"],
                               meas_df["flow_rate_lpm"], name="Flow Rate (L/min)")
        if "alarm_active" in meas_df.columns:
            alm = meas_df[meas_df["alarm_active"] == 1]
            if not alm.empty and "flow_rate_lpm" in meas_df.columns:
                _alarm_trace(fig, alm["timestamp"], alm["flow_rate_lpm"],
                             name="Alarm active")
        fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Alarm type breakdown ──────────────────────────────────────────────────
    if "alarm_type" in alarm_df.columns:
        st.markdown("### Alarm Type Breakdown")
        _note("Total number of events per alarm category during this session.")

        counts = alarm_df["alarm_type"].value_counts().reset_index()
        counts.columns = ["alarm_type", "Events"]
        counts["Alarm Category"] = counts["alarm_type"].replace({
            "overall":   "General Alarm",
            "bubble":    "Bubble / Air Alarm",
            "technical": "Technical Fault",
        })

        fig_b = _base_fig(
            title="Alarm Events by Category",
            xaxis_title="Alarm Category",
            yaxis_title="Number of Events",
            height=300,
        )
        for i, row in counts.iterrows():
            fig_b.add_trace(go.Bar(
                x=[row["Alarm Category"]],
                y=[row["Events"]],
                text=[row["Events"]],
                textposition="outside",
                textfont=dict(size=14, color=C["primary"]),
                marker_color=C["alarm"],
                name=row["Alarm Category"],
                showlegend=False,
            ))
        fig_b.update_layout(margin=dict(l=55, r=20, t=60, b=70))
        st.plotly_chart(fig_b, use_container_width=True)

    # ── Detailed log in expander ──────────────────────────────────────────────
    with st.expander("📋 Detailed Alarm Log — all events (Advanced)", expanded=False):
        st.caption("Full timestamped record of every alarm event.")
        display = alarm_df.rename(columns={
            "timestamp":   "Date & Time",
            "case_id":     "Case",
            "alarm_type":  "Alarm Category",
            "alarm_value": "Device Code",
        }, errors="ignore")
        if "Alarm Category" in display.columns:
            display["Alarm Category"] = display["Alarm Category"].replace({
                "overall": "General", "bubble": "Bubble / Air", "technical": "Technical",
            })
        st.dataframe(display, use_container_width=True, height=340)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Case Comparison  (v3 — consistent chart style, clearer overlay)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_compare(
    db: VFDatabase,
    selected_ids: list[int],
    sessions_df: pd.DataFrame,
) -> None:
    st.markdown("## Case Comparison")

    if len(selected_ids) < 2:
        st.info(
            "Select at least 2 cases in the sidebar panel (under **Cases for Comparison**) "
            "to enable side-by-side analysis."
        )
        return

    _note(
        "Side-by-side comparison of key metrics and overlaid trend charts across the selected cases. "
        "Each case is shown in a distinct colour. Use the sidebar to add or remove cases."
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### Summary Metrics per Case")

    compare_rows = []
    for sid in selected_ids:
        srow = sessions_df[sessions_df["session_id"] == sid].iloc[0]
        mdf  = db.get_measurements(sid)
        s    = compute_summary(mdf)
        row: dict = {
            "Case":            f"Case {srow['case_id']}",
            "File":            srow["filename"],
            "Duration (min)":  s.get("duration_minutes", "—"),
            "Records":         s.get("total_records", 0),
            "Alarm Events":    s.get("alarm_events", 0),
            "Alarm Rate":      _alarm_rate(s.get("alarm_events", 0), s.get("total_records", 1)),
        }
        for col in ["flow_rate_lpm", "pressure_delta_mmhg", "sat_pre_pct", "pump_speed_rpm"]:
            mean_val = s.get(f"{col}_mean")
            row[DISPLAY_LABELS.get(col, col)] = f"{mean_val:.1f}" if mean_val else "—"
        compare_rows.append(row)

    st.dataframe(
        pd.DataFrame(compare_rows).set_index("Case"),
        use_container_width=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Overlay trend ─────────────────────────────────────────────────────────
    st.markdown("### Trend Overlay — One Parameter Across Cases")
    _note(
        "Each case is plotted in a different colour. "
        "When 'Normalize time' is on, the x-axis shows session progress (0–100%) "
        "so cases of different lengths can be compared side-by-side."
    )

    available_cols = []
    for sid in selected_ids:
        mdf = db.get_measurements(sid)
        for col in MEASUREMENT_COLS:
            if col in mdf.columns and mdf[col].notna().any() and col not in available_cols:
                available_cols.append(col)

    if not available_cols:
        st.warning("No shared measurement columns found.")
        return

    label_to_col = {DISPLAY_LABELS.get(c, c): c for c in available_cols}
    ctrl_l, ctrl_r = st.columns([3, 2])
    with ctrl_l:
        chosen_label = st.selectbox(
            "Parameter to compare", options=list(label_to_col.keys()), key="compare_param",
        )
    with ctrl_r:
        use_normalized = st.checkbox(
            "Normalize time axis (0–100%)", value=True,
            help="Aligns sessions of different lengths.",
        )

    chosen_col = label_to_col[chosen_label]
    y_lbl      = YAXIS_LABELS.get(chosen_col, chosen_label)
    x_lbl      = "Session Progress (%)" if use_normalized else "Elapsed Time (min)"

    fig = _base_fig(
        title=f"{chosen_label} — Across Cases",
        xaxis_title=x_lbl,
        yaxis_title=y_lbl,
        height=400,
    )

    for i, sid in enumerate(selected_ids):
        srow  = sessions_df[sessions_df["session_id"] == sid].iloc[0]
        mdf   = db.get_measurements(sid)
        color = SESSION_COLORS[i % len(SESSION_COLORS)]

        if chosen_col not in mdf.columns or mdf[chosen_col].isna().all():
            continue
        if "timestamp" not in mdf.columns:
            continue

        mdf       = mdf.sort_values("timestamp").dropna(subset=["timestamp"])
        elapsed   = (mdf["timestamp"] - mdf["timestamp"].min()).dt.total_seconds() / 60
        x_vals    = (elapsed / elapsed.max() * 100) if use_normalized else elapsed

        fig.add_trace(go.Scatter(
            x=x_vals, y=mdf[chosen_col],
            mode="lines", name=f"Case {srow['case_id']}",
            line=dict(width=2.2, color=color),
            connectgaps=False,
        ))

    fig.update_layout(
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Alarm rate bar ────────────────────────────────────────────────────────
    st.markdown("### Alarm Rate per Case")
    _note("Percentage of each session during which at least one device alarm was active.")

    alarm_rows = []
    for i, sid in enumerate(selected_ids):
        srow    = sessions_df[sessions_df["session_id"] == sid].iloc[0]
        mdf     = db.get_measurements(sid)
        n_alarm = int(mdf["alarm_active"].sum()) if "alarm_active" in mdf.columns else 0
        alarm_rows.append({
            "Case":         f"Case {srow['case_id']}",
            "Alarm Rate":   round(n_alarm / max(len(mdf), 1) * 100, 1),
            "color":        SESSION_COLORS[i % len(SESSION_COLORS)],
        })

    alarm_bar_df = pd.DataFrame(alarm_rows)
    fig_b = _base_fig(
        title="Alarm Rate Comparison",
        xaxis_title="Case",
        yaxis_title="Alarm Rate (%)",
        height=320,
        yaxis_range=[0, max(alarm_bar_df["Alarm Rate"].max() * 1.3, 10)],
    )
    for _, r in alarm_bar_df.iterrows():
        fig_b.add_trace(go.Bar(
            x=[r["Case"]], y=[r["Alarm Rate"]],
            text=[f"{r['Alarm Rate']:.1f}%"],
            textposition="outside",
            textfont=dict(size=14),
            marker_color=r["color"],
            showlegend=False,
        ))
    st.plotly_chart(fig_b, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Cohort  (v3 — larger boxplot fonts, cleaner colours)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_cohort(
    db: VFDatabase,
    selected_ids: list[int],
    sessions_df: pd.DataFrame,
) -> None:
    st.markdown("## Cohort Insights")

    if len(selected_ids) < 2:
        st.info("Select at least 2 cases in the sidebar to view cohort-level analysis.")
        return

    _note(
        "Aggregated view across all selected cases. "
        "The box spans the middle 50% of readings for each case. "
        "The line inside is the median. Dots outside the box are outliers."
    )

    multi_df = db.get_measurements_multi(selected_ids)
    if multi_df.empty:
        st.warning("No measurements found.")
        return

    sid_to_label = {
        int(r["session_id"]): f"Case {r['case_id']}"
        for _, r in sessions_df.iterrows()
    }
    multi_df["Case"] = multi_df["session_id"].map(sid_to_label)

    available    = [c for c in MEASUREMENT_COLS if c in multi_df.columns and multi_df[c].notna().any()]
    label_to_col = {DISPLAY_LABELS.get(c, c): c for c in available}

    st.markdown("### Parameter Distributions per Case")
    selected_params = st.multiselect(
        "Select parameters for boxplots",
        options=list(label_to_col.keys()),
        default=list(label_to_col.keys())[:3],
        key="cohort_params",
    )

    for param_label in selected_params:
        col   = label_to_col[param_label]
        y_lbl = YAXIS_LABELS.get(col, param_label)

        fig = px.box(
            multi_df.dropna(subset=[col]),
            x="Case", y=col, color="Case",
            color_discrete_sequence=SESSION_COLORS,
            labels={col: y_lbl, "Case": ""},
            points="outliers",
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=40, t=60, b=60),
            title=dict(text=f"{param_label} — Distribution per Case",
                       font=dict(size=16, color=C["primary"]), x=0),
            height=360,
            showlegend=False,
        )
        fig.update_xaxes(title_text="", tickfont=dict(size=14), showgrid=False)
        fig.update_yaxes(
            title_text=y_lbl, title_font=dict(size=14),
            tickfont=dict(size=13), showgrid=True, gridcolor=C["grid"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Alarm frequency table + chart
    st.markdown("### Alarm Frequency Across Cases")
    _note("Total alarm events and alarm rate for each case in the cohort.")

    cohort_rows = []
    for sid in selected_ids:
        srow  = sessions_df[sessions_df["session_id"] == sid].iloc[0]
        mdf   = multi_df[multi_df["session_id"] == sid]
        n_alm = int(mdf["alarm_active"].sum()) if "alarm_active" in mdf.columns else 0
        cohort_rows.append({
            "Case":           f"Case {srow['case_id']}",
            "Total Records":  len(mdf),
            "Alarm Events":   n_alm,
            "Alarm Rate (%)": round(n_alm / max(len(mdf), 1) * 100, 1),
        })

    cohort_table = pd.DataFrame(cohort_rows).set_index("Case")
    st.dataframe(cohort_table, use_container_width=True)

    with st.expander("📊 Full Cohort Statistics Table (Advanced)", expanded=False):
        agg_rows = []
        for col in available:
            s = multi_df[col].dropna()
            if s.empty:
                continue
            agg_rows.append({
                "Parameter":     DISPLAY_LABELS.get(col, col),
                "Cohort Median": round(s.median(), 2),
                "IQR":           f"{round(s.quantile(.25), 1)} – {round(s.quantile(.75), 1)}",
                "Overall Min":   round(s.min(), 2),
                "Overall Max":   round(s.max(), 2),
                "N readings":    len(s),
            })
        if agg_rows:
            st.dataframe(pd.DataFrame(agg_rows).set_index("Parameter"),
                         use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Risk Prediction  (v3 — cleaner risk chart, same structure as v2)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_risk(
    db: VFDatabase,
    selected_ids: list[int],
    active_id: int,
    sessions_df: pd.DataFrame,
) -> None:
    st.markdown("## Early Warning — Risk Prediction")
    st.caption(
        "This prototype uses a machine learning model trained on device measurements "
        "to estimate the probability of a device alarm occurring in the next few time points. "
        "It is a **decision-support aid only** — not a substitute for clinical judgment."
    )

    if not selected_ids:
        st.info("Select at least one session in the sidebar to use this module.")
        return

    # ── Training controls ─────────────────────────────────────────────────────
    st.markdown("### Step 1 — Configure and Train the Model")
    _note(
        "The model learns from the sessions you select. "
        "Using more sessions produces a more reliable model. Training typically takes 5–15 seconds."
    )

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        horizon = st.slider(
            "Warning horizon (samples ahead)",
            min_value=2, max_value=20, value=5, step=1,
            help="How many samples ahead the model predicts. More = earlier warning, less precision.",
        )
    with ctrl2:
        model_type = st.radio(
            "Model type",
            options=["random_forest", "logistic_regression"],
            format_func=lambda x: "Random Forest (recommended)" if x == "random_forest"
                                   else "Logistic Regression (faster)",
        )

    train_ids = st.multiselect(
        "Sessions to train on",
        options=selected_ids,
        default=selected_ids,
        format_func=lambda k: _session_label(sessions_df[sessions_df["session_id"] == k].iloc[0]),
    )

    if st.button("🧠 Train Early Warning Model", type="primary"):
        if not train_ids:
            st.error("Select at least one session for training.")
            return
        with st.spinner("Training model…"):
            try:
                sessions_data = [db.get_measurements(sid) for sid in train_ids]
                model         = VFModel(model_type=model_type, horizon=horizon)
                metrics       = model.train(sessions_data)
                st.session_state.vf_model     = model
                st.session_state.model_trained = True
                st.session_state.pred_result  = None
                st.success(
                    f"✅ Model trained.  Detection quality (AUC): **{metrics['auc']:.3f}** "
                    f"— values above 0.70 indicate useful predictive ability."
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

    if not st.session_state.model_trained or st.session_state.vf_model is None:
        st.info("👆 Configure and train the model above to see predictions.")
        return

    model: VFModel = st.session_state.vf_model

    # ── Model performance (advanced) ──────────────────────────────────────────
    with st.expander("📈 Model Performance Details (Advanced)", expanded=False):
        m = model.metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Detection Quality (AUC)", f"{m['auc']:.3f}",
                   help="1.0 = perfect, 0.5 = random. Above 0.70 is clinically useful.")
        mc2.metric("Training samples", f"{m['train_samples']:,}")
        mc3.metric("Test samples",     f"{m['test_samples']:,}")
        if "report" in m:
            rep  = m["report"]
            prec = rep.get("1", {}).get("precision", 0)
            rec  = rep.get("1", {}).get("recall",    0)
            f1   = rep.get("1", {}).get("f1-score",  0)
            st.markdown(
                f"Alarm detection — Precision: **{prec:.2f}** · Recall: **{rec:.2f}** · F1: **{f1:.2f}**"
            )

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### What Drives Alarm Risk?")
    _note(
        "The chart below shows which measurements have the most influence on the model's "
        "alarm risk estimate. Longer bars indicate a stronger early warning signal."
    )

    if model.feature_importance:
        top  = model.feature_importance[:8]
        fi_df = pd.DataFrame(top)
        fig_fi = _base_fig(
            title="Signal Importance for Alarm Prediction",
            xaxis_title="Relative Influence (%)",
            yaxis_title="",
            height=50 + 35 * len(top),
        )
        for _, r in fi_df.iterrows():
            shade = int(40 + 180 * r["importance_pct"] / 100)
            fig_fi.add_trace(go.Bar(
                x=[r["importance_pct"]],
                y=[r["label"]],
                orientation="h",
                marker_color=f"rgb(27,{shade},{min(shade+30,200)})",
                showlegend=False,
                text=[f"{r['importance_pct']:.0f}%"],
                textposition="outside",
                textfont=dict(size=13),
            ))
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
            xaxis=dict(range=[0, 110], tickfont=dict(size=12)),
            margin=dict(l=260, r=60, t=60, b=50),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("💬 Clinical interpretation", expanded=True):
            st.markdown(model.explain())

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("### Step 2 — Generate Risk Prediction")

    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        predict_id = st.selectbox(
            "Session to predict on",
            options=selected_ids,
            format_func=lambda k: _session_label(
                sessions_df[sessions_df["session_id"] == k].iloc[0]
            ),
            key="predict_session",
        )
    with p_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_pred = st.button("▶ Run Prediction", type="secondary", use_container_width=True)

    if run_pred:
        with st.spinner("Running prediction…"):
            try:
                pred_df_raw = db.get_measurements(predict_id)
                pred_result = model.predict(pred_df_raw)
                st.session_state["pred_result"] = pred_result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

    pred_result: Optional[pd.DataFrame] = st.session_state.get("pred_result")
    if pred_result is None:
        return

    # KPIs
    high_pct = (pred_result["risk_level"] == "High").mean() * 100
    mod_pct  = (pred_result["risk_level"] == "Moderate").mean() * 100
    mean_risk = pred_result["risk_score"].mean()

    rk1, rk2, rk3 = st.columns(3)
    rk1.metric("High-Risk Periods",     f"{high_pct:.1f}% of session",
               help="Fraction of time classified as high alarm risk.")
    rk2.metric("Moderate-Risk Periods", f"{mod_pct:.1f}%")
    rk3.metric("Mean Risk Score",       f"{mean_risk:.2f}",
               help="Average predicted alarm probability (0 = none, 1 = certain).")

    # Risk chart
    st.markdown("### Predicted Risk Over Time")
    _note(
        "The blue area shows the model's estimated alarm probability at each time point. "
        "The light red zone (above 0.66) is the high-risk region. "
        "Red triangles mark where the device actually triggered an alarm."
    )

    fig_risk = _base_fig(
        title="Predicted Alarm Risk Score",
        xaxis_title="Date / Time",
        yaxis_title="Alarm Risk (0 = low, 1 = high)",
        height=400,
        yaxis_range=[0, 1.05],
    )

    fig_risk.add_hrect(y0=0.66, y1=1.0, fillcolor=C["alarm_fill"], line_width=0,
                       annotation_text="High risk zone", annotation_position="top right",
                       annotation_font=dict(size=12, color=C["alarm"]))
    fig_risk.add_hrect(y0=0.33, y1=0.66, fillcolor="rgba(202,111,30,0.07)", line_width=0)

    fig_risk.add_trace(go.Scatter(
        x=pred_result["timestamp"], y=pred_result["risk_score"],
        mode="lines", name="Predicted risk",
        line=dict(color=C["accent"], width=2.2),
        fill="tozeroy", fillcolor=C["accent_fill"],
    ))

    if "alarm_active" in pred_result.columns:
        alm = pred_result[pred_result["alarm_active"] == 1]
        if not alm.empty:
            fig_risk.add_trace(go.Scatter(
                x=alm["timestamp"], y=alm["risk_score"],
                mode="markers", name="Actual alarm",
                marker=dict(color=C["alarm"], size=8, symbol="triangle-up",
                            line=dict(width=1.5, color="white")),
            ))

    for y_val, lbl, col in [
        (0.66, "High risk threshold",      C["alarm"]),
        (0.33, "Moderate risk threshold",  C["warn"]),
    ]:
        fig_risk.add_hline(
            y=y_val, line_dash="dot", line_color=col, line_width=1.5,
            annotation_text=lbl, annotation_position="left",
            annotation_font=dict(size=12, color=col),
        )

    fig_risk.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
    st.plotly_chart(fig_risk, use_container_width=True)

    # Predicted vs actual
    if "alarm_active" in pred_result.columns:
        st.markdown("### Predicted Risk vs Actual Alarms")
        _note(
            "Top panel: the model's predicted risk score over time. "
            "Bottom panel: when the device actually triggered an alarm (1 = alarm, 0 = none). "
            "Good alignment between the panels indicates the model is detecting alarm patterns early."
        )

        fig_c = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Predicted Risk Score", "Actual Device Alarm State (0 = off, 1 = on)"],
            vertical_spacing=0.12,
        )
        for ann in fig_c.layout.annotations:
            ann.font = dict(size=15, color=C["primary"])
            ann.x    = 0; ann.xanchor = "left"

        fig_c.add_trace(go.Scatter(
            x=pred_result["timestamp"], y=pred_result["risk_score"],
            mode="lines", name="Predicted risk",
            line=dict(color=C["accent"], width=2),
            fill="tozeroy", fillcolor=C["accent_fill"],
        ), row=1, col=1)

        fig_c.add_trace(go.Scatter(
            x=pred_result["timestamp"],
            y=pred_result["alarm_active"].astype(float),
            mode="lines", name="Actual alarm",
            line=dict(color=C["alarm"], width=1.8),
            fill="tozeroy", fillcolor=C["alarm_fill"],
        ), row=2, col=1)

        fig_c.update_layout(
            **CHART_BASE_SUB,
            height=380,
            legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=13)),
        )
        fig_c.update_xaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=12))
        fig_c.update_yaxes(showgrid=True, gridcolor=C["grid"], tickfont=dict(size=12))
        fig_c.update_xaxes(title_text="Date / Time", title_font=dict(size=13), row=2, col=1)
        st.plotly_chart(fig_c, use_container_width=True)

    with st.expander("⚠️ Clinical Disclaimer", expanded=False):
        st.warning(
            "This risk prediction module is a **research prototype** and has not been validated "
            "for clinical use. All clinical decisions must remain with the responsible clinician."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Data Audit  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_audit(db: VFDatabase, session_id: int) -> None:
    st.markdown("## Data Quality & Audit Trail")
    st.caption(
        "Records every transformation applied to the raw device file. For QA review."
    )

    sessions_df = db.get_sessions()
    srow = sessions_df[sessions_df["session_id"] == session_id]
    if srow.empty:
        st.warning("Session record not found.")
        return
    row = srow.iloc[0]

    st.markdown("### File Information")
    m = st.columns(3)
    m[0].info(f"**Case ID:** {row['case_id']}")
    m[1].info(f"**File:** {row['filename']}")
    m[2].info(f"**Records loaded:** {int(row['row_count']):,}")

    st.markdown("### Cleaning Steps Applied")
    try:
        prov = json.loads(row["provenance"])
    except Exception:
        st.warning("Audit data unavailable.")
        return

    step_labels = {
        "timestamp":    "🕐 Timestamp Standardisation",
        "measurements": "📏 Measurement Conversion",
        "alarm_limits": "⚙️ Alarm Limit Conversion",
        "status_flags": "🚦 Alarm Status Processing",
        "case_id":      "🔖 Case Identification",
        "dedup":        "🧹 Duplicate Removal",
        "final_shape":  "✅ Final Dataset",
    }
    step_desc = {
        "timestamp":    "Converted device timestamp to standard date/time. Sorted chronologically.",
        "measurements": "Replaced device codes ('--','Lo','Hi') with blank values. Converted to numbers.",
        "alarm_limits": "Converted alarm thresholds. Auto/Disabled limits stored as blank.",
        "status_flags": "Encoded alarm flags as true/false.",
        "case_id":      "Identified case number from file.",
        "dedup":        "Removed exact duplicate rows from USB export glitches.",
        "final_shape":  "Summary of cleaned dataset.",
    }

    for step, details in prov.items():
        with st.expander(step_labels.get(step, f"🔧 {step}"),
                         expanded=(step == "final_shape")):
            if step in step_desc:
                st.caption(step_desc[step])
            if isinstance(details, dict):
                clean_d = {k: v for k, v in details.items() if not isinstance(v, dict)}
                if clean_d:
                    st.json(clean_d)
                if step == "measurements":
                    rows = []
                    for col, col_data in details.items():
                        if isinstance(col_data, dict) and "sentinel_count" in col_data:
                            rows.append({
                                "Parameter":             DISPLAY_LABELS.get(col, col),
                                "Unavailable readings":  col_data["sentinel_count"],
                                "Total missing":         col_data.get("total_nan_after", "—"),
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows).set_index("Parameter"),
                                     use_container_width=True)
            else:
                st.write(details)

    with st.expander("🗂️ Raw Session Record (for technical review)", expanded=False):
        st.json(dict(row))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    _init_state()
    db = st.session_state.db
    active_id, selected_ids, filters = _render_sidebar(db)

    if active_id is None:
        _render_landing()
        return

    _render_dashboard(db, active_id, selected_ids, filters)


if __name__ == "__main__":
    main()
