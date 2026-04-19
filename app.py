"""
app.py — VF Analyst Clinical Dashboard (Clinician Edition)
===========================================================
Refactored for clinical users: doctors, perfusionists, ICU technicians.

Design principles applied:
- "Make the right thing obvious" — the most important info is always visible first
- All technical labels replaced with clinical language
- Progressive disclosure: advanced data hidden in expanders
- Alarm states use restrained, professional clinical color conventions
- Minimal sidebar; filters appear contextually, not all at once
- Every chart has a one-sentence plain-language description
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import MEASUREMENT_COLS, DISPLAY_LABELS
from data_loader import load_csv
from data_cleaning import clean, compute_summary
from data_model import VFDatabase

# ─────────────────────────────────────────────────────────────────────────────
# Page & logging setup
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
# Clinical colour palette  (muted, professional — never flashy)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "primary":   "#1B4F72",
    "accent":    "#2E86C1",
    "alarm":     "#C0392B",
    "warn":      "#CA6F1E",
    "ok":        "#1E8449",
    "subtle":    "#7F8C8D",
    "bg_card":   "#F4F6F7",
    "grid":      "#E8EAEB",
    "line":      "#2E86C1",
    "line_alt":  "#117A65",
}

CHART_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Georgia, serif", size=13, color="#1B4F72"),
    margin=dict(l=50, r=30, t=50, b=50),
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Georgia', serif; }

section[data-testid="stSidebar"] { background-color: #1B4F72; }
section[data-testid="stSidebar"] * { color: #ECF0F1 !important; }
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #BDC3C7 !important; font-size: 0.82rem;
    letter-spacing: 0.04em; text-transform: uppercase;
}
section[data-testid="stSidebar"] hr { border-color: #2E86C1 !important; opacity: 0.4; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #D5D8DC; }
.stTabs [data-baseweb="tab"] {
    font-family: Georgia, serif; font-size: 0.95rem;
    color: #7F8C8D; padding: 10px 22px; border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background-color: #EAF2FB; color: #1B4F72 !important;
    font-weight: bold; border-bottom: 3px solid #2E86C1;
}

div[data-testid="metric-container"] {
    background: #F4F6F7; border: 1px solid #D5D8DC;
    border-radius: 10px; padding: 18px 20px 12px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    font-size: 0.75rem !important; letter-spacing: 0.06em;
    text-transform: uppercase; color: #7F8C8D !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: bold; color: #1B4F72 !important;
}

h1 { color: #1B4F72; font-size: 1.9rem; }
h2 { color: #1B4F72; font-size: 1.35rem; border-bottom: 1px solid #D5D8DC; padding-bottom: 6px; margin-top: 1.6rem; }
h3 { color: #2E86C1; font-size: 1.1rem; }

.chart-desc {
    background: #EBF5FB; border-left: 4px solid #2E86C1;
    padding: 8px 14px; border-radius: 0 6px 6px 0;
    font-size: 0.88rem; color: #1B4F72; margin-bottom: 12px;
}
.alarm-badge {
    display: inline-block; background: #FADBD8; color: #C0392B;
    border: 1px solid #E74C3C; border-radius: 20px;
    padding: 2px 12px; font-size: 0.82rem; font-weight: bold;
}
.ok-badge {
    display: inline-block; background: #D5F5E3; color: #1E8449;
    border: 1px solid #27AE60; border-radius: 20px;
    padding: 2px 12px; font-size: 0.82rem; font-weight: bold;
}
details summary { font-size: 0.88rem; color: #7F8C8D; letter-spacing: 0.03em; }
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chart_note(text: str) -> None:
    st.markdown(f'<div class="chart-desc">ℹ️ {text}</div>', unsafe_allow_html=True)

def _alarm_rate(alarm_count: int, total: int) -> str:
    if total == 0: return "—"
    return f"{alarm_count / total * 100:.1f}%"

def _base_fig(**extra) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**{**CHART_BASE, **extra})
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"], zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_db() -> VFDatabase:
    return VFDatabase(":memory:")

def _init_state() -> None:
    if "db" not in st.session_state:
        st.session_state.db = _get_db()
    if "active_session" not in st.session_state:
        st.session_state.active_session = None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(db: VFDatabase) -> tuple[int | None, dict]:
    with st.sidebar:
        st.markdown("""
<div style="text-align:center; padding: 18px 0 10px 0;">
    <span style="font-size:2.2rem;">🫀</span><br>
    <span style="font-size:1.25rem; font-weight:bold; letter-spacing:0.06em;">VF ANALYST</span><br>
    <span style="font-size:0.75rem; opacity:0.65; letter-spacing:0.08em;">ECMO · PERFUSION · VAD</span>
</div>""", unsafe_allow_html=True)
        st.divider()

        st.markdown("**Upload Examination File**")
        st.caption("Select the CSV file exported from the ECMO or VAD device.")
        uploaded = st.file_uploader(
            "Drag file here or click to browse",
            type=["csv"], label_visibility="collapsed",
        )
        if uploaded is not None:
            _process_upload(db, uploaded)

        st.divider()
        sessions_df = db.get_sessions()

        if sessions_df.empty:
            st.info("No examination files loaded yet.")
            return None, {}

        st.markdown("**Active Session**")
        labels = {
            row["session_id"]: f"Case {row['case_id']}  ·  {row['filename']}"
            for _, row in sessions_df.iterrows()
        }
        default_idx = 0
        if st.session_state.active_session in labels:
            default_idx = list(labels.keys()).index(st.session_state.active_session)

        selected_id = st.selectbox(
            "Session", options=list(labels.keys()),
            format_func=lambda k: labels[k],
            index=default_idx, label_visibility="collapsed",
        )

        meas_df = db.get_measurements(selected_id)
        filters: dict = {}

        if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
            st.divider()
            st.markdown("**Filter by Time Window**")
            st.caption("Drag handles to focus on a specific period.")
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
                "Show alarm periods only", value=False,
                help="When enabled, only time points with an active alarm are shown.",
            )

        st.divider()
        row = sessions_df[sessions_df["session_id"] == selected_id].iloc[0]
        st.markdown(f"""
<div style="font-size:0.78rem; opacity:0.7; line-height:1.8;">
📁 {row['filename']}<br>
📋 {int(row['row_count']):,} records<br>
🕐 Loaded {row['uploaded_at'][:16]}
</div>""", unsafe_allow_html=True)

    return selected_id, filters


def _process_upload(db: VFDatabase, uploaded) -> None:
    with st.sidebar:
        with st.status("Analysing file…", expanded=True) as status:
            try:
                st.write("Reading device data…")
                raw_df, load_warnings = load_csv(uploaded)
                st.write("Standardising measurements…")
                clean_df, provenance = clean(raw_df)
                st.write("Saving session…")
                ts_valid   = "timestamp" in clean_df.columns and clean_df["timestamp"].notna().any()
                time_start = str(clean_df["timestamp"].min()) if ts_valid else None
                time_end   = str(clean_df["timestamp"].max()) if ts_valid else None
                case_id    = clean_df["case_id"].iloc[0] if "case_id" in clean_df.columns else "unknown"
                sid = db.insert_session(
                    case_id=case_id, filename=uploaded.name, row_count=len(clean_df),
                    time_start=time_start, time_end=time_end, provenance=provenance,
                )
                db.insert_measurements(clean_df, sid)
                db.insert_alarm_events(clean_df, sid)
                st.session_state.active_session = sid
                for w in load_warnings:
                    st.warning(w)
                status.update(label="✅ File ready", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="❌ Could not read file", state="error")
                st.error(f"Error: {exc}")
                logger.exception("Upload failed")


# ─────────────────────────────────────────────────────────────────────────────
# Landing page
# ─────────────────────────────────────────────────────────────────────────────

def _render_landing() -> None:
    st.markdown("""
<div style="max-width: 640px; margin: 80px auto; text-align: center;">
    <div style="font-size: 3.5rem; margin-bottom: 16px;">🫀</div>
    <h1 style="color:#1B4F72; font-size:2rem; margin-bottom:8px;">VF Analyst</h1>
    <p style="color:#7F8C8D; font-size:1.05rem; margin-bottom:40px;">
        Clinical dashboard for ECMO &amp; VAD perfusion monitoring
    </p>
    <div style="background:#EAF2FB; border:1px solid #AED6F1; border-radius:12px;
                padding:28px 32px; text-align:left;">
        <p style="color:#1B4F72; font-weight:bold; margin-bottom:12px; font-size:1rem;">
            To get started:
        </p>
        <ol style="color:#2C3E50; line-height:2; font-size:0.95rem;">
            <li>Export the CSV log from the ECMO or VAD device via USB.</li>
            <li>Click <strong>"Upload Examination File"</strong> in the left panel.</li>
            <li>The dashboard will load automatically.</li>
        </ol>
    </div>
    <p style="color:#BDC3C7; font-size:0.78rem; margin-top:32px;">
        Compatible with Xenios · Maquet · Getinge device exports
    </p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def _render_dashboard(db: VFDatabase, session_id: int, filters: dict) -> None:
    meas_df  = db.get_measurements(session_id)
    alarm_df = db.get_alarm_events(session_id)

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

    # Case header
    sessions_df = db.get_sessions()
    srow = sessions_df[sessions_df["session_id"] == session_id].iloc[0]
    case_id = srow["case_id"]

    duration_min = None
    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        delta = meas_df["timestamp"].max() - meas_df["timestamp"].min()
        duration_min = round(delta.total_seconds() / 60, 0)

    alarm_count = int(meas_df["alarm_active"].sum()) if "alarm_active" in meas_df.columns else 0
    alarm_badge = (
        f'<span class="alarm-badge">⚠ {alarm_count:,} alarms ({_alarm_rate(alarm_count, len(meas_df))})</span>'
        if alarm_count > 0
        else '<span class="ok-badge">✓ No active alarms</span>'
    )
    dur_txt = f"· {int(duration_min)} min recorded" if duration_min else ""

    st.markdown(f"""
<div style="display:flex; align-items:baseline; gap:18px; padding:4px 0 18px 0;
            border-bottom: 2px solid #D5D8DC; margin-bottom:22px;">
    <span style="font-size:1.5rem; font-weight:bold; color:#1B4F72;">Case {case_id}</span>
    <span style="color:#7F8C8D; font-size:0.9rem;">{srow['filename']}</span>
    <span style="color:#7F8C8D; font-size:0.9rem;">{dur_txt}</span>
    <span style="margin-left:auto;">{alarm_badge}</span>
</div>""", unsafe_allow_html=True)

    tab_overview, tab_trends, tab_values, tab_alarms, tab_audit = st.tabs([
        "📋  Patient Overview",
        "📈  Vital Trends",
        "📊  Value Distributions",
        "🚨  Alarm Review",
        "🔍  Data Audit",
    ])

    with tab_overview:  _tab_overview(meas_df, alarm_df)
    with tab_trends:    _tab_trends(meas_df, alarm_df)
    with tab_values:    _tab_distributions(meas_df)
    with tab_alarms:    _tab_alarms(alarm_df, meas_df)
    with tab_audit:     _tab_audit(db, session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Patient Overview
# ─────────────────────────────────────────────────────────────────────────────

def _tab_overview(meas_df: pd.DataFrame, alarm_df: pd.DataFrame) -> None:
    summary = compute_summary(meas_df)

    st.markdown("## Session at a Glance")
    k = st.columns(4)
    k[0].metric("Total Measurements", f"{summary.get('total_records', 0):,}",
                help="Number of data points recorded by the device.")
    k[1].metric("Session Duration",
                f"{int(summary['duration_minutes'])} min" if summary.get("duration_minutes") else "—",
                help="Total time from first to last recorded measurement.")
    alarm_count = summary.get("alarm_events", 0)
    total       = summary.get("total_records", 1)
    k[2].metric("Alarm Periods", f"{alarm_count:,}",
                delta=f"{_alarm_rate(alarm_count, total)} of session" if alarm_count else None,
                delta_color="inverse",
                help="Number of time points where at least one alarm was active.")
    avg_flow = summary.get("flow_rate_lpm_mean")
    k[3].metric("Avg. Flow Rate", f"{avg_flow:.2f} L/min" if avg_flow else "—",
                help="Mean blood flow rate over the session.")

    st.markdown("## Key Measurement Summary")
    _chart_note("Average, minimum, and maximum values recorded for each parameter. "
                "Data Coverage shows what fraction of readings had a valid sensor signal.")

    stat_rows = []
    for col in MEASUREMENT_COLS:
        if col not in meas_df.columns or meas_df[col].dropna().empty:
            continue
        s = meas_df[col].dropna()
        stat_rows.append({
            "Parameter":     DISPLAY_LABELS.get(col, col),
            "Mean":          f"{s.mean():.1f}",
            "Min":           f"{s.min():.1f}",
            "Max":           f"{s.max():.1f}",
            "Std Dev":       f"{s.std():.1f}",
            "Data Coverage": f"{100*len(s)/len(meas_df):.0f}%",
        })
    if stat_rows:
        st.dataframe(
            pd.DataFrame(stat_rows).set_index("Parameter"),
            use_container_width=True,
            height=min(80 + 38 * len(stat_rows), 420),
        )

    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        st.markdown("## Quick View — Flow & Pump Speed")
        _chart_note("Compact overview of blood flow and pump speed. Red dots = alarm active.")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=["Flow Rate (L/min)", "Pump Speed (RPM)"])
        for ri, col in enumerate(["flow_rate_lpm", "pump_speed_rpm"], start=1):
            if col not in meas_df.columns: continue
            fig.add_trace(go.Scatter(
                x=meas_df["timestamp"], y=meas_df[col], mode="lines",
                line=dict(width=1.5, color=C["line"]), name=DISPLAY_LABELS.get(col, col),
                connectgaps=False,
            ), row=ri, col=1)
            if "alarm_active" in meas_df.columns:
                alm = meas_df[meas_df["alarm_active"] == 1]
                fig.add_trace(go.Scatter(
                    x=alm["timestamp"], y=alm[col], mode="markers",
                    marker=dict(color=C["alarm"], size=4),
                    name="Alarm" if ri == 1 else None, showlegend=(ri == 1),
                ), row=ri, col=1)

        fig.update_layout(**CHART_BASE, height=380, legend=dict(orientation="h", y=1.06))
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"])
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Vital Trends
# ─────────────────────────────────────────────────────────────────────────────

def _tab_trends(meas_df: pd.DataFrame, alarm_df: pd.DataFrame) -> None:
    st.markdown("## Vital Trends Over Time")

    if "timestamp" not in meas_df.columns or meas_df["timestamp"].isna().all():
        st.warning("No time information is available for this session.")
        return

    available = [c for c in MEASUREMENT_COLS if c in meas_df.columns and meas_df[c].notna().any()]
    if not available:
        st.warning("No measurement data to display.")
        return

    label_to_col = {DISPLAY_LABELS.get(c, c): c for c in available}
    _chart_note("Each panel shows how a parameter evolved during the session. "
                "Gaps indicate periods when the sensor signal was unavailable. "
                "Red dots mark active alarm periods.")

    selected_labels = st.multiselect(
        "Select parameters to display",
        options=list(label_to_col.keys()),
        default=list(label_to_col.keys())[:4],
        help="Choose one or more parameters from the device log.",
    )
    selected_cols = [label_to_col[l] for l in selected_labels]

    if not selected_cols:
        st.info("Please select at least one parameter above.")
        return

    n = len(selected_cols)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=[DISPLAY_LABELS.get(c, c) for c in selected_cols],
        vertical_spacing=max(0.04, 0.12 / n),
    )
    for i, col in enumerate(selected_cols, start=1):
        fig.add_trace(go.Scatter(
            x=meas_df["timestamp"], y=meas_df[col], mode="lines",
            name=DISPLAY_LABELS.get(col, col),
            line=dict(width=1.8, color=C["line"]), connectgaps=False,
        ), row=i, col=1)
        if "alarm_active" in meas_df.columns:
            alm = meas_df[meas_df["alarm_active"] == 1]
            if not alm.empty:
                fig.add_trace(go.Scatter(
                    x=alm["timestamp"], y=alm[col], mode="markers",
                    marker=dict(color=C["alarm"], size=5),
                    name="Alarm active" if i == 1 else None, showlegend=(i == 1),
                ), row=i, col=1)

    fig.update_layout(**CHART_BASE, height=max(240, 220 * n),
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("⚙️ Advanced — Show alarm limit bands", expanded=False):
        st.caption("Overlays configured alarm thresholds on the trend. "
                   "Values outside the dashed lines triggered an alarm.")
        _render_limits_chart(meas_df, selected_cols)


def _render_limits_chart(meas_df: pd.DataFrame, cols: list[str]) -> None:
    LIMIT_PAIRS = {
        "flow_rate_lpm":     ("alarm_flow_low_lpm",       "alarm_flow_high_lpm"),
        "pressure_pre_mmhg": ("alarm_pressure_pre_low",   "alarm_pressure_pre_high"),
        "pressure_post_mmhg":("alarm_pressure_post_low",  "alarm_pressure_post_high"),
        "sat_pre_pct":       ("alarm_sat_pre_low",        "alarm_sat_pre_high"),
        "temp_post_c":       ("alarm_temp_post_low",      "alarm_temp_post_high"),
    }
    for col in cols:
        if col not in LIMIT_PAIRS or col not in meas_df.columns: continue
        low_col, high_col = LIMIT_PAIRS[col]
        fig = _base_fig(height=250, title=DISPLAY_LABELS.get(col, col))
        fig.add_trace(go.Scatter(
            x=meas_df["timestamp"], y=meas_df[col], mode="lines",
            line=dict(width=1.5, color=C["line"]), name="Measured", connectgaps=False,
        ))
        for lim_col, lim_label in [(low_col, "Low limit"), (high_col, "High limit")]:
            if lim_col in meas_df.columns and meas_df[lim_col].notna().any():
                lim_val = meas_df[lim_col].median()
                fig.add_hline(
                    y=lim_val, line_dash="dot", line_color=C["alarm"], line_width=1.2,
                    annotation_text=f"{lim_label}: {lim_val:.1f}",
                    annotation_position="right", annotation_font_size=11,
                )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Distributions
# ─────────────────────────────────────────────────────────────────────────────

def _tab_distributions(meas_df: pd.DataFrame) -> None:
    st.markdown("## Value Distributions")
    _chart_note("These charts show how frequently each parameter was at different levels. "
                "The dashed line marks the session average.")

    available = [c for c in MEASUREMENT_COLS if c in meas_df.columns and meas_df[c].notna().any()]
    if not available:
        st.warning("No measurement data available.")
        return

    col1, col2 = st.columns(2)
    for idx, col in enumerate(available):
        values = meas_df[col].dropna()
        label  = DISPLAY_LABELS.get(col, col)
        mean_v = values.mean()

        fig = _base_fig(height=290, title=label)
        fig.add_trace(go.Histogram(x=values, nbinsx=35,
                                   marker_color=C["accent"], opacity=0.80, name=label))
        fig.add_vline(x=mean_v, line_dash="dash", line_color=C["primary"], line_width=1.5,
                      annotation_text=f"Avg {mean_v:.1f}", annotation_position="top right",
                      annotation_font_size=11)
        fig.update_layout(showlegend=False, xaxis_title=label, yaxis_title="Frequency")

        (col1 if idx % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

    with st.expander("📐 Advanced — Parameter Correlation Matrix", expanded=False):
        st.caption("Shows how strongly each pair of parameters moves together "
                   "(1 = same direction, −1 = opposite, 0 = no relationship).")
        corr_cols = [c for c in available if meas_df[c].nunique() > 1]
        if len(corr_cols) >= 2:
            corr   = meas_df[corr_cols].corr()
            labels = [DISPLAY_LABELS.get(c, c) for c in corr_cols]
            fig_c  = px.imshow(corr.values, x=labels, y=labels,
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                               text_auto=".2f", aspect="auto")
            fig_c.update_layout(**CHART_BASE, height=460)
            st.plotly_chart(fig_c, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Alarm Review
# ─────────────────────────────────────────────────────────────────────────────

def _tab_alarms(alarm_df: pd.DataFrame, meas_df: pd.DataFrame) -> None:
    st.markdown("## Alarm Review")

    if alarm_df.empty:
        st.success("✅ **No alarms were recorded** during the selected time window.")
        return

    total_alm = len(alarm_df)
    total_rec = len(meas_df)
    k1, k2, k3 = st.columns(3)
    k1.metric("Alarm Events", f"{total_alm:,}",
              help="Total alarm events in the selected period.")
    k2.metric("Alarm Coverage", _alarm_rate(total_alm, total_rec),
              help="Fraction of time points with at least one active alarm.")
    if "alarm_type" in alarm_df.columns:
        k3.metric("Alarm Types", alarm_df["alarm_type"].nunique(),
                  help="Number of distinct alarm categories detected.")

    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        st.markdown("### Alarm Timeline")
        _chart_note("Flow rate over time. Red dots indicate active alarm periods.")

        fig = _base_fig(height=280)
        fig.add_trace(go.Scatter(
            x=meas_df["timestamp"], y=meas_df.get("flow_rate_lpm", pd.Series(dtype=float)),
            mode="lines", name="Flow Rate (L/min)", line=dict(color=C["line"], width=1.5),
        ))
        if "alarm_active" in meas_df.columns:
            alm = meas_df[meas_df["alarm_active"] == 1]
            fig.add_trace(go.Scatter(
                x=alm["timestamp"], y=alm.get("flow_rate_lpm", pd.Series(dtype=float)),
                mode="markers", marker=dict(color=C["alarm"], size=5), name="Alarm active",
            ))
        fig.update_layout(yaxis_title="Flow Rate (L/min)", legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, use_container_width=True)

    if "alarm_type" in alarm_df.columns:
        st.markdown("### Alarm Type Breakdown")
        _chart_note("Number of events per alarm category.")
        counts = alarm_df["alarm_type"].value_counts().reset_index()
        counts.columns = ["Alarm Type", "Events"]
        counts["Alarm Type"] = counts["Alarm Type"].replace({
            "overall": "General Alarm", "bubble": "Bubble / Air Alarm", "technical": "Technical Fault",
        })
        fig_b = px.bar(counts, x="Alarm Type", y="Events", color="Alarm Type",
                       color_discrete_sequence=[C["alarm"], C["warn"], C["subtle"]], text="Events")
        fig_b.update_traces(textposition="outside")
        fig_b.update_layout(**CHART_BASE, showlegend=False, height=300,
                            yaxis_title="Number of Events", xaxis_title="")
        st.plotly_chart(fig_b, use_container_width=True)

    with st.expander("📋 Detailed Alarm Log (Advanced)", expanded=False):
        st.caption("Full record of each alarm event with timestamp and type.")
        display_alarm = alarm_df.rename(columns={
            "timestamp": "Date & Time", "case_id": "Case",
            "alarm_type": "Alarm Category", "alarm_value": "Status Code",
        }, errors="ignore")
        if "Alarm Category" in display_alarm.columns:
            display_alarm["Alarm Category"] = display_alarm["Alarm Category"].replace({
                "overall": "General", "bubble": "Bubble / Air", "technical": "Technical",
            })
        st.dataframe(display_alarm, use_container_width=True, height=340)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Data Audit
# ─────────────────────────────────────────────────────────────────────────────

def _tab_audit(db: VFDatabase, session_id: int) -> None:
    st.markdown("## Data Quality & Audit Trail")
    st.caption("Records every step applied to transform the raw device file into "
               "the values shown on this dashboard. Intended for QA review.")

    sessions_df = db.get_sessions()
    srow = sessions_df[sessions_df["session_id"] == session_id]
    if srow.empty:
        st.warning("Session record not found.")
        return
    row = srow.iloc[0]

    st.markdown("### File Information")
    m = st.columns(3)
    m[0].info(f"**Case ID:** {row['case_id']}")
    m[1].info(f"**File name:** {row['filename']}")
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
        "timestamp":    "Converted device timestamp format to standard date/time. Records sorted chronologically.",
        "measurements": "Replaced device codes (e.g. '--', 'Lo', 'Hi') with blank values. Converted readings to numbers.",
        "alarm_limits": "Converted alarm threshold settings. 'Auto' and 'Disabled' limits stored as blank.",
        "status_flags": "Encoded alarm flags as true/false. Classified bubble sensor status.",
        "case_id":      "Identified the case number from the device file.",
        "dedup":        "Removed duplicate rows that may arise from USB export glitches.",
        "final_shape":  "Summary of the cleaned dataset dimensions.",
    }

    for step, details in prov.items():
        with st.expander(step_labels.get(step, f"🔧 {step}"), expanded=(step == "final_shape")):
            if step in step_desc:
                st.caption(step_desc[step])
            if isinstance(details, dict):
                clean_details = {k: v for k, v in details.items() if not isinstance(v, dict)}
                if clean_details:
                    st.json(clean_details)
                if step == "measurements":
                    rows = []
                    for col, col_data in details.items():
                        if isinstance(col_data, dict) and "sentinel_count" in col_data:
                            rows.append({
                                "Parameter": DISPLAY_LABELS.get(col, col),
                                "Unavailable readings": col_data["sentinel_count"],
                                "Total missing after cleaning": col_data.get("total_nan_after", "—"),
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
    session_id, filters = _render_sidebar(db)
    if session_id is None:
        _render_landing()
        return
    _render_dashboard(db, session_id, filters)

if __name__ == "__main__":
    main()
