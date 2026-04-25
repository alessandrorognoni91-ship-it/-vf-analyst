"""
app.py — VF Analyst Clinical Dashboard  v2.0
==============================================
Extended from v1.0 to support:
  - Multi-session upload and selection
  - Case Comparison tab (side-by-side metrics + overlay charts)
  - Cohort Insights tab (boxplots, alarm frequency across cases)
  - Risk Prediction tab (Random Forest early-warning system)

Original single-session tabs (Patient Overview, Vital Trends,
Value Distributions, Alarm Review, Data Audit) are unchanged.
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
# Page config & logging
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
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "primary":  "#1B4F72",
    "accent":   "#2E86C1",
    "alarm":    "#C0392B",
    "warn":     "#CA6F1E",
    "ok":       "#1E8449",
    "subtle":   "#7F8C8D",
    "grid":     "#E8EAEB",
    "line":     "#2E86C1",
}

# Distinct colours for multi-session overlays (up to 5 sessions)
SESSION_COLORS = ["#2E86C1", "#1E8449", "#8E44AD", "#CA6F1E", "#C0392B"]

CHART_BASE = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Georgia, serif", size=13, color="#1B4F72"),
    margin=dict(l=50, r=30, t=50, b=50),
)

MAX_COMPARE = 5   # maximum simultaneous sessions for comparison


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Georgia', serif; }
section[data-testid="stSidebar"] { background-color: #1B4F72; }
section[data-testid="stSidebar"] * { color: #ECF0F1 !important; }
section[data-testid="stSidebar"] hr { border-color: #2E86C1 !important; opacity: 0.4; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #D5D8DC; }
.stTabs [data-baseweb="tab"] {
    font-family: Georgia, serif; font-size: 0.9rem;
    color: #7F8C8D; padding: 8px 18px; border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background-color: #EAF2FB; color: #1B4F72 !important;
    font-weight: bold; border-bottom: 3px solid #2E86C1;
}
div[data-testid="metric-container"] {
    background: #F4F6F7; border: 1px solid #D5D8DC;
    border-radius: 10px; padding: 16px 18px 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important; letter-spacing: 0.06em;
    text-transform: uppercase; color: #7F8C8D !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: bold; color: #1B4F72 !important;
}
h1 { color: #1B4F72; font-size: 1.8rem; }
h2 { color: #1B4F72; font-size: 1.3rem;
     border-bottom: 1px solid #D5D8DC; padding-bottom: 5px; margin-top: 1.4rem; }
h3 { color: #2E86C1; font-size: 1.05rem; }
.chart-desc {
    background: #EBF5FB; border-left: 4px solid #2E86C1;
    padding: 7px 12px; border-radius: 0 6px 6px 0;
    font-size: 0.86rem; color: #1B4F72; margin-bottom: 10px;
}
.alarm-badge {
    display:inline-block; background:#FADBD8; color:#C0392B;
    border:1px solid #E74C3C; border-radius:20px;
    padding:2px 10px; font-size:0.80rem; font-weight:bold;
}
.ok-badge {
    display:inline-block; background:#D5F5E3; color:#1E8449;
    border:1px solid #27AE60; border-radius:20px;
    padding:2px 10px; font-size:0.80rem; font-weight:bold;
}
.risk-high     { color:#C0392B; font-weight:bold; }
.risk-moderate { color:#CA6F1E; font-weight:bold; }
.risk-low      { color:#1E8449; font-weight:bold; }
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _note(text: str) -> None:
    st.markdown(f'<div class="chart-desc">ℹ️ {text}</div>', unsafe_allow_html=True)

def _alarm_rate(n: int, total: int) -> str:
    return f"{n/total*100:.1f}%" if total > 0 else "—"

def _base_fig(**kw) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**{**CHART_BASE, **kw})
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"], zeroline=False)
    return fig

def _session_label(row: pd.Series) -> str:
    return f"Case {row['case_id']}  ·  {row['filename']}"


# ─────────────────────────────────────────────────────────────────────────────
# Database & state
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_db() -> VFDatabase:
    return VFDatabase(":memory:")

def _init_state() -> None:
    defaults = {
        "db": _get_db(),
        "active_session": None,
        "selected_sessions": [],   # list[int] — for multi-session views
        "vf_model": None,          # VFModel instance
        "model_trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (EXTENDED: multi-select + multi-upload)
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(db: VFDatabase) -> tuple[Optional[int], list[int], dict]:
    """
    Returns
    -------
    active_id      : int | None   — primary session for single-session tabs
    selected_ids   : list[int]    — sessions chosen for multi-session tabs
    filters        : dict         — time range / alarm-only toggle
    """
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:16px 0 8px;">
<span style="font-size:2rem;">🫀</span><br>
<span style="font-size:1.2rem;font-weight:bold;letter-spacing:.06em;">VF ANALYST</span><br>
<span style="font-size:.72rem;opacity:.6;letter-spacing:.08em;">ECMO · PERFUSION · VAD</span>
</div>""", unsafe_allow_html=True)
        st.divider()

        # ── Multi-file upload ────────────────────────────────────────────────
        st.markdown("**Upload Examination Files**")
        st.caption("You can upload multiple CSV files at once.")
        uploaded_files = st.file_uploader(
            "Drag files here or click to browse",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            for f in uploaded_files:
                _process_upload(db, f)

        # ── Session list ─────────────────────────────────────────────────────
        st.divider()
        sessions_df = db.get_sessions()
        if sessions_df.empty:
            st.info("No files loaded yet.")
            return None, [], {}

        id_to_row = {int(r["session_id"]): r for _, r in sessions_df.iterrows()}
        all_ids   = list(id_to_row.keys())

        # Active (primary) session — single selectbox
        st.markdown("**Active Session**")
        st.caption("Used in Overview, Trends, Alarms, Audit tabs.")

        if st.session_state.active_session not in all_ids:
            st.session_state.active_session = all_ids[0]

        active_id = st.selectbox(
            "Active session",
            options=all_ids,
            format_func=lambda k: _session_label(id_to_row[k]),
            index=all_ids.index(st.session_state.active_session),
            label_visibility="collapsed",
        )
        st.session_state.active_session = active_id

        # Sessions for comparison / cohort
        st.divider()
        st.markdown("**Sessions for Comparison**")
        st.caption(f"Select up to {MAX_COMPARE} cases for the Comparison & Cohort tabs.")

        prev_sel = [i for i in st.session_state.selected_sessions if i in all_ids]
        selected_ids = st.multiselect(
            "Compare sessions",
            options=all_ids,
            default=prev_sel or all_ids[:min(2, len(all_ids))],
            format_func=lambda k: _session_label(id_to_row[k]),
            max_selections=MAX_COMPARE,
            label_visibility="collapsed",
        )
        st.session_state.selected_sessions = selected_ids

        # ── Filters (applied to active session single-session tabs) ──────────
        filters: dict = {}
        meas_df = db.get_measurements(active_id)

        if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
            st.divider()
            st.markdown("**Time Window** *(active session)*")
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
                help="Restricts single-session charts to alarm time points.",
            )

        # Footer
        st.divider()
        st.markdown(
            f'<div style="font-size:.75rem;opacity:.6;line-height:1.8;">'
            f'📁 {len(all_ids)} session(s) loaded</div>',
            unsafe_allow_html=True,
        )

    return active_id, selected_ids, filters


def _process_upload(db: VFDatabase, uploaded) -> None:
    """Ingest one CSV file; skip silently if already loaded (same filename)."""
    sessions_df = db.get_sessions()
    if not sessions_df.empty and uploaded.name in sessions_df["filename"].values:
        return   # already loaded — avoid duplicates on re-render

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
                st.session_state.active_session = sid
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
<div style="max-width:620px;margin:70px auto;text-align:center;">
  <div style="font-size:3rem;margin-bottom:14px;">🫀</div>
  <h1 style="color:#1B4F72;font-size:1.9rem;margin-bottom:6px;">VF Analyst</h1>
  <p style="color:#7F8C8D;font-size:1rem;margin-bottom:36px;">
    Clinical dashboard for ECMO &amp; VAD perfusion monitoring
  </p>
  <div style="background:#EAF2FB;border:1px solid #AED6F1;border-radius:12px;
              padding:24px 28px;text-align:left;">
    <p style="color:#1B4F72;font-weight:bold;margin-bottom:10px;">To get started:</p>
    <ol style="color:#2C3E50;line-height:2;font-size:.93rem;">
      <li>Export CSV log(s) from the ECMO / VAD device via USB.</li>
      <li>Click <strong>"Upload Examination Files"</strong> in the left panel.</li>
      <li>You can upload <strong>multiple files</strong> to enable case comparison.</li>
    </ol>
  </div>
  <p style="color:#BDC3C7;font-size:.75rem;margin-top:28px;">
    Compatible with Xenios · Maquet · Getinge device exports
  </p>
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
    meas_df  = db.get_measurements(active_id)
    alarm_df = db.get_alarm_events(active_id)

    # Apply filters to single-session data
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
    srow        = sessions_df[sessions_df["session_id"] == active_id].iloc[0]
    alarm_count = int(meas_df["alarm_active"].sum()) if "alarm_active" in meas_df.columns else 0
    alarm_badge = (
        f'<span class="alarm-badge">⚠ {alarm_count:,} alarms'
        f' ({_alarm_rate(alarm_count, len(meas_df))})</span>'
        if alarm_count > 0
        else '<span class="ok-badge">✓ No active alarms</span>'
    )
    dur_txt = ""
    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        delta = meas_df["timestamp"].max() - meas_df["timestamp"].min()
        dur_txt = f"· {int(delta.total_seconds()/60)} min"

    st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:16px;padding:4px 0 16px;
            border-bottom:2px solid #D5D8DC;margin-bottom:20px;">
  <span style="font-size:1.4rem;font-weight:bold;color:#1B4F72;">Case {srow['case_id']}</span>
  <span style="color:#7F8C8D;font-size:.88rem;">{srow['filename']} {dur_txt}</span>
  <span style="margin-left:auto;">{alarm_badge}</span>
</div>""", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    (tab_overview, tab_trends, tab_values,
     tab_alarms, tab_compare, tab_cohort,
     tab_risk, tab_audit) = st.tabs([
        "📋 Overview",
        "📈 Vital Trends",
        "📊 Distributions",
        "🚨 Alarms",
        "⚖️ Case Comparison",
        "🏥 Cohort Insights",
        "🧠 Risk Prediction",
        "🔍 Data Audit",
    ])

    with tab_overview:  _tab_overview(meas_df, alarm_df)
    with tab_trends:    _tab_trends(meas_df, alarm_df)
    with tab_values:    _tab_distributions(meas_df)
    with tab_alarms:    _tab_alarms(alarm_df, meas_df)
    with tab_compare:   _tab_compare(db, selected_ids, sessions_df)
    with tab_cohort:    _tab_cohort(db, selected_ids, sessions_df)
    with tab_risk:      _tab_risk(db, selected_ids, active_id, sessions_df)
    with tab_audit:     _tab_audit(db, active_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Patient Overview  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_overview(meas_df: pd.DataFrame, alarm_df: pd.DataFrame) -> None:
    summary = compute_summary(meas_df)
    st.markdown("## Session at a Glance")
    k = st.columns(4)
    k[0].metric("Total Measurements", f"{summary.get('total_records',0):,}",
                help="Number of data points recorded by the device.")
    k[1].metric("Duration",
                f"{int(summary['duration_minutes'])} min" if summary.get("duration_minutes") else "—")
    alarm_count = summary.get("alarm_events", 0)
    k[2].metric("Alarm Periods", f"{alarm_count:,}",
                delta=f"{_alarm_rate(alarm_count, summary.get('total_records',1))} of session"
                if alarm_count else None, delta_color="inverse")
    avg_flow = summary.get("flow_rate_lpm_mean")
    k[3].metric("Avg. Flow Rate", f"{avg_flow:.2f} L/min" if avg_flow else "—")

    st.markdown("## Key Measurement Summary")
    _note("Average, minimum, and maximum values for each parameter during this session.")
    stat_rows = []
    for col in MEASUREMENT_COLS:
        if col not in meas_df.columns or meas_df[col].dropna().empty: continue
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
        st.dataframe(pd.DataFrame(stat_rows).set_index("Parameter"),
                     use_container_width=True, height=min(80+38*len(stat_rows), 400))

    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        st.markdown("## Quick View — Flow & Pump Speed")
        _note("Blood flow and pump speed over the session. Red dots = alarm active.")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=["Flow Rate (L/min)", "Pump Speed (RPM)"])
        for ri, col in enumerate(["flow_rate_lpm", "pump_speed_rpm"], 1):
            if col not in meas_df.columns: continue
            fig.add_trace(go.Scatter(x=meas_df["timestamp"], y=meas_df[col],
                mode="lines", line=dict(width=1.5, color=C["line"]),
                name=DISPLAY_LABELS.get(col, col), connectgaps=False), row=ri, col=1)
            if "alarm_active" in meas_df.columns:
                alm = meas_df[meas_df["alarm_active"]==1]
                fig.add_trace(go.Scatter(x=alm["timestamp"], y=alm[col],
                    mode="markers", marker=dict(color=C["alarm"], size=4),
                    name="Alarm" if ri==1 else None, showlegend=(ri==1)), row=ri, col=1)
        fig.update_layout(**CHART_BASE, height=380, legend=dict(orientation="h", y=1.06))
        fig.update_xaxes(showgrid=True, gridcolor=C["grid"])
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Vital Trends  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_trends(meas_df: pd.DataFrame, alarm_df: pd.DataFrame) -> None:
    st.markdown("## Vital Trends Over Time")
    if "timestamp" not in meas_df.columns or meas_df["timestamp"].isna().all():
        st.warning("No time information available."); return
    available = [c for c in MEASUREMENT_COLS if c in meas_df.columns and meas_df[c].notna().any()]
    if not available:
        st.warning("No measurement data."); return

    label_to_col = {DISPLAY_LABELS.get(c, c): c for c in available}
    _note("Each panel shows parameter evolution over the session. "
          "Gaps = sensor unavailable. Red dots = alarm active.")
    selected_labels = st.multiselect(
        "Select parameters", options=list(label_to_col.keys()),
        default=list(label_to_col.keys())[:4])
    selected_cols = [label_to_col[l] for l in selected_labels]
    if not selected_cols:
        st.info("Select at least one parameter above."); return

    n   = len(selected_cols)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[DISPLAY_LABELS.get(c,c) for c in selected_cols],
                        vertical_spacing=max(0.04, 0.12/n))
    for i, col in enumerate(selected_cols, 1):
        fig.add_trace(go.Scatter(x=meas_df["timestamp"], y=meas_df[col],
            mode="lines", name=DISPLAY_LABELS.get(col,col),
            line=dict(width=1.8, color=C["line"]), connectgaps=False), row=i, col=1)
        if "alarm_active" in meas_df.columns:
            alm = meas_df[meas_df["alarm_active"]==1]
            if not alm.empty:
                fig.add_trace(go.Scatter(x=alm["timestamp"], y=alm[col],
                    mode="markers", marker=dict(color=C["alarm"], size=5),
                    name="Alarm active" if i==1 else None, showlegend=(i==1)),
                    row=i, col=1)
    fig.update_layout(**CHART_BASE, height=max(240, 220*n),
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Distributions  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_distributions(meas_df: pd.DataFrame) -> None:
    st.markdown("## Value Distributions")
    _note("How frequently each parameter was at different levels. "
          "Dashed line = session average.")
    available = [c for c in MEASUREMENT_COLS if c in meas_df.columns and meas_df[c].notna().any()]
    if not available:
        st.warning("No measurement data."); return
    c1, c2 = st.columns(2)
    for idx, col in enumerate(available):
        v = meas_df[col].dropna(); label = DISPLAY_LABELS.get(col, col)
        fig = _base_fig(height=280, title=label)
        fig.add_trace(go.Histogram(x=v, nbinsx=35,
                                   marker_color=C["accent"], opacity=0.80))
        fig.add_vline(x=v.mean(), line_dash="dash", line_color=C["primary"],
                      annotation_text=f"Avg {v.mean():.1f}",
                      annotation_position="top right", annotation_font_size=11)
        fig.update_layout(showlegend=False, xaxis_title=label, yaxis_title="Frequency")
        (c1 if idx%2==0 else c2).plotly_chart(fig, use_container_width=True)

    with st.expander("📐 Advanced — Correlation Matrix", expanded=False):
        corr_cols = [c for c in available if meas_df[c].nunique()>1]
        if len(corr_cols) >= 2:
            corr   = meas_df[corr_cols].corr()
            labels = [DISPLAY_LABELS.get(c,c) for c in corr_cols]
            fig_c  = px.imshow(corr.values, x=labels, y=labels,
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                               text_auto=".2f", aspect="auto")
            fig_c.update_layout(**CHART_BASE, height=440)
            st.plotly_chart(fig_c, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Alarms  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_alarms(alarm_df: pd.DataFrame, meas_df: pd.DataFrame) -> None:
    st.markdown("## Alarm Review")
    if alarm_df.empty:
        st.success("✅ No alarms recorded in the selected time window."); return

    k1, k2, k3 = st.columns(3)
    k1.metric("Alarm Events", f"{len(alarm_df):,}")
    k2.metric("Alarm Coverage", _alarm_rate(len(alarm_df), len(meas_df)))
    if "alarm_type" in alarm_df.columns:
        k3.metric("Alarm Types", alarm_df["alarm_type"].nunique())

    if "timestamp" in meas_df.columns and meas_df["timestamp"].notna().any():
        st.markdown("### Alarm Timeline")
        _note("Flow rate over time. Red dots = alarm active.")
        fig = _base_fig(height=260)
        fig.add_trace(go.Scatter(x=meas_df["timestamp"],
                                  y=meas_df.get("flow_rate_lpm", pd.Series(dtype=float)),
                                  mode="lines", name="Flow Rate (L/min)",
                                  line=dict(color=C["line"], width=1.5)))
        if "alarm_active" in meas_df.columns:
            alm = meas_df[meas_df["alarm_active"]==1]
            fig.add_trace(go.Scatter(x=alm["timestamp"],
                                      y=alm.get("flow_rate_lpm", pd.Series(dtype=float)),
                                      mode="markers",
                                      marker=dict(color=C["alarm"], size=5),
                                      name="Alarm active"))
        fig.update_layout(yaxis_title="Flow Rate (L/min)",
                          legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, use_container_width=True)

    if "alarm_type" in alarm_df.columns:
        counts = alarm_df["alarm_type"].value_counts().reset_index()
        counts.columns = ["Alarm Type", "Events"]
        counts["Alarm Type"] = counts["Alarm Type"].replace({
            "overall": "General Alarm", "bubble": "Bubble / Air", "technical": "Technical Fault"})
        fig_b = px.bar(counts, x="Alarm Type", y="Events",
                       color_discrete_sequence=[C["alarm"]], text="Events")
        fig_b.update_traces(textposition="outside")
        fig_b.update_layout(**CHART_BASE, showlegend=False, height=280,
                            yaxis_title="Events", xaxis_title="")
        st.plotly_chart(fig_b, use_container_width=True)

    with st.expander("📋 Detailed Alarm Log (Advanced)"):
        st.dataframe(alarm_df.rename(columns={
            "timestamp":"Date & Time","case_id":"Case",
            "alarm_type":"Category","alarm_value":"Status Code"}),
            use_container_width=True, height=300)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Case Comparison  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_compare(
    db: VFDatabase,
    selected_ids: list[int],
    sessions_df: pd.DataFrame,
) -> None:
    st.markdown("## Case Comparison")

    if len(selected_ids) < 2:
        st.info("Select at least 2 sessions in the sidebar to compare cases.")
        return

    _note(
        "Side-by-side comparison of key metrics and overlaid trends across "
        "the selected cases. Use the sidebar to add or remove sessions."
    )

    # ── Summary metrics table ────────────────────────────────────────────────
    st.markdown("### Summary Statistics per Case")

    compare_rows = []
    for sid in selected_ids:
        srow  = sessions_df[sessions_df["session_id"]==sid].iloc[0]
        mdf   = db.get_measurements(sid)
        adf   = db.get_alarm_events(sid)
        s     = compute_summary(mdf)

        row: dict = {
            "Case":              f"Case {srow['case_id']}",
            "File":              srow["filename"],
            "Duration (min)":    s.get("duration_minutes", "—"),
            "Records":           s.get("total_records", 0),
            "Alarm Events":      s.get("alarm_events", 0),
            "Alarm Rate":        _alarm_rate(s.get("alarm_events",0), s.get("total_records",1)),
        }
        for col in ["flow_rate_lpm", "pressure_delta_mmhg", "sat_pre_pct", "pump_speed_rpm"]:
            mean_val = s.get(f"{col}_mean")
            row[DISPLAY_LABELS.get(col, col)] = f"{mean_val:.1f}" if mean_val else "—"
        compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows).set_index("Case")
    st.dataframe(compare_df, use_container_width=True)

    # ── Overlay time-series ──────────────────────────────────────────────────
    st.markdown("### Overlay Trends")
    _note(
        "Each case is shown in a different colour. Time is shown as elapsed minutes "
        "from the start of each session so sessions with different start times align."
    )

    available_cols = []
    for sid in selected_ids:
        mdf = db.get_measurements(sid)
        for col in MEASUREMENT_COLS:
            if col in mdf.columns and mdf[col].notna().any() and col not in available_cols:
                available_cols.append(col)

    if not available_cols:
        st.warning("No shared measurement columns found."); return

    label_to_col = {DISPLAY_LABELS.get(c,c): c for c in available_cols}
    chosen_label = st.selectbox(
        "Parameter to compare", options=list(label_to_col.keys()),
        key="compare_param",
    )
    chosen_col = label_to_col[chosen_label]

    use_normalized = st.checkbox(
        "Normalize time axis (0–100% of session duration)",
        value=True,
        help="Aligns sessions of different lengths on the same x-axis.",
    )

    fig = _base_fig(height=380, title=f"{chosen_label} — Case Overlay",
                    xaxis_title="Session progress (%)" if use_normalized else "Elapsed time (min)",
                    yaxis_title=chosen_label)

    for i, sid in enumerate(selected_ids):
        srow  = sessions_df[sessions_df["session_id"]==sid].iloc[0]
        mdf   = db.get_measurements(sid)
        color = SESSION_COLORS[i % len(SESSION_COLORS)]
        label = f"Case {srow['case_id']}"

        if chosen_col not in mdf.columns or mdf[chosen_col].isna().all():
            continue
        if "timestamp" not in mdf.columns:
            continue

        mdf = mdf.sort_values("timestamp").dropna(subset=["timestamp"])
        elapsed_min = (mdf["timestamp"] - mdf["timestamp"].min()).dt.total_seconds() / 60

        x_vals = (elapsed_min / elapsed_min.max() * 100) if use_normalized else elapsed_min

        fig.add_trace(go.Scatter(
            x=x_vals, y=mdf[chosen_col],
            mode="lines", name=label,
            line=dict(width=1.8, color=color),
            connectgaps=False,
        ))

    st.plotly_chart(fig, use_container_width=True)

    # ── Alarm rate bar chart ─────────────────────────────────────────────────
    st.markdown("### Alarm Rate Comparison")
    _note("Percentage of time points during which a device alarm was active, per case.")

    alarm_bar_rows = []
    for sid in selected_ids:
        srow = sessions_df[sessions_df["session_id"]==sid].iloc[0]
        mdf  = db.get_measurements(sid)
        n_alarm = int(mdf["alarm_active"].sum()) if "alarm_active" in mdf.columns else 0
        alarm_bar_rows.append({
            "Case":       f"Case {srow['case_id']}",
            "Alarm Rate": round(n_alarm / max(len(mdf),1) * 100, 1),
        })

    alarm_bar_df = pd.DataFrame(alarm_bar_rows)
    fig_b = px.bar(alarm_bar_df, x="Case", y="Alarm Rate",
                   color="Case",
                   color_discrete_sequence=SESSION_COLORS,
                   text="Alarm Rate",
                   labels={"Alarm Rate": "Alarm Rate (%)"})
    fig_b.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_b.update_layout(**CHART_BASE, showlegend=False, height=300,
                        yaxis_title="Alarm Rate (%)", yaxis_range=[0, 100])
    st.plotly_chart(fig_b, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Cohort Insights  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_cohort(
    db: VFDatabase,
    selected_ids: list[int],
    sessions_df: pd.DataFrame,
) -> None:
    st.markdown("## Cohort Insights")

    if len(selected_ids) < 2:
        st.info("Select at least 2 sessions in the sidebar to view cohort-level analysis.")
        return

    _note(
        "Aggregated view across all selected cases. "
        "Boxplots show the distribution of values — the box spans the middle 50% of readings, "
        "the line inside is the median, and dots are outliers."
    )

    multi_df = db.get_measurements_multi(selected_ids)
    if multi_df.empty:
        st.warning("No measurements found for selected sessions."); return

    # Attach case labels
    sid_to_label = {
        int(r["session_id"]): f"Case {r['case_id']}"
        for _, r in sessions_df.iterrows()
    }
    multi_df["Case"] = multi_df["session_id"].map(sid_to_label)

    # ── Parameter boxplots ────────────────────────────────────────────────────
    st.markdown("### Parameter Distributions per Case")

    available = [c for c in MEASUREMENT_COLS
                 if c in multi_df.columns and multi_df[c].notna().any()]
    label_to_col = {DISPLAY_LABELS.get(c,c): c for c in available}

    selected_params = st.multiselect(
        "Select parameters for boxplots",
        options=list(label_to_col.keys()),
        default=list(label_to_col.keys())[:3],
        key="cohort_params",
    )

    for param_label in selected_params:
        col = label_to_col[param_label]
        fig = px.box(
            multi_df.dropna(subset=[col]),
            x="Case", y=col,
            color="Case",
            color_discrete_sequence=SESSION_COLORS,
            labels={col: param_label, "Case": ""},
            title=param_label,
            points="outliers",
        )
        fig.update_layout(**CHART_BASE, height=340, showlegend=False)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
        st.plotly_chart(fig, use_container_width=True)

    # ── Cross-case alarm frequency ────────────────────────────────────────────
    st.markdown("### Alarm Frequency Across Cases")
    _note("Total alarm events and alarm rate for each case in the cohort.")

    cohort_rows = []
    for sid in selected_ids:
        srow  = sessions_df[sessions_df["session_id"]==sid].iloc[0]
        mdf   = multi_df[multi_df["session_id"]==sid]
        n_alm = int(mdf["alarm_active"].sum()) if "alarm_active" in mdf.columns else 0
        cohort_rows.append({
            "Case":          f"Case {srow['case_id']}",
            "Total Records": len(mdf),
            "Alarm Events":  n_alm,
            "Alarm Rate (%)": round(n_alm/max(len(mdf),1)*100, 1),
        })
    cohort_table = pd.DataFrame(cohort_rows).set_index("Case")
    st.dataframe(cohort_table, use_container_width=True)

    fig_alarm = px.bar(
        cohort_table.reset_index(),
        x="Case", y="Alarm Rate (%)",
        color="Case",
        color_discrete_sequence=SESSION_COLORS,
        text="Alarm Rate (%)",
    )
    fig_alarm.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_alarm.update_layout(**CHART_BASE, showlegend=False, height=300,
                            yaxis_title="Alarm Rate (%)", xaxis_title="")
    st.plotly_chart(fig_alarm, use_container_width=True)

    # ── Aggregate statistics table ────────────────────────────────────────────
    with st.expander("📊 Advanced — Full Cohort Statistics Table", expanded=False):
        st.caption("Median and IQR of each parameter across all selected cases.")
        agg_rows = []
        for col in available:
            s = multi_df[col].dropna()
            if s.empty: continue
            agg_rows.append({
                "Parameter": DISPLAY_LABELS.get(col, col),
                "Cohort Median": round(s.median(), 2),
                "IQR (25–75%)": f"{round(s.quantile(.25),1)} – {round(s.quantile(.75),1)}",
                "Overall Min": round(s.min(), 2),
                "Overall Max": round(s.max(), 2),
                "Total readings": len(s),
            })
        if agg_rows:
            st.dataframe(pd.DataFrame(agg_rows).set_index("Parameter"),
                         use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Risk Prediction  (NEW)
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
        "to estimate the risk of an alarm occurring in the next few minutes. "
        "It is intended as a decision-support aid — **not** a substitute for clinical judgment."
    )

    if not selected_ids:
        st.info("Select at least one session in the sidebar to use the risk model.")
        return

    # ── Training panel ────────────────────────────────────────────────────────
    st.markdown("### Step 1 — Train the Model")
    _note(
        "The model learns from the sessions you select as training data. "
        "Using more sessions improves reliability. "
        "Training takes a few seconds."
    )

    horizon = st.slider(
        "Prediction horizon (number of samples ahead)",
        min_value=2, max_value=20, value=5, step=1,
        help="How many samples ahead the model tries to predict. "
             "A larger value gives more warning time but may be less precise.",
    )

    model_type = st.radio(
        "Model type",
        options=["random_forest", "logistic_regression"],
        format_func=lambda x: "Random Forest (recommended)" if x=="random_forest"
                               else "Logistic Regression (faster)",
        horizontal=True,
    )

    train_ids = st.multiselect(
        "Sessions to train on",
        options=selected_ids,
        default=selected_ids,
        format_func=lambda k: _session_label(
            sessions_df[sessions_df["session_id"]==k].iloc[0]
        ),
        help="Include all available sessions for best results.",
    )

    if st.button("🧠 Train Early Warning Model", type="primary"):
        if not train_ids:
            st.error("Select at least one session for training."); return

        with st.spinner("Training model on selected sessions…"):
            try:
                sessions_data = [db.get_measurements(sid) for sid in train_ids]
                model = VFModel(model_type=model_type, horizon=horizon)
                metrics = model.train(sessions_data)
                st.session_state.vf_model    = model
                st.session_state.model_trained = True
                st.success(
                    f"✅ Model trained successfully. "
                    f"Performance (AUC): **{metrics['auc']:.3f}** "
                    f"(0.5 = random, 1.0 = perfect)"
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                logger.exception("Training error")
                return

    if not st.session_state.model_trained or st.session_state.vf_model is None:
        st.info("👆 Configure and train the model above to see predictions.")
        return

    model: VFModel = st.session_state.vf_model

    # ── Training metrics ──────────────────────────────────────────────────────
    with st.expander("📈 Model Performance Details (Advanced)", expanded=False):
        st.caption("These metrics describe how well the model performed on held-out test data.")
        m = model.metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("AUC Score", f"{m['auc']:.3f}",
                   help="Area Under the ROC Curve. Above 0.7 is considered useful.")
        mc2.metric("Training samples", f"{m['train_samples']:,}")
        mc3.metric("Test samples",     f"{m['test_samples']:,}")

        if "report" in m:
            rep = m["report"]
            prec = rep.get("1",{}).get("precision", 0)
            rec  = rep.get("1",{}).get("recall",    0)
            f1   = rep.get("1",{}).get("f1-score",  0)
            st.markdown(
                f"Alarm detection — Precision: **{prec:.2f}** · "
                f"Recall: **{rec:.2f}** · F1: **{f1:.2f}**"
            )

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### What Drives Alarm Risk?")
    _note(
        "The bars below show which measurements have the most influence on the "
        "model's alarm risk estimate. Taller bars = stronger signal."
    )

    if model.feature_importance:
        top_n  = 8
        top    = model.feature_importance[:top_n]
        fi_df  = pd.DataFrame(top)
        fig_fi = px.bar(
            fi_df, x="importance_pct", y="label",
            orientation="h",
            labels={"importance_pct": "Relative Influence (%)", "label": ""},
            color="importance_pct",
            color_continuous_scale=["#AED6F1", "#1B4F72"],
        )
        fig_fi.update_layout(**CHART_BASE, height=40+30*top_n,
                             showlegend=False, coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("💬 Plain-language explanation", expanded=True):
            st.markdown(model.explain())

    # ── Prediction on active session ──────────────────────────────────────────
    st.markdown("### Step 2 — View Risk Prediction")
    predict_id = st.selectbox(
        "Session to predict on",
        options=selected_ids,
        format_func=lambda k: _session_label(
            sessions_df[sessions_df["session_id"]==k].iloc[0]
        ),
        key="predict_session",
    )

    if st.button("▶ Generate Risk Prediction", type="secondary"):
        with st.spinner("Running prediction…"):
            try:
                pred_df_raw = db.get_measurements(predict_id)
                pred_result = model.predict(pred_df_raw)
                st.session_state["pred_result"] = pred_result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                logger.exception("Prediction error")

    pred_result: Optional[pd.DataFrame] = st.session_state.get("pred_result")
    if pred_result is None:
        return

    # KPIs
    high_risk_pct = (pred_result["risk_level"]=="High").mean()*100
    mod_risk_pct  = (pred_result["risk_level"]=="Moderate").mean()*100
    rk1, rk2, rk3 = st.columns(3)
    rk1.metric("High Risk Periods",
               f"{high_risk_pct:.1f}% of session",
               help="Fraction of time points classified as high risk.")
    rk2.metric("Moderate Risk Periods", f"{mod_risk_pct:.1f}%")
    rk3.metric("Mean Risk Score",
               f"{pred_result['risk_score'].mean():.2f}",
               help="Average predicted probability of alarm (0 = no risk, 1 = certain alarm).")

    # Risk score over time
    st.markdown("### Predicted Risk Over Time")
    _note(
        "The blue line shows the model's estimated alarm risk at each time point. "
        "The shaded red zone marks periods of high predicted risk. "
        "Orange triangles show where the device actually triggered an alarm."
    )

    fig_risk = _base_fig(height=380,
                         title="Predicted Alarm Risk",
                         yaxis_title="Risk Score (0–1)",
                         yaxis_range=[0, 1])

    # High-risk shading
    high_mask = pred_result["risk_level"] == "High"
    fig_risk.add_hrect(y0=0.66, y1=1.0,
                       fillcolor="rgba(192,57,43,0.08)",
                       line_width=0,
                       annotation_text="High risk zone",
                       annotation_position="top right",
                       annotation_font_size=11)
    fig_risk.add_hrect(y0=0.33, y1=0.66,
                       fillcolor="rgba(202,111,30,0.06)",
                       line_width=0)

    # Risk score line
    fig_risk.add_trace(go.Scatter(
        x=pred_result["timestamp"], y=pred_result["risk_score"],
        mode="lines", name="Predicted Risk",
        line=dict(color=C["accent"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(46,134,193,0.10)",
    ))

    # Actual alarms overlay
    if "alarm_active" in pred_result.columns:
        alm = pred_result[pred_result["alarm_active"]==1]
        if not alm.empty:
            fig_risk.add_trace(go.Scatter(
                x=alm["timestamp"], y=alm["risk_score"],
                mode="markers", name="Actual alarm",
                marker=dict(color=C["alarm"], size=6,
                            symbol="triangle-up"),
            ))

    # Threshold lines
    for y_val, label, color in [
        (0.66, "High risk threshold", C["alarm"]),
        (0.33, "Moderate risk threshold", C["warn"]),
    ]:
        fig_risk.add_hline(y=y_val, line_dash="dot",
                           line_color=color, line_width=1,
                           annotation_text=label,
                           annotation_position="left",
                           annotation_font_size=10)

    fig_risk.update_layout(legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig_risk, use_container_width=True)

    # Prediction vs actual comparison
    if "alarm_active" in pred_result.columns:
        st.markdown("### Predicted vs Actual Alarm Comparison")
        _note(
            "How often the model's risk level agrees with the actual alarm state. "
            "Ideally, high-risk predictions overlap with actual alarm periods."
        )
        comp = pred_result.copy()
        comp["Actual"]    = comp["alarm_active"].map({1:"Alarm",0:"No Alarm"})
        comp["Predicted"] = comp["risk_level"].astype(str)

        fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=["Predicted Risk Score", "Actual Alarm State"],
                                 vertical_spacing=0.1)
        fig_comp.add_trace(go.Scatter(x=comp["timestamp"], y=comp["risk_score"],
            mode="lines", line=dict(color=C["accent"], width=1.5),
            name="Risk Score", fill="tozeroy",
            fillcolor="rgba(46,134,193,0.10)"), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=comp["timestamp"],
            y=comp["alarm_active"].astype(float),
            mode="lines", line=dict(color=C["alarm"], width=1.5),
            name="Actual Alarm (0/1)",
            fill="tozeroy", fillcolor="rgba(192,57,43,0.15)"), row=2, col=1)
        fig_comp.update_layout(**CHART_BASE, height=380,
                               legend=dict(orientation="h", y=1.06))
        fig_comp.update_xaxes(showgrid=True, gridcolor=C["grid"])
        fig_comp.update_yaxes(showgrid=True, gridcolor=C["grid"])
        st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("⚠️ Important Clinical Disclaimer", expanded=False):
        st.warning(
            "This risk prediction module is a **research prototype** and has not been "
            "validated for clinical use. Predictions are based on patterns in historical "
            "data and may not generalize to new patients or device configurations. "
            "All clinical decisions must remain with the responsible clinician."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Data Audit  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_audit(db: VFDatabase, session_id: int) -> None:
    st.markdown("## Data Quality & Audit Trail")
    st.caption("Records every transformation applied to the raw device file. For QA review.")

    sessions_df = db.get_sessions()
    srow = sessions_df[sessions_df["session_id"]==session_id]
    if srow.empty:
        st.warning("Session record not found."); return
    row = srow.iloc[0]

    st.markdown("### File Information")
    m = st.columns(3)
    m[0].info(f"**Case ID:** {row['case_id']}")
    m[1].info(f"**File:** {row['filename']}")
    m[2].info(f"**Records:** {int(row['row_count']):,}")

    st.markdown("### Cleaning Steps Applied")
    try:
        prov = json.loads(row["provenance"])
    except Exception:
        st.warning("Audit data unavailable."); return

    step_labels = {
        "timestamp":"🕐 Timestamp Standardisation",
        "measurements":"📏 Measurement Conversion",
        "alarm_limits":"⚙️ Alarm Limit Conversion",
        "status_flags":"🚦 Alarm Status Processing",
        "case_id":"🔖 Case Identification",
        "dedup":"🧹 Duplicate Removal",
        "final_shape":"✅ Final Dataset",
    }
    step_desc = {
        "timestamp":    "Converted device timestamp to standard date/time. Sorted chronologically.",
        "measurements": "Replaced device codes ('--','Lo','Hi') with blank. Converted to numbers.",
        "alarm_limits": "Converted alarm thresholds. Auto/Disabled limits stored as blank.",
        "status_flags": "Encoded alarm flags as true/false.",
        "case_id":      "Identified case number from file.",
        "dedup":        "Removed exact duplicate rows from USB export glitches.",
        "final_shape":  "Summary of cleaned dataset.",
    }
    for step, details in prov.items():
        with st.expander(step_labels.get(step, f"🔧 {step}"),
                         expanded=(step=="final_shape")):
            if step in step_desc:
                st.caption(step_desc[step])
            if isinstance(details, dict):
                clean_d = {k:v for k,v in details.items() if not isinstance(v,dict)}
                if clean_d: st.json(clean_d)
                if step == "measurements":
                    rows = []
                    for col, col_data in details.items():
                        if isinstance(col_data,dict) and "sentinel_count" in col_data:
                            rows.append({
                                "Parameter": DISPLAY_LABELS.get(col,col),
                                "Unavailable readings": col_data["sentinel_count"],
                                "Total missing": col_data.get("total_nan_after","—"),
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows).set_index("Parameter"),
                                     use_container_width=True)
            else:
                st.write(details)

    with st.expander("🗂️ Raw Session Record", expanded=False):
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
