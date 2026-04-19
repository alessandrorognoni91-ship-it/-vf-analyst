# 🫀 VF Analyst — Biomedical Data Platform (MVP Demo)

A lightweight, modular platform for ingesting, cleaning, and visualising
CSV exports from ECMO / VAD biomedical devices.

---

## Architecture

```
vf_analyst/
├── config.py          # Column mappings, sentinel codes, display labels
├── data_loader.py     # CSV ingestion, column renaming, timestamp detection
├── data_cleaning.py   # Type conversion, sentinel → NaN, deduplication, audit trail
├── data_model.py      # SQLite schema + VFDatabase class (sessions/measurements/alarms)
├── app.py             # Streamlit dashboard (5 tabs)
├── requirements.txt
└── README.md
```

### Data flow

```
CSV file (USB export)
      │
      ▼
data_loader.py     ← maps heterogeneous column names → canonical names
      │
      ▼
data_cleaning.py   ← sentinels→NaN, type cast, dedupe, provenance log
      │
      ▼
data_model.py      ← normalised SQLite tables: sessions / measurements / alarm_events
      │
      ▼
app.py             ← Streamlit dashboard: Summary | Time-Series | Distributions | Alarms | Provenance
```

---

## Normalised Schema

### sessions
| Column | Type | Notes |
|---|---|---|
| session_id | INTEGER PK | Auto-assigned |
| case_id | TEXT | From CSV "Case Number" |
| device_type | TEXT | e.g. "rotaflow" (extensible) |
| center_id | TEXT | Clinical centre identifier |
| filename | TEXT | Original CSV filename |
| row_count | INTEGER | |
| time_start / time_end | TEXT | ISO timestamps |
| provenance | TEXT | JSON: full cleaning audit trail |
| uploaded_at | TEXT | |

### measurements (one row per device timestamp)
Canonical columns: `timestamp`, `case_id`, `pressure_pre_mmhg`,
`pressure_post_mmhg`, `pressure_delta_mmhg`, `sat_pre_pct`, `sat_post_pct`,
`temp_post_c`, `flow_rate_lpm`, `pump_speed_rpm`, `alarm_active`,
`alarm_bubble`, `technical_fault`.

### alarm_events
Subset of measurements where any alarm was active, with `alarm_type` and `alarm_value`.

---

## Sentinel Code Mapping (caselog_29.csv)

| Raw value | Meaning | Cleaned to |
|---|---|---|
| `--` | Sensor not connected / unavailable | `NaN` |
| `Lo` | Below measurable range | `NaN` |
| `Hi` | Above measurable range | `NaN` |
| `A` | Alarm limit set to Auto | `NaN` |
| `D` | Alarm limit Disabled | `NaN` |
| `E` | Bubble sensor error | `NaN` |

---

## How to run locally

### 1. Clone / unzip the project

```bash
cd vf_analyst/
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**.

### 5. Upload data

Drag **caselog_29.csv** (or any compatible device CSV) onto the sidebar uploader.

---

## Adding a new device format

1. Open `config.py`.
2. Add the new device's raw column names to `COLUMN_MAP`, mapping them to
   the existing canonical names (or add new canonical names if needed).
3. If the device uses different sentinel strings, add them to `MEASUREMENT_SENTINELS`.
4. No other files need to change.

---

## Future improvements

1. **Multi-centre support** — add a `center_id` selector in the sidebar;
   load a `centers.yaml` config with per-centre column overrides.
2. **Persistent storage** — swap `":memory:"` in `get_database()` for a
   PostgreSQL URL; SQLAlchemy engine swap is one line.
3. **Anomaly detection** — plug an Isolation Forest or z-score detector into
   `data_cleaning.py`; flag anomalies as a new column `is_anomaly`.
4. **ML module** — add a `ml/` package with a scikit-learn pipeline for
   predicting alarm events from time-series features (rolling mean, std, etc.).
5. **Export** — add a "Download cleaned CSV" button in the Provenance tab.
6. **Authentication** — add Streamlit's built-in auth or a lightweight JWT
   layer for multi-user clinical deployments.
7. **FHIR / HL7 integration** — export sessions as FHIR Observation resources
   for EHR interoperability.
