
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_plotly_events import plotly_events  # type: ignore
except Exception:  # pragma: no cover
    plotly_events = None


st.set_page_config(
    page_title="Students' Performance Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
PALETTE = ["#488f31", "#bbba77", "#ffeeda", "#f0a281", "#de425b"]
DARK_BG = "#0b0f14"
CARD_BG = "#11161f"
CARD_BORDER = "rgba(255,255,255,0.08)"
TEXT = "#f4f4f4"
MUTED = "#b4bcc9"
GRID = "#263041"

# ---------------------------------------------------------------------
# Notebook-grounded metadata
# ---------------------------------------------------------------------
DISPLAY_MAPPINGS: Dict[str, Dict[int, str]] = {
    "Marital_status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated",
    },
    "Application_mode": {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores Island)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira Island)",
        17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
        27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)",
    },
    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening attendance)",
    },
    "Daytime_evening_attendance": {1: "Daytime", 0: "Evening"},
    "Previous_qualification": {
        1: "Secondary education",
        2: "Higher education - bachelor's degree",
        3: "Higher education - degree",
        4: "Higher education - master's",
        5: "Higher education - doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling - not completed",
        10: "11th year of schooling - not completed",
        12: "Other - 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling - not completed",
        19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
        38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
        39: "Technological specialization course",
        40: "Higher education - degree (1st cycle)",
        42: "Professional higher technical course",
        43: "Higher education - master (2nd cycle)",
    },
    "Nacionality": {
        1: "Portuguese",
        2: "German",
        6: "Spanish",
        11: "Italian",
        13: "Dutch",
        14: "English",
        17: "Lithuanian",
        21: "Angolan",
        22: "Cape Verdean",
        24: "Guinean",
        25: "Mozambican",
        26: "Santomean",
        32: "Turkish",
        41: "Brazilian",
        62: "Romanian",
        100: "Moldova (Republic of)",
        101: "Mexican",
        103: "Ukrainian",
        105: "Russian",
        108: "Cuban",
        109: "Colombian",
    },
    "Gender": {1: "Male", 0: "Female"},
    "Displaced": {1: "Yes", 0: "No"},
    "Educational_special_needs": {1: "Yes", 0: "No"},
    "Debtor": {1: "Yes", 0: "No"},
    "Tuition_fees_up_to_date": {1: "Yes", 0: "No"},
    "Scholarship_holder": {1: "Yes", 0: "No"},
    "International": {1: "Yes", 0: "No"},
    "Status": {0: "Dropout", 1: "Enrolled", 2: "Graduate"},
}

DEMOGRAPHIC_GROUP = [
    "Marital_status_label",
    "Gender_label",
    "Age_bin",
    "Nacionality_label",
    "Displaced_label",
    "Educational_special_needs_label",
]
ACADEMIC_GROUP = [
    "Admission_grade_bin",
    "approval_rate_1st_bin",
    "approval_rate_2nd_bin",
    "grade_avg_bin",
    "Previous_qualification_label",
    "Previous_qualification_grade_bin",
]
SOCIAL_GROUP = [
    "Application_mode_label",
    "Course_label",
    "Daytime_evening_attendance_label",
    "Tuition_fees_up_to_date_label",
    "Scholarship_holder_label",
    "Debtor_label",
    "Unemployment_rate_bin",
    "Inflation_rate_bin",
    "GDP_bin",
]

FEATURE_GROUP_MAP = {
    "Demographic": DEMOGRAPHIC_GROUP,
    "Academic": ACADEMIC_GROUP,
    "Social/Economic": SOCIAL_GROUP,
}

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .stApp {{
            background: {DARK_BG};
            color: {TEXT};
        }}
        div[data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 14px 16px;
        }}
        .card-block {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 18px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }}
        .card-title {{
            font-size: 1.03rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }}
        .card-subtitle {{
            color: {MUTED};
            margin-bottom: 0.65rem;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.8rem;
        }}
        .mini-box {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 0.55rem 0.7rem;
            border: 1px solid rgba(255,255,255,0.04);
        }}
        .mini-label {{
            color: {MUTED};
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .mini-value {{
            font-size: 0.92rem;
            margin-top: 0.15rem;
            line-height: 1.3;
        }}
        .section-head {{
            margin-top: 0.25rem;
            margin-bottom: 0.15rem;
            font-size: 1.15rem;
            font-weight: 700;
        }}
        .section-desc {{
            color: {MUTED};
            margin-bottom: 0.65rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

px.defaults.template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=PALETTE,
    )
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


RAW_TO_NOTEBOOK_COLS = {
    "Marital Status": "Marital_status",
    "Application mode": "Application_mode",
    "Application order": "Application_order",
    "Course": "Course",
    "Daytime/evening attendance": "Daytime_evening_attendance",
    "Previous qualification": "Previous_qualification",
    "Previous qualification (grade)": "Previous_qualification_grade",
    "Nacionality": "Nacionality",
    "Mother's qualification": "Mothers_qualification",
    "Father's qualification": "Fathers_qualification",
    "Mother's occupation": "Mothers_occupation",
    "Father's occupation": "Fathers_occupation",
    "Admission grade": "Admission_grade",
    "Displaced": "Displaced",
    "Educational special needs": "Educational_special_needs",
    "Debtor": "Debtor",
    "Tuition fees up to date": "Tuition_fees_up_to_date",
    "Gender": "Gender",
    "Scholarship holder": "Scholarship_holder",
    "Age at enrollment": "Age_at_enrollment",
    "International": "International",
    "Curricular units 1st sem (credited)": "Curricular_units_1st_sem_credited",
    "Curricular units 1st sem (enrolled)": "Curricular_units_1st_sem_enrolled",
    "Curricular units 1st sem (evaluations)": "Curricular_units_1st_sem_evaluations",
    "Curricular units 1st sem (approved)": "Curricular_units_1st_sem_approved",
    "Curricular units 1st sem (grade)": "Curricular_units_1st_sem_grade",
    "Curricular units 1st sem (without evaluations)": "Curricular_units_1st_sem_without_evaluations",
    "Curricular units 2nd sem (credited)": "Curricular_units_2nd_sem_credited",
    "Curricular units 2nd sem (enrolled)": "Curricular_units_2nd_sem_enrolled",
    "Curricular units 2nd sem (evaluations)": "Curricular_units_2nd_sem_evaluations",
    "Curricular units 2nd sem (approved)": "Curricular_units_2nd_sem_approved",
    "Curricular units 2nd sem (grade)": "Curricular_units_2nd_sem_grade",
    "Curricular units 2nd sem (without evaluations)": "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment rate": "Unemployment_rate",
    "Inflation rate": "Inflation_rate",
    "GDP": "GDP",
    "Status": "Status",
}
CANONICAL_RAW_MAP = {canonical(k): v for k, v in RAW_TO_NOTEBOOK_COLS.items()}
NOTEBOOK_COLUMNS = set(RAW_TO_NOTEBOOK_COLS.values())

ARTIFACT_CANDIDATES = [
    Path("best_model.pkl"),
    Path("model") / "best_model.pkl",
    Path("models") / "best_model.pkl",
    Path("artifacts") / "best_model.pkl",
]
FEATURE_NAMES_CANDIDATES = [
    Path("feature_names.pkl"),
    Path("model") / "feature_names.pkl",
    Path("models") / "feature_names.pkl",
    Path("artifacts") / "feature_names.pkl",
]


@st.cache_data
def load_pred_data(path: str = "pred_data.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {p.resolve()}. Run app.py first to create pred_data.csv.")
    return pd.read_csv(p)


@st.cache_resource
def load_model_artifacts():
    model_path = next((p for p in ARTIFACT_CANDIDATES if p.exists()), None)
    features_path = next((p for p in FEATURE_NAMES_CANDIDATES if p.exists()), None)
    model = joblib.load(model_path) if model_path else None
    feature_names = joblib.load(features_path) if features_path else None
    return model, feature_names


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = canonical(col)
        if key in CANONICAL_RAW_MAP:
            rename_map[col] = CANONICAL_RAW_MAP[key]
        elif col in NOTEBOOK_COLUMNS:
            rename_map[col] = col
    return df.rename(columns=rename_map).copy()


def ensure_display_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in DISPLAY_MAPPINGS.items():
        if col in out.columns and f"{col}_label" not in out.columns:
            def mapper(v):
                if pd.isna(v):
                    return v
                try:
                    return mapping.get(int(float(v)), f"Code {int(float(v))}")
                except Exception:
                    return v
            out[f"{col}_label"] = out[col].map(mapper)
    return out


def add_derived_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Age_at_enrollment" in out.columns and "Age_bin" not in out.columns:
        out["Age_bin"] = pd.cut(
            pd.to_numeric(out["Age_at_enrollment"], errors="coerce"),
            bins=[-np.inf, 18, 21, 24, 27, 30, np.inf],
            labels=["<=18", "19-21", "22-24", "25-27", "28-30", ">30"],
            include_lowest=True,
        )

    if "Admission_grade" in out.columns and "Admission_grade_bin" not in out.columns:
        out["Admission_grade_bin"] = pd.cut(
            pd.to_numeric(out["Admission_grade"], errors="coerce"),
            bins=[-np.inf, 80, 100, 120, 140, 160, np.inf],
            labels=["<=80", "81-100", "101-120", "121-140", "141-160", ">160"],
            include_lowest=True,
        )

    for base, bin_name in [
        ("approval_rate_1st", "approval_rate_1st_bin"),
        ("approval_rate_2nd", "approval_rate_2nd_bin"),
        ("grade_avg", "grade_avg_bin"),
        ("Previous_qualification_grade", "Previous_qualification_grade_bin"),
        ("Unemployment_rate", "Unemployment_rate_bin"),
        ("Inflation_rate", "Inflation_rate_bin"),
        ("GDP", "GDP_bin"),
    ]:
        if base in out.columns and bin_name not in out.columns:
            series = pd.to_numeric(out[base], errors="coerce")
            try:
                out[bin_name] = pd.qcut(series.rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
            except Exception:
                out[bin_name] = pd.cut(series, bins=4, labels=["Low", "Mid-Low", "Mid-High", "High"], include_lowest=True)
    return out


def parse_status_order(df: pd.DataFrame, col: str) -> List[str]:
    vals = [str(v) for v in df[col].dropna().unique().tolist()]
    preferred = ["Dropout", "Enrolled", "Graduate"]
    ordered = [x for x in preferred if x in vals]
    ordered.extend([x for x in sorted(vals) if x not in ordered])
    return ordered


def safe_mean(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    return float(series.mean()) if len(series) else float("nan")


def section_info(title: str, information: str, story: str, goal: str, visual_form: str) -> None:
    st.markdown(
        f"""
        <div class="card-block">
            <div class="card-title">{title}</div>
            <div class="card-grid">
                <div class="mini-box"><div class="mini-label">Information</div><div class="mini-value">{information}</div></div>
                <div class="mini-box"><div class="mini-label">Story</div><div class="mini-value">{story}</div></div>
                <div class="mini-box"><div class="mini-label">Goal</div><div class="mini-value">{goal}</div></div>
                <div class="mini-box"><div class="mini-label">Visual Form</div><div class="mini-value">{visual_form}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def series_for_chart(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype(str)
    return pd.Series(dtype=str)


def bar_agg(df: pd.DataFrame, x_col: str, value_col: str = "Prob_Dropout") -> pd.DataFrame:
    return (
        df.groupby(x_col, dropna=False)
        .agg(Count=(value_col, "size"), Avg_Dropout_Prob=(value_col, "mean"))
        .reset_index()
        .sort_values("Avg_Dropout_Prob", ascending=False)
    )


def make_bar(df: pd.DataFrame, x_col: str, title: str) -> go.Figure:
    agg = bar_agg(df, x_col)
    agg[x_col] = agg[x_col].astype(str)
    fig = px.bar(
        agg,
        x=x_col,
        y="Avg_Dropout_Prob",
        color="Avg_Dropout_Prob",
        color_continuous_scale=[(0.0, PALETTE[0]), (0.25, PALETTE[1]), (0.5, PALETTE[2]), (0.75, PALETTE[3]), (1.0, PALETTE[4])],
        text=agg["Avg_Dropout_Prob"].map(lambda v: f"{v:.1%}"),
        title=title,
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=f"{x_col}: %{{x}}<br>Avg dropout: %{{y:.2%}}<br>N: %{{customdata[0]}}<extra></extra>",
        customdata=agg[["Count"]].to_numpy(),
    )
    fig.update_layout(height=410, margin=dict(l=10, r=10, t=50, b=10), showlegend=False, xaxis_title="", yaxis_title="Avg dropout probability")
    return fig


def render_clickable(fig: go.Figure, key: str) -> List[str]:
    if plotly_events is None:
        st.plotly_chart(fig, use_container_width=True, key=key)
        return []
    events = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=key, override_height=410)
    values: List[str] = []
    for p in events or []:
        if p.get("x") is not None:
            values.append(str(p["x"]))
    return list(dict.fromkeys(values))


def update_focus(column: str, values: List[str]) -> None:
    if not values:
        return
    st.session_state.setdefault("focus_filters", {})[column] = values
    st.rerun()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    with st.sidebar:
        st.header("Filters")
        st.caption("Sidebar filters affect every chart. Click bars to add focus filters.")

        if st.button("Clear focus filters"):
            st.session_state["focus_filters"] = {}
            st.rerun()

        if "Status" in out.columns:
            opts = parse_status_order(out, "Status")
            pick = st.multiselect("Actual status", opts, default=opts)
            out = out[out["Status"].astype(str).isin(pick)]

        if "Predicted_Status" in out.columns:
            opts = parse_status_order(out, "Predicted_Status")
            pick = st.multiselect("Predicted status", opts, default=opts)
            out = out[out["Predicted_Status"].astype(str).isin(pick)]

        for label, col in [
            ("Course", "Course_label"),
            ("Gender", "Gender_label"),
            ("Nationality", "Nacionality_label"),
            ("Scholarship holder", "Scholarship_holder_label"),
        ]:
            if col in out.columns:
                opts = sorted(out[col].dropna().astype(str).unique().tolist())
                pick = st.multiselect(label, opts, default=opts)
                out = out[out[col].astype(str).isin(pick)]

        if "Prob_Dropout" in out.columns and len(out):
            pmin = float(pd.to_numeric(out["Prob_Dropout"], errors="coerce").min())
            pmax = float(pd.to_numeric(out["Prob_Dropout"], errors="coerce").max())
            rng = st.slider("Dropout probability", 0.0, 1.0, (0.0, 1.0))
            out = out[pd.to_numeric(out["Prob_Dropout"], errors="coerce").between(rng[0], rng[1])]

        if "Age_at_enrollment" in out.columns and len(out):
            amin = int(pd.to_numeric(out["Age_at_enrollment"], errors="coerce").min())
            amax = int(pd.to_numeric(out["Age_at_enrollment"], errors="coerce").max())
            arng = st.slider("Age at enrollment", max(0, amin), amax, (max(0, amin), amax))
            out = out[pd.to_numeric(out["Age_at_enrollment"], errors="coerce").between(arng[0], arng[1])]

    return out


def apply_focus_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    focus = st.session_state.get("focus_filters", {})
    if not focus:
        return out

    for col, values in focus.items():
        if col in out.columns and values:
            out = out[out[col].astype(str).isin([str(v) for v in values])]
    return out


def choose_group_feature(group_name: str, df: pd.DataFrame) -> str:
    candidates = [c for c in FEATURE_GROUP_MAP[group_name] if c in df.columns]
    if not candidates:
        return ""
    key = f"{group_name}_feature_select"
    previous = st.session_state.get(key)
    default_idx = candidates.index(previous) if previous in candidates else 0
    idx = st.selectbox(
        f"{group_name} feature",
        options=candidates,
        index=default_idx,
        key=key,
        help=f"Choose the feature to visualize in the {group_name.lower()} bar chart.",
    )
    return idx


def dropdown_for_numeric_bins(df: pd.DataFrame, base_col: str, bin_col: str) -> pd.DataFrame:
    if bin_col not in df.columns and base_col in df.columns:
        out = df.copy()
        out[bin_col] = pd.qcut(pd.to_numeric(out[base_col], errors="coerce").rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        return out
    return df


def heatmap_figure(df: pd.DataFrame, row_col: str, col_col: str, title: str) -> go.Figure:
    pivot = df.pivot_table(index=row_col, columns=col_col, values="Prob_Dropout", aggfunc="mean")
    if pivot.empty:
        return go.Figure()

    # Keep it readable.
    if pivot.shape[0] > 12:
        top_rows = df.groupby(row_col)["Prob_Dropout"].mean().sort_values(ascending=False).head(12).index
        pivot = pivot.loc[[idx for idx in pivot.index if idx in top_rows]]
    if pivot.shape[1] > 12:
        top_cols = df.groupby(col_col)["Prob_Dropout"].mean().sort_values(ascending=False).head(12).index
        pivot = pivot[[c for c in pivot.columns if c in top_cols]]

    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[(0.0, PALETTE[0]), (0.25, PALETTE[1]), (0.5, PALETTE[2]), (0.75, PALETTE[3]), (1.0, PALETTE[4])],
        title=title,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def load_feature_importance() -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    model, feature_names = load_model_artifacts()
    if model is None or feature_names is None or not hasattr(model, "feature_importances_"):
        return None, {}

    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(importances) != len(feature_names):
        return None, {}

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=False)
    groups = {
        "Demographic": sum(float(row.Importance) for row in df.itertuples() if row.Feature in {
            "Marital_status", "Gender", "Age_at_enrollment", "Nacionality",
            "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
            "Fathers_occupation", "Displaced", "Educational_special_needs",
        }),
        "Academic": sum(float(row.Importance) for row in df.itertuples() if row.Feature in {
            "Previous_qualification", "Previous_qualification_grade", "Admission_grade",
            "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_without_evaluations",
            "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_credited",
            "Curricular_units_2nd_sem_without_evaluations", "approval_rate_1st", "approval_rate_2nd",
            "grade_avg", "failure_rate_1st", "failure_rate_2nd",
        }),
        "Social/Economic": sum(float(row.Importance) for row in df.itertuples() if row.Feature in {
            "Application_mode", "Application_order", "Course", "Daytime_evening_attendance",
            "Tuition_fees_up_to_date", "Scholarship_holder", "Debtor", "Unemployment_rate",
            "Inflation_rate", "GDP",
        }),
    }
    return df, groups


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
try:
    raw = load_pred_data("pred_data.csv")
except Exception as exc:
    st.error(str(exc))
    st.stop()

raw = normalize_columns(raw)
raw = ensure_display_labels(raw)
raw = add_derived_bins(raw)
raw = dropdown_for_numeric_bins(raw, "Previous_qualification_grade", "Previous_qualification_grade_bin")
raw = dropdown_for_numeric_bins(raw, "approval_rate_1st", "approval_rate_1st_bin")
raw = dropdown_for_numeric_bins(raw, "approval_rate_2nd", "approval_rate_2nd_bin")
raw = dropdown_for_numeric_bins(raw, "grade_avg", "grade_avg_bin")
raw = dropdown_for_numeric_bins(raw, "Unemployment_rate", "Unemployment_rate_bin")
raw = dropdown_for_numeric_bins(raw, "Inflation_rate", "Inflation_rate_bin")
raw = dropdown_for_numeric_bins(raw, "GDP", "GDP_bin")

st.title("Students' Performance Dashboard")
st.caption("Notebook-grounded dashboard built from pred_data.csv and the saved Random Forest model.")

# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------
filtered = apply_filters(raw)
filtered = apply_focus_filters(filtered)

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(filtered):,}")
k2.metric("Avg Dropout Probability", f"{safe_mean(filtered['Prob_Dropout']):.2%}" if "Prob_Dropout" in filtered.columns and len(filtered) else "N/A")
k3.metric("High Risk (>0.50)", f"{int((pd.to_numeric(filtered.get('Prob_Dropout', pd.Series(dtype=float)), errors='coerce') > 0.5).sum()):,}" if "Prob_Dropout" in filtered.columns and len(filtered) else "N/A")
k4.metric("Predicted Dropout", f"{int((filtered.get('Predicted_Status', pd.Series(dtype=str)) == 'Dropout').sum()):,}" if "Predicted_Status" in filtered.columns else "N/A")

st.markdown("---")

focus_state = st.session_state.get("focus_filters", {})
if focus_state:
    st.info("Active focus filters: " + "; ".join([f"{k} = {', '.join(map(str, v))}" for k, v in focus_state.items()]))

# ---------------------------------------------------------------------
# Row 2 - three grouped bar charts
# ---------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

# Demographic
with c1:
    section_info(
        "Demographic View",
        "Average dropout probability across demographic segments.",
        "Some student profiles may require more support when demographic signals align with low persistence.",
        "Identify which demographic slice carries the highest dropout exposure.",
        "Bar chart",
    )
    demo_feature = choose_group_feature("Demographic", filtered)
    if demo_feature and len(filtered):
        fig = make_bar(filtered, demo_feature, f"Average Dropout Probability by {demo_feature}")
        clicked = render_clickable(fig, "demo_chart")
        if clicked:
            update_focus(demo_feature, clicked)
    else:
        st.info("No demographic feature available after filtering.")

# Academic
with c2:
    section_info(
        "Academic View",
        "Average dropout probability across academic-performance segments.",
        "Academic readiness and semester success are the strongest signals in the notebook.",
        "Spot the academic segments most associated with dropout risk.",
        "Bar chart",
    )
    acad_feature = choose_group_feature("Academic", filtered)
    if acad_feature and len(filtered):
        fig = make_bar(filtered, acad_feature, f"Average Dropout Probability by {acad_feature}")
        clicked = render_clickable(fig, "acad_chart")
        if clicked:
            update_focus(acad_feature, clicked)
    else:
        st.info("No academic feature available after filtering.")

# Social/Economic
with c3:
    section_info(
        "Social/Economic View",
        "Average dropout probability across social and economic segments.",
        "Financial readiness, course choice, and attendance pattern can shape persistence outcomes.",
        "Find the social/economic slices linked to higher dropout risk.",
        "Bar chart",
    )
    social_feature = choose_group_feature("Social/Economic", filtered)
    if social_feature and len(filtered):
        fig = make_bar(filtered, social_feature, f"Average Dropout Probability by {social_feature}")
        clicked = render_clickable(fig, "social_chart")
        if clicked:
            update_focus(social_feature, clicked)
    else:
        st.info("No social/economic feature available after filtering.")

st.markdown("---")

# ---------------------------------------------------------------------
# Row 3 - heatmap
# ---------------------------------------------------------------------
section_info(
    "Heatmap of Dropout Probability",
    "Shows the mean dropout probability across relevant categorical / binned features.",
    "This is the dense risk map that helps reveal concentrated pockets of dropout exposure.",
    "Prioritize the combinations that deserve intervention first.",
    "Heatmap",
)

heat_left, heat_right = st.columns([1.2, 0.8])
with heat_left:
    heat_row = social_feature if social_feature and social_feature in filtered.columns else ("Course_label" if "Course_label" in filtered.columns else "")
    heat_col = acad_feature if acad_feature and acad_feature in filtered.columns else ("Admission_grade_bin" if "Admission_grade_bin" in filtered.columns else "")
    if heat_row and heat_col and len(filtered):
        heat_df = filtered.copy()
        if heat_row == heat_col:
            heat_col = "Prob_Bin"
            heat_df["Prob_Bin"] = pd.cut(
                pd.to_numeric(heat_df["Prob_Dropout"], errors="coerce"),
                bins=[0.0, 0.25, 0.50, 0.75, 1.0],
                labels=["0-25%", "25-50%", "50-75%", "75-100%"],
                include_lowest=True,
            )
        fig = heatmap_figure(heat_df, heat_row, heat_col, f"Mean Dropout Probability: {heat_row} vs {heat_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to build the heatmap.")

with heat_right:
    st.subheader("Filtered preview")
    preview_cols = [c for c in [
        "Status", "Predicted_Status", "Prob_Dropout", "Prob_Enrolled", "Prob_Graduate",
        "Course_label", "Gender_label", "Nacionality_label", "Scholarship_holder_label",
        "Admission_grade", "Age_at_enrollment"
    ] if c in filtered.columns]
    st.dataframe(filtered[preview_cols].head(20), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------
# Notebook-style insight panels
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Visualisasi Distribusi Data",
    "Korelasi Fitur terhadap Target",
    "Heatmap Fitur Penting",
    "Feature Importance",
])

with tab1:
    section_info(
        "Visualisasi Distribusi Data",
        "Distribution of the predicted dropout probability and predicted class counts.",
        "This helps you see whether the dataset is concentrated in low-risk or high-risk students.",
        "Understand the spread before making decisions.",
        "Histogram + bar chart",
    )
    left, right = st.columns(2)
    with left:
        if "Prob_Dropout" in filtered.columns and len(filtered):
            fig = px.histogram(
                filtered,
                x="Prob_Dropout",
                nbins=20,
                title="Distribution of Dropout Probability",
                color_discrete_sequence=[PALETTE[4]],
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No probability data available.")
    with right:
        if "Predicted_Status" in filtered.columns and len(filtered):
            counts = filtered["Predicted_Status"].value_counts().reset_index()
            counts.columns = ["Predicted_Status", "Count"]
            fig = px.bar(counts, x="Predicted_Status", y="Count", text="Count", color="Predicted_Status")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction status available.")

with tab2:
    section_info(
        "Korelasi Fitur terhadap Target",
        "Correlations are computed from the numeric columns available in pred_data.csv.",
        "This mirrors the notebook’s correlation analysis used to understand which variables move with the target.",
        "Highlight the strongest numeric signals.",
        "Horizontal bar chart",
    )
    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    target_like = "Prob_Dropout" if "Prob_Dropout" in filtered.columns else None
    if target_like and len(numeric_cols) > 1:
        corr_vals = filtered[numeric_cols].corr()[target_like].abs().sort_values(ascending=False).drop(labels=[target_like], errors="ignore")
        corr_df = corr_vals.head(15).reset_index()
        corr_df.columns = ["Feature", "Abs_Correlation"]
        fig = px.bar(
            corr_df.sort_values("Abs_Correlation", ascending=True),
            x="Abs_Correlation",
            y="Feature",
            orientation="h",
            text="Abs_Correlation",
            color="Abs_Correlation",
            color_continuous_scale=[(0.0, PALETTE[0]), (0.25, PALETTE[1]), (0.5, PALETTE[2]), (0.75, PALETTE[3]), (1.0, PALETTE[4])],
        )
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10), showlegend=False, xaxis_title="Absolute correlation", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns to calculate correlation.")

with tab3:
    section_info(
        "Heatmap Fitur Penting",
        "Correlation heatmap of the top numeric features most associated with dropout probability.",
        "This gives a dense view of how the strongest features move together.",
        "Spot feature clusters that deserve closer inspection.",
        "Heatmap",
    )
    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    if "Prob_Dropout" in filtered.columns and len(numeric_cols) > 3:
        corr = filtered[numeric_cols].corr()
        top = corr["Prob_Dropout"].abs().sort_values(ascending=False).head(10).index.tolist()
        heat = corr.loc[top, top]
        fig = px.imshow(
            heat,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[(0.0, PALETTE[0]), (0.25, PALETTE[1]), (0.5, PALETTE[2]), (0.75, PALETTE[3]), (1.0, PALETTE[4])],
            title="Heatmap of Top Correlated Numeric Features",
        )
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric data for a feature heatmap.")

with tab4:
    section_info(
        "Feature Importance",
        "Top feature importances from the saved Random Forest model.",
        "This mirrors the notebook’s feature-importance analysis and helps explain model behavior.",
        "Prioritize the variables that matter most in the trained model.",
        "Bar chart",
    )
    fi_df, group_imp = load_feature_importance()
    if fi_df is not None and len(fi_df):
        top15 = fi_df.head(15).copy()
        fig = px.bar(
            top15.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[(0.0, PALETTE[0]), (0.25, PALETTE[1]), (0.5, PALETTE[2]), (0.75, PALETTE[3]), (1.0, PALETTE[4])],
            text=top15.sort_values("Importance", ascending=True)["Importance"].map(lambda v: f"{v:.3f}"),
        )
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10), showlegend=False, xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Group importance")
        group_df = pd.DataFrame({"Group": list(group_imp.keys()), "Importance": list(group_imp.values())})
        st.bar_chart(group_df.set_index("Group"))
    else:
        st.info("Model artifacts are missing or do not expose feature importance.")

st.markdown("---")
st.dataframe(filtered.head(25), use_container_width=True)
