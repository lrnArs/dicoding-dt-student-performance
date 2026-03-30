from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------
# Artifact locations
# ---------------------------------------------------------------------
ARTIFACT_CANDIDATES = [
    Path("best_model.pkl"),
    Path("model") / "best_model.pkl",
    Path("models") / "best_model.pkl",
    Path("artifacts") / "best_model.pkl",
]
BOUNDS_CANDIDATES = [
    Path("bounds.pkl"),
    Path("model") / "bounds.pkl",
    Path("models") / "bounds.pkl",
    Path("artifacts") / "bounds.pkl",
]
NUMERIC_COLS_CANDIDATES = [
    Path("numeric_cols.pkl"),
    Path("model") / "numeric_cols.pkl",
    Path("models") / "numeric_cols.pkl",
    Path("artifacts") / "numeric_cols.pkl",
]
FEATURE_NAMES_CANDIDATES = [
    Path("feature_names.pkl"),
    Path("model") / "feature_names.pkl",
    Path("models") / "feature_names.pkl",
    Path("artifacts") / "feature_names.pkl",
]
LABEL_ENCODER_CANDIDATES = [
    Path("label_encoder.pkl"),
    Path("model") / "label_encoder.pkl",
    Path("models") / "label_encoder.pkl",
    Path("artifacts") / "label_encoder.pkl",
]

# ---------------------------------------------------------------------
# Source notebook column names
# ---------------------------------------------------------------------
RAW_TO_NOTEBOOK_COLS: Dict[str, str] = {
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

# ---------------------------------------------------------------------
# Display mappings for output readability
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
    "Status": {0: "Dropout", 1: "Graduate"},
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


CANONICAL_RAW_MAP = {canonical(k): v for k, v in RAW_TO_NOTEBOOK_COLS.items()}
NOTEBOOK_COLUMNS = set(RAW_TO_NOTEBOOK_COLS.values())


def locate_file(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_resource
def load_artifacts():
    model_path = locate_file(ARTIFACT_CANDIDATES)
    bounds_path = locate_file(BOUNDS_CANDIDATES)
    numeric_cols_path = locate_file(NUMERIC_COLS_CANDIDATES)
    feature_names_path = locate_file(FEATURE_NAMES_CANDIDATES)
    label_encoder_path = locate_file(LABEL_ENCODER_CANDIDATES)

    missing = [
        name
        for name, path in {
            "best_model.pkl": model_path,
            "bounds.pkl": bounds_path,
            "numeric_cols.pkl": numeric_cols_path,
            "feature_names.pkl": feature_names_path,
            "label_encoder.pkl": label_encoder_path,
        }.items()
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Place them in the project root or in model/models/artifacts."
        )

    model = joblib.load(model_path)
    bounds = joblib.load(bounds_path)
    numeric_cols = joblib.load(numeric_cols_path)
    feature_names = joblib.load(feature_names_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, bounds, numeric_cols, feature_names, label_encoder


def read_data_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=";")
    except Exception:
        return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = canonical(col)
        if key in CANONICAL_RAW_MAP:
            rename_map[col] = CANONICAL_RAW_MAP[key]
        elif col in NOTEBOOK_COLUMNS:
            rename_map[col] = col
    return df.rename(columns=rename_map).copy()


def reverse_mapping(col: str) -> Dict[str, int]:
    forward = DISPLAY_MAPPINGS.get(col, {})
    reverse: Dict[str, int] = {}
    for k, v in forward.items():
        reverse[str(v).strip().lower()] = int(k)
    return reverse


def map_text_to_code(series: pd.Series, col: str) -> pd.Series:
    """Allow end‑users to provide labels instead of numeric codes."""
    rev = reverse_mapping(col)
    if not rev:
        return series

    def _convert(v):
        if pd.isna(v):
            return np.nan
        text = str(v).strip().lower()
        if text in rev:
            return rev[text]
        # Accept numeric text as well
        try:
            return int(float(v))
        except Exception:
            return v

    return series.map(_convert)


def coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col == "Status":
            continue
        if col in DISPLAY_MAPPINGS:
            df[col] = map_text_to_code(df[col], col)

        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="ignore")
            df[col] = converted
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly replicate the feature engineering from the notebook."""
    df = df.copy()
    eps = 1e-6

    if {"Curricular_units_1st_sem_approved", "Curricular_units_1st_sem_enrolled"}.issubset(df.columns):
        df["approval_rate_1st"] = df["Curricular_units_1st_sem_approved"] / (df["Curricular_units_1st_sem_enrolled"] + eps)
        df["failure_rate_1st"] = (df["Curricular_units_1st_sem_enrolled"] - df["Curricular_units_1st_sem_approved"]) / (
            df["Curricular_units_1st_sem_enrolled"] + eps
        )

    if {"Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_enrolled"}.issubset(df.columns):
        df["approval_rate_2nd"] = df["Curricular_units_2nd_sem_approved"] / (df["Curricular_units_2nd_sem_enrolled"] + eps)
        df["failure_rate_2nd"] = (df["Curricular_units_2nd_sem_enrolled"] - df["Curricular_units_2nd_sem_approved"]) / (
            df["Curricular_units_2nd_sem_enrolled"] + eps
        )

    if {"Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade"}.issubset(df.columns):
        df["grade_avg"] = (df["Curricular_units_1st_sem_grade"] + df["Curricular_units_2nd_sem_grade"]) / 2

    # Drop original columns used to create new features
    cols_to_drop = [
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_grade",
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


def impute_numeric_median(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Impute missing numeric values with median."""
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            median_val = series.median()
            if pd.notna(median_val):
                df[col] = series.fillna(median_val)
            else:
                df[col] = series.fillna(0)
    return df


def cap_numeric_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Apply Winsorization using pre‑computed bounds."""
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower, upper)
    return df


def align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Ensure the DataFrame has all expected features, in the correct order."""
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def to_label(col: str, value):
    """Convert numeric code to readable label using DISPLAY_MAPPINGS."""
    if pd.isna(value):
        return value
    mapping = DISPLAY_MAPPINGS.get(col, {})
    try:
        code = int(float(value))
    except Exception:
        return value
    return mapping.get(code, f"Code {code}")


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add human‑readable label columns for columns that have a mapping."""
    out = df.copy()
    for col in DISPLAY_MAPPINGS:
        if col in out.columns:
            out[f"{col}_label"] = out[col].apply(lambda v: to_label(col, v))
    return out


def round_floats(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Round all float columns to the specified number of decimals."""
    df = df.copy()
    for col in df.select_dtypes(include=[float]).columns:
        df[col] = df[col].round(decimals)
    return df


def predict_from_dataframe(input_df: pd.DataFrame):
    """Full pipeline: normalization, feature engineering, imputation, capping, prediction."""
    model, bounds, numeric_cols, feature_names, label_encoder = load_artifacts()

    # 1. Normalize column names
    df = normalize_columns(input_df)

    # 2. Convert text labels to codes where possible
    df = coerce_numeric_like_columns(df)

    # 3. Feature engineering (adds derived columns, drops raw ones)
    df = engineer_features(df)

    # 4. Prepare features for prediction (drop any target column if present)
    features_df = df.drop(columns=["Status"], errors="ignore")

    # 5. Impute missing numeric values with median
    features_df = impute_numeric_median(features_df, numeric_cols=numeric_cols)

    # 6. Apply Winsorization (capping) using pre‑computed bounds
    features_df = cap_numeric_bounds(features_df, bounds=bounds)

    # 7. Align to the feature set used during training
    features_df = align_features(features_df, feature_names=feature_names)

    # 8. Predict
    pred_codes = model.predict(features_df)
    pred_proba = model.predict_proba(features_df)
    pred_labels = label_encoder.inverse_transform(pred_codes)

    # 9. Build output DataFrame with original input + derived + predictions
    output_df = input_df.copy()
    # Add normalized columns and derived features for readability
    output_df = normalize_columns(output_df)
    output_df = coerce_numeric_like_columns(output_df)
    output_df = engineer_features(output_df)
    output_df = add_label_columns(output_df)

    output_df["Predicted_Status_Code"] = pred_codes
    output_df["Predicted_Status"] = pred_labels

    # Determine class indices (assume classes are [0,1] with 1 = Graduate)
    class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    output_df["Prob_Dropout"] = pred_proba[:, class_to_index.get(0, 0)] if 0 in class_to_index else np.nan
    output_df["Prob_Graduate"] = pred_proba[:, class_to_index.get(1, 1)] if 1 in class_to_index else np.nan
    output_df["Prediction_Confidence"] = np.max(pred_proba, axis=1)

    # Add risk group based on probability of Dropout
    def risk_level(prob_dropout):
        if pd.isna(prob_dropout):
            return "Unknown"
        if prob_dropout > 0.7:
            return "High"
        if prob_dropout > 0.3:
            return "Medium"
        return "Low"

    output_df["Risk_Level"] = output_df["Prob_Dropout"].apply(risk_level)

    # Round floats to 3 decimals
    output_df = round_floats(output_df)

    return output_df


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("Student Dropout Predictor")
st.caption("Predicts whether a student will Dropout or Graduate (binary classification) based on academic, demographic, and socio‑economic features.")

# Sidebar controls
with st.sidebar:
    st.header("Run prediction")
    csv_path = st.text_input("Input CSV", value="data.csv")
    output_path = st.text_input("Output CSV", value="pred_data.csv")
    run_btn = st.button("Process file")

    st.markdown("---")
    if st.button("Exit Application"):
        st.info("Exiting Streamlit app...")
        sys.exit(0)  # Stops the Streamlit server

st.markdown(
    """
    **How it works**
    1. Load your local CSV file (semicolon‑separated recommended).
    2. Normalize column names to match the training dataset.
    3. Convert coded values into readable labels where descriptions are available.
    4. Apply feature engineering (approval rates, failure rates, average grade).
    5. Impute missing values with median.
    6. Cap extreme values using pre‑computed bounds (Winsorization).
    7. Run the trained Random Forest model.
    8. Save predictions with probabilities and risk groups.
    """
)

if run_btn:
    try:
        input_path = Path(csv_path)
        if not input_path.exists():
            st.error(f"File not found: {input_path.resolve()}")
            st.stop()

        df_in = read_data_file(input_path)
        st.success(f"Loaded {len(df_in):,} rows × {len(df_in.columns):,} columns.")

        pred_df = predict_from_dataframe(df_in)
        out_path = Path(output_path)
        pred_df.to_csv(out_path, index=False)

        # Summary metrics (only Dropout/Graduate)
        total = len(pred_df)
        dropout_count = int((pred_df["Predicted_Status"] == "Dropout").sum())
        graduate_count = int((pred_df["Predicted_Status"] == "Graduate").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total students", f"{total:,}")
        c2.metric("Predicted Dropout", f"{dropout_count:,}")
        c3.metric("Predicted Graduate", f"{graduate_count:,}")

        st.success(f"Prediction complete. Saved to {out_path.resolve()}")

        st.subheader("Preview (first 25 rows)")
        st.dataframe(pred_df.head(25), use_container_width=True)

        st.download_button(
            "Download pred_data.csv",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="pred_data.csv",
            mime="text/csv",
        )

        st.subheader("Prediction Distribution")
        dist = pred_df["Predicted_Status"].value_counts().reset_index()
        dist.columns = ["Predicted_Status", "Count"]
        st.bar_chart(dist.set_index("Predicted_Status"))

    except Exception as exc:
        st.exception(exc)
else:
    st.info("Place your CSV file in the same folder and click **Process file**.")