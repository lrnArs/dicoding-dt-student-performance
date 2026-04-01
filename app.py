# app.py - Student Dropout Predictor
# ---------------------------------------------------------------------
# This script loads a trained Random Forest model and pre-processing
# artifacts, processes an input CSV file, and outputs predictions with
# probabilities and risk levels.
# ---------------------------------------------------------------------

from __future__ import annotations

import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------
# 1. Define artifact locations (try common paths)
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
# 2. Column mapping from raw CSV to notebook column names
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

def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

CANONICAL_RAW_MAP = {canonical(k): v for k, v in RAW_TO_NOTEBOOK_COLS.items()}
NOTEBOOK_COLUMNS = set(RAW_TO_NOTEBOOK_COLS.values())

# ---------------------------------------------------------------------
# 3. Helper functions for loading artifacts
# ---------------------------------------------------------------------
def locate_file(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Found artifact: {candidate}")
            return candidate
    return None

@st.cache_resource
def load_artifacts():
    """Load model, bounds, numeric_cols, feature_names, label_encoder."""
    model_path = locate_file(ARTIFACT_CANDIDATES)
    bounds_path = locate_file(BOUNDS_CANDIDATES)
    numeric_cols_path = locate_file(NUMERIC_COLS_CANDIDATES)
    feature_names_path = locate_file(FEATURE_NAMES_CANDIDATES)
    label_encoder_path = locate_file(LABEL_ENCODER_CANDIDATES)

    missing = []
    for name, path in {
        "best_model.pkl": model_path,
        "bounds.pkl": bounds_path,
        "numeric_cols.pkl": numeric_cols_path,
        "feature_names.pkl": feature_names_path,
        "label_encoder.pkl": label_encoder_path,
    }.items():
        if path is None:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {', '.join(missing)}. "
            "Place them in the project root or in model/models/artifacts."
        )

    model = joblib.load(model_path)
    bounds = joblib.load(bounds_path)
    numeric_cols = joblib.load(numeric_cols_path)
    feature_names = joblib.load(feature_names_path)
    label_encoder = joblib.load(label_encoder_path)

    logger.info("All artifacts loaded successfully.")
    return model, bounds, numeric_cols, feature_names, label_encoder

# ---------------------------------------------------------------------
# 4. Data preprocessing (mirroring notebook)
# ---------------------------------------------------------------------
def read_data_file(path: Path) -> pd.DataFrame:
    """Read CSV with semicolon separator, fallback to comma."""
    try:
        return pd.read_csv(path, sep=";")
    except Exception:
        return pd.read_csv(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match notebook names using canonical mapping."""
    rename_map = {}
    for col in df.columns:
        key = canonical(col)
        if key in CANONICAL_RAW_MAP:
            rename_map[col] = CANONICAL_RAW_MAP[key]
        elif col in NOTEBOOK_COLUMNS:
            rename_map[col] = col
    return df.rename(columns=rename_map).copy()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features exactly as in the notebook.
    The notebook uses:
        approval_rate_1st = approved / (enrolled + epsilon)
        failure_rate_1st = (enrolled - approved) / (enrolled + epsilon)
        similarly for 2nd semester
        grade_avg = (1st_grade + 2nd_grade) / 2
    Then drops original columns used to create them.
    """
    df = df.copy()
    eps = 1e-6

    # 1st semester rates
    if "Curricular_units_1st_sem_approved" in df.columns and "Curricular_units_1st_sem_enrolled" in df.columns:
        enrolled1 = df["Curricular_units_1st_sem_enrolled"]
        approved1 = df["Curricular_units_1st_sem_approved"]
        df["approval_rate_1st"] = approved1 / (enrolled1 + eps)
        df["failure_rate_1st"] = (enrolled1 - approved1) / (enrolled1 + eps)

    # 2nd semester rates
    if "Curricular_units_2nd_sem_approved" in df.columns and "Curricular_units_2nd_sem_enrolled" in df.columns:
        enrolled2 = df["Curricular_units_2nd_sem_enrolled"]
        approved2 = df["Curricular_units_2nd_sem_approved"]
        df["approval_rate_2nd"] = approved2 / (enrolled2 + eps)
        df["failure_rate_2nd"] = (enrolled2 - approved2) / (enrolled2 + eps)

    # Average grade
    if "Curricular_units_1st_sem_grade" in df.columns and "Curricular_units_2nd_sem_grade" in df.columns:
        df["grade_avg"] = (df["Curricular_units_1st_sem_grade"] + df["Curricular_units_2nd_sem_grade"]) / 2

    # Drop columns that were used to create derived features
    cols_to_drop = [
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_grade",
    ]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df

def impute_numeric_median(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Impute missing numeric values with median."""
    df = df.copy()
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
    """Apply winsorization using pre‑computed bounds."""
    df = df.copy()
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(low, high)
    return df

def align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Ensure DataFrame has all expected features in correct order."""
    df = df.copy()
    # Add missing columns with default 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    # Ensure correct order
    return df[feature_names]

# ---------------------------------------------------------------------
# 5. Prediction pipeline
# ---------------------------------------------------------------------
def predict_from_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full prediction pipeline:
        - Normalize columns
        - Feature engineering
        - Impute missing numeric values with median
        - Winsorization using bounds
        - Align features
        - Predict
        - Add probability columns and risk level
    """
    model, bounds, numeric_cols, feature_names, label_encoder = load_artifacts()

    # 1. Normalize column names
    df = normalize_columns(input_df)

    # 2. Feature engineering (adds derived columns)
    df = engineer_features(df)

    # 3. Separate features for prediction
    features_df = df.drop(columns=["Status"], errors="ignore")

    # 4. Impute missing values
    features_df = impute_numeric_median(features_df, numeric_cols)

    # 5. Cap extreme values
    features_df = cap_numeric_bounds(features_df, bounds)

    # 6. Align to training features
    features_df = align_features(features_df, feature_names)

    # 7. Predict
    pred_codes = model.predict(features_df)
    pred_proba = model.predict_proba(features_df)
    pred_labels = label_encoder.inverse_transform(pred_codes)

    # 8. Build output DataFrame
    output_df = input_df.copy()
    # Add normalized and engineered columns for readability (optional)
    output_df = normalize_columns(output_df)
    output_df = engineer_features(output_df)   # this will add derived features again

    output_df["Predicted_Status_Code"] = pred_codes
    output_df["Predicted_Status"] = pred_labels

    # Map class indices (model classes are [0,1] where 1=Graduate, 0=Dropout)
    class_to_idx = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    output_df["Prob_Dropout"] = pred_proba[:, class_to_idx.get(0, 0)]
    output_df["Prob_Graduate"] = pred_proba[:, class_to_idx.get(1, 1)]
    output_df["Prediction_Confidence"] = np.max(pred_proba, axis=1)

    # Risk level based on dropout probability (as in notebook: High >0.7, Medium 0.3‑0.7, Low <0.3)
    def risk_level(prob_dropout):
        if pd.isna(prob_dropout):
            return "Unknown"
        if prob_dropout > 0.7:
            return "High"
        if prob_dropout > 0.3:
            return "Medium"
        return "Low"

    output_df["Risk_Level"] = output_df["Prob_Dropout"].apply(risk_level)

    # Round floats to 3 decimals for readability
    for col in output_df.select_dtypes(include=[float]).columns:
        output_df[col] = output_df[col].round(3)

    return output_df

# ---------------------------------------------------------------------
# 6. Manual prediction helper
# ---------------------------------------------------------------------
def create_manual_input_df(**kwargs) -> pd.DataFrame:
    """Create a DataFrame from manual inputs."""
    # Use all feature names; missing will be set to 0 later in pipeline
    data = {col: [kwargs.get(col, 0)] for col in feature_names}
    return pd.DataFrame(data)

# ---------------------------------------------------------------------
# 7. Streamlit UI
# ---------------------------------------------------------------------
st.title("Student Dropout Predictor")
st.caption("Predicts whether a student will Dropout or Graduate based on academic, demographic, and socio‑economic features.")

# Tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Upload CSV File", "Manual Input", "Sample File"])

# ---------------------------------------------------------------------
# Tab 1: Upload CSV
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Predict from CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the file
        try:
            df_in = pd.read_csv(uploaded_file, sep=";")
        except:
            df_in = pd.read_csv(uploaded_file)

        st.success(f"Loaded {len(df_in):,} rows × {len(df_in.columns):,} columns.")
        if st.button("Process uploaded file"):
            try:
                pred_df = predict_from_dataframe(df_in)

                # Summary metrics
                total = len(pred_df)
                dropout_count = int((pred_df["Predicted_Status"] == "Dropout").sum())
                graduate_count = int((pred_df["Predicted_Status"] == "Graduate").sum())

                c1, c2, c3 = st.columns(3)
                c1.metric("Total students", f"{total:,}")
                c2.metric("Predicted Dropout", f"{dropout_count:,}")
                c3.metric("Predicted Graduate", f"{graduate_count:,}")

                st.subheader("Preview (first 25 rows)")
                st.dataframe(pred_df.head(25), use_container_width=True)

                st.download_button(
                    "Download predictions as CSV",
                    data=pred_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                st.subheader("Prediction Distribution")
                dist = pred_df["Predicted_Status"].value_counts().reset_index()
                dist.columns = ["Predicted_Status", "Count"]
                st.bar_chart(dist.set_index("Predicted_Status"))

            except Exception as e:
                st.exception(e)

# ---------------------------------------------------------------------
# Tab 2: Manual Input
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Predict for a Single Student")
    # Create a form for manual input of key features
    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age at enrollment", min_value=15, max_value=100, value=20)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            marital = st.selectbox("Marital Status", options=[1, 2, 3, 4], format_func=lambda x: {1: "Single", 2: "Married", 3: "Divorced", 4: "Widowed"}[x])
            debtor = st.selectbox("Debtor (has outstanding fees)", options=[0, 1])
            scholarship = st.selectbox("Scholarship holder", options=[0, 1])
            tuition_up_to_date = st.selectbox("Tuition fees up to date", options=[0, 1])

        with col2:
            admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, value=130.0)
            curricular_units_1st_sem_credited = st.number_input("Curricular units 1st sem credited", min_value=0, value=0)
            curricular_units_1st_sem_enrolled = st.number_input("Curricular units 1st sem enrolled", min_value=0, value=5)
            curricular_units_1st_sem_evaluations = st.number_input("Curricular units 1st sem evaluations", min_value=0, value=5)
            curricular_units_1st_sem_approved = st.number_input("Curricular units 1st sem approved", min_value=0, value=5)
            curricular_units_1st_sem_grade = st.number_input("Curricular units 1st sem grade", min_value=0.0, max_value=20.0, value=12.0)

        # Additional features can be added if needed, but these are the main ones.
        # We'll use the pipeline to handle missing features (set to 0).
        submitted = st.form_submit_button("Predict")
        if submitted:
            # Build a dictionary of all features expected by the model
            # For simplicity, we'll create a DataFrame with all required columns, filling missing with 0
            # But we'll set the ones we have.
            # Load feature names
            _, _, _, feature_names, _ = load_artifacts()
            # Create an empty DataFrame with all features
            data_dict = {col: [0] for col in feature_names}
            # Fill in the ones we have
            data_dict["Age_at_enrollment"] = [age]
            data_dict["Gender"] = [1 if gender == "Male" else 0]  # Assuming 1=Male, 0=Female
            data_dict["Marital_status"] = [marital]
            data_dict["Debtor"] = [debtor]
            data_dict["Scholarship_holder"] = [scholarship]
            data_dict["Tuition_fees_up_to_date"] = [tuition_up_to_date]
            data_dict["Admission_grade"] = [admission_grade]
            data_dict["Curricular_units_1st_sem_credited"] = [curricular_units_1st_sem_credited]
            data_dict["Curricular_units_1st_sem_enrolled"] = [curricular_units_1st_sem_enrolled]
            data_dict["Curricular_units_1st_sem_evaluations"] = [curricular_units_1st_sem_evaluations]
            data_dict["Curricular_units_1st_sem_approved"] = [curricular_units_1st_sem_approved]
            data_dict["Curricular_units_1st_sem_grade"] = [curricular_units_1st_sem_grade]

            # Create DataFrame
            input_df = pd.DataFrame(data_dict)
            # Run prediction
            try:
                pred_df = predict_from_dataframe(input_df)
                result = pred_df.iloc[0]
                st.success(f"Prediction: **{result['Predicted_Status']}**")
                st.write(f"Probability of Dropout: {result['Prob_Dropout']:.2%}")
                st.write(f"Probability of Graduate: {result['Prob_Graduate']:.2%}")
                st.write(f"Risk Level: {result['Risk_Level']}")
            except Exception as e:
                st.exception(e)

# ---------------------------------------------------------------------
# Tab 3: Sample CSV
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Sample CSV Format")
    st.write("Download a sample CSV file with the correct column names and example data.")
    # Create a sample DataFrame with a few rows
    sample_data = {
        "Marital Status": [1, 2, 1],
        "Application mode": [1, 17, 15],
        "Application order": [1, 1, 2],
        "Course": [33, 171, 8014],
        "Daytime/evening attendance": [1, 1, 0],
        "Previous qualification": [1, 2, 1],
        "Previous qualification (grade)": [130.0, 145.0, 120.0],
        "Nacionality": [1, 1, 1],
        "Mother's qualification": [2, 3, 1],
        "Father's qualification": [2, 3, 1],
        "Mother's occupation": [3, 2, 1],
        "Father's occupation": [3, 2, 1],
        "Admission grade": [130.0, 145.0, 120.0],
        "Displaced": [0, 0, 1],
        "Educational special needs": [0, 0, 0],
        "Debtor": [0, 0, 1],
        "Tuition fees up to date": [1, 1, 0],
        "Gender": [1, 1, 0],
        "Scholarship holder": [0, 1, 0],
        "Age at enrollment": [19, 22, 20],
        "International": [0, 0, 0],
        "Curricular units 1st sem (credited)": [0, 0, 0],
        "Curricular units 1st sem (enrolled)": [5, 6, 4],
        "Curricular units 1st sem (evaluations)": [5, 6, 4],
        "Curricular units 1st sem (approved)": [5, 6, 4],
        "Curricular units 1st sem (grade)": [12.5, 14.2, 10.0],
        "Curricular units 1st sem (without evaluations)": [0, 0, 0],
        "Curricular units 2nd sem (credited)": [0, 0, 0],
        "Curricular units 2nd sem (enrolled)": [5, 6, 4],
        "Curricular units 2nd sem (evaluations)": [5, 6, 4],
        "Curricular units 2nd sem (approved)": [5, 6, 4],
        "Curricular units 2nd sem (grade)": [12.5, 14.2, 10.0],
        "Curricular units 2nd sem (without evaluations)": [0, 0, 0],
        "Unemployment rate": [10.0, 9.5, 11.0],
        "Inflation rate": [2.5, 2.3, 2.7],
        "GDP": [0.5, 0.6, 0.4],
        "Status": ["Graduate", "Graduate", "Dropout"],  # optional, will be ignored
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    st.download_button(
        "Download Sample CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_input.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# (Optional) legacy file path input (kept for compatibility)
# ---------------------------------------------------------------------
with st.expander("Advanced: Specify file path (alternative)"):
    csv_path = st.text_input("Input CSV file path", value="")
    if st.button("Process file from path") and csv_path:
        try:
            input_path = Path(csv_path)
            if not input_path.exists():
                st.error(f"File not found: {input_path.resolve()}")
            else:
                df_in = read_data_file(input_path)
                st.success(f"Loaded {len(df_in):,} rows × {len(df_in.columns):,} columns.")
                pred_df = predict_from_dataframe(df_in)
                # Show results (similar to above)
                # ... (omitted for brevity, similar to tab1)
        except Exception as e:
            st.exception(e)