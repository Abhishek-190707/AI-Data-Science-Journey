
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #f4f7fb;
    border: 1px solid #e3eaf3;
    padding: 14px 16px;
    border-radius: 12px;
}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance Prediction")
st.caption("Explore student data, evaluate a Random Forest model, and estimate a student's final Mathematics grade.")

EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"
]
TARGET = "G3"

@st.cache_data
def load_data(uploaded_file, local_file_mtime):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        data = pd.read_csv("student-mat.csv", sep=";")
    data.columns = [c.strip() for c in data.columns]
    return data

@st.cache_resource
def train_model(data):
    X = data.drop(columns=[TARGET]).copy()
    y = data[TARGET].copy()

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=1,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R²": r2_score(y_test, predictions)
    }
    return model, X, y, X_train, X_test, y_train, y_test, predictions, metrics, categorical_cols, numeric_cols

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader(
        "Upload student-mat.csv",
        type=["csv"],
        help="Use the Mathematics file from the UCI Student Performance dataset."
    )
    st.caption("If student-mat.csv is in the same folder as app.py, it loads automatically.")
    st.divider()
    st.markdown("**Target:** `G3` (final Mathematics grade)")
    st.markdown("**Dataset:** UCI Student Performance – Mathematics")

try:
    data = load_data(uploaded, None)
except FileNotFoundError:
    st.warning("Upload `student-mat.csv` using the sidebar to start the dashboard.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

missing_cols = [c for c in EXPECTED_COLUMNS if c not in data.columns]
if missing_cols:
    st.error(
        "This does not look like the UCI Mathematics dataset. "
        "Missing columns: " + ", ".join(missing_cols)
    )
    st.stop()

if data.empty or TARGET not in data.columns:
    st.error("The dataset is empty or does not contain the target column G3.")
    st.stop()

data = data[EXPECTED_COLUMNS].copy()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype(str).str.strip()

if data[TARGET].isna().any():
    st.error("The target column G3 contains missing values. Please check the dataset.")
    st.stop()

try:
    (model, X, y, X_train, X_test, y_train, y_test, predictions,
     metrics, categorical_cols, numeric_cols) = train_model(data)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "📊 Student Data Analysis",
    "🧪 Model Performance",
    "🔮 Grade Prediction"
])

with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student records", f"{len(data):,}")
    c2.metric("Input features", f"{len(X.columns)}")
    c3.metric("Average final grade", f"{data[TARGET].mean():.2f} / 20")
    c4.metric("Median final grade", f"{data[TARGET].median():.1f} / 20")

    left, right = st.columns(2)
    with left:
        grade_counts = data[TARGET].value_counts().sort_index().rename_axis("Final grade").reset_index(name="Students")
        fig = px.bar(
            grade_counts, x="Final grade", y="Students",
            title="Distribution of final Mathematics grades (G3)",
            color="Students", color_continuous_scale="Blues"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        study_summary = data.groupby("studytime", as_index=False)[TARGET].mean()
        fig = px.bar(
            study_summary, x="studytime", y=TARGET,
            title="Average final grade by weekly study time",
            labels={"studytime": "Study time category", TARGET: "Average G3"},
            text_auto=".2f", color=TARGET, color_continuous_scale="Teal"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        data, x="absences", y=TARGET, color="studytime",
        title="Absences vs final grade",
        labels={"absences": "Number of absences", TARGET: "Final grade (G3)", "studytime": "Study time"}
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview dataset"):
        st.dataframe(data.head(10), use_container_width=True)

with tab2:
    st.subheader("Random Forest model evaluation")
    st.caption("The dashboard trains a fresh Random Forest model using an 80/20 split and random_state=42. Its scores may differ slightly from the notebook if preprocessing or model settings differ.")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.3f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c3.metric("R² score", f"{metrics['R²']:.3f}")

    result_frame = pd.DataFrame({
        "Actual G3": y_test.to_numpy(),
        "Predicted G3": predictions
    })
    fig = px.scatter(
        result_frame, x="Actual G3", y="Predicted G3",
        title="Actual vs predicted final grades",
        opacity=0.75
    )
    fig.add_trace(go.Scatter(
        x=[0, 20], y=[0, 20], mode="lines",
        name="Perfect prediction", line=dict(dash="dash")
    ))
    fig.update_xaxes(range=[0, 20])
    fig.update_yaxes(range=[0, 20])
    st.plotly_chart(fig, use_container_width=True)

    residuals = result_frame["Actual G3"] - result_frame["Predicted G3"]
    residual_frame = pd.DataFrame({"Predicted G3": predictions, "Residual": residuals})
    fig = px.scatter(
        residual_frame, x="Predicted G3", y="Residual",
        title="Residual analysis",
        labels={"Residual": "Residual (actual − predicted)"}
    )
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    # Permutation importance is calculated only when requested to keep initial load quick.
    if st.button("Calculate permutation feature importance"):
        from sklearn.inspection import permutation_importance
        with st.spinner("Calculating permutation importance..."):
            result = permutation_importance(
                model, X_test, y_test, n_repeats=10,
                random_state=42, scoring="r2", n_jobs=-1
            )
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance (R² decrease)": result.importances_mean
        }).sort_values("Importance (R² decrease)", ascending=True)
        fig = px.bar(
            importance.tail(15), x="Importance (R² decrease)", y="Feature",
            orientation="h", title="Top 15 permutation importances"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Permutation importance measures how much the model score changes when one feature is shuffled. It does not establish causation.")

with tab3:
    st.subheader("Estimate a student's final Mathematics grade")
    st.info("Enter the student's available information. The model returns an estimate, not an official grade or a guaranteed outcome.")

    defaults = {}
    for col in numeric_cols:
        defaults[col] = float(pd.to_numeric(X[col], errors="coerce").median())
    for col in categorical_cols:
        modes = X[col].mode(dropna=True)
        defaults[col] = str(modes.iloc[0]) if not modes.empty else str(X[col].dropna().iloc[0])

    with st.form("prediction_form"):
        st.markdown("#### Academic information")
        academic_cols = st.columns(4)
        user_values = {}
        for i, col in enumerate(["G1", "G2", "studytime", "failures", "absences", "age", "Medu", "Fedu"]):
            if col in X.columns:
                with academic_cols[i % 4]:
                    min_v = float(pd.to_numeric(X[col], errors="coerce").min())
                    max_v = float(pd.to_numeric(X[col], errors="coerce").max())
                    default_v = float(defaults[col])
                    user_values[col] = st.number_input(
                        col, min_value=min_v, max_value=max_v,
                        value=min(max(default_v, min_v), max_v), step=1.0,
                        help=f"Observed range in this dataset: {min_v:g} to {max_v:g}"
                    )

        st.markdown("#### Student and family information")
        categorical_inputs = st.columns(3)
        for i, col in enumerate(categorical_cols):
            options = sorted(X[col].dropna().astype(str).unique().tolist())
            if options:
                with categorical_inputs[i % 3]:
                    default = defaults[col] if defaults[col] in options else options[0]
                    user_values[col] = st.selectbox(col, options, index=options.index(default))

        # Any remaining numeric fields are presented in a compact advanced section.
        remaining_numeric = [c for c in numeric_cols if c not in user_values]
        if remaining_numeric:
            with st.expander("Other numeric inputs"):
                cols = st.columns(4)
                for i, col in enumerate(remaining_numeric):
                    min_v = float(pd.to_numeric(X[col], errors="coerce").min())
                    max_v = float(pd.to_numeric(X[col], errors="coerce").max())
                    default_v = float(defaults[col])
                    with cols[i % 4]:
                        user_values[col] = st.number_input(
                            col, min_value=min_v, max_value=max_v,
                            value=min(max(default_v, min_v), max_v), step=1.0
                        )

        submitted = st.form_submit_button("Predict final grade", type="primary")

    if submitted:
        input_row = pd.DataFrame([{col: user_values[col] for col in X.columns}])
        prediction = float(model.predict(input_row)[0])
        prediction = min(max(prediction, 0), 20)
        st.success(f"Estimated final Mathematics grade: **{prediction:.1f} / 20**")
        st.caption("This estimate is based on patterns in the historical dataset. It should not be used for consequential decisions about a student.")

st.divider()
st.caption("Built with Streamlit, Pandas, Plotly and Scikit-learn • Dataset: UCI Student Performance (Mathematics)")
