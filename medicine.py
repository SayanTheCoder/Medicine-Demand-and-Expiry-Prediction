from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from clean_data import clean_dataset

RAW_FILE = ROOT / "data" / "raw" / "medicine_demand_expiry_raw.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "medicine_demand_expiry_cleaned.csv"

st.set_page_config(page_title="Medicine Demand & Expiry Prediction", page_icon="💊", layout="wide")

FEATURES = [
    "Price", "Current_Stock", "Previous_Stock", "Quantity_Sold",
    "Quantity_Used", "Supplier_Lead_Days", "Demand_Lag_1",
    "Demand_Lag_2", "Demand_Lag_3", "Demand_Rolling_3",
    "Forecast_Month", "Forecast_Year", "Category"
]
TARGET = "Next_Month_Demand"

@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_FILE)

@st.cache_data
def load_data():
    if not CLEAN_FILE.exists():
        clean_dataset(RAW_FILE, CLEAN_FILE)
    return pd.read_csv(CLEAN_FILE, parse_dates=["Record_Date", "Expiry_Date"])

def build_model(model_name):
    numeric_features = [c for c in FEATURES if c != "Category"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("category", categorical_pipe, ["Category"])
    ])
    models = {
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=7, min_samples_leaf=4, random_state=42
        ),
        "Multiple Linear Regression": LinearRegression(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=2, random_state=42
        )
    }
    return Pipeline([("preprocess", prep), ("model", models[model_name])])

def train_and_compare(df):
    usable = df.dropna(subset=[TARGET]).copy().sort_values("Record_Date")
    unique_dates = np.sort(usable["Record_Date"].dropna().unique())
    if len(unique_dates) < 5:
        raise ValueError("Not enough dated records for a chronological train/test split.")
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train = usable[usable["Record_Date"] < split_date]
    test = usable[usable["Record_Date"] >= split_date]
    if train.empty or test.empty:
        raise ValueError("Unable to create chronological train/test data.")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    results, fitted = [], {}
    for name in [
        "Decision Tree Regressor",
        "Multiple Linear Regression",
        "Gradient Boosting Regressor"
    ]:
        pipe = build_model(name)
        pipe.fit(X_train, y_train)
        pred = np.maximum(0, pipe.predict(X_test))
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R² Score": r2_score(y_test, pred)
        })
        fitted[name] = pipe
    return pd.DataFrame(results).sort_values("RMSE"), fitted, split_date

def expiry_label(days):
    if days < 0:
        return "Expired"
    if days <= 30:
        return "Critical (0-30 days)"
    if days <= 90:
        return "Near Expiry (31-90 days)"
    return "Safe"

st.title("💊 Medicine Demand and Expiry Prediction")
st.caption("Forecast next-month medicine demand and generate upcoming expiry lists.")

try:
    raw_df = load_raw_data()
    df = load_data()
except Exception as e:
    st.error(f"Could not load project dataset: {e}")
    st.stop()

today = pd.Timestamp.today().normalize()
latest_records = (
    df.sort_values("Record_Date")
      .groupby("Medicine_ID", as_index=False)
      .tail(1)
      .copy()
)
latest_records["Days_To_Expiry"] = (latest_records["Expiry_Date"] - today).dt.days
latest_records["Expiry_Status"] = latest_records["Days_To_Expiry"].apply(expiry_label)

# Dashboard summary
c1, c2, c3 = st.columns(3)
c1.metric("Cleaned records", f"{len(df):,}")
c2.metric("Medicines", f"{df['Medicine_ID'].nunique():,}")
c3.metric("Forecast training rows", f"{df[TARGET].notna().sum():,}")

st.subheader("1. Raw and cleaned datasets")
st.write("Compare the original source records with the dataset prepared for analysis.")
raw_tab, cleaned_tab = st.tabs(["Raw data", "Cleaned data"])
with raw_tab:
    st.caption(f"Showing all {len(raw_df):,} raw records and {len(raw_df.columns)} columns.")
    st.dataframe(
        raw_df,
        width="stretch",
        height=520,
        hide_index=True,
        alt="Complete raw medicine demand and expiry dataset",
    )
with cleaned_tab:
    st.caption(f"Showing all {len(df):,} cleaned records and {len(df.columns)} columns.")
    st.dataframe(
        df,
        width="stretch",
        height=520,
        hide_index=True,
        alt="Complete cleaned medicine demand and expiry dataset",
    )

st.subheader("2. Historical demand analysis")
demand_by_month = df.groupby("Record_Date", as_index=False)["Monthly_Demand"].sum()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(demand_by_month["Record_Date"], demand_by_month["Monthly_Demand"])
ax.set_title("Total Recorded Monthly Medicine Demand")
ax.set_xlabel("Record month")
ax.set_ylabel("Units demanded")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("3. Train and compare the three models")
try:
    metrics, fitted_models, split_date = train_and_compare(df)
    st.dataframe(
        metrics.round(3),
        hide_index=True,
        alt="Performance metrics comparing the three demand forecasting models",
    )
    best_name = metrics.iloc[0]["Model"]
    st.success(f"Best model by test RMSE: {best_name}")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(metrics["Model"], metrics["RMSE"])
    ax.set_title("Model comparison – RMSE (lower is better)")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

st.subheader("4. Predict next month's medicine demand")
medicine_names = sorted(df["Medicine_Name"].dropna().unique())
selected_name = st.selectbox("Select medicine name", medicine_names)
selected_rows = df[df["Medicine_Name"] == selected_name]
selected_id = selected_rows["Medicine_ID"].iloc[0]
st.text_input("Medicine ID (auto-generated)", value=selected_id, disabled=True)

latest = selected_rows.sort_values("Record_Date").tail(1).copy()
if latest[["Demand_Lag_1", "Demand_Lag_2", "Demand_Lag_3"]].isna().any(axis=None):
    st.warning("Not enough historical months are available for a three-month lag-based forecast.")
else:
    best_model = fitted_models[best_name]
    forecast = max(0, float(best_model.predict(latest[FEATURES])[0]))
    forecast_month = pd.Timestamp(latest["Record_Date"].iloc[0]) + pd.offsets.MonthBegin(1)
    a, b, c = st.columns(3)
    a.metric("Predicted next-month demand", f"{forecast:,.0f} units")
    b.metric("Current stock", f"{latest['Current_Stock'].iloc[0]:,.0f} units")
    c.metric("Estimated stock gap", f"{max(0, forecast-latest['Current_Stock'].iloc[0]):,.0f} units")
    st.caption(f"Forecast month: {forecast_month.strftime('%B %Y')} | Model: {best_name}")
    if latest["Current_Stock"].iloc[0] < forecast:
        st.warning("Current stock is below the predicted next-month demand. Review replenishment.")
    else:
        st.info("Current stock is at or above the predicted next-month demand.")

st.subheader("5. Generate medicines expiring this month or next month")
st.write("Choose a month and click **Generate** to list medicines whose latest recorded batch expiry date falls within that calendar month.")
month_choice = st.selectbox(
    "Select expiry period",
    ["This Month", "Next Month"],
    index=0
)
if month_choice == "This Month":
    period_start = today.replace(day=1)
else:
    period_start = (today.replace(day=1) + pd.offsets.MonthBegin(1))
period_end = period_start + pd.offsets.MonthBegin(1)

if st.button("Generate", type="primary"):
    expiry_list = latest_records[
        (latest_records["Expiry_Date"] >= period_start) &
        (latest_records["Expiry_Date"] < period_end)
    ].copy()
    expiry_list = expiry_list.sort_values("Expiry_Date")
    period_label = period_start.strftime("%B %Y")
    st.markdown(f"**Expiry report: {period_label}**")
    st.metric("Medicines expiring in selected month", int(len(expiry_list)))
    if expiry_list.empty:
        st.info(f"No medicines are scheduled to expire in {period_label} in the current dataset.")
    else:
        expiry_output = expiry_list[[
            "Medicine_ID", "Medicine_Name", "Category", "Current_Stock",
            "Expiry_Date", "Days_To_Expiry", "Expiry_Status"
        ]].reset_index(drop=True)
        st.dataframe(
            expiry_output,
            hide_index=True,
            alt="Medicines expiring during the selected month",
        )
        st.caption(f"Generated {len(expiry_output)} medicine record(s) for {period_label}.")
