from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "medicine_demand_expiry_raw.csv"
PROCESSED_FILE = ROOT / "data" / "processed" / "medicine_demand_expiry_cleaned.csv"

REQUIRED_COLUMNS = [
    "Record_Date", "Medicine_ID", "Medicine_Name", "Category", "Price",
    "Current_Stock", "Previous_Stock", "Quantity_Sold", "Quantity_Used",
    "Supplier_Lead_Days", "Expiry_Date", "Monthly_Demand"
]

def clean_dataset(input_file=RAW_FILE, output_file=PROCESSED_FILE):
    df = pd.read_csv(input_file)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.drop_duplicates()
    for col in ["Medicine_ID", "Medicine_Name", "Category"]:
        df[col] = df[col].astype("string").str.strip()
    df["Medicine_Name"] = df["Medicine_Name"].str.title()
    df["Category"] = df["Category"].replace("", pd.NA).fillna("Other")

    for col in ["Record_Date", "Expiry_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = [
        "Price", "Current_Stock", "Previous_Stock", "Quantity_Sold",
        "Quantity_Used", "Supplier_Lead_Days", "Monthly_Demand"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby("Medicine_ID")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].clip(lower=0)

    df = df.dropna(subset=["Record_Date", "Expiry_Date", "Medicine_ID"])
    df = df.sort_values(["Medicine_ID", "Record_Date"]).reset_index(drop=True)

    # Forecasting features: only information available at the current record date.
    grouped = df.groupby("Medicine_ID", group_keys=False)
    df["Demand_Lag_1"] = grouped["Monthly_Demand"].shift(1)
    df["Demand_Lag_2"] = grouped["Monthly_Demand"].shift(2)
    df["Demand_Lag_3"] = grouped["Monthly_Demand"].shift(3)
    df["Demand_Rolling_3"] = grouped["Monthly_Demand"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    next_month = df["Record_Date"] + pd.offsets.MonthBegin(1)
    df["Forecast_Month"] = next_month.dt.month
    df["Forecast_Year"] = next_month.dt.year

    # Target is the next month's observed demand, not the demand already in this row.
    df["Next_Month_Demand"] = grouped["Monthly_Demand"].shift(-1)

    # Keep the earliest rows for history/monitoring; ML rows need lag history and target.
    df["Expiry_Days_Remaining"] = (df["Expiry_Date"] - df["Record_Date"]).dt.days
    df["Expiry_Status"] = np.select(
        [
            df["Expiry_Days_Remaining"] < 0,
            df["Expiry_Days_Remaining"] <= 30,
            df["Expiry_Days_Remaining"] <= 90
        ],
        ["Expired", "Critical (0-30 days)", "Near Expiry (31-90 days)"],
        default="Safe"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Raw rows after cleaning: {len(df):,}")
    print(f"Rows usable for forecasting: {df['Next_Month_Demand'].notna().sum():,}")
    print(f"Saved cleaned dataset: {output_file}")
    return df

if __name__ == "__main__":
    clean_dataset()
