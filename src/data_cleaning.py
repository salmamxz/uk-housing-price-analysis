## Data cleaning and preprocessing pipeline for London House Price dataset.
## Prepares clean datasets for:
#   Time-series market trend analysis
#   Machine learning price regression


from pathlib import Path
import numpy as np
import pandas as pd


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    #Load the raw CSV dataset.
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    # Clean, filter, and feature-engineer the property dataset.
    df_clean = df.copy()

    # 1. Standardise column names
    df_clean.columns = df_clean.columns.str.strip()

    # 2. Drop duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"[INFO] Dropped {initial_rows - len(df_clean):,} duplicate rows")

    # 3. Identify and validate target price
    # Supports either 'price' or 'saleEstimate_currentPrice'
    price_col = "saleEstimate_currentPrice" if "saleEstimate_currentPrice" in df_clean.columns else "price"
    df_clean = df_clean.rename(columns={price_col: "price"})

    # Ensure price is numeric and filter extreme anomalies/non-residential outliers
    df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")
    df_clean = df_clean.dropna(subset=["price"])
    # Typical residential bounds for London: £50k to £15M
    df_clean = df_clean[(df_clean["price"] >= 50_000) & (df_clean["price"] <= 15_000_000)]

    # 4. Clean physical feature columns
    numeric_cols = ["floorAreaSqM", "bedrooms", "bathrooms", "livingRooms"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Filter physically plausible values
    if "floorAreaSqM" in df_clean.columns:
        df_clean = df_clean[(df_clean["floorAreaSqM"].isna()) |
                            ((df_clean["floorAreaSqM"] >= 15) & (df_clean["floorAreaSqM"] <= 1000))]

    if "bedrooms" in df_clean.columns:
        df_clean = df_clean[(df_clean["bedrooms"].isna()) |
                            ((df_clean["bedrooms"] >= 0) & (df_clean["bedrooms"] <= 15))]

    # 5. Clean categorical features
    cat_cols = ["propertyType", "tenure", "outcode"]
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace(["Nan", "None", ""], np.nan)

    # 6. Parse dates for trend analysis (if transaction date exists)
    date_cols = [c for c in df_clean.columns if "date" in c.lower()]
    if date_cols:
        primary_date = date_cols[0]
        df_clean["sale_date"] = pd.to_datetime(df_clean[primary_date], errors="coerce")
        df_clean["sale_year"] = df_clean["sale_date"].dt.year
        df_clean["sale_year_month"] = df_clean["sale_date"].dt.to_period("M")

    # 7. Prevent target leakage: drop automated valuation and rental estimate fields from ML features
    leakage_cols = [
        col for col in df_clean.columns
        if any(term in col.lower() for term in ["rentestimate", "saleestimate_valuechange", "confidencelevel"])
    ]
    if leakage_cols:
        print(f"[INFO] Dropping potential leakage columns: {leakage_cols}")
        df_clean = df_clean.drop(columns=leakage_cols)

    print(f"[INFO] Cleaned dataset: {df_clean.shape[0]:,} rows remaining")
    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned data to processed directory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Processed dataset saved to: {output_path}")


if __name__ == "__main__":
    # raw_path = Path("data/raw/kaggle_london_house_price_data.csv")
    processed_path = Path("data/processed/cleaned_london_house_prices.csv")

    #if raw_path.exists():
     #   raw_df = load_raw_data(raw_path)
      ##  clean_df = clean_housing_data(raw_df)
        #save_processed_data(clean_df, processed_path)
   # else:
   #     print(f"[WARNING] File not found at {raw_path}. Update the path or place your file in data/raw/")