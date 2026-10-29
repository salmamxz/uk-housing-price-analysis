from pathlib import Path
import re
import numpy as np
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "London.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "london_cleaned.csv"


def clean_london_data():

    ## Cleans raw London housing data and prepares it for regression and trend analysis.
    ##Covers:
    ##  - Task 1: Loading raw data and understanding schema
    ##  - Task 2: Cleaning nulls, fixing text fields, and trimming outliers
    ##  - Task 3: Feature engineering for X/y selection


    # Task 1: Get & understand the data

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file missing at: {RAW_PATH}")

    print(f"Loading raw data from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Initial raw records: {df.shape[0]} rows, {df.shape[1]} columns")

    # Standardise column headers to snake_case for consistency
    df.columns = [
        col.strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        for col in df.columns
    ]

    # Shorter variable names for modelling
    rename_dict = {
        "area_in_sq_ft": "area_sqft",
        "no_of_bedrooms": "bedrooms",
        "no_of_bathrooms": "bathrooms",
        "no_of_receptions": "receptions",
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})


    # Task 2: Explore & clean the data

    # Strip whitespace and treat empty strings as null
    text_cols = ["house_type", "location", "city_county", "postal_code"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    # The dataset has roughly 28% missing locations.
    # Pull the outward postcode (e.g. SW19, E1W, CR0) to retain geographical information.
    def extract_outward(pc):
        if pd.isna(pc) or pc == "nan":
            return "Unknown"
        match = re.match(r"^([A-Z]{1,2}\d[A-Z\d]?)", str(pc).upper().strip())
        return match.group(1) if match else "Unknown"

    df["outward_code"] = df["postal_code"].apply(extract_outward)

    # Imputation hierarchy: Location -> Outward Code -> County -> 'London'
    df["location_clean"] = df["location"].fillna(df["outward_code"])
    df["location_clean"] = (
        df["location_clean"]
        .replace("Unknown", np.nan)
        .fillna(df["city_county"])
        .fillna("London")
    )

    # Avoid high cardinality in OneHotEncoding later by grouping small areas (<12 listings) into 'Other'
    loc_counts = df["location_clean"].value_counts()
    top_locs = loc_counts[loc_counts >= 12].index
    df["location_group"] = df["location_clean"].apply(lambda x: x if x in top_locs else "Other")

    # Group rare property types to keep the category levels tidy
    main_types = ["House", "Flat / Apartment", "New development", "Duplex", "Penthouse"]
    df["house_type_clean"] = df["house_type"].apply(lambda x: x if x in main_types else "Other")

    # Cast room counts and prices to numeric, dealing with any messy strings
    numeric_cols = ["price", "area_sqft", "bedrooms", "bathrooms", "receptions"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any missing room values with minimum baseline of 1
    df["bedrooms"] = df["bedrooms"].fillna(1).astype(int)
    df["bathrooms"] = df["bathrooms"].fillna(1).astype(int)
    df["receptions"] = df["receptions"].fillna(1).astype(int)


    # Task 3: Feature engineering (Preparing X and y)

    # Room counts tend to be collinear; total_rooms acts as a combined capacity metric
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["receptions"]
    df["price_per_sqft"] = (df["price"] / df["area_sqft"]).round(2)


    # Task 2 (cont.): Filter severe outliers
    # Filter non-viable listings and ultra-luxury homes (>£12m) that break OLS residual normality
    initial_count = len(df)
    df = df[
        (df["price"] >= 150_000) & (df["price"] <= 12_000_000) &
        (df["area_sqft"] >= 250) & (df["area_sqft"] <= 10_000) &
        (df["bedrooms"] <= 10) & (df["bathrooms"] <= 8)
    ]
    dropped = initial_count - len(df)
    print(f"Removed {dropped} outlier rows ({dropped / initial_count:.2%})")

    # Export clean dataset for exploratory analysis and modelling
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset to: {PROCESSED_PATH}")
    print(f"Final dataset dimensions: {df.shape}")


if __name__ == "__main__":
    clean_london_data()