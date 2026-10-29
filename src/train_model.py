#Handles Steps 3, 5, and 6:
#- Selects feature set X and target y
#- Splits into Train (80%) and Test (20%)
#- Constructs ColumnTransformer and Pipeline (Imputation + OneHotEncoding + LinearRegression)
# - Fits the model and persists artifacts for evaluation


from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_and_train_model():
    project_root = Path(__file__).resolve().parent.parent
    data_path = (
        project_root / "data" / "processed" / "cleaned_london_house_prices.csv"
    )
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    models_dir.mkdir(exist_ok=True)

    print(f"[INFO] Loading processed dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Step 3: Choose target and predictors
    target = "price"
    numeric_features = [
        col
        for col in ["floorAreaSqM", "bedrooms", "bathrooms", "livingRooms"]
        if col in df.columns
    ]
    categorical_features = [
        col for col in ["propertyType", "tenure", "outcode"] if col in df.columns
    ]

    feature_cols = numeric_features + categorical_features
    X = df[feature_cols]
    y = df[target]

    print(f"[INFO] Predictors ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Target: {target}")

    # Step 5: Split train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(
        f"[INFO] Training set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows"
    )

    # Step 6: Build preprocessing & model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    print("[INFO] Fitting Linear Regression pipeline...")
    pipeline.fit(X_train, y_train)

    # Save artifacts for evaluate_model.py
    joblib.dump(pipeline, models_dir / "baseline_linear_model.joblib")

    # Persist the test set so evaluation runs on identical data
    test_data = X_test.copy()
    test_data["actual_price"] = y_test
    test_data.to_csv(processed_dir / "test_set_eval.csv", index=False)

    print(f"[SUCCESS] Trained model saved to: {models_dir / 'baseline_linear_model.joblib'}")
    print(f"[SUCCESS] Held-out test set saved to: {processed_dir / 'test_set_eval.csv'}")


if __name__ == "__main__":
    build_and_train_model()